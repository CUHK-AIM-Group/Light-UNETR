import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CGLU(nn.Module):
    def __init__(self, channels):
        super(CGLU, self).__init__()
        expansion = 4
        self.identity = nn.Conv3d(
            channels, channels * expansion // 2, kernel_size=1, bias=False
        )
        self.gating = nn.Conv3d(
            channels, channels * expansion // 2, kernel_size=1, bias=False
        )
        self.act = nn.Sigmoid()
        self.recover = nn.Conv3d(
            channels * expansion // 2, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.identity(x) * self.act(self.gating(x))
        x = self.recover(x)
        return x
    

class SpatialRecover(nn.Module):
    def __init__(self, channels, r):
        super(SpatialRecover, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(channels, channels, kernel_size=r, stride=r, groups=channels),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class LightweightDimensionReductiveAttention(nn.Module):
    def __init__(self, channels, r, heads, attn_scale=False):
        super(LightweightDimensionReductiveAttention, self).__init__()
        self.head_dim = channels // (3*heads)
        self.scale = self.head_dim**-0.5
        self.attn_scale = attn_scale
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
        # qkv
        self.qkv = nn.Conv3d(channels // 3, (channels // 3) * 3, kernel_size=1, bias=False)
        self.high_freq = torch.nn.Sequential(
            nn.Conv3d(channels // 3, (channels // 2) // 3, kernel_size=1, bias=False),
            nn.BatchNorm3d((channels // 2) // 3),
            nn.GELU(),
            nn.Conv3d((channels // 2) // 3, channels // 3, kernel_size=3, padding=1, groups=channels // 6, bias=False), nn.GELU())
        self.high_freq3 = torch.nn.Sequential(
            nn.Conv3d(channels // 3, (channels // 2) // 3, kernel_size=1, bias=False),
            nn.BatchNorm3d((channels // 2) // 3),
            nn.GELU(),
            nn.Conv3d((channels // 2) // 3, channels // 3, kernel_size=5, padding=2, groups=channels // 6, bias=False), nn.GELU())

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W, Z = x.shape
        x_low, x_high, x_high2 = x.split([C // 3, C // 3, C // 3], dim=1)
        q, k, v = (
            self.qkv(x_low)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        if self.attn_scale:
            # attention with seq_len aware scaling
            scale = 1.0 / math.sqrt(self.head_dim * (H * W * Z))
            attn = (q.transpose(-1, -2) @ k) * scale  # (B, H, S, S)
            attn = attn.softmax(dim=-1)
        else:
            attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x_low = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
        x_high = self.high_freq(x_high)
        x_high2 = self.high_freq3(x_high2)
        x = torch.cat([x_low, x_high, x_high2], dim=1)
        return x


class OverlapPatchEmbedding2(nn.Module):
    def __init__(self, in_channels, embed_dim, stride):
        super(OverlapPatchEmbedding2, self).__init__()
        self.proj1 = nn.Conv3d(in_channels, embed_dim // 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=embed_dim // 4)
        self.proj2 = nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=embed_dim // 2)        
        self.proj3 = nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.proj3(x)
        x = self.norm3(x)
        return x

        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class LightUNETRBlock(nn.Module):
    def __init__(self, channels, r, heads):
        super(LightUNETRBlock, self).__init__()

        self.conv0 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.lidr1 = LightweightDimensionReductiveAttention(channels, r, heads)
        self.rec1 = SpatialRecover(channels, r)
        self.cglu1 = CGLU(channels)
        self.lidr2 = LightweightDimensionReductiveAttention(channels, r, heads)
        self.rec2 = SpatialRecover(channels, r)
        self.cglu2 = CGLU(channels)

    def forward(self, x):
        x = self.conv0(x) + x
        x = self.rec1(self.lidr1(x)) + x
        x = self.cglu1(x) + x
        x = self.rec2(self.lidr2(x)) + x
        x = self.cglu2(x) + x
        return x

class LightweightDimensionReductiveAttention_Final(LightweightDimensionReductiveAttention):
    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W, Z = x.shape
        x_low, x_high, x_high2 = x.split([C // 3, C // 3, C // 3], dim=1)
        q, k, v = (
            self.qkv(x_low)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        att_new = attn.detach().clone().mean(dim=(1,2))
        att_new = att_new.view(B, -1, H, W, Z)
        att_new = nn.functional.interpolate(att_new, size=(112, 112, 80), mode='trilinear', align_corners=True) # B, 1, 112, 112, 80
        x_low = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
        x_high = self.high_freq(x_high)
        x_high2 = self.high_freq3(x_high2)
        x = torch.cat([x_low, x_high, x_high2], dim=1)
        return x, att_new

class Block_Final(nn.Module):
    def __init__(self, channels, r, heads):
        super(Block_Final, self).__init__()

        self.conv0 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.lidr1 = LightweightDimensionReductiveAttention(channels, r, heads)
        self.rec1 = SpatialRecover(channels, r)
        self.cglu1 = CGLU(channels)
        self.lidr2 = LightweightDimensionReductiveAttention_Final(channels, r, heads)
        self.rec2 = SpatialRecover(channels, r)
        self.cglu2 = CGLU(channels)


    def forward(self, x):
        x = self.conv0(x) + x
        x = self.rec1(self.lidr1(x)) + x
        x = self.cglu1(x) + x
        x_tmp, att_map = self.lidr2(x)
        x = self.rec2(x_tmp) + x
        x = self.cglu2(x) + x
        return x, att_map

class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DepthwiseConvLayer, self).__init__()
        self.depth_wise = nn.Conv3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.se = SELayer(dim_out)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.norm(x)
        x = self.se(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dim=384,
        embedding_dim=27,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        super(Encoder, self).__init__()
        self.DWconv1 = OverlapPatchEmbedding2(in_channels=in_channels, embed_dim=channels[0], stride=2)
        self.DWconv2 = DepthwiseConvLayer(dim_in=channels[0], dim_out=channels[1], r=2)
        self.DWconv3 = DepthwiseConvLayer(dim_in=channels[1], dim_out=channels[2], r=2)
        self.DWconv4 = DepthwiseConvLayer(dim_in=channels[2], dim_out=embed_dim, r=2)
        block = []
        for _ in range(blocks[0]):
            block.append(LightUNETRBlock(channels=channels[0], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[1]):
            block.append(LightUNETRBlock(channels=channels[1], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[2]):
            block.append(LightUNETRBlock(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[3]):
            block.append(LightUNETRBlock(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, embedding_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden_states_out = []
        x = self.DWconv1(x)
        x = self.block1(x)
        hidden_states_out.append(x)
        x = self.DWconv2(x)
        x = self.block2(x)
        hidden_states_out.append(x)
        x = self.DWconv3(x)
        if x.shape[2:] == (7, 7, 5):
            x = nn.functional.pad(x, (0, 1, 0, 1, 0, 1))
        x = self.block3(x)
        hidden_states_out.append(x)
        x = self.DWconv4(x)
        B, C, W, H, Z = x.shape
        x = self.block4(x)
        x = x.flatten(2).transpose(-1, -2)
        # print(x.shape, self.position_embeddings.shape)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x, hidden_states_out, (B, C, W, H, Z)
    

class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(TransposedConvLayer, self).__init__()
        self.transposed = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.se = SELayer(dim_out)

    def forward(self, x):
        x = self.transposed(x)
        x = self.norm(x)
        x = self.se(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        embed_dim=384,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        super(Decoder, self).__init__()
        self.SegHead = nn.ConvTranspose3d(channels[0], out_channels, kernel_size=4, stride=4)
        self.TSconv3 = TransposedConvLayer(dim_in=channels[1], dim_out=channels[0], r=2)
        self.TSconv2 = TransposedConvLayer(dim_in=channels[2], dim_out=channels[1], r=2)
        self.TSconv1 = TransposedConvLayer(dim_in=embed_dim, dim_out=channels[2], r=2)

        block = []
        for _ in range(blocks[0]):
            block.append(Block_Final(channels=channels[0], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[1]):
            block.append(LightUNETRBlock(channels=channels[1], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[2]):
            block.append(LightUNETRBlock(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[3]):
            block.append(LightUNETRBlock(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)

    def forward(self, x, hidden_states_out, x_shape):
        B, C, W, H, Z = x_shape
        x = x.reshape(B, C, W, H, Z)
        x = self.block4(x)
        x = self.TSconv1(x)
        x = x + hidden_states_out[2]
        x = self.block3(x)
        if x.shape[2:] == (8, 8, 6):
            x = nn.functional.pad(x, (0, -1, 0, -1, 0, -1))
        x = self.TSconv2(x)
        x = x + hidden_states_out[1]
        x = self.block2(x)
        x = self.TSconv3(x)
        x = x + hidden_states_out[0]
        x, att_map = self.block1(x)
        x = self.SegHead(x)
        return x, att_map



class LightUNETR(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        embed_dim=96,
        embedding_dim=27,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            embed_dim: deepest semantic channels
            embedding_dim: position code length
            channels: selection list of downsampling feature channel
            blocks: depth list of slim blocks
            heads: multiple set list of attention computations in parallel
            r: list of stride rate
            dropout: dropout rate
        Examples::
            # for 3D single channel input with size (128, 128, 128), 3-channel output.
            >>> net = SlimUNETR(in_channels=4, out_channels=3, embedding_dim=64)

            # for 3D single channel input with size (96, 96, 96), 2-channel output.
            >>> net = SlimUNETR(in_channels=1, out_channels=2, embedding_dim=27)

        """
        super(LightUNETR, self).__init__()
        self.Encoder = Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )
        self.Decoder = Decoder(
            out_channels=out_channels,
            embed_dim=embed_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )

    def forward(self, x):
        embeding, hidden_states_out, (B, C, W, H, Z) = self.Encoder(x)
        x, att_map = self.Decoder(embeding, hidden_states_out, (B, C, W, H, Z))
        return x
