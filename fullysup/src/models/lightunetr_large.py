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
    def __init__(self, 
                in_channels,
                out_channels,
                exp_r=None,
                kernel_size=None,
                r=2,
                heads=2,
                ):
        super().__init__()
        channels = in_channels
        self.conv0 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.lidr1 = LightweightDimensionReductiveAttention(channels, r, heads)
        self.rec1 = SpatialRecover(channels, r)
        self.cglu1 = CGLU(channels)
        self.lidr2 = LightweightDimensionReductiveAttention(channels, r, heads)
        self.rec2 = SpatialRecover(channels, r)
        self.cglu2 = CGLU(channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x) + x
        x = self.rec1(self.lidr1(x)) + x
        x = self.cglu1(x) + x
        x = self.rec2(self.lidr2(x)) + x
        x = self.cglu2(x) + x
        return x
    

class ConvBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                exp_r:int=4, 
                kernel_size:int=7, 
                n_groups:int or None = None,
                r=0,
                heads=0,
                ):

        super().__init__()

        # conv1-norm1-conv2-act-conv3
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels if n_groups is None else n_groups),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.Conv3d(in_channels, exp_r*in_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv3d(exp_r*in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        x1 = x
        x1 = self.conv_block(x1)
        x1 = x + x1  
        return x1

class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2, groups=in_channels),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.Conv3d(in_channels, exp_r*in_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv3d(exp_r*in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        
        x1 = self.conv_block(x)
        res = self.res_conv(x)
        x1 = x1 + res

        return x1


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2, groups=in_channels),
            nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
            nn.Conv3d(in_channels, exp_r*in_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv3d(exp_r*in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.res_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        
        x1 = self.conv_block(x)
        x1 = torch.nn.functional.pad(x1, (1,0,1,0,1,0))
        res = self.res_conv(x)
        res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
        x1 = x1 + res

        return x1


class LightUNETRLarge(nn.Module):

    def __init__(self, 
        in_channels: int, 
        n_classes: int, 
        embedding_dim: int = 27,
        exp_r: int = 2,
        block_cfg = ['c', 'c', 'l', 'l', 'l', 'l', 'l', 'c', 'c'],
        channels_cfg = 	[32, 60, 96, 150, 198, 150, 96, 60, 32],
        block_counts = [2,2,2,2,2,2,2,2,2],
        head_counts = [0,0,1,2,2,2,1,0,0],
        r_ratios = [4, 2, 2],
    ):

        super().__init__()

        # map block types
        block_mapper = {'c': ConvBlock, 'l': LightUNETRBlock}
        enc_kernel_size = dec_kernel_size = 3
        exp_r = [exp_r for _ in range(len(block_counts))]
            
        # stem
        self.stem = nn.Conv3d(in_channels, channels_cfg[0], kernel_size=1)

        # Small helpers inside __init__ to keep things tidy
        def stage_heads(stage: int) -> int:
            # Preserve original head assignment behavior
            mapping = {0: 2, 1: 2, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 2, 8: 2}
            idx = mapping.get(stage, 2)
            return head_counts[idx] if idx < len(head_counts) else head_counts[-1]

        def stage_r(stage: int) -> int:
            # Preserve original r assignment behavior across stages
            if stage in [0, 1, 2, 7, 8]:
                return r_ratios[0]
            if stage in [3, 5, 6]:
                return r_ratios[1]
            return r_ratios[2]

        def make_seq(stage: int, ksz: int) -> nn.Sequential:
            blk_cls = block_mapper[block_cfg[stage]]
            seq = [
                blk_cls(
                    in_channels=channels_cfg[stage],
                    out_channels=channels_cfg[stage],
                    exp_r=exp_r[stage],
                    kernel_size=ksz,
                    r=stage_r(stage),
                    heads=stage_heads(stage),
                )
                for _ in range(block_counts[stage])
            ]
            return nn.Sequential(*seq)

        # Encoder
        self.enc_block_0 = make_seq(stage=0, ksz=3)
        for i in range(4):
            setattr(
                self,
                f"down_{i}",
                DownBlock(
                    in_channels=channels_cfg[i],
                    out_channels=channels_cfg[i + 1],
                    exp_r=exp_r[i + 1],
                    kernel_size=3 if i == 0 else enc_kernel_size,
                ),
            )
            if i < 3:
                setattr(self, f"enc_block_{i+1}", make_seq(stage=i + 1, ksz=enc_kernel_size))

        # Bottleneck (stage 4)
        self.bottleneck = make_seq(stage=4, ksz=dec_kernel_size)

        # Decoder: up_3 -> dec_block_3, ..., up_0 -> dec_block_0
        for i in range(4):
            # up layers: (in, out) = (channels[4+i], channels[5+i])
            setattr(
                self,
                f"up_{3 - i}",
                UpBlock(
                    in_channels=channels_cfg[4 + i],
                    out_channels=channels_cfg[5 + i],
                    exp_r=exp_r[5 + i],  # match original pattern: up_3->exp_r[5], up_2->exp_r[6], ...
                    kernel_size=dec_kernel_size if i < 3 else 3,
                ),
            )
            # dec blocks use stages 5..8
            dec_stage = 5 + i
            setattr(
                self,
                f"dec_block_{3 - i}",
                make_seq(stage=dec_stage, ksz=dec_kernel_size if i < 3 else 3),
            )

        # Output
        self.out_0 = nn.ConvTranspose3d(channels_cfg[8], n_classes, kernel_size=1)


    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Encoder: collect skip connections in a list
        skips = []
        x = self.enc_block_0(x)
        skips.append(x)
        for i in range(3):  # stages 0->1->2->3
            x = getattr(self, f'down_{i}')(x)
            x = getattr(self, f'enc_block_{i+1}')(x)
            skips.append(x)
        # Final down to bottleneck input
        x = self.down_3(skips[-1])

        # Optional padding around bottleneck to handle odd shapes
        if x.shape[2:] == (7, 7, 5):
            x = nn.functional.pad(x, (0, 1, 0, 1, 0, 1))

        x = self.bottleneck(x)

        if x.shape[2:] == (8, 8, 6):
            x = nn.functional.pad(x, (0, -1, 0, -1, 0, -1))

        # Decoder: traverse skips in reverse with corresponding up/dec blocks
        for i in range(3, -1, -1):  # 3,2,1,0
            x_up = getattr(self, f'up_{i}')(x)
            x = getattr(self, f'dec_block_{i}')(skips[i] + x_up)

        # Head
        x = self.out_0(x)
        return x

