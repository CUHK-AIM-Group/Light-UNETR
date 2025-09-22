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