import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.modules import (
    SpatialRecover,
    LightweightDimensionReductiveAttention,
    CGLU,
)

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
