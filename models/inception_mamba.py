"""
models/inception_mamba.py -- InceptionMamba student encoder for FedMamba-SALT.

Reference backbone:
  "InceptionMamba: A Lightweight and Effective Model for Medical Image
  Classification Revealing Mamba's Low-Frequency Bias", Neural Processing
  Letters 58:15, 2026.

Paper-aligned block features preserved here:
  1. Dual branch channel split: C/2 Inception path + C/2 SSM path.
  2. Inception local path with four branches:
     1x1, 1x1->3x3, 1x1->3x3->3x3, and AvgPool3x3->1x1.
  3. Channel attention after multi-scale Inception fusion.
  4. SSM path with LayerNorm, Linear, DWConv3x3, SiLU gating,
     four-direction SS2D-style scans, and output projection.
  5. Channel concat, channel shuffle, and residual fusion.

FedMamba-SALT adaptations:
  - Uses 16x16 patches and no patch merging so the student keeps the same
    14x14 = 196 dense token grid as the frozen MAE ViT-B/16 teacher.
  - Projects every student patch token to 768 dimensions for dense SALT
    distillation against the ViT teacher.
  - Replaces BatchNorm2d/ReLU with GroupNorm/GELU in convolution blocks to
    avoid client-specific running-stat drift in federated training.
  - The original paper classifier is replaced downstream by SALT pretraining,
    linear probing, and federated attention-pooling fine-tuning heads.
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mamba import with CPU-only mock fallback
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba-ssm not installed. Using a LINEAR MOCK for shape testing only.",
        stacklevel=2,
    )

    class Mamba(nn.Module):
        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                     expand: int = 2, **kwargs):
            super().__init__()
            d_inner = d_model * expand
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                    groups=d_inner, padding=d_conv - 1)
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L, D = x.shape
            xz = self.in_proj(x)
            x_, z = xz.chunk(2, dim=-1)
            h = x_.transpose(1, 2)
            h = self.conv1d(h)[:, :, :L]
            h = h.transpose(1, 2)
            return self.out_proj(h * F.silu(z))


# ======================================================================
# Utilities
# ======================================================================
def _get_groups(channels: int) -> int:
    """Pick a valid GroupNorm group count for the given channel count."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel shuffle from ShuffleNet to mix parallel branch features."""
    B, C, H, W = x.shape
    channels_per_group = C // groups
    x = x.view(B, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(B, -1, H, W)


class BasicConv2d(nn.Module):
    """Conv + GroupNorm + GELU (federated-safe, no BatchNorm)."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)
        self.norm = nn.GroupNorm(_get_groups(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DropPath(nn.Module):
    """Stochastic depth: drop entire residual branch during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.drop_prob
        return x * keep / (1.0 - self.drop_prob)


# ======================================================================
# Branch 1: Paper's Inception Module + Channel Attention
# ======================================================================
class InceptionBranch(nn.Module):
    """Authentic multi-scale feature extractor with 1x1 bottlenecks."""
    def __init__(self, in_channels: int):
        super().__init__()
        branch_c = in_channels // 4

        self.branch1 = BasicConv2d(in_channels, branch_c, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, branch_c, kernel_size=1),
            BasicConv2d(branch_c, branch_c, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, branch_c, kernel_size=1),
            BasicConv2d(branch_c, branch_c, kernel_size=3, padding=1),
            BasicConv2d(branch_c, branch_c, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, branch_c, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class ChannelAttention(nn.Module):
    """Dual-Pool Channel Attention to filter multi-scale Inception features."""
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


# ======================================================================
# Branch 2: SS2D-Style Mamba Branch with Gating & Directional Weights
# ======================================================================
class SSMBranch(nn.Module):
    """
    Paper-accurate SSM branch: DWConv enhancement, 4-directional scan,
    element-wise SiLU gating, and learnable directional fusion weights.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)

        self.proj_main = nn.Linear(d_model, d_model)
        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=3,
                                padding=1, groups=d_model)

        mamba_kwargs = dict(d_model=d_model, d_state=d_state,
                            d_conv=4, expand=expand)
        self.mamba_lr = Mamba(**mamba_kwargs)
        self.mamba_rl = Mamba(**mamba_kwargs)
        self.mamba_tb = Mamba(**mamba_kwargs)
        self.mamba_bt = Mamba(**mamba_kwargs)
        
        # Learnable weights for the 4 directions (preventing blind fusion)
        self.dir_weights = nn.Parameter(torch.ones(4))
        
        self.norm2 = nn.LayerNorm(d_model)

        self.proj_gate = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, L, D = x_seq.shape
        H = W = int(math.isqrt(L))

        x_norm = self.norm1(x_seq)

        gate = F.silu(self.proj_gate(x_norm))

        main = self.proj_main(x_norm)
        main_2d = main.transpose(1, 2).reshape(B, D, H, W)
        main_2d = self.dwconv(main_2d)
        main = F.silu(main_2d.flatten(2).transpose(1, 2))

        out_lr = self.mamba_lr(main)
        out_rl = self.mamba_rl(main.flip(dims=[1])).flip(dims=[1])

        main_col = main.reshape(B, H, W, D).transpose(1, 2) \
                       .contiguous().reshape(B, L, D)
        out_tb = self.mamba_tb(main_col)
        out_tb = out_tb.reshape(B, W, H, D).transpose(1, 2) \
                       .contiguous().reshape(B, L, D)

        out_bt = self.mamba_bt(main_col.flip(dims=[1])).flip(dims=[1])
        out_bt = out_bt.reshape(B, W, H, D).transpose(1, 2) \
                       .contiguous().reshape(B, L, D)

        # Weighted fusion instead of simple addition
        w = F.softmax(self.dir_weights, dim=0)
        ssm_out = self.norm2(w[0] * out_lr + w[1] * out_rl + w[2] * out_tb + w[3] * out_bt)
        return self.proj_out(ssm_out * gate)


# ======================================================================
# The Full InceptionMamba Block (Parallel Split + Shuffle)
# ======================================================================
class InceptionMambaBlock(nn.Module):
    """
    Paper's core block:
      1. Split channels C → C/2 (Inception) + C/2 (SSM)
      2. Process in parallel
      3. Concat → Channel Shuffle → Residual
    """
    def __init__(self, embed_dim: int = 256, drop_path: float = 0.0):
        super().__init__()
        half_dim = embed_dim // 2

        self.inception = InceptionBranch(half_dim)
        self.channel_attn = ChannelAttention(half_dim)
        self.ssm_branch = SSMBranch(half_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = W = int(math.isqrt(L))

        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x1_inc, x2_ssm = torch.split(x_2d, C // 2, dim=1)

        out_inc = self.channel_attn(self.inception(x1_inc))

        x2_seq = x2_ssm.flatten(2).transpose(1, 2)
        out_ssm_seq = self.ssm_branch(x2_seq)
        out_ssm = out_ssm_seq.transpose(1, 2).reshape(B, C // 2, H, W)

        out_merged = channel_shuffle(
            torch.cat([out_inc, out_ssm], dim=1), groups=2,
        )
        return x + self.drop_path(out_merged.flatten(2).transpose(1, 2))


# ======================================================================
# Patch Embedding & Full Encoder
# ======================================================================
class PatchMerging(nn.Module):
    """Halves spatial resolution and doubles channels."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        import math
        H = W = int(math.isqrt(L))
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_merged = self.proj(x_2d).flatten(2).transpose(1, 2)
        return self.norm(x_merged)


class PatchEmbedding(nn.Module):
    """Non-overlapping patch projection with LayerNorm."""
    def __init__(self, patch_size: int = 4, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class InceptionMambaEncoder(nn.Module):
    """
    Paper-accurate hierarchical encoder: 4 stages with Patch Merging.
    """
    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 4, 2],
        dims: list[int] = [96, 192, 384, 768],
        out_dim: int = 768,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        num_stages = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        for i in range(num_stages):
            stage = nn.ModuleList([
                InceptionMambaBlock(dims[i], drop_path=dpr[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            
            if i < num_stages - 1:
                self.downsamples.append(PatchMerging(dims[i], dims[i+1]))
            else:
                self.downsamples.append(nn.Identity())

        self.norm = nn.LayerNorm(dims[-1])
        self.proj = nn.Linear(dims[-1], out_dim)

    def forward(self, x: torch.Tensor, return_patches: bool = False,
                mask_ratio: float = 0.0) -> torch.Tensor:
        x = self.patch_embed(x)

        if mask_ratio > 0.0 and self.training:
            B, L, D = x.shape
            mask = torch.rand(B, L, device=x.device) < mask_ratio
            x = torch.where(mask.unsqueeze(-1), torch.zeros_like(x), x)

        for i in range(len(self.stages)):
            for blk in self.stages[i]:
                x = blk(x)
            if i < len(self.stages) - 1:
                x = self.downsamples[i](x)

        x = self.norm(x)
        x = self.proj(x)

        if return_patches:
            return x

        return x.mean(dim=1)
