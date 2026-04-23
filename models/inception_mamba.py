"""
models/inception_mamba.py -- Inception-Mamba student encoder for FedMamba-SALT.

Architecture overview (3 stages):
  Stage 1 -- Patch Embedding:
      Conv2d(3, 256, 16, 16) -> BN -> GELU -> reshape -> +pos_embed
  Stage 2 -- N x InceptionMambaBlock, each containing:
      InceptionConvBlock  (local multi-scale features via parallel 3/5/7 DW convs)
      MambaBlock          (global sequence modeling via 4-directional SSM scan)
      FFN                 (2-layer MLP with 4x expansion)
  Stage 3 -- Projection:
      LayerNorm -> Linear(embed_dim, 768) -> global mean pool -> (B, 768)

Reference implementations studied:
  - Mamba SSM:  state-spaces/mamba  (Mamba class: d_model, d_state, d_conv, expand)
  - VMamba:     MzeroMiko/VMamba    (SS2D 4-directional cross-scan strategy)
  - InceptionV3: torchvision        (parallel conv branches + concat + project)
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Mamba import with CPU-only mock fallback for shape testing
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba-ssm not installed.  Using a LINEAR MOCK for shape testing only.  "
        "Install mamba-ssm==1.2.0 and causal-conv1d==1.2.0 for real training.",
        stacklevel=2,
    )

    class Mamba(nn.Module):
        """
        Lightweight mock that mimics the real Mamba(d_model, d_state, d_conv,
        expand) interface with a gated linear unit.  Parameter count is
        deliberately close to the real module so shape/grad tests remain
        meaningful.  **NOT suitable for actual training.**
        """

        def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            **kwargs,
        ):
            super().__init__()
            d_inner = d_model * expand
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, d_conv,
                groups=d_inner, padding=d_conv - 1,
            )
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, L, D)
            B, L, D = x.shape
            xz = self.in_proj(x)            # (B, L, 2*d_inner)
            x_, z = xz.chunk(2, dim=-1)     # each (B, L, d_inner)
            # causal conv1d
            h = x_.transpose(1, 2)          # (B, d_inner, L)
            h = self.conv1d(h)[:, :, :L]    # trim causal padding
            h = h.transpose(1, 2)           # (B, L, d_inner)
            return self.out_proj(h * F.silu(z))


# ======================================================================
# Stage 1 -- Patch Embedding
# ======================================================================
class PatchEmbedding(nn.Module):
    """
    Non-overlapping patch projection: Conv2d(3, embed_dim, P, P) -> BN -> GELU.
    Reshapes to (B, num_patches, embed_dim) and adds learnable positional
    embeddings initialised with truncated normal (std=0.02).
    """

    def __init__(self, patch_size: int = 16, embed_dim: int = 256, img_size: int = 224):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224/16

        _get_groups = lambda c: next((g for g in [32, 16, 8, 4, 2, 1] if c % g == 0), 1)

        self.proj = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(_get_groups(embed_dim), embed_dim),
            nn.GELU(),
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        x = self.proj(x)                               # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)               # (B, 196, embed_dim)
        x = x + self.pos_embed                          # add positional embeddings
        return x


# ======================================================================
# Stage 2a -- InceptionConvBlock
# ======================================================================
class InceptionConvBlock(nn.Module):
    """
    Multi-scale local feature extraction with 4 parallel branches:
      - 3x3 depthwise separable conv
      - 5x5 depthwise separable conv
      - 7x7 depthwise separable conv
      - AdaptiveAvgPool2d(1) + 1x1 conv (global context)

    Each branch outputs embed_dim // 4 channels.  Outputs are concatenated
    and merged back to embed_dim via a 1x1 convolution.

    Input/output shape: (B, L, D) -- reshapes internally to 2D and back.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        branch_dim = embed_dim // 4

        _get_groups = lambda c: next((g for g in [32, 16, 8, 4, 2, 1] if c % g == 0), 1)

        # Helper: pointwise reduction + depthwise spatial conv + GN + GELU
        def _make_dw_branch(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(embed_dim, branch_dim, 1),                             # pointwise reduce
                nn.Conv2d(branch_dim, branch_dim, kernel_size,
                          padding=kernel_size // 2, groups=branch_dim),           # depthwise
                nn.GroupNorm(_get_groups(branch_dim), branch_dim),
                nn.GELU(),
            )

        self.branch3 = _make_dw_branch(3)
        self.branch5 = _make_dw_branch(5)
        self.branch7 = _make_dw_branch(7)

        # Global-context branch: pool -> 1x1 conv -> broadcast
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, branch_dim, 1),
            nn.GroupNorm(_get_groups(branch_dim), branch_dim),
            nn.GELU(),
        )

        # Merge projection: concat(4 * branch_dim) == embed_dim -> embed_dim
        self.merge = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        H = W = int(math.isqrt(L))
        x_2d = x.transpose(1, 2).reshape(B, D, H, W)      # (B, D, H, W)

        b3 = self.branch3(x_2d)                             # (B, branch_dim, H, W)
        b5 = self.branch5(x_2d)
        b7 = self.branch7(x_2d)
        bp = self.branch_pool(x_2d)                         # (B, branch_dim, 1, 1)
        bp = bp.expand_as(b3)                               # broadcast to (B, branch_dim, H, W)

        merged = torch.cat([b3, b5, b7, bp], dim=1)         # (B, embed_dim, H, W)
        out = self.merge(merged)                             # (B, embed_dim, H, W)

        return out.flatten(2).transpose(1, 2)                # (B, L, D)


# ======================================================================
# Stage 2b -- MambaBlock (4-directional scan)
# ======================================================================
class MambaBlock(nn.Module):
    """
    4-directional SSM scan inspired by VMamba's cross-scan strategy.

    Directions:
      1. Row-major left-to-right   (default sequence order)
      2. Row-major right-to-left   (flip -> Mamba -> flip)
      3. Column-major top-to-bottom (transpose H/W -> flatten -> Mamba -> undo)
      4. Column-major bottom-to-top (transpose H/W -> flip -> Mamba -> flip -> undo)

    Outputs are merged by element-wise summation.
    Pre-norm + residual connection + linear output projection.

    Input/output shape: (B, L, D).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

        # Four independent Mamba instances, one per scan direction
        mamba_kwargs = dict(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_lr = Mamba(**mamba_kwargs)   # left-to-right
        self.mamba_rl = Mamba(**mamba_kwargs)   # right-to-left
        self.mamba_tb = Mamba(**mamba_kwargs)   # top-to-bottom
        self.mamba_bt = Mamba(**mamba_kwargs)   # bottom-to-top

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = W = int(math.isqrt(L))

        residual = x
        x = self.norm(x)

        # Direction 1: Left-to-right (row-major, natural order)
        out_lr = self.mamba_lr(x)

        # Direction 2: Right-to-left (reverse row-major)
        out_rl = self.mamba_rl(x.flip(dims=[1])).flip(dims=[1])

        # Direction 3: Top-to-bottom (column-major)
        # Reshape to 2D, transpose H<->W, flatten back to sequence
        x_2d = x.reshape(B, H, W, D)
        x_col = x_2d.transpose(1, 2).contiguous().reshape(B, L, D)   # column-major
        out_tb = self.mamba_tb(x_col)
        # Undo: reshape column-major back to row-major
        out_tb = out_tb.reshape(B, W, H, D).transpose(1, 2).contiguous().reshape(B, L, D)

        # Direction 4: Bottom-to-top (column-major reversed)
        out_bt = self.mamba_bt(x_col.flip(dims=[1])).flip(dims=[1])
        out_bt = out_bt.reshape(B, W, H, D).transpose(1, 2).contiguous().reshape(B, L, D)

        # Merge all directions by element-wise sum
        merged = out_lr + out_rl + out_tb + out_bt

        # Output projection + residual
        return residual + self.out_proj(merged)


# ======================================================================
# Stage 2c -- InceptionMambaBlock (full encoder block)
# ======================================================================
class InceptionMambaBlock(nn.Module):
    """
    Composes InceptionConvBlock + MambaBlock + FFN into one encoder block.

    Forward pass:
      1. x = x + InceptionConvBlock(x)           -- local multi-scale features
      2. x = MambaBlock(x)                        -- global sequence modeling
                                                     (MambaBlock has internal pre-norm + residual)
      3. x = x + FFN(LayerNorm(x))               -- channel mixing

    Input/output shape: (B, L, D).
    """

    def __init__(self, embed_dim: int = 256, ffn_ratio: int = 4):
        super().__init__()

        # Local multi-scale convolution with pre-norm
        self.norm_conv = nn.LayerNorm(embed_dim)
        self.conv_block = InceptionConvBlock(embed_dim)

        # Global 4-directional Mamba scan (has internal pre-norm + residual)
        self.mamba_block = MambaBlock(embed_dim)

        # FFN with pre-norm
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv_block(self.norm_conv(x))   # pre-norm + residual from conv block
        x = self.mamba_block(x)                      # MambaBlock internal residual
        x = x + self.ffn(self.norm_ffn(x))           # FFN with pre-norm + residual
        return x


# ======================================================================
# Full Encoder
# ======================================================================
class InceptionMambaEncoder(nn.Module):
    """
    Inception-Mamba student encoder for SALT distillation.

    Takes (B, 3, 224, 224) images and produces (B, out_dim) embeddings
    that will be aligned to the frozen ViT teacher's CLS-token output
    via the projection head.

    Args:
        patch_size: Patch tokenisation stride (default 16 -> 196 patches).
        embed_dim:  Channel width of the encoder (default 256).
        depth:      Number of stacked InceptionMambaBlocks (default 4).
        out_dim:    Output embedding dimension, must match teacher (default 768).
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        out_dim: int = 768,
    ):
        super().__init__()

        # Stage 1: Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, embed_dim=embed_dim, img_size=224,
        )

        # Stage 2: InceptionMamba blocks
        self.blocks = nn.ModuleList([
            InceptionMambaBlock(embed_dim) for _ in range(depth)
        ])

        # Stage 3: Final norm + projection to teacher-matching dimension
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) image batch.
            return_patches: If True, returns all 196 patch tokens.
                            If False, returns the Global Average Pooled (GAP) token.
        Returns:
            If return_patches=False: (B, out_dim) embedding vector per image.
            If return_patches=True: (B, 196, out_dim) patch embeddings per image.
        """
        # Stage 1: (B, 3, 224, 224) -> (B, 196, embed_dim)
        x = self.patch_embed(x)

        # Stage 2: repeated InceptionMambaBlocks
        for blk in self.blocks:
            x = blk(x)                              # (B, 196, embed_dim)

        # Stage 3: norm -> project
        x = self.norm(x)                             # (B, 196, embed_dim)
        x = self.proj(x)                             # (B, 196, out_dim)
        
        if return_patches:
            return x

        # Default: global mean pool
        x = x.mean(dim=1)                            # (B, out_dim)

        return x


# ===========================================================================
# Predictor Mamba (for Latent Feature Masking)
# ===========================================================================
class PredictorMamba(nn.Module):
    """
    Lightweight predictor for Latent Feature Masking (LFM).
    Takes a dense sequence, randomly masks out a percentage of tokens,
    and uses a Mamba block to predict the full unmasked sequence.
    """
    def __init__(self, embed_dim: int = 768, depth: int = 1):
        super().__init__()
        # Learnable mask token to replace dropped features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Mamba blocks for prediction
        self.blocks = nn.ModuleList([
            Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
            if MAMBA_AVAILABLE else MockMamba(d_model=embed_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.6) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) sequence
            mask_ratio: fraction of tokens to replace with mask_token
            
        Returns:
            out: (B, L, D) predicted sequence
            mask: (B, L) boolean mask indicating which tokens were dropped
        """
        B, L, D = x.shape
        
        # 1. Generate random mask
        noise = torch.rand(B, L, device=x.device)
        # mask is True for tokens that should be DROPPED
        mask = noise < mask_ratio
        
        # 2. Replace masked tokens with the learnable mask token
        x_masked = x.clone()
        expanded_mask_token = self.mask_token.expand(B, L, D)
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        
        # If mask is True, use mask_token, else use original feature
        x_masked = torch.where(mask_expanded, expanded_mask_token, x_masked)
        
        # 3. Pass through predictor
        out = x_masked
        for blk in self.blocks:
            out = blk(out)
            
        out = self.norm(out)
        return out, mask
