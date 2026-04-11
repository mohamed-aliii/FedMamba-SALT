"""
models/vit_teacher.py -- Frozen MAE-pretrained ViT-B/16 teacher encoder.

The teacher is a ViT-B/16 constructed explicitly via timm 0.3.2's
VisionTransformer class (NOT timm.create_model, which is version-registry
dependent).  Weights are loaded from an MAE checkpoint that stores encoder
parameters with an ``encoder.`` prefix on every key.

The module is permanently frozen:
  - All parameters have requires_grad = False.
  - eval() is called in __init__ AND defensively at the top of every
    forward() call (guards against accidental model.train() elsewhere).
  - forward() is decorated with @torch.no_grad() so no graph is built.

Its sole output is the CLS-token embedding -- a (B, 768) tensor that serves
as the learning target for the Inception-Mamba student.
"""

import functools
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

# --- timm 0.3.2 compatibility shim for PyTorch 2.x ---
# timm 0.3.2 imports torch._six which was removed in PyTorch 2.0.
# This must run before any timm import.
import collections.abc, sys, types
if "torch._six" not in sys.modules:
    _mock = types.ModuleType("torch._six")
    _mock.container_abcs = collections.abc
    sys.modules["torch._six"] = _mock

from timm.models.vision_transformer import VisionTransformer

from utils.ckpt_compat import safe_torch_load


class FrozenViTTeacher(nn.Module):
    """
    Frozen ViT-B/16 teacher that loads MAE-pretrained weights and produces
    768-dimensional CLS-token embeddings.  Never trains, never updates,
    always eval.

    In production, ``ckpt_path`` is required.  For unit tests that do not
    need real weights, use the ``FrozenViTTeacher.for_testing()`` classmethod.
    """

    EMBED_DIM: int = 768

    def __init__(self, ckpt_path: Union[str, Path]):
        super().__init__()

        # ------------------------------------------------------------------
        # Construct ViT-B/16 explicitly (matches MAE pretraining config)
        # ------------------------------------------------------------------
        self.encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,           # drop the classification head
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        )

        # ------------------------------------------------------------------
        # Load MAE checkpoint (required)
        # ------------------------------------------------------------------
        self._load_mae_checkpoint(str(ckpt_path))

        # ------------------------------------------------------------------
        # Freeze every parameter -- teacher never receives gradients
        # ------------------------------------------------------------------
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    @classmethod
    def for_testing(cls) -> "FrozenViTTeacher":
        """
        Create a FrozenViTTeacher with random weights (no checkpoint).

        This is the ONLY escape hatch for unit tests that do not need a real
        MAE checkpoint.  Do NOT use this in production training code.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        )

        # Freeze all parameters
        for param in instance.parameters():
            param.requires_grad = False

        instance.eval()
        return instance

    # ------------------------------------------------------------------
    # Checkpoint loading with prefix stripping
    # ------------------------------------------------------------------
    def _load_mae_checkpoint(self, ckpt_path: str) -> None:
        """Load an MAE checkpoint, handling both prefixed and unprefixed formats.

        Format A (Facebook MAE):  keys start with ``encoder.`` — strip prefix.
        Format B (SSL-FL):        keys have no prefix — filter out decoder keys.
        """
        ckpt = safe_torch_load(ckpt_path, map_location="cpu")

        # MAE checkpoints nest the state dict under "model"
        state_dict = ckpt.get("model", ckpt)

        # --- Try Format A: strip 'encoder.' prefix ---
        prefix = "encoder."
        stripped = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

        if len(stripped) > 0:
            # Format A: Facebook MAE checkpoint
            encoder_state = stripped
            fmt = "Format A (encoder.* prefix)"
        else:
            # --- Format B: no prefix, filter out decoder/mask keys ---
            DECODER_PREFIXES = ("decoder_", "decoder.", "mask_token")
            encoder_state = {
                key: value
                for key, value in state_dict.items()
                if not any(key.startswith(p) for p in DECODER_PREFIXES)
            }
            fmt = "Format B (no prefix, decoder keys filtered)"

        result = self.encoder.load_state_dict(encoder_state, strict=False)

        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)

        print(f"[FrozenViTTeacher] Loaded checkpoint: {ckpt_path}")
        print(f"  Detected: {fmt}")
        print(f"  Encoder keys loaded: {len(encoder_state)}")
        print(f"  Missing keys:    {n_missing}")
        print(f"  Unexpected keys: {n_unexpected}")

        if n_missing > 20:
            print(
                f"  WARNING: {n_missing} missing keys is suspicious -- "
                "check checkpoint format."
            )
        if n_unexpected > 0:
            print(
                f"  WARNING: unexpected keys found: "
                f"{result.unexpected_keys[:5]}"
            )

    # ------------------------------------------------------------------
    # Forward pass -- always eval, always no_grad
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the CLS-token embedding from a batch of images.

        Args:
            x: Image batch of shape ``(B, 3, 224, 224)``.

        Returns:
            CLS-token embeddings of shape ``(B, 768)``.
        """
        # Defensive: re-assert eval mode in case .train() was called on a
        # parent module that recursively set this module to train mode.
        self.eval()

        # timm 0.3.2: forward_features returns pre_logits(x[:, 0]) = (B, 768)
        # i.e. the CLS token is already extracted and normalized.
        return self.encoder.forward_features(x)  # (B, 768)
