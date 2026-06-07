"""
tests/test_student.py -- Smoke tests for the Inception-Mamba student encoder.

Three checks:
  1. Output shape is (B, 768) for a batch of random inputs
  2. Gradients flow through every trainable parameter
  3. Total parameter count matches the FedMamba-SALT paper config

Run from the project root:
    python -m tests.test_student
"""

import os
import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.inception_mamba import InceptionMambaEncoder, MAMBA_AVAILABLE

BATCH = int(os.environ.get("FEDMAMBA_SMOKE_BATCH", "1"))
IMG_SHAPE = (3, 224, 224)
OUT_DIM = 768
PATCH_SIZE = 16
EMBED_DIM = 448
DEPTH = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_encoder() -> InceptionMambaEncoder:
    """Build the same encoder configuration used by training and the notebook."""
    return InceptionMambaEncoder().to(DEVICE)


# =====================================================================
#  Test 1 -- Output shape
# =====================================================================
def test_output_shape() -> bool:
    encoder = _build_encoder()
    x = torch.randn(BATCH, *IMG_SHAPE, device=DEVICE)
    out = encoder(x)

    expected = (BATCH, OUT_DIM)
    passed = out.shape == torch.Size(expected)
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 1 -- Output shape: {tuple(out.shape)}  (expected {expected})")
    return passed


# =====================================================================
#  Test 2 -- Gradient flow
# =====================================================================
def test_gradient_flow() -> bool:
    """After output.sum().backward(), every trainable param must have a gradient."""
    encoder = _build_encoder()
    x = torch.randn(BATCH, *IMG_SHAPE, device=DEVICE)
    out = encoder(x, mask_ratio=0.5)
    out.sum().backward()

    no_grad_params = []
    for name, p in encoder.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad_params.append(name)

    passed = len(no_grad_params) == 0
    tag = "PASS" if passed else "FAIL"
    if not passed:
        print(f"  [{tag}] Test 2 -- Gradient flow: {len(no_grad_params)} params missing gradients:")
        for name in no_grad_params[:10]:
            print(f"          - {name}")
        if len(no_grad_params) > 10:
            print(f"          ... and {len(no_grad_params) - 10} more")
    else:
        print(f"  [{tag}] Test 2 -- Gradient flow: all trainable params received gradients")
    return passed


# =====================================================================
#  Test 3 -- Parameter count
# =====================================================================
def test_param_count() -> bool:
    encoder = _build_encoder()
    total = _count_params(encoder)
    trainable = _count_trainable(encoder)
    total_m = total / 1e6
    trainable_m = trainable / 1e6

    # Encoder-only count for embed_dim=448/depth=6 is about 10.1M with the
    # local mock and about 10.9M with real mamba-ssm. The SALT projection head
    # is tested separately with the loss objective and is not counted here.
    lo, hi = 9.5, 12.0
    passed = lo <= total_m <= hi
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 3 -- Parameter count: {total_m:.2f}M total, "
          f"{trainable_m:.2f}M trainable  (expected {lo:.1f}--{hi:.1f}M)")
    if not MAMBA_AVAILABLE:
        print(f"          NOTE: using mock Mamba -- count will differ from real mamba-ssm")
    return passed


# =====================================================================
#  Main
# =====================================================================
if __name__ == "__main__":
    print("=" * 62)
    if MAMBA_AVAILABLE:
        print("  Inception-Mamba Encoder -- Smoke Tests  (real mamba-ssm)")
    else:
        print("  Inception-Mamba Encoder -- Smoke Tests  (MOCK Mamba)")
    print("=" * 62)

    results = [
        test_output_shape(),
        test_gradient_flow(),
        test_param_count(),
    ]

    print("=" * 62)
    n_passed = sum(results)
    print(f"  Result: {n_passed}/3 tests passed")
    if n_passed == 3:
        print("  [OK] All tests PASSED -- student encoder is ready.")
    else:
        print("  [!!] SOME TESTS FAILED -- review before proceeding.")
    print("=" * 62)

    sys.exit(0 if n_passed == 3 else 1)
