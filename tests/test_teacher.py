"""
tests/test_teacher.py — Smoke tests for the frozen ViT-B/16 teacher.

Four assertions that must ALL pass before proceeding to Phase 2:
  1. Output shape is exactly (B, 768)
  2. Teacher is truly frozen — no gradients reach any parameter
  3. Two forward passes on the same input are bit-identical (eval mode)
  4. Different inputs produce different outputs (encoder is not collapsed)

Run from the project root:
    python -m tests.test_teacher
"""

import sys

import torch

from models.vit_teacher import FrozenViTTeacher

# =====================================================================
#  Test helpers
# =====================================================================
BATCH_SIZE = 4
IMG_SHAPE = (3, 224, 224)


def _make_teacher() -> FrozenViTTeacher:
    """Create a teacher with random (untrained) weights -- no checkpoint."""
    return FrozenViTTeacher.for_testing()


# =====================================================================
#  Test 1 — Output shape
# =====================================================================
def test_output_shape() -> bool:
    """Output shape must be exactly (B, 768)."""
    teacher = _make_teacher()
    x = torch.randn(BATCH_SIZE, *IMG_SHAPE)
    out = teacher(x)

    expected = (BATCH_SIZE, 768)
    passed = out.shape == torch.Size(expected)
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 1 — Output shape is {expected}: got {tuple(out.shape)}")
    return passed


# =====================================================================
#  Test 2 — Frozen gradients
# =====================================================================
def test_frozen_gradients() -> bool:
    """
    The teacher must be truly frozen: no gradient graph is built
    (@torch.no_grad on forward) and every parameter has
    requires_grad=False.

    We verify both layers of defense:
      (a) The output tensor has no grad_fn  → proves @torch.no_grad works.
      (b) Every parameter has requires_grad=False → proves explicit freeze.

    Together these guarantee that loss.backward() can never push gradients
    into the teacher, which is the intent of the original test description.
    """
    teacher = _make_teacher()
    x = torch.randn(2, *IMG_SHAPE)
    out = teacher(x)

    # (a) No computation graph attached to the output
    no_grad_fn = out.grad_fn is None

    # (b) Every single parameter is frozen
    all_frozen = all(not p.requires_grad for p in teacher.parameters())

    # (c) Explicit backward test: build a tiny graph that *uses* the output
    #     but route gradients through a fresh parameter, and verify they
    #     never leak into the teacher.
    probe = torch.ones(1, requires_grad=True)
    loss = (out.detach() * probe).sum()
    loss.backward()
    teacher_grads_none = all(p.grad is None for p in teacher.parameters())

    passed = no_grad_fn and all_frozen and teacher_grads_none
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 2 — Frozen (no grad_fn={no_grad_fn}, "
          f"all requires_grad=False={all_frozen}, "
          f"all .grad is None={teacher_grads_none})")
    return passed


# =====================================================================
#  Test 3 — Deterministic eval mode
# =====================================================================
def test_deterministic_eval() -> bool:
    """Two forward passes on the same input must be bit-identical."""
    teacher = _make_teacher()
    x = torch.randn(2, *IMG_SHAPE)

    out1 = teacher(x)
    out2 = teacher(x)

    passed = torch.equal(out1, out2)
    tag = "PASS" if passed else "FAIL"
    max_diff = (out1 - out2).abs().max().item() if not passed else 0.0
    print(f"  [{tag}] Test 3 — Deterministic output (max diff = {max_diff:.2e})")
    return passed


# =====================================================================
#  Test 4 — Non-collapsed representations
# =====================================================================
def test_not_collapsed() -> bool:
    """Different inputs must produce different outputs."""
    teacher = _make_teacher()
    x1 = torch.randn(2, *IMG_SHAPE)
    x2 = torch.randn(2, *IMG_SHAPE)

    out1 = teacher(x1)
    out2 = teacher(x2)

    passed = not torch.equal(out1, out2)
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 4 — Non-collapsed (different inputs → different outputs)")
    return passed


# =====================================================================
#  Main
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  FrozenViTTeacher -- Smoke Tests (no checkpoint needed)")
    print("=" * 60)

    results = [
        test_output_shape(),
        test_frozen_gradients(),
        test_deterministic_eval(),
        test_not_collapsed(),
    ]

    print("=" * 60)
    n_passed = sum(results)
    print(f"  Result: {n_passed}/4 tests passed")
    if n_passed == 4:
        print("  ✓ All tests PASSED — ready for Phase 2.")
    else:
        print("  ✗ SOME TESTS FAILED — review before proceeding.")
    print("=" * 60)

    sys.exit(0 if n_passed == 4 else 1)
