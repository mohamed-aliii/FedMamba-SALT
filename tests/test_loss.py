"""
tests/test_loss.py -- Tests for the SALT loss and projection head.

Five checks:
  1. Loss is a scalar tensor with requires_grad=True
  2. Loss ~ 0.0 for identical normalised vectors
  3. Loss ~ 1.0 for orthogonal normalised vectors  (cosine sim = 0)
  4. Loss ~ 2.0 for opposite normalised vectors    (cosine sim = -1)
  5. .backward() produces grads on student_proj but NOT on teacher_emb

Run from the project root:
    python -m tests.test_loss
"""

import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from objectives.salt_loss import embedding_std, salt_loss

DIM = 768
BATCH = 8
TOL = 1e-5  # tolerance for floating-point comparisons


# =====================================================================
#  Test 1 -- Loss is a scalar with requires_grad=True
# =====================================================================
def test_loss_scalar_and_grad() -> bool:
    """Loss must be a 0-d tensor that tracks gradients."""
    student_proj = torch.randn(BATCH, DIM, requires_grad=True)
    teacher_emb = torch.randn(BATCH, DIM)

    loss, align, var = salt_loss(student_proj, teacher_emb)

    is_scalar = loss.dim() == 0
    has_grad = loss.requires_grad is True

    passed = is_scalar and has_grad
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 1 -- Scalar={is_scalar}, requires_grad={has_grad}")
    return passed


# =====================================================================
#  Test 2 -- Identical normalised vectors -> loss ~ 0.0
# =====================================================================
def test_loss_identical() -> bool:
    """Cosine loss between two identical unit vectors should be 0."""
    v = torch.randn(BATCH, DIM)

    # Pass the same tensor as both arguments.  salt_loss will normalise
    # and detach internally.
    loss, align, var = salt_loss(v.clone().requires_grad_(True), v.clone())

    passed = align.item() < TOL
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 2 -- Identical vectors: align_loss = {align.item():.6e}  "
          f"(expected ~0.0)")
    return passed


# =====================================================================
#  Test 3 -- Orthogonal normalised vectors -> loss ~ 1/768
# =====================================================================
# Smooth L1 evaluates Mean(0.5 * (a - b)^2). For orthogonal unit vectors,
# sum((a - b)^2) = 2.0. Mean is 2.0 / (2 * 768).
EXPECTED_ORTHO = 1.0 / DIM


def test_loss_orthogonal() -> bool:
    """Cosine loss for orthogonal unit vectors = 1.0."""
    # Construct a pair of exactly orthogonal vectors using QR decomposition.
    q, _ = torch.linalg.qr(torch.randn(DIM, 2))
    a = q[:, 0].unsqueeze(0).expand(BATCH, -1)  # (B, DIM), unit norm
    b = q[:, 1].unsqueeze(0).expand(BATCH, -1)  # (B, DIM), unit norm

    loss, align, var = salt_loss(a.clone().requires_grad_(True), b.clone())

    passed = abs(align.item() - EXPECTED_ORTHO) < TOL
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 3 -- Orthogonal vectors: align_loss = {align.item():.6f}  "
          f"(expected {EXPECTED_ORTHO:.1f})")
    return passed


# =====================================================================
#  Test 4 -- Opposite normalised vectors -> loss ~ 2/768
# =====================================================================
# For opposite unit vectors b = -a, sum((a - b)^2) = 4.0.
# Mean of 0.5 * 4.0 across 768 dims is 2.0 / 768.
EXPECTED_OPP = 2.0 / DIM


def test_loss_opposite() -> bool:
    """Cosine loss for opposite unit vectors = 2.0."""
    v = torch.randn(BATCH, DIM)
    v = v / v.norm(dim=-1, keepdim=True)  # unit norm

    loss, align, var = salt_loss(v.clone().requires_grad_(True), -v.clone())

    passed = abs(align.item() - EXPECTED_OPP) < TOL
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 4 -- Opposite vectors: align_loss = {align.item():.6f}  "
          f"(expected {EXPECTED_OPP:.1f})")
    return passed


# =====================================================================
#  Test 5 -- Gradients flow to student only, not to teacher
# =====================================================================
def test_gradient_isolation() -> bool:
    """
    Even if teacher_emb is constructed with requires_grad=True, the
    explicit .detach() inside salt_loss must prevent any gradient from
    reaching it.
    """
    student_proj = torch.randn(BATCH, DIM, requires_grad=True)
    teacher_emb = torch.randn(BATCH, DIM, requires_grad=True)

    loss, align, var = salt_loss(student_proj, teacher_emb)
    loss.backward()

    student_has_grad = student_proj.grad is not None
    teacher_no_grad = teacher_emb.grad is None

    passed = student_has_grad and teacher_no_grad
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 5 -- Student grad exists={student_has_grad}, "
          f"teacher grad is None={teacher_no_grad}")
    return passed


# =====================================================================
#  Bonus -- ProjectionHead and embedding_std quick checks
# =====================================================================



def test_embedding_std_healthy() -> bool:
    """Random embeddings should have std well above 0.01."""
    embeddings = torch.randn(BATCH, DIM)
    std = embedding_std(embeddings)
    passed = std > 0.1
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Bonus A -- embedding_std = {std:.4f}  (healthy > 0.1)")
    return passed


def test_embedding_std_collapsed() -> bool:
    """Constant embeddings should have std ~ 0 (collapsed)."""
    embeddings = torch.ones(BATCH, DIM) * 0.42
    std = embedding_std(embeddings)
    passed = std < 0.01
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Bonus A -- embedding_std (collapsed) = {std:.6f}  "
          f"(expected < 0.01)")
    return passed


# =====================================================================
#  Main
# =====================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("  SALT Loss -- Tests")
    print("=" * 62)

    core_results = [
        test_loss_scalar_and_grad(),
        test_loss_identical(),
        test_loss_orthogonal(),
        test_loss_opposite(),
        test_gradient_isolation(),
    ]

    print()
    print("  -- Bonus checks --")
    bonus_results = [
        test_embedding_std_healthy(),
        test_embedding_std_collapsed(),
    ]

    all_results = core_results + bonus_results
    print("=" * 62)
    n_passed = sum(all_results)
    n_total = len(all_results)
    n_core = sum(core_results)
    print(f"  Core:  {n_core}/5 passed")
    print(f"  Bonus: {sum(bonus_results)}/2 passed")
    print(f"  Total: {n_passed}/{n_total} passed")
    if n_core == 5:
        print("  [OK] All core tests PASSED -- loss function is ready.")
    else:
        print("  [!!] CORE TESTS FAILED -- fix before proceeding.")
    print("=" * 62)

    sys.exit(0 if n_core == 5 else 1)
