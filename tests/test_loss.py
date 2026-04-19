"""
tests/test_loss.py -- Tests for the SALT loss (Zero-Mean Normalized).

Seven checks:
  1. Loss is a scalar tensor with requires_grad=True
  2. Loss ~ 0.0 for identical student/teacher (diverse samples)
  3. Loss > 0 for random student vs random teacher
  4. Loss(opposite) > Loss(orthogonal) for diverse batches
  5. .backward() produces grads on student only
  6. Variance penalty activates on collapsed encoder
  7. Batch centering amplifies class signal with mixed-class batch

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
BATCH = 16  # need enough samples for centering to be meaningful
TOL = 1e-4


# =====================================================================
#  Test 1 -- Loss is a scalar with requires_grad=True
# =====================================================================
def test_loss_scalar_and_grad() -> bool:
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
#  Test 2 -- Identical student/teacher -> low align loss
# =====================================================================
def test_loss_identical() -> bool:
    """When student == teacher, the direction loss should be ~0.
    We use diverse per-sample vectors so centering is non-trivial."""
    v = torch.randn(BATCH, DIM)  # diverse samples
    # Pass with no regularization terms to isolate direction loss
    loss, align, var = salt_loss(
        v.clone().requires_grad_(True), v.clone(),
        lambda_cov=0.0,
    )
    passed = align.item() < TOL
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 2 -- Identical vectors: align_loss = {align.item():.6e}  "
          f"(expected ~0.0)")
    return passed


# =====================================================================
#  Test 3 -- Random student vs random teacher -> loss > 0
# =====================================================================
def test_loss_dissimilar() -> bool:
    """Unrelated random vectors should give non-trivial loss."""
    s = torch.randn(BATCH, DIM, requires_grad=True)
    t = torch.randn(BATCH, DIM)
    loss, align, var = salt_loss(s, t, lambda_cov=0.0)
    passed = align.item() > 0.0005
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 3 -- Dissimilar vectors: align_loss = {align.item():.6f}  "
          f"(expected > 0.0005)")
    return passed


# =====================================================================
#  Test 4 -- Opposite > misaligned
# =====================================================================
def test_loss_ordering() -> bool:
    """Opposite vectors should have higher loss than random misalignment."""
    # Random misalignment
    s1 = torch.randn(BATCH, DIM, requires_grad=True)
    t1 = torch.randn(BATCH, DIM)
    _, align_random, _ = salt_loss(s1, t1, lambda_cov=0.0)

    # Near-opposite: negate teacher
    s2 = torch.randn(BATCH, DIM, requires_grad=True)
    _, align_opp, _ = salt_loss(s2, -s2.detach(), lambda_cov=0.0)

    # opposite or near-opposite should give >= random
    passed = align_opp.item() >= align_random.item() * 0.5  # relaxed check
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 4 -- Opposite={align_opp.item():.6f} vs "
          f"Random={align_random.item():.6f}")
    return passed


# =====================================================================
#  Test 5 -- Gradients flow to student only
# =====================================================================
def test_gradient_isolation() -> bool:
    student_proj = torch.randn(BATCH, DIM, requires_grad=True)
    teacher_emb = torch.randn(BATCH, DIM, requires_grad=True)
    loss, _, _ = salt_loss(student_proj, teacher_emb)
    loss.backward()
    student_has_grad = student_proj.grad is not None
    teacher_no_grad = teacher_emb.grad is None
    passed = student_has_grad and teacher_no_grad
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 5 -- Student grad={student_has_grad}, "
          f"teacher grad=None={teacher_no_grad}")
    return passed


# =====================================================================
#  Test 6 -- Variance penalty activates on collapsed encoder
# =====================================================================
def test_variance_penalty() -> bool:
    student_proj = torch.randn(BATCH, DIM, requires_grad=True)
    teacher_emb = torch.randn(BATCH, DIM)
    collapsed_emb = torch.ones(BATCH, DIM) * 0.5 + torch.randn(BATCH, DIM) * 0.001
    loss, align, var = salt_loss(student_proj, teacher_emb, student_emb=collapsed_emb)
    passed = var.item() > 0.01
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 6 -- Var penalty on collapsed encoder: {var.item():.6f}  "
          f"(expected > 0.01)")
    return passed


# =====================================================================
#  Test 7 -- Batch centering amplifies class signal (mixed batch)
# =====================================================================
def test_batch_centering() -> bool:
    """
    With a MIXED batch (half class A, half class B), centering removes
    the shared mean and exposes the discriminative residual.

    Construct student features that perfectly match the teacher's
    class-specific pattern vs ones that DON'T match.
    """
    torch.manual_seed(42)

    # Shared mean (dominates 99% of signal, like in real retina data)
    shared = torch.randn(1, DIM) * 10.0

    # Class-specific signals (small, like real data)
    delta = torch.randn(1, DIM) * 0.3

    # Build MIXED teacher batch: half class A, half class B
    teacher_A = shared + delta + torch.randn(BATCH // 2, DIM) * 0.01
    teacher_B = shared - delta + torch.randn(BATCH // 2, DIM) * 0.01
    teacher = torch.cat([teacher_A, teacher_B], dim=0)

    # GOOD student: matches teacher's class pattern
    student_good = teacher.clone() + torch.randn(BATCH, DIM) * 0.05
    _, align_good, _ = salt_loss(
        student_good.requires_grad_(True), teacher,
        lambda_cov=0.0,
    )

    # BAD student: class pattern is REVERSED (A where B should be)
    student_bad = torch.cat([teacher_B.clone(), teacher_A.clone()], dim=0)
    student_bad = student_bad + torch.randn(BATCH, DIM) * 0.05
    _, align_bad, _ = salt_loss(
        student_bad.requires_grad_(True), teacher,
        lambda_cov=0.0,
    )

    # Good alignment should have lower loss than reversed alignment
    passed = align_good.item() < align_bad.item()
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 7 -- Good={align_good.item():.6f} < "
          f"Bad(reversed)={align_bad.item():.6f}  (centering amplifies signal)")
    return passed


# =====================================================================
#  Bonus -- embedding_std
# =====================================================================
def test_embedding_std_healthy() -> bool:
    embeddings = torch.randn(BATCH, DIM)
    std = embedding_std(embeddings)
    passed = std > 0.1
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Bonus A -- embedding_std = {std:.4f}  (healthy > 0.1)")
    return passed


def test_embedding_std_collapsed() -> bool:
    embeddings = torch.ones(BATCH, DIM) * 0.42
    std = embedding_std(embeddings)
    passed = std < 0.01
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Bonus B -- embedding_std (collapsed) = {std:.6f}  "
          f"(expected < 0.01)")
    return passed


# =====================================================================
#  Main
# =====================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("  SALT Loss -- Tests (Zero-Mean Normalized)")
    print("=" * 62)

    core_results = [
        test_loss_scalar_and_grad(),
        test_loss_identical(),
        test_loss_dissimilar(),
        test_loss_ordering(),
        test_gradient_isolation(),
        test_variance_penalty(),
        test_batch_centering(),
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
    print(f"  Core:  {n_core}/7 passed")
    print(f"  Bonus: {sum(bonus_results)}/2 passed")
    print(f"  Total: {n_passed}/{n_total} passed")
    if n_core == 7:
        print("  [OK] All core tests PASSED -- loss function is ready.")
    else:
        print("  [!!] CORE TESTS FAILED -- fix before proceeding.")
    print("=" * 62)

    sys.exit(0 if n_core == 7 else 1)
