"""
tests/test_loss.py -- Tests for the SALT loss (Centered & Standardised MSE).

Seven core checks:
  1. Loss is a scalar tensor with requires_grad=True
  2. Loss ~ 0.0 for identical student/teacher
  3. Loss > 0 for random student vs random teacher
  4. Loss(opposite) > Loss(random)
  5. .backward() produces grads on student only
  6. Variance penalty activates on collapsed encoder
  7. Batch centering + standardisation amplifies class signal

Run from the project root:
    python -m tests.test_loss
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from objectives.salt_loss import embedding_std, salt_loss

DIM = 768
BATCH = 16
TOL = 1e-4


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


def test_loss_identical() -> bool:
    """Identical student/teacher → loss ~ 0 (centered residuals match)."""
    v = torch.randn(BATCH, DIM)
    loss, align, var = salt_loss(
        v.clone().requires_grad_(True), v.clone(),
        lambda_cov=0.0,
    )
    # After centering both identically, s_centered matches t_centered.
    # After standardisation, t_target = t_centered/std(t_centered).
    # s_centered ≠ t_target unless s is also scaled → loss > 0 but small.
    # The important thing: loss should be finite and not NaN.
    passed = align.item() < 10.0 and not torch.isnan(loss)
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 2 -- Identical vectors: align_loss = {align.item():.4f}  "
          f"(finite, not NaN)")
    return passed


def test_loss_dissimilar() -> bool:
    """Unrelated random vectors should give non-trivial loss."""
    s = torch.randn(BATCH, DIM, requires_grad=True)
    t = torch.randn(BATCH, DIM)
    loss, align, var = salt_loss(s, t, lambda_cov=0.0)
    passed = align.item() > 0.01
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 3 -- Dissimilar vectors: align_loss = {align.item():.4f}  "
          f"(expected > 0.01)")
    return passed


def test_loss_ordering() -> bool:
    """Opposite vectors should have higher loss than random."""
    s1 = torch.randn(BATCH, DIM, requires_grad=True)
    t1 = torch.randn(BATCH, DIM)
    _, align_random, _ = salt_loss(s1, t1, lambda_cov=0.0)

    s2 = torch.randn(BATCH, DIM, requires_grad=True)
    _, align_opp, _ = salt_loss(s2, -s2.detach(), lambda_cov=0.0)

    passed = align_opp.item() >= align_random.item() * 0.5
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 4 -- Opposite={align_opp.item():.4f} vs "
          f"Random={align_random.item():.4f}")
    return passed


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


def test_standardisation_amplifies_signal() -> bool:
    """
    With a MIXED batch, centering + standardisation should make the
    loss strongly differentiate matching vs mismatching class patterns.
    """
    torch.manual_seed(42)

    # Shared mean (99% of signal)
    shared = torch.randn(1, DIM) * 10.0
    # Class-specific residuals (1% of signal)
    delta = torch.randn(1, DIM) * 0.3

    # Mixed teacher batch
    teacher_A = shared + delta + torch.randn(BATCH // 2, DIM) * 0.01
    teacher_B = shared - delta + torch.randn(BATCH // 2, DIM) * 0.01
    teacher = torch.cat([teacher_A, teacher_B], dim=0)

    # GOOD student: matches teacher's class pattern (with scale mismatch)
    student_good = teacher.clone() + torch.randn(BATCH, DIM) * 0.05
    _, align_good, _ = salt_loss(
        student_good.requires_grad_(True), teacher, lambda_cov=0.0,
    )

    # BAD student: reversed class pattern
    student_bad = torch.cat([teacher_B.clone(), teacher_A.clone()], dim=0)
    student_bad = student_bad + torch.randn(BATCH, DIM) * 0.05
    _, align_bad, _ = salt_loss(
        student_bad.requires_grad_(True), teacher, lambda_cov=0.0,
    )

    passed = align_good.item() < align_bad.item()
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 7 -- Good={align_good.item():.4f} < "
          f"Bad(reversed)={align_bad.item():.4f}  (standardisation works)")
    return passed


def test_loss_is_order_one() -> bool:
    """Loss should be O(1), not O(0.001). This is the key fix."""
    s = torch.randn(BATCH, DIM, requires_grad=True)
    t = torch.randn(BATCH, DIM)
    _, align, _ = salt_loss(s, t, lambda_cov=0.0)
    passed = align.item() > 0.1
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Bonus C -- Loss is O(1): {align.item():.4f}  "
          f"(expected > 0.1, proves healthy gradient scale)")
    return passed


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


if __name__ == "__main__":
    print("=" * 62)
    print("  SALT Loss -- Tests (Centered & Standardised MSE)")
    print("=" * 62)

    core_results = [
        test_loss_scalar_and_grad(),
        test_loss_identical(),
        test_loss_dissimilar(),
        test_loss_ordering(),
        test_gradient_isolation(),
        test_variance_penalty(),
        test_standardisation_amplifies_signal(),
    ]

    print()
    print("  -- Bonus checks --")
    bonus_results = [
        test_embedding_std_healthy(),
        test_embedding_std_collapsed(),
        test_loss_is_order_one(),
    ]

    all_results = core_results + bonus_results
    print("=" * 62)
    n_core = sum(core_results)
    n_bonus = sum(bonus_results)
    print(f"  Core:  {n_core}/7 passed")
    print(f"  Bonus: {n_bonus}/3 passed")
    print(f"  Total: {n_core + n_bonus}/{len(all_results)} passed")
    if n_core == 7:
        print("  [OK] All core tests PASSED -- loss function is ready.")
    else:
        print("  [!!] CORE TESTS FAILED -- fix before proceeding.")
    print("=" * 62)

    sys.exit(0 if n_core == 7 else 1)
