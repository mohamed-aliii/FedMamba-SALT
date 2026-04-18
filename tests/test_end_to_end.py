"""
tests/test_end_to_end.py -- Full pipeline smoke test.

Runs a complete forward pass + one gradient step of the SALT training
pipeline on a synthetic batch (no real data needed).  Confirms that
all components from Phases 1-6 integrate correctly before committing
to a multi-epoch experiment.

Should complete in under 60 seconds on any CUDA-capable machine
(or CPU with mock Mamba).

Run from the project root:
    python -m tests.test_end_to_end
"""

import copy
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.inception_mamba import InceptionMambaEncoder
from models.vit_teacher import FrozenViTTeacher
from objectives.salt_loss import embedding_std, salt_loss

BATCH = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_end_to_end() -> bool:
    """
    Full integration test:
      1. Instantiate teacher, student, projector
      2. Create synthetic teacher_view and student_view  (4, 3, 224, 224)
      3. Full forward pass through all three models
      4. Compute SALT loss
      5. loss.backward()
      6. One optimizer step
      7. Assert loss is finite (not NaN, not inf)
      8. Assert student parameters changed (optimizer step worked)
    """
    t0 = time.time()
    results = []

    print("  [1/8] Instantiating models...")
    teacher = FrozenViTTeacher.for_testing().to(DEVICE)
    student = InceptionMambaEncoder(
        patch_size=16, embed_dim=256, depth=4, out_dim=768,
    ).to(DEVICE)
    # Verify teacher is frozen
    teacher_trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    ok = teacher_trainable == 0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Teacher has 0 trainable params: {teacher_trainable}")
    results.append(ok)

    # Snapshot student params BEFORE the optimizer step
    student_params_before = {
        name: p.clone().detach()
        for name, p in student.named_parameters()
        if p.requires_grad
    }

    # Optimizer on student + projector only
    params = list(student.parameters())
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.05)

    # ----- Synthetic batch -----
    print("  [2/8] Creating synthetic batch (4, 3, 224, 224)...")
    teacher_view = torch.randn(BATCH, 3, 224, 224, device=DEVICE)
    student_view = torch.randn(BATCH, 3, 224, 224, device=DEVICE)

    # ----- Forward pass -----
    print("  [3/8] Teacher forward (frozen)...")
    with torch.no_grad():
        t_emb = teacher(teacher_view)            # (B, 768)

    print("  [4/8] Student + projector forward...")
    student.train()
    s_emb = student(student_view)                 # (B, 768)
    s_proj = s_emb                     # (B, 768)

    # ----- SALT loss -----
    print("  [5/8] Computing SALT loss...")
    loss, align_loss, var_loss = salt_loss(s_proj, t_emb)

    # ----- Check loss is finite -----
    loss_val = loss.item()
    is_finite = torch.isfinite(loss).item()
    tag = "PASS" if is_finite else "FAIL"
    print(f"  [{tag}] Loss is finite: {loss_val:.6f} (align={align_loss.item():.6f}, var={var_loss.item():.6f})")
    results.append(is_finite)

    # ----- Check embedding_std -----
    s_std = embedding_std(s_proj.detach())
    t_std = embedding_std(t_emb.detach())
    std_ok = s_std > 0.0 and t_std > 0.0
    tag = "PASS" if std_ok else "FAIL"
    print(f"  [{tag}] Embedding std: student={s_std:.4f}, teacher={t_std:.4f}")
    results.append(std_ok)

    # ----- Backward -----
    print("  [6/8] loss.backward()...")
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist on student
    grad_count = sum(1 for p in student.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in student.parameters() if p.requires_grad)
    grads_ok = grad_count == total_trainable
    tag = "PASS" if grads_ok else "FAIL"
    print(f"  [{tag}] Gradients: {grad_count}/{total_trainable} student params have grads")
    results.append(grads_ok)

    # ----- Optimizer step -----
    print("  [7/8] Optimizer step + grad clipping...")
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()

    # ----- Check params changed -----
    print("  [8/8] Checking student params changed after step...")
    any_changed = False
    for name, p in student.named_parameters():
        if p.requires_grad and name in student_params_before:
            if not torch.equal(p.data, student_params_before[name]):
                any_changed = True
                break

    tag = "PASS" if any_changed else "FAIL"
    print(f"  [{tag}] Student parameters updated by optimizer")
    results.append(any_changed)

    elapsed = time.time() - t0
    return all(results), elapsed


def main() -> None:
    print("=" * 60)
    print(f"  End-to-End Smoke Test  (device={DEVICE})")
    print("=" * 60)

    all_passed, elapsed = test_end_to_end()

    print("=" * 60)
    if all_passed:
        print(f"  [OK] ALL CHECKS PASSED in {elapsed:.1f}s")
        print("  Pipeline is ready for multi-epoch training.")
    else:
        print(f"  [!!] SOME CHECKS FAILED in {elapsed:.1f}s")
        print("  Fix issues before launching training.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
