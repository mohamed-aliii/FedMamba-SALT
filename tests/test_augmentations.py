"""
tests/test_augmentations.py — Smoke tests for the dual augmentation pipeline.

Creates a temporary directory with synthetic images, wraps them in a
DualViewDataset, and verifies:
  1. Both views are tensors of shape (3, 224, 224).
  2. The two views are NOT identical (augmentations are random & different).
  3. Both views fall within a reasonable normalised pixel range (≈ -3 to +3).
  4. Saves a side-by-side teacher/student visualisation to tests/aug_sample.png
     for manual inspection.

Run from the project root:
    python -m tests.test_augmentations
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations.medical_aug import (
    AddGaussianNoise,
    DualViewDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_student_transform,
    get_teacher_transform,
)

# =====================================================================
#  Helpers
# =====================================================================
OUTPUT_DIR = PROJECT_ROOT / "tests"
SAMPLE_PATH = OUTPUT_DIR / "aug_sample.png"


def _create_test_images(directory: str, n: int = 5, size: int = 256) -> None:
    """Create ``n`` random RGB images inside ``directory/class0/``."""
    class_dir = os.path.join(directory, "class0")
    os.makedirs(class_dir, exist_ok=True)
    for i in range(n):
        arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img.save(os.path.join(class_dir, f"img_{i:03d}.png"))


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and convert to (H, W, 3) uint8 for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# =====================================================================
#  Tests
# =====================================================================
def test_output_shape(teacher_view: torch.Tensor, student_view: torch.Tensor) -> bool:
    """Test 1: Both views must be (3, 224, 224)."""
    expected = (3, 224, 224)
    t_ok = teacher_view.shape == torch.Size(expected)
    s_ok = student_view.shape == torch.Size(expected)
    passed = t_ok and s_ok
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 1 — Shape: teacher={tuple(teacher_view.shape)}, "
          f"student={tuple(student_view.shape)}  (expected {expected})")
    return passed


def test_views_differ(teacher_view: torch.Tensor, student_view: torch.Tensor) -> bool:
    """Test 2: The two views must NOT be identical."""
    passed = not torch.equal(teacher_view, student_view)
    tag = "PASS" if passed else "FAIL"
    max_diff = (teacher_view - student_view).abs().max().item()
    print(f"  [{tag}] Test 2 — Views differ (max |diff| = {max_diff:.4f})")
    return passed


def test_pixel_range(teacher_view: torch.Tensor, student_view: torch.Tensor) -> bool:
    """Test 3: Normalised values should be roughly in [-3, +3]."""
    lo, hi = -3.5, 3.5  # generous margin for noise
    t_ok = teacher_view.min().item() >= lo and teacher_view.max().item() <= hi
    s_ok = student_view.min().item() >= lo and student_view.max().item() <= hi
    passed = t_ok and s_ok
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] Test 3 — Pixel range: teacher=[{teacher_view.min():.2f}, "
          f"{teacher_view.max():.2f}], student=[{student_view.min():.2f}, "
          f"{student_view.max():.2f}]  (expected within [{lo}, {hi}])")
    return passed


def test_save_visual(teacher_view: torch.Tensor, student_view: torch.Tensor) -> bool:
    """Test 4: Save a side-by-side PNG for manual inspection."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(_denormalize(teacher_view))
        axes[0].set_title("Teacher view (clean)", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(_denormalize(student_view))
        axes[1].set_title("Student view (corrupted)", fontsize=13)
        axes[1].axis("off")

        fig.suptitle(
            "Asymmetric Augmentation — Same source image",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(str(SAMPLE_PATH), dpi=150)
        plt.close(fig)

        passed = SAMPLE_PATH.exists()
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] Test 4 — Saved visual to {SAMPLE_PATH}")
    except ImportError:
        print("  [SKIP] Test 4 — matplotlib not installed, cannot save visual")
        passed = True  # non-fatal
    return passed


# =====================================================================
#  Main
# =====================================================================
def main() -> None:
    print("=" * 62)
    print("  Dual Augmentation Pipeline — Smoke Tests")
    print("=" * 62)

    # Create a temporary image folder
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_test_images(tmpdir, n=5, size=256)

        # Use torchvision ImageFolder as the base dataset
        from torchvision.datasets import ImageFolder

        base_ds = ImageFolder(root=tmpdir)
        dual_ds = DualViewDataset(base_ds)

        print(f"\n  Dataset length: {len(dual_ds)}")
        print(f"  Teacher transform:\n    {dual_ds.teacher_transform}")
        print(f"  Student transform:\n    {dual_ds.student_transform}\n")

        # Fetch one sample
        teacher_view, student_view = dual_ds[0]

        results = [
            test_output_shape(teacher_view, student_view),
            test_views_differ(teacher_view, student_view),
            test_pixel_range(teacher_view, student_view),
            test_save_visual(teacher_view, student_view),
        ]

    print("=" * 62)
    n_passed = sum(results)
    print(f"  Result: {n_passed}/4 tests passed")
    if n_passed == 4:
        print("  ✓ All tests PASSED — augmentation pipelines are ready.")
    else:
        print("  ✗ SOME TESTS FAILED — review before proceeding.")
    print("=" * 62)

    sys.exit(0 if n_passed == 4 else 1)


if __name__ == "__main__":
    main()
