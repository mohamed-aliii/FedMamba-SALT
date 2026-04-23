"""
augmentations/medical_aug.py — Asymmetric augmentation pipelines for SALT.

The entire learning signal in FedMamba-SALT comes from the *gap* between the
teacher view (minimally augmented, semantically clear) and the student view
(heavily corrupted to simulate real-world medical-imaging variation).  The
loss forces the student to produce the same 768-d embedding despite seeing
only the corrupted version.

Key design constraint:
    • Teacher pipeline must be **minimal** — every augmentation added to the
      teacher shrinks the semantic gap and makes the student's task trivially
      easy.
    • Student pipeline must be **aggressive** — simulate scanner protocol
      differences (contrast, brightness), motion blur, sensor noise, and
      field-of-view variability.
    • Spatial augmentations (crop, flip) are applied **independently** to
      teacher and student. The teacher uses GAP (global average pooling),
      so different crops of the same image produce similar embeddings.
      Independent crops force the student to learn semantic invariance,
      not just photometric denoising.
"""

from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ======================================================================
# Normalization constants
# ======================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Retina-specific stats (from SSL-FL datasets.py)
RETINA_MEAN = [0.5007, 0.5010, 0.5019]
RETINA_STD = [0.0342, 0.0535, 0.0484]


# ======================================================================
# Custom transform: additive Gaussian noise
# ======================================================================
class AddGaussianNoise:
    """
    Add i.i.d. Gaussian noise to a tensor.

    Compatible with ``torchvision.transforms.Compose``.  Applied *after*
    ``ToTensor`` and normalisation so the noise sits on top of the
    normalised pixel distribution.

    Args:
        std: Standard deviation of the noise.
    """

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


# ======================================================================
# Teacher augmentation pipeline (minimal)
# ======================================================================
def get_teacher_transform(dataset: str = "imagenet") -> transforms.Compose:
    """
    Minimal augmentation for the teacher view.

    Only applies ToTensor and Normalize. Spatial transforms (crop/flip)
    are handled externally by the DualViewDataset to ensure strict alignment
    with the student view for dense distillation.

    Args:
        dataset: ``'imagenet'`` or ``'retina'`` to select normalization stats.
    """
    mean = RETINA_MEAN if dataset == "retina" else IMAGENET_MEAN
    std = RETINA_STD if dataset == "retina" else IMAGENET_STD
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ======================================================================
# Student augmentation pipeline (medical-safe)
# ======================================================================
def get_student_transform(dataset: str = "imagenet") -> transforms.Compose:
    """
    Moderate augmentation for the student view.

    Applies photometric corruption (color jitter, grayscale, blur, noise)
    to force hard visual inference. Spatial transforms are handled externally
    to maintain strict alignment with the teacher.

    Args:
        dataset: ``'imagenet'`` or ``'retina'`` to select normalization stats.
    """
    mean = RETINA_MEAN if dataset == "retina" else IMAGENET_MEAN
    std = RETINA_STD if dataset == "retina" else IMAGENET_STD
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.02,
        ),
        transforms.RandomGrayscale(p=0.02),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        AddGaussianNoise(std=0.02),
    ])


# ======================================================================
# DualViewDataset — wraps any torchvision-style image dataset
# ======================================================================
class DualViewDataset(Dataset):
    """
    Wraps a torchvision-style dataset and returns paired
    ``(teacher_view, student_view)`` for each image.

    Crucially, it applies a shared spatial transform (Crop and Flip) to the
    underlying PIL image *before* passing it to the teacher and student
    pipelines. This guarantees strict spatial alignment (Student Patch i
    maps to Teacher Patch i), which is mathematically mandatory for
    Dense Distillation. Labels are loaded internally but **discarded**.

    Args:
        base_dataset: Any dataset whose ``__getitem__`` returns
            ``(PIL.Image, label)`` (e.g. ``torchvision.datasets.ImageFolder``).
        teacher_transform: Transform pipeline for the clean teacher view.
            Defaults to :func:`get_teacher_transform`.
        student_transform: Transform pipeline for the corrupted student view.
            Defaults to :func:`get_student_transform`.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        teacher_transform: Optional[Callable] = None,
        student_transform: Optional[Callable] = None,
    ):
        self.base_dataset = base_dataset
        
        # Shared spatial transform to ensure alignment for dense distillation
        self.shared_spatial = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        
        self.teacher_transform = teacher_transform or get_teacher_transform()
        self.student_transform = student_transform or get_student_transform()

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ``(teacher_view, student_view)`` — both tensors of shape
            ``(3, 224, 224)``.  Labels are discarded.
        """
        img, _label = self.base_dataset[index]

        # 1. Apply shared spatial transform (strict pixel alignment)
        base_img = self.shared_spatial(img)

        # 2. Apply independent photometric/tensor pipelines
        teacher_view: torch.Tensor = self.teacher_transform(base_img)
        student_view: torch.Tensor = self.student_transform(base_img)

        return teacher_view, student_view
