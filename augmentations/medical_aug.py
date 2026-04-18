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

    Only geometric invariances (tight crop + horizontal flip) are applied.
    **No** colour jitter, blur, or noise — every augmentation added here
    makes the student's task easier and weakens the learning signal.

    Args:
        dataset: ``'imagenet'`` or ``'retina'`` to select normalization stats.
    """
    mean = RETINA_MEAN if dataset == "retina" else IMAGENET_MEAN
    std = RETINA_STD if dataset == "retina" else IMAGENET_STD
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ======================================================================
# Student augmentation pipeline (aggressive)
# ======================================================================
def get_student_transform(dataset: str = "imagenet") -> transforms.Compose:
    """
    Heavy augmentation for the student view.

    Simulates real-world sources of variation in medical imaging:
        • Aggressive random crop       → field-of-view variability
        • ColorJitter                   → scanner protocol differences
        • RandomGrayscale              → single-channel modalities
        • GaussianBlur                  → motion blur / defocus
        • AddGaussianNoise              → sensor noise

    Args:
        dataset: ``'imagenet'`` or ``'retina'`` to select normalization stats.
    """
    mean = RETINA_MEAN if dataset == "retina" else IMAGENET_MEAN
    std = RETINA_STD if dataset == "retina" else IMAGENET_STD
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.6,
            contrast=0.6,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.5, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        AddGaussianNoise(std=0.05),
    ])


# ======================================================================
# DualViewDataset — wraps any torchvision-style image dataset
# ======================================================================
class DualViewDataset(Dataset):
    """
    Wraps a torchvision-style dataset and returns paired
    ``(teacher_view, student_view)`` for each image.

    Both transforms are applied **independently** to the same underlying
    ``PIL.Image`` object so their randomness is uncorrelated.  Labels are
    loaded internally but **discarded** — they are not needed during
    self-supervised pre-training.

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

        # Both transforms receive the SAME PIL image; randomness inside
        # each Compose pipeline is independent (different crops, flips, etc).
        teacher_view: torch.Tensor = self.teacher_transform(img)
        student_view: torch.Tensor = self.student_transform(img)

        return teacher_view, student_view
