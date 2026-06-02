"""
augmentations/retina_dataset.py -- SSL-FL image dataset loader.

The SSL-FL datasets use a shared metadata format:
  - Images are stored in ``train/`` and ``test/``. Retina uses .npy arrays;
    COVID-FL and other image datasets use standard image files.
  - Labels are stored in ``data/Retina/labels.csv`` as ``filename,class_id``
  - Train/test splits are defined by CSV files listing filenames:
    ``data/Retina/central/train.csv`` (centralized split) or
    ``data/Retina/5_clients/split_*/client_*.csv`` (federated splits)

This module provides :class:`RetinaDataset` which loads images from
a split CSV, maps labels from ``labels.csv``, and converts arrays/images
to PIL Images for compatibility with torchvision transforms. The class name
is retained for backward compatibility with the existing training scripts.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    from skimage.transform import resize as sk_resize
except ImportError:  # pragma: no cover - exercised in minimal test envs
    sk_resize = None


# ======================================================================
# Retina-specific normalization constants (from SSL-FL datasets.py)
# ======================================================================
RETINA_MEAN = (0.5007, 0.5010, 0.5019)
RETINA_STD = (0.0342, 0.0535, 0.0484)


def _array_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale/RGB numpy image to uint8 without changing scale twice."""
    if img.max() <= 1.0:
        return (img * 255).clip(0, 255).astype(np.uint8)
    return img.clip(0, 255).astype(np.uint8)


def _resize_array(img: np.ndarray, size: int) -> np.ndarray:
    """Resize an ndarray image; use skimage when available, PIL as fallback."""
    if sk_resize is not None:
        return sk_resize(img, (size, size), preserve_range=True)

    img_uint8 = _array_to_uint8(img)
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(pil_img)


class RetinaDataset(Dataset):
    """
    Dataset loader for SSL-FL image classification format.

    Each item returns ``(PIL.Image, int_label)``, compatible with
    :class:`DualViewDataset` and torchvision transforms.

    Args:
        data_path: Root directory containing ``train/``, ``test/``,
            ``labels.csv``, and ``central/``.
        phase: ``'train'`` or ``'test'``.
        split_type: ``'central'`` for centralized training.
        split_csv: Name of the split CSV file within the split directory.
            For centralized: ``'train.csv'``; for federated: ``'client_1.csv'``.
        transform: Optional torchvision transform to apply.
        resize_to: Resize .npy images to this size (default 256 per SSL-FL).
            Standard image files are left to torchvision transforms.
    """

    def __init__(
        self,
        data_path: str,
        phase: str = "train",
        split_type: str = "central",
        split_csv: str = "train.csv",
        transform: Optional[Callable] = None,
        resize_to: int = 256,
    ):
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.resize_to = resize_to

        # --- Load labels ---
        labels_path = os.path.join(data_path, "labels.csv")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(
                f"labels.csv not found at: {labels_path}\n"
                f"Download the SSL-FL dataset and ensure labels.csv exists."
            )
        self.labels = {}
        with open(labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    self.labels[parts[0]] = int(float(parts[1]))
                except ValueError:
                    # Skip a possible CSV header.
                    continue

        # --- Load split file ---
        if split_type == "central":
            # Some SSL-FL releases keep train.csv/test.csv at the dataset root,
            # while the local Retina notebooks use central/train.csv.
            candidates = [
                os.path.join(data_path, split_type, split_csv),
                os.path.join(data_path, split_csv),
            ]
            split_path = next((p for p in candidates if os.path.isfile(p)), candidates[0])
        else:
            # e.g. data/Retina/5_clients/split_1/client_1.csv
            split_path = os.path.join(data_path, split_csv)

        if not os.path.isfile(split_path):
            raise FileNotFoundError(
                f"Split CSV not found: {split_path}\n"
                f"For centralized training, ensure {split_type}/{split_csv} exists."
            )

        self.img_paths = []
        with open(split_path, "r") as f:
            for line in f:
                fname = line.strip().split(",")[0]
                if fname:
                    self.img_paths.append(fname)
        # Deduplicate while preserving order
        self.img_paths = list(dict.fromkeys(self.img_paths))

        # --- Get classes ---
        self.class_set = sorted(set(
            self.labels[f] for f in self.img_paths if f in self.labels
        ))
        self.num_classes = len(self.class_set)
        self.targets = [self.labels.get(f, 0) for f in self.img_paths]

        print(f"[RetinaDataset] {phase}: {len(self.img_paths)} images, "
              f"{self.num_classes} classes, split={split_type}/{split_csv}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        max_retries = 10
        for attempt in range(max_retries):
            idx = index % len(self.img_paths) if attempt == 0 else random.randint(0, len(self.img_paths) - 1)
            fname = self.img_paths[idx]
            path = fname if os.path.isabs(fname) else os.path.join(self.data_path, self.phase, fname)
            if not os.path.exists(path):
                alt = os.path.join(self.data_path, fname)
                if os.path.exists(alt):
                    path = alt

            try:
                if Path(path).suffix.lower() == ".npy":
                    # Load .npy and resize
                    img = np.load(path)
                    img = _resize_array(img, self.resize_to)
                else:
                    pil_img = Image.open(path).convert("RGB")
                    img = np.asarray(pil_img)
            except (ValueError, Exception) as e:
                warnings.warn(
                    f"[RetinaDataset] Corrupt file skipped: {fname} ({e})",
                    stacklevel=2,
                )
                continue

            # Ensure 3-channel
            if img.ndim < 3:
                img = np.stack((img,) * 3, axis=-1)
            elif img.shape[2] >= 3:
                img = img[:, :, :3]

            # Convert to PIL intelligently based on the mathematical data bounds
            # skimage.resize outputs float type. If original array natively spanned
            # 0.0 - 255.0, multiplying by 255 corrupts it into pure white noise.
            img_uint8 = _array_to_uint8(img)
                
            pil_img = Image.fromarray(img_uint8)

            # Label
            target = self.labels.get(fname, 0)

            if self.transform is not None:
                pil_img = self.transform(pil_img)

            return pil_img, target

        raise RuntimeError(
            f"[RetinaDataset] Failed to load a valid sample after {max_retries} retries "
            f"(last index={index}). Too many corrupt .npy files in the dataset."
        )

    @property
    def classes(self):
        """Return sorted list of class names (as strings) for compatibility."""
        return [str(c) for c in self.class_set]
