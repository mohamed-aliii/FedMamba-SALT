"""
augmentations/retina_dataset.py -- Retina dataset loader for SSL-FL format.

The SSL-FL Retina dataset uses a custom format:
  - Images are stored as .npy files in ``data/Retina/train/`` and
    ``data/Retina/test/``
  - Labels are stored in ``data/Retina/labels.csv`` as ``filename,class_id``
  - Train/test splits are defined by CSV files listing filenames:
    ``data/Retina/central/train.csv`` (centralized split) or
    ``data/Retina/5_clients/split_*/client_*.csv`` (federated splits)

This module provides :class:`RetinaDataset` which loads images from
a split CSV, maps labels from ``labels.csv``, and converts .npy arrays
to PIL Images for compatibility with torchvision transforms.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.transform import resize as sk_resize
from torch.utils.data import Dataset


# ======================================================================
# Retina-specific normalization constants (from SSL-FL datasets.py)
# ======================================================================
RETINA_MEAN = (0.5007, 0.5010, 0.5019)
RETINA_STD = (0.0342, 0.0535, 0.0484)


class RetinaDataset(Dataset):
    """
    Dataset loader for SSL-FL Retina format.

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
                f"Download the Retina dataset and ensure labels.csv exists."
            )
        self.labels = {}
        with open(labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                self.labels[parts[0]] = int(float(parts[1]))

        # --- Load split file ---
        if split_type == "central":
            split_path = os.path.join(data_path, split_type, split_csv)
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

        print(f"[RetinaDataset] {phase}: {len(self.img_paths)} images, "
              f"{self.num_classes} classes, split={split_type}/{split_csv}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        index = index % len(self.img_paths)
        fname = self.img_paths[index]
        path = os.path.join(self.data_path, self.phase, fname)

        # Load .npy and resize
        img = np.load(path)
        img = sk_resize(img, (self.resize_to, self.resize_to))

        # Ensure 3-channel
        if img.ndim < 3:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        # Convert to PIL intelligently based on the mathematical data bounds
        # skimage.resize outputs float type. If original array natively spanned
        # 0.0 - 255.0, multiplying by 255 corrupts it into pure white noise.
        if img.max() <= 1.0:
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = img.clip(0, 255).astype(np.uint8)
            
        pil_img = Image.fromarray(img_uint8)

        # Label
        target = self.labels.get(fname, 0)

        if self.transform is not None:
            pil_img = self.transform(pil_img)

        return pil_img, target

    @property
    def classes(self):
        """Return sorted list of class names (as strings) for compatibility."""
        return [str(c) for c in self.class_set]
