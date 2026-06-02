import json
from pathlib import Path


SRC = Path("notebooks/FedMamba_SALT_Federated_momentum_Prox_split1.ipynb")
DST = Path("notebooks/FedMamba_SALT_Federated_momentum_Prox_COVIDFL.ipynb")


def as_source(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def set_cell(nb: dict, idx: int, text: str) -> None:
    nb["cells"][idx]["source"] = as_source(text)


with SRC.open("r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

set_cell(
    nb,
    0,
    """
# FedMamba-SALT: COVID-FL Real-World Split Federated Experiment

> Clean paper notebook for the COVID-FL real-world non-IID federated experiment, derived from the Split 1 federated momentum/prox notebook.

This notebook runs:
1. Environment setup and checkpoint verification.
2. COVID-FL split inspection and per-site distribution plots.
3. Federated SALT pre-training on `12_clients/split_real`.
4. Linear probe, federated fine-tuning, TTA, diagnostics, and artifact backup.
""",
)

set_cell(
    nb,
    4,
    """
### 1.2 Set Paths (EDIT THESE)

Configure the repository location, COVID-FL dataset root, teacher checkpoint, and real-world split.

Expected COVID-FL layout:

```text
COVID-FL/
  train/
  test/
  labels.csv
  train.csv
  test.csv
  12_clients/split_real/*.csv
```
""",
)

set_cell(
    nb,
    5,
    """
# ============================================================
# EDIT THESE PATHS TO MATCH YOUR GOOGLE DRIVE LAYOUT
# ============================================================
DRIVE_REPO           = "/content/drive/MyDrive/fedmamba_salt"
DRIVE_DATASET_DRIVE  = "/content/drive/MyDrive/COVID-FL"      # must have train/, test/, labels.csv
DRIVE_CKPT           = "/content/drive/MyDrive/original/covidfl/mae_vit_base.pth"
ZIP_PATH             = "/content/drive/MyDrive/COVID-FL.zip"  # optional local-copy source

DATASET_NAME    = "COVID-FL"
DATASET_PRESET  = "covidfl"
DATASET_DIRNAME = "COVID-FL"

# ============================================================
# COVID-FL REAL-WORLD FEDERATED PRETRAINING SETTINGS
# ============================================================
SPLIT_TYPE  = "split_real"
N_CLIENTS   = 12
MAX_ROUNDS  = 400
E_EPOCH     = 1
MU          = 0
BATCH_SIZE  = 64
LR          = 3.75e-4
MASK_RATIO  = 0.30
NUM_CLASSES = 3
NUM_WORKERS = 10

ALGO_NAME = "FedProx" if MU > 0 else "FedAvg"
print(f"Dataset:    {DATASET_NAME}")
print(f"Algorithm:  {ALGO_NAME} (mu={MU})")
print(f"Split:      {SPLIT_TYPE}")
print(f"Clients:    {N_CLIENTS}")
print(f"Rounds:     {MAX_ROUNDS}")
print(f"E_epoch:    {E_EPOCH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"LR:         {LR}")
print(f"Mask ratio: {MASK_RATIO}")
""",
)

set_cell(
    nb,
    6,
    """
### 1.2.b Copy COVID-FL Dataset Locally

Copying the zip to `/content/` reduces Google Drive I/O stalls during training. If the zip is not present, the notebook uses the Drive directory directly.
""",
)

set_cell(
    nb,
    7,
    """
import os
import time
import zipfile

LOCAL_DATASET = "/content/COVIDFL_local"

start_time = time.time()
if os.path.exists(ZIP_PATH):
    print(f"Preparing local dataset from {ZIP_PATH} -> {LOCAL_DATASET}")
    if not os.path.exists(LOCAL_DATASET):
        os.makedirs(LOCAL_DATASET, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_DATASET)
    else:
        print(f"Directory {LOCAL_DATASET} already exists. Skipping extraction.")

    candidates = [
        os.path.join(LOCAL_DATASET, DATASET_DIRNAME),
        os.path.join(LOCAL_DATASET, "COVID-FL"),
        os.path.join(LOCAL_DATASET, "COVIDFL"),
        os.path.join(LOCAL_DATASET, "COVID_FL"),
    ]
    DRIVE_DATASET = next((p for p in candidates if os.path.exists(p)), LOCAL_DATASET)
elif os.path.exists(DRIVE_DATASET_DRIVE):
    print("Dataset zip not found; using Drive dataset directory directly.")
    DRIVE_DATASET = DRIVE_DATASET_DRIVE
else:
    raise FileNotFoundError(
        f"Neither ZIP_PATH nor DRIVE_DATASET_DRIVE exists:\\n"
        f"  ZIP_PATH={ZIP_PATH}\\n"
        f"  DRIVE_DATASET_DRIVE={DRIVE_DATASET_DRIVE}"
    )

labels_path = os.path.join(DRIVE_DATASET, "labels.csv")
if os.path.exists(labels_path):
    detected_labels = set()
    with open(labels_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    detected_labels.add(int(float(parts[1])))
                except ValueError:
                    pass
    if detected_labels:
        NUM_CLASSES = len(detected_labels)
        print(f"Detected {NUM_CLASSES} classes from labels.csv: {sorted(detected_labels)}")

elapsed = time.time() - start_time
print(f"Dataset ready in {elapsed:.1f} seconds")
print(f"Dataset path for training: {DRIVE_DATASET}")
""",
)

set_cell(
    nb,
    8,
    """
import os
import shutil

# Some SSL-FL releases keep central train/test CSVs at the dataset root.
# If a central/ directory exists in Drive but not in the local copy, mirror it.
source_central = os.path.join(DRIVE_DATASET_DRIVE, "central")
local_central = os.path.join(DRIVE_DATASET, "central")

if os.path.exists(source_central):
    os.makedirs(local_central, exist_ok=True)
    for item in os.listdir(source_central):
        src = os.path.join(source_central, item)
        dst = os.path.join(local_central, item)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"Verified central split files in {local_central}")
else:
    print(f"Central split source not found: {source_central}; root train.csv/test.csv fallback is supported.")
""",
)

set_cell(
    nb,
    12,
    """
import shutil, os

# Copy the MAE checkpoint to the expected local path
CKPT_DIR = os.path.join(LOCAL_REPO, "data", "ckpts")
CKPT_PATH = os.path.join(CKPT_DIR, "mae_vit_base.pth")
os.makedirs(CKPT_DIR, exist_ok=True)
if not os.path.exists(CKPT_PATH):
    shutil.copy2(DRIVE_CKPT, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1e6
    print(f"Checkpoint copied: {size_mb:.1f} MB")
else:
    print(f"Checkpoint already present: {CKPT_PATH}")

# Define output paths for federated experiments
OUTPUT_RUN_NAME = f"{DATASET_PRESET}_{ALGO_NAME.lower()}_{SPLIT_TYPE}"
OUTPUT_DIR = os.path.join(LOCAL_REPO, "outputs", OUTPUT_RUN_NAME)
DRIVE_OUTPUT = os.path.join(DRIVE_REPO, "outputs", OUTPUT_RUN_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DRIVE_OUTPUT, exist_ok=True)

print(f"Local output:  {OUTPUT_DIR}")
print(f"Drive output:  {DRIVE_OUTPUT}")
""",
)

set_cell(
    nb,
    28,
    """
import os, sys
sys.path.insert(0, LOCAL_REPO)

# COVID-FL uses site-named CSVs under 12_clients/split_real.
split_dir = os.path.join(DRIVE_DATASET, f"{N_CLIENTS}_clients", SPLIT_TYPE)
print(f"Looking for client splits in: {split_dir}")
print()

if os.path.isdir(split_dir):
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".csv"))
    total_samples = 0
    for f in files:
        fp = os.path.join(split_dir, f)
        with open(fp) as fh:
            n = len([l for l in fh if l.strip()])
        total_samples += n
        print(f"  {f}: {n} samples")
    print(f"\\n  Found {len(files)} clients")
    print(f"  Total across all clients: {total_samples} samples")
    if len(files) != N_CLIENTS:
        print(f"  WARNING: expected {N_CLIENTS} client CSVs.")
else:
    print(f"  Directory not found: {split_dir}")
    print("  Check that COVID-FL is extracted with 12_clients/split_real/*.csv.")
""",
)

set_cell(
    nb,
    32,
    """
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from augmentations.medical_aug import (
    DualViewDataset,
    get_teacher_transform,
    get_student_transform,
    get_normalization_stats,
)
from augmentations.retina_dataset import RetinaDataset

NORM_MEAN, NORM_STD = get_normalization_stats(DATASET_PRESET)

def denormalize(tensor):
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(NORM_STD).view(3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)

split_dir = os.path.join(DRIVE_DATASET, f"{N_CLIENTS}_clients", SPLIT_TYPE)
client_csv = sorted(f for f in os.listdir(split_dir) if f.endswith(".csv"))[0]
split_csv = os.path.join(f"{N_CLIENTS}_clients", SPLIT_TYPE, client_csv)

base_ds = RetinaDataset(
    data_path=DRIVE_DATASET, phase="train",
    split_type="federated", split_csv=split_csv,
)
dual_ds = DualViewDataset(
    base_ds,
    teacher_transform=get_teacher_transform(dataset=DATASET_PRESET),
    student_transform=get_student_transform(dataset=DATASET_PRESET),
    dataset=DATASET_PRESET,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f"Augmentation Pairs ({client_csv}, {SPLIT_TYPE})", fontsize=14, fontweight='bold')

for i in range(4):
    idx = min(i * 10, len(dual_ds) - 1)
    t_view, s_view = dual_ds[idx]

    t_clean = TF.to_pil_image(denormalize(t_view))
    s_clean = TF.to_pil_image(denormalize(s_view))

    axes[0, i].imshow(t_clean)
    axes[0, i].set_title(f"Teacher #{i}", fontsize=10)
    axes[0, i].axis('off')

    axes[1, i].imshow(s_clean)
    axes[1, i].set_title(f"Student #{i}", fontsize=10)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_augmentation_pairs.png"), dpi=150)
plt.show()
""",
)

set_cell(
    nb,
    34,
    """
import os
from pathlib import Path
from PIL import Image
import numpy as np

split_dir = os.path.join(DRIVE_DATASET, f"{N_CLIENTS}_clients", SPLIT_TYPE)
sample_csv = sorted(f for f in os.listdir(split_dir) if f.endswith(".csv"))[0]
sample_csv_path = os.path.join(split_dir, sample_csv)

with open(sample_csv_path) as f:
    sample_fname = next(line.strip().split(",")[0] for line in f if line.strip())

sample_path = os.path.join(DRIVE_DATASET, "train", sample_fname)
print(f"Sample client CSV: {sample_csv}")
print(f"Sample image:      {sample_path}")

if not os.path.exists(sample_path):
    raise FileNotFoundError(sample_path)

if Path(sample_path).suffix.lower() == ".npy":
    img = np.load(sample_path)
    print(f"Loaded npy sample with shape={img.shape}, dtype={img.dtype}")
else:
    with Image.open(sample_path) as img:
        print(f"Loaded image sample with size={img.size}, mode={img.mode}")
""",
)

set_cell(
    nb,
    36,
    """
!python train_fedavg.py \\
    --data_path "{DRIVE_DATASET}" \\
    --dataset "{DATASET_PRESET}" \\
    --teacher_ckpt "{CKPT_PATH}" \\
    --output_dir "{DRIVE_OUTPUT}" \\
    --split_type {SPLIT_TYPE} \\
    --n_clients {N_CLIENTS} \\
    --max_rounds {MAX_ROUNDS} \\
    --E_epoch {E_EPOCH} \\
    --mu {MU} \\
    --batch_size {BATCH_SIZE} \\
    --lr {LR} \\
    --mask_ratio {MASK_RATIO} \\
    --num_workers {NUM_WORKERS} \\
    --save_every 10 \\
    --device cuda
""",
)

set_cell(
    nb,
    62,
    """
!python -m eval.linear_probe \\
    --encoder_ckpt "{FINAL_CKPT}" \\
    --data_path "{DRIVE_DATASET}" \\
    --dataset "{DATASET_PRESET}" \\
    --num_classes {NUM_CLASSES} \\
    --output_dir "{EVAL_LP_DIR}" \\
    --epochs 200 \\
    --batch_size 64 \\
    --lr 1.0e-3 \\
    --mode linear_probe
""",
)

set_cell(
    nb,
    68,
    """
mixup_flag = "--use_mixup" if FED_FT_USE_MIXUP else ""
focal_flag = "--use_focal_loss" if FED_FT_FOCAL else ""

cmd = (
    f"python train_fed_finetune.py"
    f' --encoder_ckpt "{FINAL_CKPT}"'
    f' --data_path "{DRIVE_DATASET}"'
    f' --dataset "{DATASET_PRESET}"'
    f" --num_classes {NUM_CLASSES}"
    f' --output_dir "{EVAL_FED_FT_DIR}"'
    f" --n_clients {N_CLIENTS}"
    f" --split_type {SPLIT_TYPE}"
    f" --max_rounds {FED_FT_MAX_ROUNDS}"
    f" --E_epoch {FED_FT_E_EPOCH}"
    f" --lr {FED_FT_LR}"
    f" --batch_size {FED_FT_BATCH_SIZE}"
    f" --mode {FED_FT_MODE}"
    f" --algo {FED_FT_ALGO}"
    f" --mu {FED_FT_MU}"
    f" {mixup_flag} {focal_flag}"
)

print(cmd)
!{cmd}
""",
)

set_cell(
    nb,
    78,
    """
import os
import shutil

TTA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "eval_tta")
os.makedirs(TTA_OUTPUT_DIR, exist_ok=True)

fed_ft_mode = globals().get("FED_FT_MODE", "federated_finetune")
fed_ft_local_dir = globals().get("EVAL_FED_FT_DIR", os.path.join(OUTPUT_DIR, f"eval_{fed_ft_mode}"))
fed_ft_drive_dir = os.path.join(DRIVE_OUTPUT, f"eval_{fed_ft_mode}")

TTA_CKPT_CANDIDATES = [
    os.path.join(fed_ft_local_dir, "ckpt_best_finetune.pth"),
    os.path.join(fed_ft_local_dir, "ckpt_latest.pth"),
    os.path.join(fed_ft_drive_dir, "ckpt_best_finetune.pth"),
    os.path.join(fed_ft_drive_dir, "ckpt_latest.pth"),
]
TTA_CKPT = next((p for p in TTA_CKPT_CANDIDATES if os.path.exists(p)), None)
if TTA_CKPT is None:
    raise FileNotFoundError(
        "No federated fine-tune checkpoint found for TTA. Run Section 8 first or verify the Drive backup path."
    )

print(f"TTA checkpoint: {TTA_CKPT}")
print(f"TTA output:     {TTA_OUTPUT_DIR}")

tta_cmd = (
    f"python -m eval.eval_tta"
    f' --ckpt "{TTA_CKPT}"'
    f' --data_path "{DRIVE_DATASET}"'
    f' --dataset "{DATASET_PRESET}"'
    f" --num_classes {NUM_CLASSES}"
    f' --split_type "{SPLIT_TYPE}"'
    f" --split_csv test.csv"
    f" --batch_size 64"
    f" --n_tta 4"
    f" --threshold_sweep"
    f' --output_dir "{TTA_OUTPUT_DIR}"'
    f" --device cuda"
)

print(tta_cmd)
!{tta_cmd}
""",
)

src82 = "".join(nb["cells"][82]["source"])
src82 = src82.replace('baseline_acc = 81.93', 'baseline_acc = 91.47')
src82 = src82.replace('centralized_ref_acc = 82.23', 'centralized_ref_acc = 95.77')
src82 = src82.replace('FedMamba-SALT: {ALGO_NAME} Split 1 Results', 'FedMamba-SALT: {ALGO_NAME} COVID-FL {SPLIT_TYPE} Results')
src82 = src82.replace('Centralized ref:', 'COVID-FL central ref:')
src82 = src82.replace('Fed-MAE baseline:', 'Fed-MAE split ref:')
src82 = src82.replace("'Centralized\\nReference', 'Fed-MAE\\nBaseline'", "'COVID-FL\\nCentral Ref', 'Fed-MAE\\nSplit Ref'")
src82 = src82.replace("FedMamba-SALT Split 1 Evaluation", "FedMamba-SALT COVID-FL Evaluation")
nb["cells"][82]["source"] = as_source(src82)

with DST.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"Wrote {DST}")
