# FedMamba-SALT

**Self-supervised Asymmetric Learning via Teacher–Student Distillation**

A centralized self-supervised learning experiment that trains a lightweight
**Inception-Mamba** student encoder to produce embeddings matching those of a
frozen **MAE-pretrained ViT-B/16** teacher. The learning signal comes from
**asymmetric augmentation**: teacher and student see differently augmented views
of the same image, and the student must learn to produce teacher-aligned
representations despite the augmentation gap.

---

## Project Layout

```
fedmamba_salt/
├── models/            # Student (Inception-Mamba) and Teacher (ViT-B/16) definitions
├── objectives/        # Loss functions (cosine embedding loss, etc.)
├── augmentations/     # Asymmetric augmentation pipelines (teacher vs. student views)
├── eval/              # Downstream evaluation (linear probing, kNN, etc.)
├── tests/             # Unit & integration tests
├── data/
│   └── ckpts/         # Pretrained checkpoints (MAE ViT-B/16, etc.)
├── configs/           # YAML/JSON experiment configurations
├── scripts/           # Shell helpers (launch training, download ckpts, …)
├── train_centralized.py   # Main training entry-point
├── requirements.txt       # Pinned dependencies
└── README.md
```

## Quick Start

```bash
# 1. Create and activate a conda environment (recommended)
conda create -n fedmamba python=3.10 -y
conda activate fedmamba

# 2. Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Place the MAE ViT-B/16 checkpoint in data/ckpts/
#    (see scripts/ for download helpers)

# 5. Run the smoke test to verify everything works
python -m tests.test_end_to_end

# 6. Run centralized training
python train_centralized.py \
    --data_path /path/to/retina \
    --teacher_ckpt data/ckpts/mae_vit_base.pth \
    --output_dir outputs/retina_centralized \
    --epochs 100

# 7. Evaluate the trained encoder
python -m eval.linear_probe \
    --encoder_ckpt outputs/retina_centralized/ckpt_latest.pth \
    --data_path /path/to/retina \
    --num_classes 5
```

Or use the all-in-one launch script:

```bash
bash scripts/run_centralized_retina.sh
```

## Running the Experiment

### 1. Smoke Test

Before committing to a full training run, verify all components integrate:

```bash
python -m tests.test_end_to_end
```

This instantiates the full pipeline (teacher + student + projector), runs a
forward pass and one gradient step on synthetic data, and confirms everything
is wired correctly. Should finish in under 60 seconds.

### 2. Pre-training

```bash
python train_centralized.py \
    --data_path /path/to/retina \
    --epochs 100 --batch_size 128 --lr 1e-3
```

All hyperparameters are also stored in `configs/retina_centralized.yaml`.
Training automatically resumes from `ckpt_latest.pth` if interrupted.

### 3. Evaluation

```bash
# Linear probe (diagnostic)
python -m eval.linear_probe \
    --encoder_ckpt outputs/retina_centralized/ckpt_latest.pth \
    --data_path /path/to/retina --num_classes 5 --mode linear_probe

# Full fine-tuning (apples-to-apples comparison with Fed-MAE)
python -m eval.linear_probe \
    --encoder_ckpt outputs/retina_centralized/ckpt_latest.pth \
    --data_path /path/to/retina --num_classes 5 --mode full_finetune
```

## Success Criteria

A successful centralized experiment must meet **all three** conditions:

| # | Condition | Threshold | Why |
|---|---|---|---|
| 1 | **SALT loss at epoch 100** | < 0.3 | Confirms the student is learning to match the teacher's embedding space |
| 2 | **embedding_std throughout training** | never < 0.05 | Ensures no representation collapse (all embeddings collapsing to the same vector) |
| 3 | **Linear probe accuracy on Retina test set** | within 5% of 77.43% | Fed-MAE achieves 77.43% with full fine-tuning on Retina Split-3. Since linear probing is strictly weaker than fine-tuning, reaching ~72%+ with a frozen encoder indicates strong representations |

## Expected Training Behavior

Loss = 1 - cosine_similarity (range [0, 2])

| Epoch | Expected Loss | Notes |
|---|---|---|
| 1 | ~0.9 - 1.0 | Random initialization; cosine similarity near 0 for uncorrelated vectors |
| 10 | ~0.4 - 0.7 | Warmup complete, student starting to align |
| 30 | ~0.15 - 0.4 | Student learning meaningful features |
| 100 | ~0.05 - 0.2 | Plateau; diminishing returns |

**Warning signs:**
- Loss stuck above 0.8 after epoch 20 --> check learning rate, augmentations, or data loading
- embedding_std drops below 0.05 --> representation collapse, stop and debug
- Loss goes to NaN --> gradient explosion, reduce learning rate or check for data issues

## Dependency Notes

| Package | Pin | Reason |
|---|---|---|
| `timm` | `0.3.2` | MAE ViT-B/16 checkpoint was saved against this version's `VisionTransformer` class; newer versions changed the interface and break weight loading |
| `mamba-ssm` | `1.2.0` | Mamba CUDA kernels have breaking API changes across versions |
| `causal-conv1d` | `1.2.0` | Hard dependency of `mamba-ssm`; must version-match |

## License

TBD
