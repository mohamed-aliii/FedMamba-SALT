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

# 5. Run centralized training
python train_centralized.py --config configs/default.yaml
```

## Dependency Notes

| Package | Pin | Reason |
|---|---|---|
| `timm` | `0.3.2` | MAE ViT-B/16 checkpoint was saved against this version's `VisionTransformer` class; newer versions changed the interface and break weight loading |
| `mamba-ssm` | `1.2.0` | Mamba CUDA kernels have breaking API changes across versions |
| `causal-conv1d` | `1.2.0` | Hard dependency of `mamba-ssm`; must version-match |

## License

TBD
