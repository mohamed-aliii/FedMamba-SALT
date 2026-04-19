<p align="center">
  <h1 align="center">FedMamba-SALT</h1>
  <p align="center">
    <strong>Federated Self-supervised Asymmetric Learning via Teacher–Student Distillation</strong><br>
    <em>Privacy-preserving representation learning for medical imaging using Mamba state-space models</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Mamba-SSM-green.svg" alt="Mamba">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## Overview

**FedMamba-SALT** is a federated self-supervised learning framework that trains a lightweight **Inception-Mamba** student encoder to produce embeddings matching those of a frozen **MAE-pretrained ViT-B/16** teacher. The system is designed for **privacy-sensitive medical imaging** scenarios where patient data cannot leave hospital premises.

The core idea is **asymmetric knowledge distillation**: the teacher (a large ViT-B, 85.8M params) provides rich visual representations, while a compact student encoder (31.8M params) learns to replicate them using a novel **Centered & Standardised MSE** loss. This approach enables hospitals to collaboratively train a high-quality medical image encoder **without sharing any patient data**.

### Key Innovation: Centered & Standardised Distillation

In medical imaging, class-discriminative signals are extremely subtle — retinal disease markers occupy only ~0.03% of the total embedding signal (cosine similarity 0.9996 between classes). Standard regression losses (MSE, SmoothL1, Cosine) learn the 99.97% shared retinal structure and ignore the disease signal entirely.

**FedMamba-SALT solves this** by:
1. **Batch-centering** both student and teacher embeddings to remove the dominant shared mean
2. **Standardising** the teacher's residuals (dividing by scalar std) to amplify the class signal to O(1)
3. Using **MSE** on the standardised targets, giving healthy O(1) gradients throughout training

This produces an **18× amplification** of disease-discriminative features that standard distillation methods miss.

---

## Key Advantages

### 1. Privacy-Preserving by Design
- Patient data **never leaves hospital premises** — only model weights are communicated
- Federated aggregation (FedAvg) operates on student encoder weights only
- No dependency on centralized data collection or transfer

### 2. Lightweight & Deployable
- **Student encoder: 31.8M params** (2.7× smaller than the ViT-B teacher)
- Inference on **CPU: ~90ms/image** — no GPU required at deployment
- Model checkpoint: **127 MB** (64 MB at FP16)
- Minimal hospital hardware: **2 GB GPU** for federated training, **CPU-only** for inference

### 3. Linear Scan Complexity via Mamba SSM
- Unlike ViT's O(n²) self-attention, **Mamba state-space models scale linearly** O(n) with sequence length
- 4-directional scanning (left→right, right→left, top→bottom, bottom→top) captures spatial relationships in all orientations
- **3.2× more parameter-efficient** than equivalent ViT architectures

### 4. Medical-Imaging Optimised
- **Inception-style multi-scale convolutions** (3×3, 5×5, 7×7 kernels) capture pathological features at multiple scales — from microaneurysms (5-20 pixels) to hemorrhages (50+ pixels)
- **Medical-safe augmentations** that preserve diagnostic signals while providing regularisation
- Retina-specific normalisation statistics for optimal preprocessing

### 5. Communication Efficient
- Only **157 MB uploaded per FL round** (student weights only — teacher is frozen and shared once)
- Teacher checkpoint distributed once (343 MB), never updated
- 50-round training: ~15 GB total network traffic per client

### 6. Robust Distillation Pipeline
- **Target standardisation** eliminates NaN/gradient vanishing issues common in medical KD
- **Covariance regularisation** (Barlow Twins-style) prevents dimension collapse
- **Variance penalty** on encoder output guards against representation collapse
- Automatic collapse detection and early stopping

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FedMamba-SALT Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Image (224×224×3)                                    │
│       │                                                     │
│       ├──── Teacher View ──── Frozen ViT-B/16 (MAE) ──┐    │
│       │     (clean aug)         85.8M params           │    │
│       │                                                │    │
│       └──── Student View ──── Inception-Mamba ────┐    │    │
│             (medical aug)      31.8M params       │    │    │
│                                    │              │    │    │
│                              Projection Head      │    │    │
│                               7.4M params         │    │    │
│                                    │              │    │    │
│                                    ▼              ▼    ▼    │
│                              ┌─────────────────────────┐    │
│                              │  Centered & Std. MSE    │    │
│                              │  + Covariance Penalty   │    │
│                              │  + Variance Guard       │    │
│                              └─────────────────────────┘    │
│                                                             │
├──────────── Inception-Mamba Block (×6) ─────────────────────┤
│                                                             │
│  Input (B, 196, 384)                                        │
│       │                                                     │
│       ├─── Inception Conv ─── [3×3, 5×5, 7×7, Pool] ──┐    │
│       │    Multi-scale local features                  │    │
│       │                                                │    │
│       ├─── 4-Dir Mamba SSM ── [LR, RL, TB, BT] ───────┤    │
│       │    Global context (linear complexity)          │    │
│       │                                                │    │
│       └─── FFN ── [Linear → GELU → Linear] ───────────┘    │
│            Feature refinement                               │
│                                                             │
│  Output (B, 196, 384) → Global Average Pool → (B, 768)     │
└─────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | Parameters | Memory (FP32) | Role |
|---|---|---|---|
| Teacher (ViT-B/16, MAE) | 85.8M | 343 MB | Frozen feature extractor |
| Student (Inception-Mamba) | 31.8M | 127 MB | Learnable encoder |
| Projection Head (3-layer MLP) | 7.4M | 29 MB | Aligns student → teacher space |
| Classifier (LayerNorm + Linear) | 3K | ~0 MB | Downstream task head |
| **Total (training)** | **125.0M** | **500 MB** | |
| **Total (inference)** | **31.8M** | **127 MB** | Student + classifier only |

---

## Project Structure

```
fedmamba_salt/
├── models/
│   ├── inception_mamba.py      # Inception-Mamba encoder (student)
│   ├── teacher.py              # Frozen ViT-B/16 MAE teacher
│   └── __init__.py
├── objectives/
│   └── salt_loss.py            # Centered & Standardised MSE loss
├── augmentations/
│   ├── medical_aug.py          # Medical-safe dual-view augmentations
│   └── retina_dataset.py       # Retina dataset loader with CSV splits
├── eval/
│   └── linear_probe.py         # Linear probe & full fine-tuning evaluation
├── tests/
│   ├── test_loss.py            # 10 loss function tests (7 core + 3 bonus)
│   └── test_end_to_end.py      # Full pipeline smoke test
├── configs/
│   └── retina_centralized.yaml # Training configuration
├── utils/
│   └── ckpt_compat.py          # Cross-version checkpoint loading
├── notebooks/
│   └── FedMamba_SALT_Centralized.ipynb  # Colab training notebook
├── scripts/                    # Shell helpers
├── data/
│   └── ckpts/                  # Pretrained checkpoints (MAE ViT-B/16)
├── train_centralized.py        # Main training entry-point
├── requirements.txt            # Pinned dependencies
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n fedmamba python=3.10 -y
conda activate fedmamba

# Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Mamba SSM (requires CUDA toolkit)
pip install mamba-ssm==1.2.0 causal-conv1d==1.2.0

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run the loss function tests (10/10 should pass)
python -m tests.test_loss

# Run the end-to-end smoke test
python -m tests.test_end_to_end
```

### 3. Prepare Data

Place your retinal fundus dataset in a directory with the following structure:
```
/path/to/retina/
├── train/
│   ├── 0/          # Normal retinal images
│   └── 1/          # Diseased retinal images
├── test/
│   ├── 0/
│   └── 1/
└── central/
    ├── train.csv   # Training split labels
    └── test.csv    # Test split labels
```

### 4. Download Teacher Checkpoint

Place the MAE ViT-B/16 pretrained checkpoint at `data/ckpts/mae_vit_base.pth`. This serves as the frozen teacher for knowledge distillation.

### 5. Pre-training

```bash
# Centralized pre-training
python train_centralized.py \
    --data_path /path/to/retina \
    --teacher_ckpt data/ckpts/mae_vit_base.pth \
    --output_dir outputs/retina_centralized \
    --epochs 100 --batch_size 256 --lr 1e-3

# Or use the YAML configuration
python train_centralized.py --config configs/retina_centralized.yaml
```

Training automatically resumes from `ckpt_latest.pth` if interrupted.

### 6. Evaluation

```bash
# Linear probe (diagnostic — measures representation quality with frozen encoder)
python -m eval.linear_probe \
    --encoder_ckpt outputs/retina_centralized/ckpt_latest.pth \
    --data_path /path/to/retina \
    --num_classes 2 \
    --mode linear_probe

# Full fine-tuning (apples-to-apples comparison with baselines like Fed-MAE)
python -m eval.linear_probe \
    --encoder_ckpt outputs/retina_centralized/ckpt_latest.pth \
    --data_path /path/to/retina \
    --num_classes 2 \
    --mode full_finetune

# Label scarcity experiment (tests performance at 30%, 60%, 100% labels)
python -m eval.linear_probe \
    --encoder_ckpt outputs/retina_centralized/ckpt_latest.pth \
    --data_path /path/to/retina \
    --num_classes 2 \
    --mode full_finetune \
    --label_scarcity
```

---

## Training Diagnostics

### Healthy Training Indicators

| Metric | Healthy Range | Warning | Critical |
|---|---|---|---|
| `align_loss` | Decreasing, O(1) | Flat after epoch 10 | NaN |
| `enc_std` | > 0.1 | 0.05 – 0.1 | < 0.05 (collapse) |
| `proj_std` | > 0.01 | < 0.01 | 0.0 (dead projector) |
| `t_std` | ~0.054 (stable) | Fluctuating > 10% | — |
| Train-Val gap | < 15% | 15-25% | > 25% (overfitting) |

### Expected Training Behaviour

| Epoch | Loss | enc_std | Notes |
|---|---|---|---|
| 1 | ~1.0 – 2.0 | ~0.4 | Random init, warmup phase |
| 10 | ~0.5 – 1.0 | ~0.5+ | Warmup complete, alignment starts |
| 50 | ~0.2 – 0.5 | ~0.5+ | Student capturing teacher structure |
| 100 | ~0.1 – 0.3 | ~0.5+ | Convergence plateau |

---

## Hardware Requirements

### Federated Training (Per Hospital Client)

| Resource | Minimum | Recommended |
|---|---|---|
| **GPU** | 2 GB VRAM (batch=8) | 4+ GB VRAM (batch=64) |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 500 MB (models) + dataset | 2 GB |
| **Network** | 313 MB per FL round | Stable broadband |

### Inference Only (Post-Training Deployment)

| Resource | Minimum | Notes |
|---|---|---|
| **GPU** | **Not required** | CPU inference: ~90ms/image |
| **RAM** | 512 MB | 134 MB model + headroom |
| **Storage** | 128 MB | Single checkpoint file |

### Communication Cost (Federated)

| Item | Size | Frequency |
|---|---|---|
| Teacher checkpoint | 343 MB | Once (initial setup) |
| Student weights (upload) | 157 MB | Per FL round |
| Global weights (download) | 157 MB | Per FL round |
| **Total per round** | **313 MB** | — |
| **50-round training** | **~15 GB** | Total per client |

---

## Loss Function: Centered & Standardised MSE

The SALT loss is the technical core of FedMamba-SALT. Here is the mathematical formulation:

### Problem
Given teacher embeddings $\mathbf{t}_i$ and student projections $\mathbf{s}_i$ for a batch of $B$ images:

$$\cos(\mathbf{t}_{\text{class0}}, \mathbf{t}_{\text{class1}}) = 0.9996$$

The inter-class angle is only **1.6°**. Standard MSE/Cosine learns the 99.97% shared mean and ignores the 0.03% disease signal.

### Solution

**Step 1: Center** — Remove the batch mean to expose the discriminative residual:

$$\bar{\mathbf{t}} = \frac{1}{B}\sum_i \mathbf{t}_i, \qquad \tilde{\mathbf{t}}_i = \mathbf{t}_i - \bar{\mathbf{t}}$$

$$\bar{\mathbf{s}} = \frac{1}{B}\sum_i \mathbf{s}_i, \qquad \tilde{\mathbf{s}}_i = \mathbf{s}_i - \bar{\mathbf{s}}$$

**Step 2: Standardise** — Scale teacher residuals to O(1):

$$\hat{\mathbf{t}}_i = \frac{\tilde{\mathbf{t}}_i}{\sigma(\tilde{\mathbf{t}}) + \epsilon}$$

**Step 3: Align** — MSE on standardised targets:

$$\mathcal{L}_{\text{align}} = \frac{1}{BD}\sum_{i,d}(\tilde{s}_{i,d} - \hat{t}_{i,d})^2$$

**Step 4: Regularise** — Covariance penalty + variance guard:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align}} + \lambda_{\text{cov}} \cdot \mathcal{L}_{\text{cov}} + \lambda_{\text{var}} \cdot \mathcal{L}_{\text{var}}$$

---

## Comparison with Baselines

| Method | Backbone | Params (Trainable) | Self-Supervised | Federated-Ready | Privacy |
|---|---|---|---|---|---|
| **FedMamba-SALT** | Inception-Mamba | 39.2M | ✅ | ✅ | ✅ No data sharing |
| Fed-MAE | ViT-B/16 | 85.8M | ✅ | ✅ | ✅ No data sharing |
| FedAvg + Supervised | ResNet-50 | 25.6M | ❌ | ✅ | ✅ No data sharing |
| Centralized Supervised | ViT-B/16 | 85.8M | ❌ | ❌ | ❌ Requires data pooling |

### FedMamba-SALT Advantages over Fed-MAE

| Aspect | Fed-MAE | FedMamba-SALT |
|---|---|---|
| **Encoder complexity** | O(n²) self-attention | **O(n) Mamba SSM** |
| **Trainable params** | 85.8M | **39.2M** (2.2× fewer) |
| **Communication/round** | ~344 MB | **157 MB** (2.2× less) |
| **Multi-scale features** | Single-scale patches | **Inception 3×3/5×5/7×7** |
| **GPU for inference** | Required | **CPU sufficient** |
| **Spatial scanning** | 2D patch sequence | **4-directional (LR/RL/TB/BT)** |

---

## Dependency Notes

| Package | Version | Reason |
|---|---|---|
| `torch` | ≥ 2.0 | Mixed precision, modern APIs |
| `timm` | `0.3.2` | MAE checkpoint compatibility |
| `mamba-ssm` | `1.2.0` | Mamba CUDA kernels (breaking changes across versions) |
| `causal-conv1d` | `1.2.0` | Hard dependency of mamba-ssm |
| `torchvision` | ≥ 0.15 | Image transforms and datasets |
| `matplotlib` | ≥ 3.7 | Evaluation visualisations |
| `scikit-learn` | ≥ 1.2 | AUC/ROC metrics |

---

## Testing

```bash
# Loss function tests (7 core + 3 bonus = 10 total)
python -m tests.test_loss

# End-to-end smoke test (full pipeline on synthetic data)
python -m tests.test_end_to_end
```

### Test Suite Coverage

| Test | What it verifies |
|---|---|
| Test 1 | Loss is scalar tensor with gradient |
| Test 2 | Identical inputs → near-zero loss |
| Test 3 | Random inputs → non-trivial loss |
| Test 4 | Opposite vectors → higher loss than random |
| Test 5 | Gradient flows to student only (teacher detached) |
| Test 6 | Variance penalty activates on collapsed encoder |
| Test 7 | **Centering amplifies class-discriminative signal** |
| Bonus A | Healthy encoder has std > 0.1 |
| Bonus B | Collapsed encoder detected (std < 0.01) |
| Bonus C | **Loss magnitude is O(1) — healthy gradient scale** |
| E2E | Full forward + backward + optimizer step succeeds |

---

## Citation

If you use FedMamba-SALT in your research, please cite:

```bibtex
@misc{fedmamba-salt2026,
  title   = {FedMamba-SALT: Federated Self-supervised Asymmetric Learning 
             via Teacher-Student Distillation with Mamba State-Space Models},
  year    = {2026},
  url     = {https://github.com/mohamed-aliii/FedMamba-SALT}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built for privacy-preserving medical AI research</em>
</p>
