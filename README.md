<p align="center">
  <h1 align="center">FedMamba-SALT</h1>
  <p align="center">
    <strong>Federated Self-Supervised Asymmetric Learning via Teacher–Student Distillation</strong><br>
    <em>Privacy-Preserving, High-Performance Representation Learning for Medical Diagnostics</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Mamba-SSM-green.svg" alt="Mamba">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

> [!NOTE]
> **Repository Scope**: This documentation exclusively covers the **`fedmamba_salt`** system folder, detailing its standalone architecture, models, training scripts, and evaluation pipelines.

## 🎯 Primary Objective

The primary objective of **FedMamba-SALT** is to enable hospitals and clinical institutions to collaboratively train a high-quality medical image encoder **without sharing any patient data**. 

We achieve this by optimizing a federated representation learning pipeline that employs a compact, state-space **Inception-Mamba** student model distilling knowledge from a frozen **MAE-pretrained ViT-B/16** teacher. The system is specifically engineered to overcome the representation collapse often seen in medical distillation, providing a highly efficient, CPU-deployable backbone for downstream clinical diagnostic tasks.

---

## 🚀 Key Innovations & Architectural Highlights

### 1. Centered & Standardised Distillation (SALT Loss)
Medical imaging pathologies (e.g., microaneurysms) often constitute a microscopic fraction of the image, leading to a cosine similarity of >0.9996 between healthy and diseased embeddings. Standard distillation losses fail by merely learning the dominant shared structural mean.
**Our Solution (`objectives/salt_loss.py`):**
- **Batch-center** embeddings to isolate the class-discriminative residual.
- **Standardise** teacher residuals to amplify subtle disease signals by ~18×.
- Align using **MSE** on standardised targets to ensure stable, $O(1)$ gradients without dimension collapse.
- Incorporates **Covariance Regularisation** and **Variance Penalties** to guarantee robust representations.

### 2. Dense Patch-Level Distillation & Attention Pooling
Traditional Global Average Pooling (GAP) destroys spatial variance, capping performance. FedMamba-SALT transitions to **Dense Spatial Distillation**, where the student predicts the teacher's dense spatial feature map (196 individual patches) rather than a single global vector. During full fine-tuning, a custom **Attention-Pooling** classifier adaptively weighs diagnostically relevant patches to maximize clinical accuracy.

### 3. Asymmetric Knowledge Distillation
To force hard visual inference rather than trivial mimicry, we decouple the student and teacher views (`augmentations/medical_aug.py`):
- **Teacher View (Minimal):** Processes clean images to extract pristine representations.
- **Student View (Medical-Safe Augmentation):** Processes heavily corrupted views (blur, noise, color jitter) combined with **Latent Masking** (dropping 50% of tokens internally). The student must reconstruct the pristine teacher features from degraded inputs.

### 4. Inception-Mamba Backbone
The student encoder (`models/inception_mamba.py`) replaces standard self-attention with a highly efficient hybrid architecture:
- **Mamba State-Space Models (SSM):** Scales linearly $O(N)$ with sequence length, employing 4-directional cross-scanning (Left-to-Right, Right-to-Left, Top-to-Bottom, Bottom-to-Top) for global context.
- **Inception Multi-Scale Convolutions:** Captures pathological features at multiple scales ($3\times3, 5\times5, 7\times7$ spatial depthwise convolutions).
- **Efficiency:** Only 31.8M parameters allowing rapid CPU inference (~90ms/image), significantly outperforming larger ViT alternatives.

---

## 🛠️ Project Structure

The `fedmamba_salt/` directory contains the complete end-to-end pipeline:

```text
fedmamba_salt/
├── models/
│   ├── inception_mamba.py      # Inception-Mamba encoder (Student)
│   ├── vit_teacher.py          # Frozen ViT-B/16 MAE (Teacher)
│   └── __init__.py
├── objectives/
│   └── salt_loss.py            # Centered & Standardised MSE + Variance Guards
├── augmentations/
│   ├── medical_aug.py          # Asymmetric dual-view augmentations
│   └── retina_dataset.py       # Retina dataset loader with splits
├── eval/
│   └── linear_probe.py         # Linear probing & Full fine-tuning evaluation
├── configs/
│   └── retina_centralized.yaml # Experiment configurations
├── tests/
│   ├── test_loss.py            # Comprehensive loss unit tests
│   └── test_end_to_end.py      # Full pipeline smoke test
├── scripts/
│   └── run_centralized_retina.sh # End-to-end launch script
├── train_centralized.py        # Main centralized pre-training script
└── requirements.txt            # Pinned environment dependencies
```

---

## ⚙️ Quick Start Guide

### 1. Environment Setup

```bash
conda create -n fedmamba python=3.10 -y
conda activate fedmamba

# Install PyTorch with CUDA (e.g., cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Mamba SSM
pip install mamba-ssm==1.2.0 causal-conv1d==1.4.0

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

Run the test suite to ensure the environment and custom loss functions are functioning properly:
```bash
python -m tests.test_loss
python -m tests.test_end_to_end
```

### 3. Data & Checkpoint Preparation

- **Dataset**: Place your dataset in `data/Retina/` containing `train/`, `test/`, and a `labels.csv`.
- **Teacher Model**: Download the MAE ViT-B/16 pretrained checkpoint to `data/ckpts/mae_vit_base.pth`.

### 4. Training (Centralized Pre-training)

Start the representation learning process:
```bash
python train_centralized.py \
    --config configs/retina_centralized.yaml \
    --data_path /path/to/retina \
    --teacher_ckpt data/ckpts/mae_vit_base.pth \
    --output_dir outputs/retina_centralized
```
*Note: Training will automatically resume from `ckpt_latest.pth` if interrupted.*

### 5. Evaluation

Evaluate the learned representations using either a linear probe or full end-to-end fine-tuning.

**Linear Probe (Representation Quality):**
```bash
python -m eval.linear_probe \
    --encoder_ckpt outputs/retina_centralized/ckpt_latest.pth \
    --data_path /path/to/retina \
    --num_classes 2 \
    --mode linear_probe
```

**Full Fine-Tuning (Clinical Diagnostics):**
```bash
python -m eval.linear_probe \
    --encoder_ckpt outputs/retina_centralized/ckpt_latest.pth \
    --data_path /path/to/retina \
    --num_classes 2 \
    --mode full_finetune
```

---

## 📊 Training Diagnostics & Health Indicators

Monitor the following indicators during pre-training to ensure convergence and prevent representation collapse:

| Metric | Target/Healthy Range | Warning | Critical Action |
|--------|----------------------|---------|-----------------|
| **Loss** | Decreasing towards $\sim 0.05 - 0.2$ | Flat > 0.4 | Check learning rate |
| **Encoder Std (`enc_std`)** | $> 0.1$ | $0.02 - 0.1$ | $< 0.02$ (Collapse detected, abort) |
| **Projector Std (`proj_std`)** | $> 0.01$ | $< 0.01$ | $0.0$ (Dead projector) |
| **Teacher Std (`t_std`)** | $\sim 0.054$ (Stable) | Fluctuates | Ensure teacher is frozen |

---

## 🛡️ Federated Learning Readiness

FedMamba-SALT is specifically engineered for strict privacy-preserving environments:
- **No Data Sharing:** Patient data never leaves the hospital. Only the lightweight student model weights are transmitted during federated rounds.
- **Communication Efficiency:** $\sim 157$ MB per upload round (Student only).
- **Client Hardware:** 4GB VRAM GPU (Batch size 64) for training; CPU-only for downstream inference ($\sim 90$ms/image).

---

## 📝 License & Citation

MIT License — see [LICENSE](LICENSE) for details.

If this framework aids your research, please cite:
```bibtex
@misc{fedmamba-salt2026,
  title   = {FedMamba-SALT: Federated Self-supervised Asymmetric Learning via Teacher-Student Distillation with Mamba State-Space Models},
  year    = {2026},
  url     = {https://github.com/mohamed-aliii/FedMamba-SALT}
}
```
