# FedMamba-SALT System Architecture & Alignment

## 1. Reference Architecture & Core Design
The baseline model for this project is **InceptionMamba**:
- **Reference:** *Bingquan Huang, Yue Liu, Bin Tang, Gang Fang. "InceptionMamba: A Lightweight and Effective Model for Medical Image Classification Revealing Mamba's Low-Frequency Bias." Neural Processing Letters 58:15, 2026.*
- **Core Mechanism:** The architecture leverages a dual-branch channel split. One branch handles local, multi-scale feature extraction via an Inception-style CNN pathway (`1x1`, `3x3`, and `AvgPool`). The other branch captures global, long-range dependencies via a State Space Model (SSM / Mamba) with four-directional scanning.
- **Federated Adaptations:**
  - `GroupNorm` replaces all instances of `BatchNorm2d`. This prevents catastrophic covariate shift and running-stat drift during non-IID federated training.
  - The model uses `16x16` patches without patch merging to maintain a dense `14x14 = 196` token grid, explicitly matching the frozen teacher's resolution.

## 2. The Teacher-Student Knowledge Distillation Paradigm
The primary learning objective before supervised fine-tuning is **SALT (Scaffold-Accelerated Local Tuning)**, a specialized form of self-supervised dense distillation.
- **The Teacher:** A completely frozen **Masked Autoencoder (MAE) ViT-B/16**.
- **The Challenge of Medical Features:** Diagnostics (`diagnostic_teacher_probe.py`) proved that the teacher's raw representations are **not linearly separable**. The embeddings for completely different medical classes possess a cosine similarity of `0.9996`. This means $99.97\%$ of the signal represents the shared anatomical structure (e.g., the mean retinal fundus or chest wall), and only $0.03\%$ of the signal contains the actual disease-discriminative information.
- **The Distillation Task:** The student (InceptionMamba) must map its output patches through a projection head to dense-match the 768-D token outputs of the ViT teacher, thereby learning the spatial structure of the medical domain without labels.

## 3. The SALT Loss Formulation (Centered & Standardised MSE)
Because the discriminative signal is microscopic ($0.03\%$), standard loss functions completely fail:
- *Raw SmoothL1/MSE:* Simply learns the global anatomical mean and ignores the disease features.
- *Cosine/L2-Normalized Loss:* Causes gradients to vanish or explode as vectors collapse toward zero.

**The Solution:** The SALT objective employs **Target Standardisation**.
1. **Center Teacher:** Subtract the global mean of the teacher's batch to remove the $99.97\%$ anatomical overlap.
2. **Standardise Teacher:** Divide the centered teacher embeddings by their scalar standard deviation ($\approx 0.054$). This mathematically amplifies the microscopic disease residual by nearly $18\times$, turning it into an $O(1)$ target.
3. **Center Student:** Remove the student's batch mean.
4. **Compute MSE:** Calculate the Mean Squared Error between the centered student and the *standardised* teacher.
This provides $O(1)$ gradients that cleanly extract the $0.03\%$ discriminative signal without vanishing or exploding. Additionally, variance and covariance regularizations (like VICReg/Barlow Twins) are applied to the raw encoder output to prevent informational collapse.

## 4. Federated Algorithms & The "Deep Problem" Pathology
The ultimate goal is to fine-tune the distilled FedMamba-SALT model on a 12-client federated COVID-19 dataset with **Extreme Label Skew** (mono-class and missing-class clients). This process exposed a critical, foundational paradox in Federated State Space Models.

### A. The Local Optimization Pathology (Extreme Skew)
Under extreme non-IID settings, local optimization dynamics break the global model:
- **Mono-Class Clients:** Having no negative examples, the local Cross-Entropy loss forces the discriminator's logits to $+\infty$.
- **Missing-Class Clients:** Driven by Label Smoothing, clients explicitly push the logits of missing classes to $-\infty$.
When aggregated via standard **FedAvg**, the massive $-\infty$ pushes from clients with large datasets mathematically destroy the boundaries of the missing classes (causing global predictions for that class to collapse to zero). Alternatively, attempts to use `class_head_only` row-wise aggregations structurally decouple the Softmax function, creating "Frankenstein classifiers" where logits are artificially inflated, causing massive over-predictions (e.g., predicting Class 2 $3030$ times).

### B. The Deep Origin: The Capacity vs. Federation Deadlock
The catastrophic failure of the linear classifier is actually a *symptom* of an underlying encoder bottleneck.
1. **The Mamba FedAvg Incompatibility:** We cannot fully unfreeze the encoder. The Mamba blocks rely on continuous-time ODEs ($\Delta, A, B, C$). Linearly averaging these discrete transition matrices across clients using FedAvg destroys the ODE dynamics, causing representation collapse.
2. **The "Federated Branch Protect" Trap:** To protect the ODEs, the `mamba_` parameters are frozen, and only the shallow CNN layers are fine-tuned. 
3. **The Paradox:** The pre-trained SSL features are highly entangled (proven by the teacher diagnostic). The shallow CNN layers alone lack the non-linear capacity to disentangle them. Because the Mamba sequence modeling engine is frozen, the encoder fails to map the images into a linearly separable space. 
4. **The Explosion:** Confronted with linearly inseparable data, the `nn.Linear` classifier has no geometric way to lower the Cross-Entropy loss other than resorting to pathological magnitude explosions ($+\infty$ / $-\infty$). 

