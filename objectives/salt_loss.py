"""
objectives/salt_loss.py -- Learning objective for FedMamba-SALT.

Loss formulation: Centered & Standardised MSE Distillation.

The critical insight: teacher embeddings for class 0 and class 1 have
cosine similarity 0.9996 (angle = 1.6 degrees) in RAW space.  99.97%
of the signal is shared structure (mean retinal fundus) and only 0.03%
is class-discriminative.

Previous failure modes:
    1. Raw SmoothL1/MSE: learned the 99.97% mean, ignored the 0.03%.
    2. Cosine on raw:  gradient vanishes as ||s|| grows → NaN.
    3. Centered + L2-norm: near-zero centered vectors → NaN from normalize.
    4. Centered SmoothL1 (no norm): residuals are O(0.001) → microscopic
       gradients, model cannot learn.

Solution: **Target Standardisation**.
    1. Center teacher: remove the global mean.
    2. Standardise teacher: divide by scalar std → targets become O(1).
    3. Center student: remove student's own mean.
    4. MSE(s_centered, t_standardised): loss is O(1) with healthy gradients.
    5. Teacher std acts like a natural temperature: small teacher variance
       amplifies the signal; large variance dampens it.

Why this works:
    - The teacher's centered std is ~0.054 (from diagnostics).
    - Dividing by 0.054 amplifies the class residual by ~18x.
    - MSE on O(1) targets gives O(1) gradients — no NaN, no vanishing.
    - Only the TEACHER is standardised (detached, no gradient).
    - The student just centers and matches.

References:
    - Barlow Twins (Zbontar et al., 2021) -- centering + decorrelation
    - VICReg (Bardes et al., 2022) -- variance regularisation
    - DINO (Caron et al., 2021) -- centering for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Projection Head (BYOL-style 3-layer MLP)
# ======================================================================
class ProjectionHead(nn.Module):
    """
    3-layer MLP that maps the student encoder's output to a projected
    embedding aligned with the teacher's representation space.

    LayerNorm is used (not BatchNorm) so the module is federated-safe.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 2048,
        out_dim: int = 768,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# Per-dimension variance penalty (applied to ENCODER output)
# ======================================================================
def variance_loss(
    encoder_embeddings: torch.Tensor,
    target_std: float = 0.1,
) -> torch.Tensor:
    """
    Per-dimension variance penalty to prevent encoder collapse.
    Applied to the RAW encoder output (before the projection head).
    """
    s_std = encoder_embeddings.std(dim=0)
    return F.relu(target_std - s_std).mean()


# ======================================================================
# Covariance regularisation (off-diagonal penalty)
# ======================================================================
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Penalise off-diagonal elements of the feature covariance matrix.
    Encourages different dimensions to encode independent information.

    Args:
        embeddings: ``(B, D)`` batch of embedding vectors (will be centered).

    Returns:
        Scalar loss: mean of squared off-diagonal covariance elements.
    """
    B, D = embeddings.shape
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(B - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / (D * D)


# ======================================================================
# SALT loss function (Centered & Standardised MSE Distillation)
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor = None,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
) -> tuple:
    """
    Centered & Standardised MSE Distillation for FedMamba-SALT.

    Steps:
        1. Detach teacher embeddings (no gradient to teacher).
        2. Center teacher: subtract batch mean.
        3. Standardise teacher: divide by scalar std (+ eps).
           This amplifies the tiny class residuals to O(1).
        4. Center student: subtract batch mean.
        5. MSE(s_centered, t_standardised) gives O(1) loss & gradients.
        6. Covariance penalty on student prevents dim collapse.
        7. Variance penalty on encoder output prevents encoder collapse.

    The standardisation is only applied to the TEACHER (which is detached).
    The student learns to output features whose centered version matches
    the teacher's standardised residuals.

    Args:
        student_proj: ``(B, D)`` or ``(B, N, D)`` output of the projection head.
        teacher_emb:  ``(B, D)`` or ``(B, N, D)`` embedding from frozen teacher.
        student_emb:  ``(B, D)`` or ``(B, N, D)`` raw encoder output (before proj head).
        lambda_var:   Weight for the variance penalty.
        lambda_cov:   Weight for the covariance penalty.

    Returns:
        Tuple of ``(total_loss, align_loss, var_loss)``.
    """
    # --- Flatten spatial dimensions if using dense patch distillation ---
    if student_proj.dim() == 3:
        student_proj = student_proj.flatten(0, 1)  # (B*N, D)
        teacher_emb = teacher_emb.flatten(0, 1)
        if student_emb is not None:
            student_emb = student_emb.flatten(0, 1)

    # --- Detach the teacher ---
    teacher_emb = teacher_emb.detach()

    # --- Center both ---
    t_centered = teacher_emb - teacher_emb.mean(dim=0, keepdim=True)
    s_centered = student_proj - student_proj.mean(dim=0, keepdim=True)

    # --- Standardise teacher residuals ---
    # The teacher's per-batch std is ~0.054 (from diagnostics).
    # Dividing by this amplifies the class signal by ~18x,
    # making targets O(1) instead of O(0.001).
    t_std = t_centered.std() + 1e-6  # scalar std across all elements
    t_target = t_centered / t_std

    # --- MSE alignment on standardised targets ---
    # Loss is now O(1) with healthy gradients.
    # MSE (not SmoothL1) because the targets are well-scaled.
    align_loss = F.mse_loss(s_centered, t_target)

    # --- Covariance penalty on centered student ---
    cov_loss = covariance_loss(s_centered)

    # --- Combined alignment ---
    total_align = align_loss + lambda_cov * cov_loss

    # --- Variance penalty on ENCODER output ---
    if student_emb is not None:
        var_loss_val = variance_loss(student_emb)
    else:
        var_loss_val = torch.tensor(0.0, device=student_proj.device)

    # --- Total loss ---
    total_loss = total_align + (lambda_var * var_loss_val)

    return total_loss, total_align, var_loss_val


# ======================================================================
# Collapse detection utility
# ======================================================================
def embedding_std(embeddings: torch.Tensor) -> float:
    """
    Compute the average standard deviation across embedding dimensions.

    A single scalar diagnostic for representation collapse:
        • Healthy:   > 0.1
        • Warning:   0.01 – 0.1
        • Collapsed: < 0.01

    Args:
        embeddings: ``(B, D)`` or ``(B, N, D)`` batch of embedding vectors.

    Returns:
        Average per-dimension standard deviation (scalar float).
    """
    if embeddings.dim() == 3:
        embeddings = embeddings.flatten(0, 1)
    return embeddings.std(dim=0).mean().item()

