"""
objectives/salt_loss.py -- Learning objective for FedMamba-SALT.

Loss formulation: Centered Smooth L1 Distillation.

The critical insight: teacher embeddings for class 0 and class 1 have
cosine similarity 0.9996 (angle = 1.6°) in RAW space.  This means 99.97%
of the signal is shared structure (mean retinal fundus) and only 0.03%
is class-discriminative.  Any regression loss in raw space will learn the
shared mean and ignore the discriminative residual.

Solution: **Batch-center** both student and teacher embeddings before
alignment.  Centering removes the global mean, so the remaining signal
IS the discriminative residual.  Then SmoothL1 matches these residuals
directly — no L2-normalization needed (which caused NaN when centered
vectors were near-zero early in training).

Total loss::

    t_centered = t_emb - mean(t_emb)       # remove global mean
    s_centered = s_proj - mean(s_proj)      # remove global mean
    L = SmoothL1(s_centered, t_centered)    # match residuals
        + lambda_cov * off_diag_penalty     # prevent dimension collapse
        + lambda_var * var_loss(s_emb)      # prevent encoder collapse

Why NOT normalise after centering?
    Early in training, all student outputs are similar → centered vectors
    are near-zero → F.normalize divides by ~1e-6 → gradient explosion → NaN.
    SmoothL1 on unnormalised centered vectors is stable AND constrains
    magnitude naturally (any deviation from teacher's residual is penalised).

References:
    - Barlow Twins (Zbontar et al., 2021) -- centering + off-diagonal penalty
    - VICReg (Bardes et al., 2022) -- variance regularisation
    - DINO (Caron et al., 2021) -- centering for distillation stability
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
        embeddings: ``(B, D)`` batch of CENTERED embedding vectors.

    Returns:
        Scalar loss: mean of squared off-diagonal covariance elements.
    """
    B, D = embeddings.shape
    # embeddings should already be centered, but center again for safety
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    # Covariance matrix (D, D)
    cov = (x.T @ x) / max(B - 1, 1)
    # Zero the diagonal (we only penalize off-diagonal correlations)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / (D * D)


# ======================================================================
# SALT loss function (Centered Smooth L1 Distillation)
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor = None,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
) -> tuple:
    """
    Centered Smooth L1 Distillation loss for FedMamba-SALT.

    Steps:
        1. Detach teacher embeddings.
        2. Subtract batch mean from both student and teacher vectors.
           This removes the dominant shared structure (99.97% of signal)
           and exposes the discriminative residual AS the dominant signal.
        3. Apply Smooth L1 on the centered vectors (NO normalisation).
        4. Add covariance penalty to prevent dimension collapse.
        5. Add variance penalty on encoder output.

    No L2-normalisation: early in training, centered vectors can be
    near-zero (all outputs similar).  F.normalize would divide by ~eps
    and cause NaN.  SmoothL1 on unnormalised centered vectors is stable
    and naturally constrains magnitude.

    Args:
        student_proj: ``(B, D)`` output of the projection head.
        teacher_emb:  ``(B, D)`` embedding from frozen teacher.
        student_emb:  ``(B, D)`` raw encoder output (before proj head).
        lambda_var:   Weight for the variance penalty.
        lambda_cov:   Weight for the covariance penalty.

    Returns:
        Tuple of ``(total_loss, align_loss, var_loss)``.
    """
    # --- Detach the teacher ---
    teacher_emb = teacher_emb.detach()

    # --- Batch centering (the critical step) ---
    t_centered = teacher_emb - teacher_emb.mean(dim=0, keepdim=True)
    s_centered = student_proj - student_proj.mean(dim=0, keepdim=True)

    # --- Directional alignment on centered vectors (no normalization!) ---
    direction_loss = F.smooth_l1_loss(s_centered, t_centered)

    # --- Covariance penalty on centered student (prevent dim collapse) ---
    cov_loss = covariance_loss(s_centered)

    # --- Combined alignment ---
    align_loss = direction_loss + lambda_cov * cov_loss

    # --- Variance penalty on ENCODER output ---
    if student_emb is not None:
        var_loss_val = variance_loss(student_emb)
    else:
        var_loss_val = torch.tensor(0.0, device=student_proj.device)

    # --- Total loss ---
    total_loss = align_loss + (lambda_var * var_loss_val)

    return total_loss, align_loss, var_loss_val


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
        embeddings: ``(B, D)`` batch of embedding vectors.

    Returns:
        Average per-dimension standard deviation (scalar float).
    """
    return embeddings.std(dim=0).mean().item()
