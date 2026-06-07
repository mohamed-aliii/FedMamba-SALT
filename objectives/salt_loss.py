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

SALT_NORM_MODES = ("batch", "instance", "global_teacher")


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
    s_std = encoder_embeddings.std(dim=0, unbiased=False)
    return F.relu(target_std - s_std).mean()


# ======================================================================
# Covariance regularisation (off-diagonal penalty)
# ======================================================================
_COV_MAX_SAMPLES = 1024  # subsample if batch is larger (for dense distillation)

def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Penalise off-diagonal elements of the feature covariance matrix.
    Encourages different dimensions to encode independent information.
    
    When using dense patch distillation the effective batch can be
    B*196 = 50 000+.  Computing a full (D, D) covariance matrix from
    50 000 samples is both slow and unnecessary — 1024 random samples
    provide a reliable covariance estimate.

    Must be computed in float32 to prevent FP16 overflow during large sums.

    Args:
        embeddings: ``(B, D)`` batch of embedding vectors (will be centered).

    Returns:
        Scalar loss: mean of squared off-diagonal covariance elements.
    """
    # Force float32 to prevent FP16 overflow in .pow(2).sum()
    embeddings = embeddings.float()

    B, D = embeddings.shape
    # Subsample to keep the covariance computation tractable
    if B > _COV_MAX_SAMPLES:
        idx = torch.randperm(B, device=embeddings.device)[:_COV_MAX_SAMPLES]
        embeddings = embeddings[idx]
        B = _COV_MAX_SAMPLES
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(B - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / (D * D)


def _canonical_norm_mode(norm_mode: str) -> str:
    mode = norm_mode.lower().replace("-", "_")
    aliases = {
        "global": "global_teacher",
        "global_stats": "global_teacher",
        "teacher_global": "global_teacher",
    }
    mode = aliases.get(mode, mode)
    if mode not in SALT_NORM_MODES:
        raise ValueError(
            f"Unknown SALT norm_mode={norm_mode!r}. "
            f"Expected one of {SALT_NORM_MODES}."
        )
    return mode


def _broadcast_stat(stat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast a saved teacher statistic to ``target`` shape."""
    stat = stat.to(device=target.device, dtype=target.dtype)
    while stat.dim() < target.dim():
        stat = stat.unsqueeze(0)
    return stat


def _finite_summary(tensor: torch.Tensor) -> dict:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    if not finite.any():
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "finite": 0.0,
        }
    values = values[finite]
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "finite": finite.float().mean().item(),
    }


def _salt_normalize(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    norm_mode: str,
    teacher_stats: dict | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return student residuals, teacher targets, and teacher std tensor."""
    mode = _canonical_norm_mode(norm_mode)

    if mode == "batch":
        # Modified behavior to prevent non-IID local batch mean from destroying class boundaries.
        # We now use LayerNorm-style (instance-level) centering over features (dim=-1).
        t_centered = teacher_emb - teacher_emb.mean(dim=-1, keepdim=True)
        s_centered = student_proj - student_proj.mean(dim=-1, keepdim=True)
        t_std = t_centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return s_centered, t_centered / t_std, t_std

    if mode == "instance":
        # Evidence-gated option: no cross-sample batch statistics.
        t_centered = teacher_emb - teacher_emb.mean(dim=-1, keepdim=True)
        s_centered = student_proj - student_proj.mean(dim=-1, keepdim=True)
        t_std = t_centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return s_centered, t_centered / t_std, t_std

    if teacher_stats is None or "mean" not in teacher_stats or "std" not in teacher_stats:
        raise ValueError(
            "SALT norm_mode='global_teacher' requires teacher_stats with "
            "'mean' and 'std' tensors."
        )

    t_mean = _broadcast_stat(teacher_stats["mean"], teacher_emb)
    t_std = _broadcast_stat(teacher_stats["std"], teacher_emb).clamp_min(eps)
    t_target = (teacher_emb - t_mean) / t_std
    # Avoid local batch-stat leakage on the student side in this mode.
    s_centered = student_proj - student_proj.mean(dim=-1, keepdim=True)
    return s_centered, t_target, t_std


# ======================================================================
# SALT loss function (Centered & Standardised MSE Distillation)
# ======================================================================
def salt_loss(
    student_proj: torch.Tensor,
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor = None,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
    norm_mode: str = "batch",
    teacher_stats: dict | None = None,
    eps: float = 1e-6,
    return_stats: bool = False,
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
        norm_mode:    ``batch`` keeps the current SALT behavior; ``instance``
                      normalizes per sample/token over features; ``global_teacher``
                      uses fixed teacher mean/std statistics supplied via
                      ``teacher_stats``.
        teacher_stats: Dict with ``mean`` and ``std`` tensors for global-teacher
                       mode. Dense patch stats may be shaped ``(N, D)`` or
                       ``(1, N, D)``.
        eps:          Minimum teacher std for division.
        return_stats: If True, append a diagnostics dict to the return tuple.

    Returns:
        Tuple of ``(total_loss, align_loss, var_loss)``. If ``return_stats`` is
        True, returns ``(total_loss, align_loss, var_loss, stats)``.
    """
    # --- Handle dense patch distillation (B, N, D) -> (B*N, D) ---
    # For variance_loss we need image-level features (GAP of patches)
    # to maintain meaningful collapse detection, so save them separately.
    student_emb_for_var = student_emb
    if student_proj.dim() == 3 and student_emb is not None:
        # Collapse detection: use per-image GAP, not per-patch
        student_emb_for_var = student_emb.mean(dim=1)  # (B, D)

    # --- Detach the teacher ---
    teacher_emb = teacher_emb.detach()

    # --- Pool for Global Semantic Distillation (BEFORE NORMALIZATION) ---
    if student_proj.dim() == 3:
        student_proj = student_proj.mean(dim=1)
    if teacher_emb.dim() == 3:
        teacher_emb = teacher_emb.mean(dim=1)

    s_centered, t_target, t_std = _salt_normalize(
        student_proj=student_proj,
        teacher_emb=teacher_emb,
        norm_mode=norm_mode,
        teacher_stats=teacher_stats,
        eps=eps,
    )


    # Force float32 for loss computation to prevent FP16 overflow
    s_centered_f32 = s_centered.float()
    t_target_f32 = t_target.float()

    # --- MSE alignment on standardised targets ---
    # Loss is now O(1) with healthy gradients.
    # MSE (not SmoothL1) because the targets are well-scaled.
    align_loss = F.mse_loss(s_centered_f32, t_target_f32)

    # --- Covariance penalty on centered student ---
    cov_loss = covariance_loss(s_centered_f32)

    # --- Combined alignment ---
    total_align = align_loss + lambda_cov * cov_loss

    # --- Variance penalty on ENCODER output ---
    if student_emb_for_var is not None:
        var_loss_val = variance_loss(student_emb_for_var.float())
    else:
        var_loss_val = torch.tensor(0.0, device=student_proj.device)

    # --- Total loss ---
    total_loss = total_align + (lambda_var * var_loss_val)

    if not torch.isfinite(total_loss):
        stats = {
            "salt_norm_mode": _canonical_norm_mode(norm_mode),
            "teacher_std": _finite_summary(t_std),
            "teacher_target": _finite_summary(t_target_f32),
            "student_centered": _finite_summary(s_centered_f32),
        }
        raise FloatingPointError(f"SALT produced a non-finite loss: {stats}")

    if return_stats:
        stats = {
            "salt_norm_mode": _canonical_norm_mode(norm_mode),
            "teacher_std": _finite_summary(t_std),
            "teacher_target": _finite_summary(t_target_f32),
            "student_centered": _finite_summary(s_centered_f32),
            "encoder_embedding": (
                _finite_summary(student_emb_for_var)
                if student_emb_for_var is not None else None
            ),
        }
        return total_loss, total_align, var_loss_val, stats

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
    return embeddings.std(dim=0, unbiased=False).mean().item()
