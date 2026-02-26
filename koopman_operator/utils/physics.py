"""
Physics Utilities — Heskestad Correlations & Loss Functions
============================================================

Implements fire physics correlations used for:
  1. Feature engineering (input channels 3–5)
  2. Physics-informed loss terms

Reference:
    Heskestad, G. (1984). "Engineering relations for fire plumes."
    Fire Safety Journal, 7(1), 25-32.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


# ══════════════════════════════════════════════════════════════════════
# HESKESTAD CORRELATIONS
# ══════════════════════════════════════════════════════════════════════

def heskestad_flame_height(
    Q: Union[float, np.ndarray, torch.Tensor],
    D: float = 0.3,
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Mean flame height via Heskestad's correlation:

        L_f = −1.02 D + 0.235 Q_c^(2/5)

    where Q_c = 0.7 Q  (convective fraction).

    Args:
        Q:  Total heat release rate (kW).
        D:  Fire diameter (m).

    Returns:
        L_f:  Flame height (m), clamped ≥ 0.
    """
    if isinstance(Q, torch.Tensor):
        Qc = torch.clamp(Q * 0.7, min=1e-6)
        Lf = -1.02 * D + 0.235 * torch.pow(Qc, 0.4)
        return torch.clamp(Lf, min=0.0)

    if isinstance(Q, np.ndarray):
        Qc = np.maximum(Q * 0.7, 1e-6)
        Lf = -1.02 * D + 0.235 * np.power(Qc, 0.4)
        return np.maximum(Lf, 0.0)

    # scalar
    Qc = max(Q * 0.7, 1e-6)
    Lf = -1.02 * D + 0.235 * (Qc ** 0.4)
    return max(Lf, 0.0)


def compute_heskestad_features(
    hrr: np.ndarray,
    fire_diameter: float = 0.3,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Derive three physics channels from an HRR time-series.

    Returns [T, 3]:
        col 0  –  Flame height  L_f  (m)
        col 1  –  Flame height rate  dL_f/dt  (m/s)
        col 2  –  Deviation from mean flame height  (m)
    """
    Lf = heskestad_flame_height(hrr, D=fire_diameter)
    dLf_dt = np.gradient(Lf, dt)
    Lf_dev = Lf - np.mean(Lf)

    return np.stack([Lf, dLf_dt, Lf_dev], axis=1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# PHYSICS-INFORMED LOSS TERMS
# ══════════════════════════════════════════════════════════════════════

def physics_consistency_loss(
    pred_hrr: torch.Tensor,
    target_hrr: torch.Tensor,
    fire_diameter: float = 0.3,
    weight: float = 0.1,
) -> torch.Tensor:
    """
    Penalise predictions whose implied flame height deviates from the
    target's flame height (via Heskestad's correlation).

    Args:
        pred_hrr:     [B, T, 1]  predicted HRR
        target_hrr:   [B, T, 1]  ground-truth HRR
        fire_diameter: metres
        weight:       scaling factor λ_physics

    Returns:
        Scalar loss.
    """
    Lf_pred   = heskestad_flame_height(pred_hrr.squeeze(-1),   D=fire_diameter)
    Lf_target = heskestad_flame_height(target_hrr.squeeze(-1), D=fire_diameter)
    return weight * F.mse_loss(Lf_pred, Lf_target)


def monotonicity_loss(
    pred_hrr: torch.Tensor,
    weight: float = 0.05,
    max_rate: float = 5.0,
) -> torch.Tensor:
    """
    Penalise non-physical behaviour:
      - Negative HRR values
      - Extreme frame-to-frame jumps (> max_rate per 0.1 s ≈ 50 kW/s)
    """
    neg_penalty  = torch.mean(torch.relu(-pred_hrr))
    diff         = pred_hrr[:, 1:, :] - pred_hrr[:, :-1, :]
    jump_penalty = torch.mean(torch.relu(torch.abs(diff) - max_rate))
    return weight * (neg_penalty + jump_penalty)
