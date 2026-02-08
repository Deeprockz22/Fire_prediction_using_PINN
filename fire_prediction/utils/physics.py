"""
Physics-Informed Utilities for Fire Prediction

This module implements fire physics correlations, primarily Heskestad's
correlations for flame height and plume behavior.

References:
- Heskestad, G. (1984). "Engineering relations for fire plumes"
- Fire Safety Journal, 7(1), 25-32
"""

import torch
import numpy as np
from typing import Union, Tuple

# ═══════════════════════════════════════════════════════════════════════
# HESKESTAD CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════

def heskestad_flame_height(
    Q_c: Union[float, np.ndarray, torch.Tensor],
    D: float = 0.3,
    return_tensor: bool = False
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calculate mean flame height using Heskestad's correlation.
    
    Heskestad's correlation:
        L_f = -1.02D + 0.235 Q_c^(2/5)
    
    Args:
        Q_c: Convective heat release rate (kW)
             Typically Q_c ≈ 0.7 * Q_total for most fuels
        D: Fire diameter (m), default 0.3m (typical burner size)
        return_tensor: If True, return torch.Tensor (for gradient computation)
    
    Returns:
        L_f: Mean flame height (m)
    
    Physical Interpretation:
        - Flame height increases with HRR^(2/5) (sublinear growth)
        - Larger diameter fires have shorter flames (spreading effect)
        - Valid for pool fires and burner fires
    
    Example:
        >>> Q_c = 100  # kW
        >>> L_f = heskestad_flame_height(Q_c, D=0.3)
        >>> print(f"Flame height: {L_f:.2f} m")
        Flame height: 1.22 m
    """
    if isinstance(Q_c, torch.Tensor):
        # Ensure positive HRR (physical constraint)
        Q_c_safe = torch.clamp(Q_c, min=1e-6)
        L_f = -1.02 * D + 0.235 * torch.pow(Q_c_safe, 2/5)
        return torch.clamp(L_f, min=0.0)  # Flame height must be non-negative
    
    elif isinstance(Q_c, np.ndarray):
        Q_c_safe = np.maximum(Q_c, 1e-6)
        L_f = -1.02 * D + 0.235 * np.power(Q_c_safe, 2/5)
        result = np.maximum(L_f, 0.0)
        
        if return_tensor:
            return torch.from_numpy(result).float()
        return result
    
    else:  # scalar
        Q_c_safe = max(Q_c, 1e-6)
        L_f = -1.02 * D + 0.235 * (Q_c_safe ** (2/5))
        result = max(L_f, 0.0)
        
        if return_tensor:
            return torch.tensor(result, dtype=torch.float32)
        return result


def heskestad_plume_temperature(
    Q_c: Union[float, np.ndarray, torch.Tensor],
    z: float,
    T_ambient: float = 293.15,
    return_tensor: bool = False
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calculate plume centerline temperature at height z.
    
    Heskestad's plume temperature correlation:
        ΔT = 9.1 * (T_ambient / g * c_p^2 * ρ_ambient^2)^(1/3) * Q_c^(2/3) / z^(5/3)
    
    Simplified for air at standard conditions:
        ΔT ≈ 25 * Q_c^(2/3) / z^(5/3)
    
    Args:
        Q_c: Convective heat release rate (kW)
        z: Height above fire source (m)
        T_ambient: Ambient temperature (K), default 293.15K (20°C)
        return_tensor: If True, return torch.Tensor
    
    Returns:
        T: Plume temperature (K)
    """
    if isinstance(Q_c, torch.Tensor):
        Q_c_safe = torch.clamp(Q_c, min=1e-6)
        z_safe = max(z, 0.1)  # Avoid division by zero
        delta_T = 25.0 * torch.pow(Q_c_safe, 2/3) / (z_safe ** (5/3))
        return T_ambient + delta_T
    
    elif isinstance(Q_c, np.ndarray):
        Q_c_safe = np.maximum(Q_c, 1e-6)
        z_safe = max(z, 0.1)
        delta_T = 25.0 * np.power(Q_c_safe, 2/3) / (z_safe ** (5/3))
        result = T_ambient + delta_T
        
        if return_tensor:
            return torch.from_numpy(result).float()
        return result
    
    else:
        Q_c_safe = max(Q_c, 1e-6)
        z_safe = max(z, 0.1)
        delta_T = 25.0 * (Q_c_safe ** (2/3)) / (z_safe ** (5/3))
        result = T_ambient + delta_T
        
        if return_tensor:
            return torch.tensor(result, dtype=torch.float32)
        return result


def heskestad_flame_height_derivative(
    Q_c: torch.Tensor,
    D: float = 0.3
) -> torch.Tensor:
    """
    Calculate the derivative of flame height with respect to HRR.
    
    dL_f/dQ_c = 0.235 * (2/5) * Q_c^(-3/5)
              = 0.094 * Q_c^(-3/5)
    
    This is useful for understanding how sensitive flame height is to HRR changes.
    
    Args:
        Q_c: Convective heat release rate (kW), must be torch.Tensor
        D: Fire diameter (m)
    
    Returns:
        dL_f/dQ_c: Rate of change of flame height with HRR (m/kW)
    """
    Q_c_safe = torch.clamp(Q_c, min=1e-6)
    return 0.094 * torch.pow(Q_c_safe, -3/5)


# ═══════════════════════════════════════════════════════════════════════
# PHYSICS-INFORMED LOSS COMPONENTS
# ═══════════════════════════════════════════════════════════════════════

def physics_consistency_loss(
    predicted_hrr: torch.Tensor,
    target_hrr: torch.Tensor,
    fire_diameter: float = 0.3,
    lambda_physics: float = 0.1
) -> torch.Tensor:
    """
    Physics-informed loss that penalizes predictions violating Heskestad's correlation.
    
    The loss encourages the model to predict HRR values that result in
    physically consistent flame heights.
    
    Loss = MSE(predicted, target) + λ * physics_penalty
    
    where physics_penalty measures deviation from expected flame height growth.
    
    Args:
        predicted_hrr: Model predictions [batch, timesteps, 1]
        target_hrr: Ground truth [batch, timesteps, 1]
        fire_diameter: Fire diameter in meters
        lambda_physics: Weight for physics penalty term
    
    Returns:
        loss: Combined loss value
    """
    # Standard MSE loss
    mse_loss = torch.nn.functional.mse_loss(predicted_hrr, target_hrr)
    
    # Calculate expected flame heights
    # Convert HRR to convective (assume 70% convective)
    Q_c_pred = predicted_hrr * 0.7
    Q_c_target = target_hrr * 0.7
    
    # Heskestad flame heights
    L_f_pred = heskestad_flame_height(Q_c_pred.squeeze(-1), D=fire_diameter, return_tensor=True)
    L_f_target = heskestad_flame_height(Q_c_target.squeeze(-1), D=fire_diameter, return_tensor=True)
    
    # Physics penalty: flame heights should match if HRR is correct
    physics_penalty = torch.nn.functional.mse_loss(L_f_pred, L_f_target)
    
    # Combined loss
    total_loss = mse_loss + lambda_physics * physics_penalty
    
    return total_loss


def monotonicity_loss(
    predicted_hrr: torch.Tensor,
    lambda_monotonic: float = 0.05
) -> torch.Tensor:
    """
    Penalize non-physical HRR changes (e.g., negative HRR, extreme jumps).
    
    This encourages smooth, physically plausible predictions.
    
    Args:
        predicted_hrr: Model predictions [batch, timesteps, 1]
        lambda_monotonic: Weight for monotonicity penalty
    
    Returns:
        penalty: Monotonicity penalty value
    """
    # Penalize negative HRR (unphysical)
    negative_penalty = torch.mean(torch.relu(-predicted_hrr))
    
    # Penalize extreme changes (dHRR/dt should be bounded)
    # Fire growth/decay rates are typically < 50 kW/s
    hrr_diff = predicted_hrr[:, 1:, :] - predicted_hrr[:, :-1, :]
    extreme_change_penalty = torch.mean(torch.relu(torch.abs(hrr_diff) - 5.0))  # 5 kW per 0.1s = 50 kW/s
    
    return lambda_monotonic * (negative_penalty + extreme_change_penalty)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def compute_heskestad_features(
    hrr_sequence: np.ndarray,
    fire_diameter: float = 0.3
) -> np.ndarray:
    """
    Compute Heskestad-derived features from HRR sequence.
    
    These features provide physics-informed context to the model:
    1. Expected flame height
    2. Flame height growth rate
    3. Deviation from steady-state flame height
    
    Args:
        hrr_sequence: HRR time series [timesteps] in kW
        fire_diameter: Fire diameter in meters
    
    Returns:
        features: Array of shape [timesteps, 3] with:
                  - Column 0: Flame height (m)
                  - Column 1: Flame height growth rate (m/s)
                  - Column 2: Deviation from mean flame height (m)
    """
    # Convert to convective HRR (70% of total)
    Q_c = hrr_sequence * 0.7
    
    # Feature 1: Flame height
    L_f = heskestad_flame_height(Q_c, D=fire_diameter)
    
    # Feature 2: Flame height growth rate (dL_f/dt)
    # Sampling rate is 10 Hz (0.1s per timestep)
    dt = 0.1
    dL_f_dt = np.gradient(L_f, dt)
    
    # Feature 3: Deviation from mean (indicates fire phase)
    L_f_mean = np.mean(L_f)
    L_f_deviation = L_f - L_f_mean
    
    # Stack features
    features = np.stack([L_f, dL_f_dt, L_f_deviation], axis=1)
    
    return features.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def validate_physics_consistency(
    predicted_hrr: np.ndarray,
    actual_hrr: np.ndarray,
    fire_diameter: float = 0.3,
    tolerance: float = 0.2
) -> Tuple[bool, dict]:
    """
    Validate if predictions are physically consistent with Heskestad's correlation.
    
    Args:
        predicted_hrr: Predicted HRR values [timesteps]
        actual_hrr: Actual HRR values [timesteps]
        fire_diameter: Fire diameter in meters
        tolerance: Acceptable relative error (0.2 = 20%)
    
    Returns:
        is_valid: True if predictions are physically consistent
        metrics: Dictionary with validation metrics
    """
    # Calculate flame heights
    L_f_pred = heskestad_flame_height(predicted_hrr * 0.7, D=fire_diameter)
    L_f_actual = heskestad_flame_height(actual_hrr * 0.7, D=fire_diameter)
    
    # Compute relative error
    relative_error = np.abs(L_f_pred - L_f_actual) / (L_f_actual + 1e-6)
    mean_error = np.mean(relative_error)
    max_error = np.max(relative_error)
    
    # Check for unphysical predictions
    has_negative = np.any(predicted_hrr < 0)
    has_extreme_jumps = np.any(np.abs(np.diff(predicted_hrr)) > 5.0)  # > 50 kW/s
    
    is_valid = (mean_error < tolerance) and (not has_negative) and (not has_extreme_jumps)
    
    metrics = {
        'mean_flame_height_error': mean_error,
        'max_flame_height_error': max_error,
        'has_negative_hrr': has_negative,
        'has_extreme_jumps': has_extreme_jumps,
        'is_physically_consistent': is_valid
    }
    
    return is_valid, metrics
