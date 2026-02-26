"""
Physics-Informed Utilities for Fire Prediction

This module implements fire physics correlations for flame and plume behavior.

References:
- Heskestad, G. (1984). "Engineering relations for fire plumes"
  Fire Safety Journal, 7(1), 25-32
- McCaffrey, B. J. (1979). "Purely buoyant diffusion flames"
  National Bureau of Standards, NBSIR 79-1910
- Thomas, P. H. (1963). "The size of flames from natural fires"
  Symposium (International) on Combustion, 9(1), 844-859
"""

import torch
import numpy as np
from typing import Union, Tuple

# ═══════════════════════════════════════════════════════════════════════
# HESKESTAD CORRELATIONS (Flame Height)
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

def mccaffrey_plume_region(
    Q: Union[float, np.ndarray, torch.Tensor],
    z: float
) -> Union[str, np.ndarray]:
    """
    Determine McCaffrey plume region (continuous, intermittent, or far-field).
    
    McCaffrey's regions based on z/Q^(2/5):
    - Continuous flame: z/Q^(2/5) < 0.08
    - Intermittent: 0.08 ≤ z/Q^(2/5) < 0.20
    - Far-field plume: z/Q^(2/5) ≥ 0.20
    
    Args:
        Q: Heat release rate (kW)
        z: Height above fire source (m)
    
    Returns:
        Normalized region indicator: 0.0 (continuous), 0.5 (intermittent), 1.0 (far-field)
    """
    if isinstance(Q, torch.Tensor):
        Q_safe = torch.clamp(Q, min=1e-6)
        z_star = z / torch.pow(Q_safe, 2/5)
        # Continuous: 0, Intermittent: 0.5, Far-field: 1.0
        region = torch.where(z_star < 0.08, torch.tensor(0.0),
                           torch.where(z_star < 0.20, torch.tensor(0.5), torch.tensor(1.0)))
        return region
    
    elif isinstance(Q, np.ndarray):
        Q_safe = np.maximum(Q, 1e-6)
        z_star = z / np.power(Q_safe, 2/5)
        region = np.where(z_star < 0.08, 0.0, np.where(z_star < 0.20, 0.5, 1.0))
        return region
    
    else:
        Q_safe = max(Q, 1e-6)
        z_star = z / (Q_safe ** (2/5))
        if z_star < 0.08:
            return 0.0
        elif z_star < 0.20:
            return 0.5
        else:
            return 1.0


def thomas_window_flow(
    Q: Union[float, np.ndarray, torch.Tensor],
    A_w: float,
    H_w: float
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Calculate mass flow rate through window/opening using Thomas correlation.
    
    Mass flow rate through vertical opening:
        m_dot ≈ 0.5 * A_w * sqrt(H_w) (kg/s)
    
    This represents natural ventilation driven by buoyancy.
    
    Args:
        Q: Heat release rate (kW) - used for buoyancy calculation
        A_w: Window/opening area (m²)
        H_w: Window/opening height (m)
    
    Returns:
        m_dot: Mass flow rate through opening (kg/s)
    """
    # Simplified Thomas correlation for natural ventilation
    # Mass flow proportional to area and sqrt(height)
    if isinstance(Q, torch.Tensor):
        H_safe = max(H_w, 0.1)
        m_dot = 0.5 * A_w * torch.sqrt(torch.tensor(H_safe))
        return m_dot * torch.ones_like(Q)  # Broadcast to match Q shape
    
    elif isinstance(Q, np.ndarray):
        H_safe = max(H_w, 0.1)
        m_dot = 0.5 * A_w * np.sqrt(H_safe)
        return m_dot * np.ones_like(Q)
    
    else:
        H_safe = max(H_w, 0.1)
        return 0.5 * A_w * np.sqrt(H_safe)


def ventilation_factor(
    A_w: float,
    H_w: float,
    room_area: float = 9.0
) -> float:
    """
    Calculate ventilation factor for compartment fires.
    
    Ventilation factor = A_w * sqrt(H_w) / room_area
    Higher values indicate better ventilation.
    
    Args:
        A_w: Opening area (m²)
        H_w: Opening height (m)
        room_area: Floor area of room (m²)
    
    Returns:
        Ventilation factor (dimensionless)
    """
    return (A_w * np.sqrt(max(H_w, 0.1))) / max(room_area, 1.0)


def compute_enhanced_features(
    hrr_sequence: np.ndarray,
    fire_diameter: float = 0.3,
    room_dims: dict = None
) -> np.ndarray:
    """
    Compute enhanced physics features from HRR sequence.
    
    Combines multiple fire physics correlations:
    1. Heskestad flame height
    2. Flame height growth rate
    3. Flame height deviation
    4. McCaffrey plume region indicator
    5. Window/ventilation flow factor
    6. Buoyancy indicator (Q^(2/5))
    
    Args:
        hrr_sequence: HRR time series [timesteps] in kW
        fire_diameter: Fire diameter in meters
        room_dims: Optional dict with 'opening_area', 'opening_height', 'room_area'
    
    Returns:
        features: Array of shape [timesteps, 6] with all physics features
    """
    # Convert to convective HRR (70% of total)
    Q_c = hrr_sequence * 0.7
    
    # Feature 1: Heskestad flame height
    L_f = heskestad_flame_height(Q_c, D=fire_diameter)
    
    # Feature 2: Flame height growth rate (dL_f/dt)
    dt = 0.1  # Sampling rate: 10 Hz
    dL_f_dt = np.gradient(L_f, dt)
    
    # Feature 3: Deviation from mean (indicates fire phase)
    L_f_mean = np.mean(L_f)
    L_f_deviation = L_f - L_f_mean
    
    # Feature 4: McCaffrey plume region at mid-height (z = 1.2m)
    z_measure = 1.2  # Typical measurement height
    plume_region = mccaffrey_plume_region(hrr_sequence, z_measure)
    
    # Feature 5: Ventilation flow indicator (if room dims available)
    if room_dims and 'opening_area' in room_dims:
        vent_flow = thomas_window_flow(hrr_sequence, 
                                       room_dims['opening_area'],
                                       room_dims.get('opening_height', 1.0))
        # Normalize by typical values (0-2 kg/s range)
        vent_flow_norm = vent_flow / 2.0
    else:
        # Default: assume moderate ventilation
        vent_flow_norm = np.ones_like(hrr_sequence) * 0.5
    
    # Feature 6: Buoyancy power indicator (Q^(2/5) - fundamental scaling)
    buoyancy_power = np.power(np.maximum(hrr_sequence, 1e-6), 2/5)
    # Normalize by typical value (100 kW -> ~6.3)
    buoyancy_norm = buoyancy_power / 6.3
    
    # Stack all features
    features = np.stack([
        L_f,                # Flame height (m)
        dL_f_dt,           # Flame growth rate (m/s)
        L_f_deviation,     # Flame deviation (m)
        plume_region,      # McCaffrey region (0-1)
        vent_flow_norm,    # Ventilation flow (normalized)
        buoyancy_norm      # Buoyancy power (normalized)
    ], axis=1)
    
    return features.astype(np.float32)


def compute_heskestad_features(
    hrr_sequence: np.ndarray,
    fire_diameter: float = 0.3
) -> np.ndarray:
    """
    Compute Heskestad-derived features from HRR sequence.
    
    LEGACY FUNCTION: Kept for backward compatibility.
    For new models, use compute_enhanced_features() which includes
    additional correlations (McCaffrey, ventilation).
    
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
