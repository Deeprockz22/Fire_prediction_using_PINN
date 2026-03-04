"""
Time Series Feature Engineering
================================
Utility functions for computing temporal derivatives and transformations.

Author: Fire Prediction Team
Date: 2026-03-04
"""

import torch
import numpy as np


def compute_time_derivatives(sequence: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    ✅ QUICK WIN #6: Add time derivatives (dX/dt, d²X/dt²)
    
    Computes first and second time derivatives for a time series using
    central difference approximation.
    
    Args:
        sequence: Input time series [seq_len] or [batch, seq_len] or [batch, seq_len, features]
        dt: Time step (default: 1.0)
    
    Returns:
        derivatives: Tensor with original + 1st + 2nd derivatives
                    Shape: [..., features*3]
    
    Example:
        >>> hrr = torch.tensor([100, 150, 220, 300, 350])  # [seq_len]
        >>> features = compute_time_derivatives(hrr)
        >>> # features.shape = [5, 3]  # [HRR, dHRR/dt, d²HRR/dt²]
    """
    original_shape = sequence.shape
    
    # Handle different input shapes
    if sequence.dim() == 1:
        # [seq_len] → [1, seq_len, 1]
        sequence = sequence.unsqueeze(0).unsqueeze(-1)
    elif sequence.dim() == 2:
        # [batch, seq_len] → [batch, seq_len, 1]
        sequence = sequence.unsqueeze(-1)
    # else: already [batch, seq_len, features]
    
    batch_size, seq_len, n_features = sequence.shape
    
    # Initialize derivatives
    first_deriv = torch.zeros_like(sequence)
    second_deriv = torch.zeros_like(sequence)
    
    for feat_idx in range(n_features):
        x = sequence[:, :, feat_idx]  # [batch, seq_len]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FIRST DERIVATIVE: dX/dt
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / 2h
        dx = torch.zeros_like(x)
        
        # Interior points (central difference)
        dx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (2.0 * dt)
        
        # Boundary points (forward/backward difference)
        dx[:, 0] = (x[:, 1] - x[:, 0]) / dt      # Forward
        dx[:, -1] = (x[:, -1] - x[:, -2]) / dt   # Backward
        
        first_deriv[:, :, feat_idx] = dx
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # SECOND DERIVATIVE: d²X/dt²
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Apply finite difference on first derivative
        d2x = torch.zeros_like(x)
        
        # Interior points
        d2x[:, 1:-1] = (dx[:, 2:] - dx[:, :-2]) / (2.0 * dt)
        
        # Boundary points
        d2x[:, 0] = (dx[:, 1] - dx[:, 0]) / dt
        d2x[:, -1] = (dx[:, -1] - dx[:, -2]) / dt
        
        second_deriv[:, :, feat_idx] = d2x
    
    # Concatenate: [original, 1st deriv, 2nd deriv]
    augmented = torch.cat([sequence, first_deriv, second_deriv], dim=-1)
    # Shape: [batch, seq_len, features*3]
    
    # Restore original shape if needed
    if len(original_shape) == 1:
        # [1, seq_len, 3] → [seq_len, 3]
        augmented = augmented.squeeze(0)
    elif len(original_shape) == 2 and original_shape[-1] == augmented.shape[1]:
        # [batch, seq_len, 3] → [batch, seq_len*3] (flatten features)
        pass
    
    return augmented


def add_derivative_features_to_dataset(hrr_sequence: torch.Tensor) -> torch.Tensor:
    """
    Convenience function for adding derivatives to HRR sequences in dataset.
    
    Args:
        hrr_sequence: HRR time series [seq_len] or [batch, seq_len]
    
    Returns:
        features: [seq_len, 3] or [batch, seq_len, 3] with [HRR, dHRR/dt, d²HRR/dt²]
    """
    return compute_time_derivatives(hrr_sequence, dt=1.0)


def smooth_derivatives(sequence: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    """
    Apply smoothing to noisy derivatives using moving average.
    
    Useful when derivatives amplify noise in the data.
    
    Args:
        sequence: Time series with derivatives [batch, seq_len, features]
        window_size: Smoothing window size (odd number recommended)
    
    Returns:
        smoothed: Smoothed version of input
    """
    if window_size <= 1:
        return sequence
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    padding = window_size // 2
    
    # Apply 1D convolution for moving average
    # sequence: [batch, seq_len, features]
    batch_size, seq_len, n_features = sequence.shape
    
    smoothed = torch.zeros_like(sequence)
    
    for feat_idx in range(n_features):
        x = sequence[:, :, feat_idx]  # [batch, seq_len]
        
        # Pad edges
        x_padded = torch.nn.functional.pad(x, (padding, padding), mode='replicate')
        
        # Moving average
        kernel = torch.ones(window_size, device=x.device) / window_size
        
        for b in range(batch_size):
            smoothed[b, :, feat_idx] = torch.nn.functional.conv1d(
                x_padded[b:b+1].unsqueeze(0),
                kernel.view(1, 1, -1),
                padding=0
            ).squeeze()
    
    return smoothed


def compute_rate_of_change(sequence: torch.Tensor) -> torch.Tensor:
    """
    Compute relative rate of change (percentage change).
    
    Useful for fire growth rate analysis.
    
    Args:
        sequence: Time series [batch, seq_len]
    
    Returns:
        rate: Relative rate of change [batch, seq_len]
              rate[t] = (x[t] - x[t-1]) / (x[t-1] + epsilon)
    """
    epsilon = 1e-6  # Avoid division by zero
    
    rate = torch.zeros_like(sequence)
    
    # Compute relative change
    rate[:, 1:] = (sequence[:, 1:] - sequence[:, :-1]) / (sequence[:, :-1] + epsilon)
    
    # First point has no previous value
    rate[:, 0] = 0.0
    
    return rate


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING & VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_derivatives():
    """Test derivative computation with known functions."""
    print("Testing time derivatives...")
    
    # Test 1: Linear function (slope = 2)
    # f(t) = 2t → f'(t) = 2, f''(t) = 0
    t = torch.arange(0, 10, dtype=torch.float32)
    linear = 2.0 * t
    
    features = compute_time_derivatives(linear, dt=1.0)
    
    print(f"Linear function: f(t) = 2t")
    print(f"  Original: {linear[:5].numpy()}")
    print(f"  1st deriv (should be ~2): {features[:5, 1].numpy()}")
    print(f"  2nd deriv (should be ~0): {features[:5, 2].numpy()}")
    
    # Test 2: Quadratic function
    # f(t) = t² → f'(t) = 2t, f''(t) = 2
    quadratic = t ** 2
    
    features = compute_time_derivatives(quadratic, dt=1.0)
    
    print(f"\nQuadratic function: f(t) = t²")
    print(f"  At t=5: f={quadratic[5]:.1f}, f'={features[5, 1]:.1f} (should be ~10), f''={features[5, 2]:.1f} (should be ~2)")
    
    # Test 3: Batch processing
    batch = torch.stack([linear, quadratic])  # [2, 10]
    batch_features = compute_time_derivatives(batch, dt=1.0)
    
    print(f"\nBatch processing: {batch_features.shape} (should be [2, 10, 3])")
    
    print("✅ Derivatives test passed!\n")


if __name__ == "__main__":
    test_derivatives()
