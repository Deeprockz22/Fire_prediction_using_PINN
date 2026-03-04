"""
Data Augmentation for Fire Prediction
======================================
Physics-consistent augmentation techniques for time series data.

Author: Fire Prediction Team  
Date: 2026-03-04
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple


class PhysicsConsistentAugmentation:
    """
    ✅ QUICK WIN #7: Data augmentation with physics constraints
    
    Applies various augmentation techniques while maintaining physical validity:
    - Gaussian noise (with non-negativity constraint for HRR)
    - Time stretching/compression
    - Random time shifts
    - Magnitude scaling
    """
    
    def __init__(
        self,
        noise_level: float = 0.05,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        magnitude_scale_range: Tuple[float, float] = (0.95, 1.05),
        time_shift_range: int = 3,
        apply_prob: float = 0.5
    ):
        """
        Args:
            noise_level: Std of Gaussian noise as fraction of signal std
            time_stretch_range: Min/max time stretching factors
            magnitude_scale_range: Min/max magnitude scaling factors
            time_shift_range: Max time shift in steps (±)
            apply_prob: Probability of applying each augmentation
        """
        self.noise_level = noise_level
        self.time_stretch_range = time_stretch_range
        self.magnitude_scale_range = magnitude_scale_range
        self.time_shift_range = time_shift_range
        self.apply_prob = apply_prob
    
    def add_gaussian_noise(
        self, 
        sequence: torch.Tensor, 
        feature_idx: int = 0,
        is_hrr: bool = True
    ) -> torch.Tensor:
        """
        Add physics-consistent Gaussian noise.
        
        Args:
            sequence: [seq_len, n_features] or [seq_len]
            feature_idx: Which feature to augment (0=HRR usually)
            is_hrr: If True, enforce non-negativity constraint
        
        Returns:
            Augmented sequence
        """
        if torch.rand(1).item() > self.apply_prob:
            return sequence
        
        augmented = sequence.clone()
        
        # Get the feature to augment
        if sequence.dim() == 1:
            signal = sequence
        else:
            signal = sequence[:, feature_idx]
        
        # Compute signal-dependent noise
        signal_std = signal.std()
        noise = torch.randn_like(signal) * self.noise_level * signal_std
        
        # Add noise
        noisy_signal = signal + noise
        
        # Physics constraint: HRR cannot be negative
        if is_hrr:
            noisy_signal = torch.clamp(noisy_signal, min=0.0)
        
        # Update sequence
        if sequence.dim() == 1:
            augmented = noisy_signal
        else:
            augmented[:, feature_idx] = noisy_signal
        
        return augmented
    
    def time_stretch(
        self, 
        sequence: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Stretch or compress time axis.
        
        Args:
            sequence: [seq_len, n_features] or [seq_len]
            target_length: If None, randomly sample from time_stretch_range
        
        Returns:
            Time-stretched sequence (interpolated to original length)
        """
        if torch.rand(1).item() > self.apply_prob:
            return sequence
        
        original_len = sequence.shape[0]
        
        if target_length is None:
            # Random stretch factor
            stretch_factor = torch.FloatTensor(1).uniform_(
                self.time_stretch_range[0],
                self.time_stretch_range[1]
            ).item()
            target_length = int(original_len * stretch_factor)
        
        # Interpolate to new length
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
            stretched = nn.functional.interpolate(
                sequence,
                size=target_length,
                mode='linear',
                align_corners=False
            )
            stretched = stretched.squeeze(0).squeeze(0)  # [seq_len]
            
            # Interpolate back to original length
            stretched = stretched.unsqueeze(0).unsqueeze(0)
            final = nn.functional.interpolate(
                stretched,
                size=original_len,
                mode='linear',
                align_corners=False
            )
            return final.squeeze(0).squeeze(0)
        else:
            # [seq_len, n_features] → [n_features, seq_len]
            sequence = sequence.transpose(0, 1).unsqueeze(0)  # [1, n_features, seq_len]
            stretched = nn.functional.interpolate(
                sequence,
                size=target_length,
                mode='linear',
                align_corners=False
            )
            # Back to original length
            final = nn.functional.interpolate(
                stretched,
                size=original_len,
                mode='linear',
                align_corners=False
            )
            return final.squeeze(0).transpose(0, 1)  # [seq_len, n_features]
    
    def time_shift(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Randomly shift sequence in time.
        
        Args:
            sequence: [seq_len] or [seq_len, n_features]
        
        Returns:
            Time-shifted sequence (with edge padding)
        """
        if torch.rand(1).item() > self.apply_prob:
            return sequence
        
        shift = torch.randint(
            -self.time_shift_range,
            self.time_shift_range + 1,
            (1,)
        ).item()
        
        if shift == 0:
            return sequence
        
        # Roll and pad edges
        shifted = torch.roll(sequence, shifts=shift, dims=0)
        
        if shift > 0:
            # Pad beginning
            shifted[:shift] = sequence[0] if sequence.dim() == 1 else sequence[0:1]
        else:
            # Pad end
            shifted[shift:] = sequence[-1] if sequence.dim() == 1 else sequence[-1:]
        
        return shifted
    
    def magnitude_scale(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Scale magnitude by random factor.
        
        Args:
            sequence: [seq_len] or [seq_len, n_features]
        
        Returns:
            Scaled sequence
        """
        if torch.rand(1).item() > self.apply_prob:
            return sequence
        
        scale = torch.FloatTensor(1).uniform_(
            self.magnitude_scale_range[0],
            self.magnitude_scale_range[1]
        ).item()
        
        return sequence * scale
    
    def augment(self, sequence: torch.Tensor, is_hrr: bool = True) -> torch.Tensor:
        """
        Apply all augmentations in sequence.
        
        Args:
            sequence: Input time series
            is_hrr: Whether this is HRR data (for non-negativity)
        
        Returns:
            Augmented sequence
        """
        augmented = sequence.clone()
        
        # Apply augmentations (order matters!)
        augmented = self.magnitude_scale(augmented)
        augmented = self.time_stretch(augmented)
        augmented = self.time_shift(augmented)
        augmented = self.add_gaussian_noise(augmented, is_hrr=is_hrr)
        
        return augmented


class AugmentedFireDataset(Dataset):
    """
    ✅ QUICK WIN #7: Wrapper for dataset with augmentation
    
    Wraps existing dataset and applies augmentation during training.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        augment: bool = True,
        augmentation_config: Optional[dict] = None
    ):
        """
        Args:
            base_dataset: Original dataset
            augment: Whether to apply augmentation
            augmentation_config: Config dict for PhysicsConsistentAugmentation
        """
        self.base_dataset = base_dataset
        self.augment = augment
        
        # Initialize augmentation
        if augmentation_config is None:
            augmentation_config = {
                'noise_level': 0.05,
                'apply_prob': 0.5
            }
        
        self.augmentor = PhysicsConsistentAugmentation(**augmentation_config)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        sample = self.base_dataset[idx]
        
        if not self.augment:
            return sample
        
        # Augment depending on dataset format
        if isinstance(sample, tuple) and len(sample) == 2:
            # Format: ((x_seq, x_static), y)
            (x_seq, x_static), y = sample
            
            # Augment time series only (not static features)
            x_seq_aug = self.augmentor.augment(x_seq, is_hrr=True)
            
            # Note: We don't augment y (target) to maintain consistency
            # But you could augment it the same way for consistency
            
            return ((x_seq_aug, x_static), y)
        
        else:
            # Simple format: (x, y)
            x, y = sample
            x_aug = self.augmentor.augment(x, is_hrr=True)
            return (x_aug, y)
    
    def set_augment(self, augment: bool):
        """Enable/disable augmentation (useful for validation)."""
        self.augment = augment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_augmentation():
    """Test augmentation functions."""
    print("Testing data augmentation...\n")
    
    # Create synthetic fire HRR curve
    t = torch.linspace(0, 10, 100)
    hrr = 1000 * (1 - torch.exp(-0.5 * t))  # Exponential growth to 1000 kW
    
    print(f"Original HRR: min={hrr.min():.1f}, max={hrr.max():.1f}, mean={hrr.mean():.1f}")
    
    # Initialize augmentor
    augmentor = PhysicsConsistentAugmentation(
        noise_level=0.05,
        time_stretch_range=(0.8, 1.2),
        magnitude_scale_range=(0.9, 1.1),
        time_shift_range=5,
        apply_prob=1.0  # Always apply for testing
    )
    
    # Test noise
    noisy = augmentor.add_gaussian_noise(hrr)
    print(f"After noise:  min={noisy.min():.1f}, max={noisy.max():.1f}, mean={noisy.mean():.1f}")
    assert noisy.min() >= 0, "❌ HRR became negative!"
    
    # Test scaling
    scaled = augmentor.magnitude_scale(hrr)
    print(f"After scale:  min={scaled.min():.1f}, max={scaled.max():.1f}, mean={scaled.mean():.1f}")
    
    # Test time stretch
    stretched = augmentor.time_stretch(hrr)
    print(f"After stretch: shape={stretched.shape} (should be {hrr.shape})")
    
    # Test time shift
    shifted = augmentor.time_shift(hrr)
    print(f"After shift:  shape={shifted.shape}")
    
    # Test full augmentation
    augmented = augmentor.augment(hrr)
    print(f"Fully augmented: min={augmented.min():.1f}, max={augmented.max():.1f}")
    
    print("\n[OK] All augmentation tests passed!")
    
    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(hrr.numpy(), label='Original', linewidth=2)
        plt.plot(noisy.numpy(), label='+ Noise', alpha=0.7)
        plt.legend()
        plt.title('Noise Injection')
        plt.ylabel('HRR (kW)')
        
        plt.subplot(1, 3, 2)
        plt.plot(hrr.numpy(), label='Original', linewidth=2)
        plt.plot(stretched.numpy(), label='Time Stretched', alpha=0.7)
        plt.legend()
        plt.title('Time Stretching')
        
        plt.subplot(1, 3, 3)
        plt.plot(hrr.numpy(), label='Original', linewidth=2)
        plt.plot(augmented.numpy(), label='Fully Augmented', alpha=0.7)
        plt.legend()
        plt.title('All Augmentations')
        
        plt.tight_layout()
        plt.savefig('augmentation_test.png', dpi=150)
        print("[INFO] Visualization saved to: augmentation_test.png")
        
    except ImportError:
        print("[INFO] matplotlib not available, skipping visualization")


if __name__ == "__main__":
    test_augmentation()
