"""
Enhanced Dataset with Multiple Physics Correlations

This dataset extends the base FireTimeSeriesDataset to include
physics-informed features from multiple correlations.

Input channels expanded from 3 to 9:
- Original: [HRR, Q_RADI, MLR]
- Heskestad: [Flame_Height, dFlame_Height/dt, Flame_Height_Deviation]
- McCaffrey: [Plume_Region]
- Thomas: [Ventilation_Flow]
- Buoyancy: [Buoyancy_Power]
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fire_prediction.data.dataset import FireTimeSeriesDataset
from fire_prediction.utils.physics import compute_heskestad_features, compute_enhanced_features


class PhysicsInformedDataset(FireTimeSeriesDataset):
    """
    Dataset with enhanced physics features from multiple correlations.
    
    This class extends the base dataset to compute and include
    six additional physics-informed features based on:
    - Heskestad's flame height correlation
    - McCaffrey's plume region classification
    - Thomas ventilation flow correlation
    - Buoyancy power scaling
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        input_seq_len: int = 30,
        pred_horizon: int = 10,
        standardize: bool = True,
        fire_diameter: float = 0.3,  # Fire diameter for Heskestad (m)
        include_heskestad: bool = True,  # Toggle enhanced physics features
        train_stats: dict = None,  # Pass train stats for val/test splits
        room_dims: dict = None  # Room dimensions for ventilation correlation
    ):
        """
        Initialize physics-informed dataset.
        
        Args:
            data_dir: Path to ml_dataset.json
            split: 'train', 'val', or 'test'
            input_seq_len: Number of timesteps for input
            pred_horizon: Number of timesteps to predict
            standardize: Whether to normalize data
            fire_diameter: Fire diameter for physics correlations (m)
            include_heskestad: If True, add 6 enhanced physics features
            train_stats: For val/test splits, pass the train split's stats
            room_dims: Optional room dimensions for ventilation correlation
        """
        # Store parameters before calling parent init
        self.fire_diameter = fire_diameter
        self.include_heskestad = include_heskestad
        self.train_stats = train_stats
        self.room_dims = room_dims if room_dims else {
            'opening_area': 0.8,  # Default 0.8 m² opening
            'opening_height': 1.0,  # Default 1.0 m height
            'room_area': 9.0  # Default 3m x 3m room
        }
        
        # Initialize base dataset
        super().__init__(
            data_dir=data_dir,
            split=split,
            input_seq_len=input_seq_len,
            pred_horizon=pred_horizon,
            standardize=standardize
        )
        
        # Add enhanced physics features AFTER parent initialization
        if self.include_heskestad:
            print(f"Computing enhanced physics features (fire diameter={fire_diameter}m)...")
            self._add_heskestad_features()
            print(f"✅ Dataset now has 9 input channels (3 original + 6 physics features)")
    
    def _add_heskestad_features(self):
        """
        Compute and append enhanced physics features to all processed scenarios.
        
        For each scenario, compute:
        1. Heskestad flame height from HRR
        2. Flame height growth rate
        3. Deviation from mean flame height
        4. McCaffrey plume region indicator
        5. Thomas ventilation flow factor
        6. Buoyancy power scaling
        
        This is done in two passes:
        - Pass 1 (train only): Collect all features and compute normalization stats
        - Pass 2: Apply normalization and concatenate features
        """
        # First pass: Collect all Heskestad features
        all_heskestad_features = []
        
        for scenario in self.processed_scenarios:
            # Get the data tensor [timesteps, 3]
            data_tensor = scenario["data"]
            data_np = data_tensor.numpy()
            
            # Extract HRR channel (index 0)
            # Need to un-normalize first if standardized
            if self.standardize:
                hrr_normalized = data_np[:, 0]
                hrr_sequence = hrr_normalized * self.stats['std'][0] + self.stats['mean'][0]
            else:
                hrr_sequence = data_np[:, 0]
            
            # Compute enhanced physics features [timesteps, 6]
            heskestad_feats = compute_enhanced_features(
                hrr_sequence,
                fire_diameter=self.fire_diameter,
                room_dims=self.room_dims
            )
            
            all_heskestad_features.append(heskestad_feats)
        
        # Compute normalization stats for Heskestad features (train split only)
        if self.standardize and self.split == "train":
            self._compute_heskestad_stats_from_features(all_heskestad_features)
        elif self.standardize and self.train_stats is not None:
            # For val/test, use the stats from train split
            self.stats = self.train_stats.copy()
            print(f"Using train split's Heskestad normalization stats")
        
        # Second pass: Normalize and concatenate
        for i, scenario in enumerate(self.processed_scenarios):
            data_np = scenario["data"].numpy()
            heskestad_feats = all_heskestad_features[i]
            
            # Normalize Heskestad features if stats are available
            if self.standardize and len(self.stats['mean']) > 3:
                heskestad_mean = self.stats['mean'][3:]
                heskestad_std = self.stats['std'][3:]
                heskestad_feats = (heskestad_feats - heskestad_mean) / heskestad_std
            
            # Concatenate with original data
            # Original: [timesteps, 3] (HRR, Q_RADI, MLR)
            # Enhanced: [timesteps, 9] (HRR, Q_RADI, MLR, L_f, dL_f/dt, L_f_dev, plume, vent, buoyancy)
            enhanced_data = np.concatenate([data_np, heskestad_feats], axis=1)
            
            # Update the scenario's data tensor
            scenario["data"] = torch.from_numpy(enhanced_data).float()
    
    def _compute_heskestad_stats_from_features(self, all_heskestad_features):
        """
        Compute normalization statistics for enhanced physics features from collected features.
        
        Args:
            all_heskestad_features: List of numpy arrays, each [timesteps, 6]
        """
        # Collect all 6 features
        feature_arrays = [[] for _ in range(6)]
        
        for heskestad_feats in all_heskestad_features:
            for i in range(6):
                feature_arrays[i].append(heskestad_feats[:, i])
        
        # Compute stats for each feature
        feature_means = []
        feature_stds = []
        
        feature_names = [
            'Flame height', 'Flame rate', 'Flame deviation',
            'Plume region', 'Ventilation flow', 'Buoyancy power'
        ]
        
        print(f"Computed normalization stats for enhanced physics features:")
        for i, name in enumerate(feature_names):
            mean_val = np.mean(np.concatenate(feature_arrays[i]))
            std_val = np.std(np.concatenate(feature_arrays[i]))
            feature_means.append(mean_val)
            feature_stds.append(std_val)
            if i < 3:  # Only print first 3 to keep output concise
                print(f"  {name}: μ={mean_val:.3f}, σ={std_val:.3f}")
        
        # Extend stats arrays
        self.stats['mean'] = np.concatenate([
            self.stats['mean'],
            np.array(feature_means, dtype=np.float32)
        ])
        
        self.stats['std'] = np.concatenate([
            self.stats['std'],
            np.array(feature_stds, dtype=np.float32) + 1e-6
        ])
        
        print(f"  ... and {len(feature_names) - 3} more correlation features")


if __name__ == "__main__":
    # Test the enhanced dataset
    print("="*60)
    print("Testing Physics-Informed Dataset")
    print("="*60)
    
    DATA_DIR = r"D:\FDS\Small_project\ml_data"
    
    # Create dataset with Heskestad features
    dataset = PhysicsInformedDataset(
        DATA_DIR,
        split="train",
        input_seq_len=30,
        pred_horizon=10,
        include_heskestad=True,
        fire_diameter=0.3
    )
    
    print(f"\nDataset size: {len(dataset)} samples")
    
    # Get a sample
    (x, static), y = dataset[0]
    
    print(f"\nSample shapes:")
    print(f"  Input (x): {x.shape} - includes 6 channels:")
    print(f"    [0] HRR")
    print(f"    [1] Q_RADI")
    print(f"    [2] MLR")
    print(f"    [3] Flame Height (Heskestad)")
    print(f"    [4] Flame Height Rate (Heskestad)")
    print(f"    [5] Flame Height Deviation (Heskestad)")
    print(f"  Static features: {static.shape}")
    print(f"  Target (y): {y.shape} - HRR only")
    
    print(f"\nNormalization stats shape: {dataset.stats['mean'].shape}")
    print("="*60)
