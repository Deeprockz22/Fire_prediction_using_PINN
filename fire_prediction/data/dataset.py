import torch
from torch.utils.data import Dataset
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from fire_prediction.data.feature_extractor import StaticFeatureExtractor

class FireTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for Fire Heat Release Rate (HRR) prediction using Multivariate Physics Inputs.
    
    This dataset loads standardized simulation data and static scenario features.
    
    Shapes:
        Input Sequence (X): (batch, input_seq_len, 3) 
            - Channels: [HRR, Q_RADI, MLR]
        Static Features (S): (batch, static_dim)
            - Dim: 12 (One-hot Fuel/Room + scalars)
        Target Sequence (Y): (batch, pred_horizon, 1)
            - Channel: [HRR]
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train", 
        input_seq_len: int = 50, # 5 seconds
        pred_horizon: int = 10,  # 1 second
        standardize: bool = True
    ):
        """
        Args:
            data_dir: Directory containing 'ml_dataset.json' and 'splits/'
            split: Dataset split ('train', 'val', 'test')
            input_seq_len: Length of input time window
            pred_horizon: Length of prediction horizon
            standardize: If True, applies Z-score normalization using Train set statistics
        """
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.split: str = split
        self.input_seq_len: int = input_seq_len
        self.pred_horizon: int = pred_horizon
        self.standardize: bool = standardize
        
        # Initialize Feature Extractor
        self.feature_extractor: StaticFeatureExtractor = StaticFeatureExtractor()
        
        # Load Raw Dataset
        dataset_path: Path = self.data_dir / "ml_dataset.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with open(dataset_path, "r") as f:
            full_data: Dict[str, Any] = json.load(f)
        
        # Index all scenarios by name
        self.all_scenarios: Dict[str, Dict] = {s["scenario"]: s for s in full_data["scenarios"]}
        
        # Load Split
        split_file: Path = self.data_dir / "splits" / f"{split}_split.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found.")
            
        with open(split_file, "r") as f:
            self.scenario_names: List[str] = json.load(f)
            
        print(f"Loading {split} set: {len(self.scenario_names)} scenarios (Multivariate: HRR, Q_RADI, MLR)")
        
        # Compute Normalization Statistics (Train Only)
        self.stats: Dict[str, np.ndarray] = {
            'mean': np.zeros(3, dtype=np.float32), 
            'std': np.ones(3, dtype=np.float32)
        }
        
        if self.standardize:
            self._compute_stats()

        # Prepare Samples
        self.samples: List[Tuple[int, int]] = [] # Index pairs (scenario_idx, start_time_idx)
        self.processed_scenarios: List[Dict[str, Any]] = []

        self._process_scenarios()
        print(f"Generated {len(self.samples)} samples for {split} set.")

    def _compute_stats(self) -> None:
        """Compute mean and std for HRR, Q_RADI, MLR from Training split."""
        train_vals: Dict[str, List[float]] = {'hrr': [], 'q_radi': [], 'mlr': []}
        train_split_file = self.data_dir / "splits" / "train_split.json"
        
        if not train_split_file.exists():
             # Fallback if train split missing (should not happen in prod)
             print("Warning: Train split not found for stats. Using default.")
             return

        with open(train_split_file, "r") as f:
            train_scenarios_list: List[str] = json.load(f)
        
        for name in train_scenarios_list:
            s = self.all_scenarios.get(name)
            if s:
                train_vals['hrr'].extend(s["hrr_series"])
                train_vals['q_radi'].extend(s.get("q_radi_series", np.zeros_like(s["hrr_series"]).tolist()))
                train_vals['mlr'].extend(s.get("mlr_series", np.zeros_like(s["hrr_series"]).tolist()))
        
        # Compute stats [HRR, Q_RADI, MLR]
        means = [np.mean(train_vals['hrr']), np.mean(train_vals['q_radi']), np.mean(train_vals['mlr'])]
        stds = [np.std(train_vals['hrr']), np.std(train_vals['q_radi']), np.std(train_vals['mlr'])]
        
        self.stats['mean'] = np.array(means, dtype=np.float32)
        self.stats['std'] = np.array(stds, dtype=np.float32) + 1e-6
        
        print(f"Stats [HRR, Q_RADI, MLR]: Mean={self.stats['mean']}, Std={self.stats['std']}")

    def _process_scenarios(self) -> None:
        """Pre-process all scenarios: Feature extraction, normalization, and windowing."""
        for name in self.scenario_names:
            if name not in self.all_scenarios:
                continue
            
            s = self.all_scenarios[name]
            
            # Create (Time, 3) array
            hrr = np.array(s["hrr_series"], dtype=np.float32)
            q_radi = np.array(s.get("q_radi_series", np.zeros_like(hrr)), dtype=np.float32)
            mlr = np.array(s.get("mlr_series", np.zeros_like(hrr)), dtype=np.float32)
            
            combined = np.stack([hrr, q_radi, mlr], axis=1) # Shape: (L, 3)
            
            # Normalize
            if self.standardize:
                combined = (combined - self.stats['mean']) / self.stats['std']
            
            # Extract Static Features using Robust Extractor
            # Note: The feature extractor handles errors internally
            static_feats = self.feature_extractor.extract(name)
            
            self.processed_scenarios.append({
                "name": name,
                "data": torch.tensor(combined).float(),
                "static": static_feats
            })
            
            # Generate Sliding Windows
            total_len = len(combined)
            valid_starts = total_len - self.input_seq_len - self.pred_horizon
            
            # Only add samples if the sequence is long enough
            if valid_starts > 0:
                current_scenario_idx = len(self.processed_scenarios) - 1
                for i in range(valid_starts):
                    self.samples.append((current_scenario_idx, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns:
            ((x_seq, x_static), y_seq)
        """
        sc_idx, start_idx = self.samples[idx]
        scenario = self.processed_scenarios[sc_idx]
        
        data = scenario["data"] # (L, 3)
        static = scenario["static"] # (12,)
        
        # Input Window
        input_end = start_idx + self.input_seq_len
        x_seq = data[start_idx : input_end, :] # (seq_len, 3)
        
        # Target Window (Only HRR - Channel 0)
        target_start = input_end
        target_end = target_start + self.pred_horizon
        y_seq = data[target_start : target_end, 0:1] # (pred_horizon, 1)
        
        return (x_seq, static), y_seq

if __name__ == "__main__":
    # Smoke Test
    try:
        ds = FireTimeSeriesDataset(r"D:\FDS\Small_project\ml_data", split="train")
        if len(ds) > 0:
            (x, s), y = ds[0]
            print(f"Test Passed!")
            print(f"Input shape: {x.shape} (Exp: 50, 3)")
            print(f"Static shape: {s.shape} (Exp: 12)")
            print(f"Target shape: {y.shape} (Exp: 10, 1)")
        else:
            print("Dataset empty.")
            
    except Exception as e:
        print(f"Test Failed: {e}")
