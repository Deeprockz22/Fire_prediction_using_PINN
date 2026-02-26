"""
Fire Time-Series Dataset for Koopman-Enhanced LSTM
===================================================

Loads the existing ml_dataset.json and prepares sequences for training.
Computes Heskestad-derived features as additional input channels.

Input Channels (6):
    [0] HRR         - Heat Release Rate (kW)
    [1] Q_RADI      - Radiative Heat Flux (kW/m²)
    [2] MLR         - Mass Loss Rate (kg/s)
    [3] L_f         - Heskestad Flame Height (m)
    [4] dL_f/dt     - Flame Height Growth Rate (m/s)
    [5] L_f_dev     - Flame Height Deviation from Mean (m)

Target:
    Future HRR values [pred_horizon, 1]
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from koopman_fire.utils.physics import compute_heskestad_features


class FireTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for fire HRR prediction with Heskestad physics features.

    Loads simulation data from ml_dataset.json, computes physics-derived
    features, applies normalization, and creates sliding-window samples.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        input_seq_len: int = 30,
        pred_horizon: int = 10,
        fire_diameter: float = 0.3,
        include_heskestad: bool = True,
        train_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            data_dir:           Path to folder containing ml_dataset.json and splits/
            split:              'train', 'val', or 'test'
            input_seq_len:      Number of past timesteps as model input
            pred_horizon:       Number of future timesteps to predict
            fire_diameter:      Fire source diameter in metres (for Heskestad)
            include_heskestad:  If True, append 3 Heskestad-derived channels
            train_stats:        Pass the training split's stats for val/test normalisation
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.input_seq_len = input_seq_len
        self.pred_horizon = pred_horizon
        self.fire_diameter = fire_diameter
        self.include_heskestad = include_heskestad
        self.n_base_channels = 3
        self.n_channels = 6 if include_heskestad else 3

        # ----- load raw JSON -----
        dataset_path = self.data_dir / "ml_dataset.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, "r") as f:
            full_data = json.load(f)

        self.all_scenarios = {s["scenario"]: s for s in full_data["scenarios"]}

        # ----- load split -----
        split_path = self.data_dir / "splits" / f"{split}_split.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            self.scenario_names: List[str] = json.load(f)

        # ----- compute normalisation stats -----
        if train_stats is not None:
            self.stats = train_stats.copy()
        else:
            self.stats = self._compute_stats()

        # ----- process scenarios into tensors -----
        self.processed: List[Dict] = []
        self.samples: List[Tuple[int, int]] = []
        self._process_all()

        print(
            f"[{split.upper()}] {len(self.scenario_names)} scenarios -> "
            f"{len(self.samples)} samples  |  {self.n_channels} input channels"
        )

    # ------------------------------------------------------------------
    # Normalisation statistics (computed from TRAIN split only)
    # ------------------------------------------------------------------
    def _compute_stats(self) -> Dict[str, np.ndarray]:
        """Z-score statistics over the training split."""
        train_path = self.data_dir / "splits" / "train_split.json"
        with open(train_path, "r") as f:
            train_names = json.load(f)

        all_hrr, all_qradi, all_mlr = [], [], []

        for name in train_names:
            s = self.all_scenarios.get(name)
            if s is None:
                continue
            all_hrr.extend(s["hrr_series"])
            all_qradi.extend(s.get("q_radi_series", [0.0] * len(s["hrr_series"])))
            all_mlr.extend(s.get("mlr_series", [0.0] * len(s["hrr_series"])))

        means = [np.mean(all_hrr), np.mean(all_qradi), np.mean(all_mlr)]
        stds  = [np.std(all_hrr),  np.std(all_qradi),  np.std(all_mlr)]

        stats = {
            "mean": np.array(means, dtype=np.float32),
            "std":  np.array(stds,  dtype=np.float32) + 1e-6,
        }

        # If we need Heskestad stats, compute them here too
        if self.include_heskestad:
            all_feats = []
            for name in train_names:
                s = self.all_scenarios.get(name)
                if s is None:
                    continue
                hrr = np.array(s["hrr_series"], dtype=np.float32)
                feats = compute_heskestad_features(hrr, self.fire_diameter)
                all_feats.append(feats)

            all_feats = np.concatenate(all_feats, axis=0)  # [total_timesteps, 3]
            h_mean = all_feats.mean(axis=0)
            h_std  = all_feats.std(axis=0) + 1e-6

            stats["mean"] = np.concatenate([stats["mean"], h_mean])
            stats["std"]  = np.concatenate([stats["std"],  h_std])

        return stats

    # ------------------------------------------------------------------
    # Build tensor representations for every scenario
    # ------------------------------------------------------------------
    def _process_all(self):
        for name in self.scenario_names:
            s = self.all_scenarios.get(name)
            if s is None:
                continue

            hrr   = np.array(s["hrr_series"],                                dtype=np.float32)
            qradi = np.array(s.get("q_radi_series", np.zeros_like(hrr)),     dtype=np.float32)
            mlr   = np.array(s.get("mlr_series",    np.zeros_like(hrr)),     dtype=np.float32)

            # Stack base channels  [T, 3]
            data = np.stack([hrr, qradi, mlr], axis=1)

            # Optionally append Heskestad features  [T, 3] → concat → [T, 6]
            if self.include_heskestad:
                h_feats = compute_heskestad_features(hrr, self.fire_diameter)
                data = np.concatenate([data, h_feats], axis=1)

            # Normalise (Z-score with train stats)
            data = (data - self.stats["mean"]) / self.stats["std"]

            data_tensor = torch.from_numpy(data).float()

            idx = len(self.processed)
            self.processed.append({"name": name, "data": data_tensor})

            # Sliding-window sample indices
            n_valid = len(data) - self.input_seq_len - self.pred_horizon
            for t in range(max(n_valid, 0)):
                self.samples.append((idx, t))

    # ------------------------------------------------------------------
    # PyTorch interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x : [input_seq_len, n_channels]   input window
            y : [pred_horizon,  1]            target HRR values
        """
        sc_idx, t0 = self.samples[idx]
        data = self.processed[sc_idx]["data"]

        t1 = t0 + self.input_seq_len
        t2 = t1 + self.pred_horizon

        x = data[t0:t1, :]       # [seq, C]
        y = data[t1:t2, 0:1]     # [horizon, 1]  (HRR channel only)

        return x, y


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    DATA_DIR = r"D:\FDS\Small_project\ml_data"

    ds = FireTimeSeriesDataset(DATA_DIR, split="train")
    x, y = ds[0]
    print(f"Input shape:  {x.shape}  (expect [30, 6])")
    print(f"Target shape: {y.shape}  (expect [10, 1])")
    print(f"Stats mean:   {ds.stats['mean']}")
    print(f"Stats std:    {ds.stats['std']}")
