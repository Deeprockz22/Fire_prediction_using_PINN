"""
Retrain Model Script
====================
Retrains the Physics-Informed LSTM model using data in training_data/ folder.
This script is self-contained and does not require external imports.

Usage:
    python retrain_model.py [--epochs N] [--batch-size N]

Example:
    python retrain_model.py --epochs 50
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
TRAINING_DATA_DIR = SCRIPT_DIR / "training_data"
MODEL_DIR = SCRIPT_DIR / "model"
MANIFEST_FILE = TRAINING_DATA_DIR / "manifest.json"


# ============================================================================
# INLINE MODEL DEFINITION (Self-contained)
# ============================================================================

class PhysicsLSTM(nn.Module):
    """
    Self-contained LSTM model for fire prediction.
    Compatible with the existing checkpoint format.
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, 
                 output_dim=3, pred_horizon=10, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        self.output_dim = output_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection (use 'head' to match predict.py expectations)
        self.head = nn.Linear(hidden_dim, output_dim * pred_horizon)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Project to prediction
        out = self.head(last_hidden)  # [batch, output_dim * pred_horizon]
        
        # Reshape to [batch, pred_horizon, output_dim]
        out = out.view(-1, self.pred_horizon, self.output_dim)
        
        return out


# ============================================================================
# HESKESTAD FEATURE COMPUTATION (Inline)
# ============================================================================

def compute_heskestad_features(hrr, fire_diameter=0.3):
    """
    Compute Heskestad flame height correlation features.
    
    Features:
    1. HRR (original)
    2. Q_RADI (placeholder - 0.35 * HRR)
    3. MLR (placeholder - HRR / 50000)
    4. Flame Height (Heskestad)
    5. Flame Height Rate
    6. Flame Height Deviation
    """
    Q = np.array(hrr, dtype=np.float32)
    n = len(Q)
    
    # Heskestad correlation: L_f = 0.235 * Q^(2/5) - 1.02 * D
    Q_positive = np.maximum(Q, 0.1)  # Avoid negative/zero
    L_f = 0.235 * np.power(Q_positive * 1000, 0.4) - 1.02 * fire_diameter
    L_f = np.maximum(L_f, 0.0)  # Non-negative flame height
    
    # Rate of change
    L_f_rate = np.zeros(n, dtype=np.float32)
    L_f_rate[1:] = L_f[1:] - L_f[:-1]
    
    # Deviation from mean
    mean_L_f = np.mean(L_f)
    L_f_dev = L_f - mean_L_f
    
    # Placeholder features
    Q_radi = 0.35 * Q  # Radiative fraction
    MLR = Q / 50000.0  # Approximate mass loss rate
    
    # Stack features: [HRR, Q_RADI, MLR, L_f, L_f_rate, L_f_dev]
    features = np.stack([Q, Q_radi, MLR, L_f, L_f_rate, L_f_dev], axis=1)
    
    return features


# ============================================================================
# DATASET
# ============================================================================

class FireDataset(Dataset):
    """Simple dataset for fire HRR sequences."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_process_csv(filepath, seq_len=30, pred_len=10):
    """Load a CSV and create training sequences."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"    ⚠️ Error reading {filepath}: {e}")
        return [], []
    
    # Find HRR column
    hrr_col = None
    for col in df.columns:
        if 'hrr' in col.lower() or 'kw' in col.lower():
            hrr_col = col
            break
    
    if hrr_col is None:
        return [], []
    
    # Convert to numeric, coercing errors to NaN
    hrr = pd.to_numeric(df[hrr_col], errors='coerce').values
    
    # Remove NaN values
    hrr = hrr[~np.isnan(hrr)]
    
    # Skip if not enough data
    if len(hrr) < seq_len + pred_len:
        return [], []
    
    # Compute Heskestad features
    features = compute_heskestad_features(hrr)
    
    # Normalize (simple z-score)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std
    
    # Create sequences
    sequences = []
    targets = []
    
    total_len = seq_len + pred_len
    for i in range(0, len(hrr) - total_len + 1, 5):  # Step by 5 to reduce overlap
        seq = features[i:i+seq_len]
        tgt = features[i+seq_len:i+total_len, :3]  # HRR, Q_RADI, MLR only
        sequences.append(seq)
        targets.append(tgt)
    
    return sequences, targets



def load_training_data():
    """Load all training data from manifest."""
    if not MANIFEST_FILE.exists():
        print("❌ No manifest found. Run 'python add_training_data.py' first.")
        return None, None
    
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    
    if not manifest["files"]:
        print("❌ No training files in manifest.")
        return None, None
    
    print(f"📊 Loading {len(manifest['files'])} training files...")
    
    all_sequences = []
    all_targets = []
    loaded = 0
    
    for filename in manifest["files"]:
        filepath = TRAINING_DATA_DIR / filename
        if filepath.exists():
            seqs, tgts = load_and_process_csv(filepath)
            if seqs:
                all_sequences.extend(seqs)
                all_targets.extend(tgts)
                loaded += 1
    
    print(f"✅ Loaded {loaded} files, created {len(all_sequences)} training sequences")
    
    if not all_sequences:
        print("❌ No valid sequences created!")
        return None, None
    
    return np.array(all_sequences), np.array(all_targets)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(sequences, targets, epochs=30, batch_size=32, lr=0.001):
    """Train the model."""
    
    # Shuffle data
    indices = np.random.permutation(len(sequences))
    sequences = sequences[indices]
    targets = targets[indices]
    
    # Split data (80/20)
    split_idx = int(len(sequences) * 0.8)
    train_seqs, val_seqs = sequences[:split_idx], sequences[split_idx:]
    train_tgts, val_tgts = targets[:split_idx], targets[split_idx:]
    
    print(f"📊 Train: {len(train_seqs)} | Val: {len(val_seqs)}")
    
    # Create datasets
    train_dataset = FireDataset(train_seqs, train_tgts)
    val_dataset = FireDataset(val_seqs, val_tgts)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = PhysicsLSTM(
        input_dim=6,
        hidden_dim=128,
        num_layers=2,
        output_dim=3,
        pred_horizon=10
    )
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    print(f"\n🚀 Training for {epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'input_dim': 6,
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'output_dim': 3,
                    'pred_horizon': 10
                }
            }, MODEL_DIR / "best_model.ckpt")
    
    print("-" * 50)
    print(f"✅ Training complete! Best Val Loss: {best_val_loss:.6f}")
    
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Retrain the fire prediction model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("RETRAIN FIRE PREDICTION MODEL")
    print("=" * 60)
    
    # Ensure model directory exists
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    sequences, targets = load_training_data()
    if sequences is None:
        return
    
    # Train
    model = train_model(
        sequences, targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    print(f"\n📁 Model saved to: {MODEL_DIR / 'best_model.ckpt'}")
    print("\n" + "=" * 60)
    print("You can now use 'python predict.py' or 'python batch_predict.py'")
    print("=" * 60)


if __name__ == "__main__":
    main()
