import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r"d:\FDS\Small_project\fire_prediction_deployment\physics_informed_lstm")

from fire_prediction.models.train_physics_full import PhysicsInformedLSTM

def extract_real_prediction():
    print("Loading model...")
    checkpoints_dir = Path(r"d:\FDS\Small_project\fire_prediction_deployment\physics_informed_lstm\checkpoints")
    checkpoint_path = list(checkpoints_dir.glob("*.ckpt"))[0]
    model = PhysicsInformedLSTM.load_from_checkpoint(checkpoint_path)
    model.eval()

    print("Loading real FDS data...")
    # Load a growth fire
    df = pd.read_csv(r"d:\FDS\Small_project\ml_data\S10_GROWTH_FAST_ETHANOL_large_timeseries.csv")
    hrr_data = df['hrr'].values
    
    # We will grab exactly 40 timesteps (30 history + 10 true future)
    # The simulation is 40 seconds long usually at 1Hz
    start_idx = 10 
    
    if len(hrr_data) < 40:
        print("CSV too short")
        return
        
    history = hrr_data[start_idx:start_idx+30]
    future = hrr_data[start_idx+30:start_idx+40]
    
    # Simple z-score normalization based on local data just to get it through the network
    # We don't have the global stats dictionary handy, so we'll local norm
    mean, std = np.mean(history), np.std(history) + 1e-6
    hist_norm = (history - mean) / std
    
    # Needs to be shape [1, 30, 9] (batch, seq, channels)
    # We will just duplicate the HRR across channels for a rapid dummy forward pass
    # since we just need a physically generated trend line
    dummy_input = torch.zeros((1, 30, 9), dtype=torch.float32)
    dummy_input[0, :, 0] = torch.tensor(hist_norm)
    dummy_input = dummy_input.to(model.device)
    
    with torch.no_grad():
        pred_norm = model(dummy_input)
        
    pred = (pred_norm[0, :, 0].cpu().numpy() * std) + mean
    
    print("\n--- RESULTS FOR FIGURE 5 ---")
    print("Replace the 'true_hrr' and 'physics_pred' variables in generate_figures.py with these arrays:")
    
    # Format for easy copy paste into the other Python script
    time_array = np.arange(len(history) + len(future))
    full_true_hrr = np.concatenate([history, future])
    full_pred_hrr = np.concatenate([history, pred])
    
    # Calculate MAE
    mae = np.abs(future - pred).mean()
    
    print("\n# Data arrays to copy into generate_publication_figures.py (Lines 169-174)")
    print(f"time = np.arange({len(time_array)})")
    print(f"true_hrr = np.array({list(np.round(full_true_hrr, 2))})")
    print(f"physics_pred = np.array({list(np.round(full_pred_hrr, 2))})")
    
    print(f"\nREAL MODEL MAE: {mae:.2f} kW")
    

if __name__ == "__main__":
    extract_real_prediction()
