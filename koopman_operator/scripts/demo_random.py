"""
Random Scenario Demo
====================
Picks a random scenario from the TEST set and runs the Koopman prediction.
Demonstrates the model's generalization to unseen fires.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from koopman_fire.data.dataset import FireTimeSeriesDataset
from koopman_fire.models.koopman_lstm import KoopmanLSTM
from koopman_fire.predict import find_checkpoint, denormalise_hrr

def run_random_demo():
    # 1. Load Model
    ckpt_path = find_checkpoint()
    print(f"Loading model: {Path(ckpt_path).name}")
    model = KoopmanLSTM.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.cpu()
    
    # 2. Load Test Dataset
    DATA_DIR = r"D:\FDS\Small_project\ml_data"
    # We need the train stats to denormalize
    train_ds = FireTimeSeriesDataset(DATA_DIR, split="train")
    test_ds = FireTimeSeriesDataset(DATA_DIR, split="test", train_stats=train_ds.stats)
    
    # 3. Pick Random Scenario
    idx = random.randint(0, len(test_ds.processed) - 1)
    scenario = test_ds.processed[idx]
    name = scenario["name"]
    data = scenario["data"]
    
    print(f"\nSelected Random Scenario: {name}")
    print(f"Duration: {len(data)/10:.1f} seconds")

    # 4. Prepare Input (First 3s)
    seq_len = test_ds.input_seq_len  # 30 steps = 3s
    x = data[0:seq_len, :].unsqueeze(0) # [1, 30, 6]
    
    # 5. Predict Future (Next 10s = 100 steps)
    input_hrr = denormalise_hrr(data[0:seq_len, 0].numpy(), train_ds.stats)
    
    PRED_STEPS = 100
    with torch.no_grad():
        # Koopman Multi-Step Prediction
        y_pred = model.predict_sequence(x, steps=PRED_STEPS)
        y_pred = y_pred.squeeze().cpu().numpy()
        pred_hrr = denormalise_hrr(y_pred, train_ds.stats)

    # 6. Get Actual Future
    future_len = min(PRED_STEPS, len(data) - seq_len)
    actual_data = data[seq_len : seq_len+future_len, 0].numpy()
    actual_hrr = denormalise_hrr(actual_data, train_ds.stats)

    # 7. Plot
    dt = 0.1
    t_input = np.arange(seq_len) * dt
    t_pred = np.arange(seq_len, seq_len + PRED_STEPS) * dt
    t_actual = np.arange(seq_len, seq_len + future_len) * dt

    plt.figure(figsize=(10, 6))
    
    # Plot Context
    plt.plot(t_input, input_hrr, 'k-', linewidth=2, label="Observed (Past)")
    
    # Plot Actual Future
    plt.plot(t_actual, actual_hrr, 'k--', alpha=0.5, label="Actual Future (Ground Truth)")
    
    # Plot Prediction
    plt.plot(t_pred, pred_hrr, 'r-', linewidth=2, label="Koopman Prediction")
    
    # Uncertainty/Error shading (heuristic)
    plt.fill_between(t_pred[:len(actual_hrr)], 
                     pred_hrr[:len(actual_hrr)], 
                     actual_hrr, color='red', alpha=0.1)

    plt.axvline(x=3.0, color='gray', linestyle=':', label="Predict Start")
    plt.title(f"Random Test Scenario: {name}\nKoopman Forecasting (10s Horizon)", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Heat Release Rate (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path(__file__).parent / "predictions" / f"random_demo_{name}.png"
    plt.savefig(output_path)
    print(f"\n✅ Plot saved to: {output_path}")

if __name__ == "__main__":
    run_random_demo()
