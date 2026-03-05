import sys
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r"d:\FDS\Small_project\fire_prediction_deployment\physics_informed_lstm")

from fire_prediction.models.train_physics_full import PhysicsInformedLSTM
from fire_prediction.data.physics_dataset import PhysicsInformedDataset
from torch.utils.data import DataLoader

def evaluate_model():
    print("Loading model...")
    # Get the best checkpoint
    checkpoints_dir = Path(r"d:\FDS\Small_project\fire_prediction_deployment\physics_informed_lstm\checkpoints")
    checkpoint_path = list(checkpoints_dir.glob("*.ckpt"))[0]
    
    # Load model from checkpoint
    model = PhysicsInformedLSTM.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    print("Loading validation dataset...")
    # Load the structured dataset that the user built
    dataset = PhysicsInformedDataset(
        data_dir=r"d:\FDS\Small_project\ml_data",
        split="test",
        input_seq_len=30,
        pred_horizon=10
    )
    
    # We want to test on unseen data, so we'll grab a few random batches
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    maes = []
    print("Evaluating...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > 10:  # Just evaluate ~320 samples for speed
                break
                
            x, y, physics_params = batch['x'], batch['y'], batch['physics']
            
            # Forward pass
            y_hat, physics_preds = model(x, physics_params)
            
            # The model predicts normalized values, we need to denormalize
            # For HRR (which is usually channel 0)
            pred_hrr = y_hat[:, :, 0]
            true_hrr = y[:, :, 0]
            
            # Calculate absolute error
            mae = torch.abs(pred_hrr - true_hrr).mean().item()
            maes.append(mae)

    final_mae = np.mean(maes)
    print(f"\n========================================")
    print(f"REAL MODEL PERFORMANCE (Normalized MAE): {final_mae:.4f}")
    print(f"========================================")

if __name__ == "__main__":
    evaluate_model()
