"""
Fire Prediction Tool - Production Version
Predicts future Heat Release Rate (HRR) from FDS simulation data

Usage:
    python predict.py path/to/scenario_hrr.csv

Example:
    python predict.py examples/sample_scenario_hrr.csv
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# Get script directory for portable paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# Import required modules
from fire_prediction.models.physics_informed import PhysicsInformedLSTM
from fire_prediction.utils.physics import compute_heskestad_features

# Configuration - use absolute paths relative to script location
MODEL_PATH = SCRIPT_DIR / "model" / "best_model.ckpt"
INPUT_SEQ_LEN = 30
PRED_HORIZON = 10
FIRE_DIAMETER = 0.3  # meters

# Normalization stats from training (DO NOT MODIFY)
STATS = {
    'mean': np.array([1.6312595e+02, -4.2468037e+01, 3.9271861e-03, 
                      1.2081864e+00, -1.6674624e-08, -1.2529218e-02], dtype=np.float32),
    'std': np.array([8.8223785e+01, 2.5670046e+01, 1.6978320e-03,
                     3.1914881e-01, 3.6280316e-01, 1.2922239e-01], dtype=np.float32)
}

def load_model():
    """Load the pre-trained model"""
    print("\n🔧 Loading model...")
    model = PhysicsInformedLSTM(
        input_dim=6,
        hidden_dim=128,
        num_layers=2,
        output_dim=3,  # HRR, Q_RADI, MLR
        dropout=0.1,
        lr=0.001,
        pred_horizon=PRED_HORIZON,
        use_physics_loss=True,
        lambda_physics=0.1,
        lambda_monotonic=0.05,
        fire_diameter=FIRE_DIAMETER,
        validate_physics=True
    )
    
    checkpoint = torch.load(str(MODEL_PATH), map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Handle legacy checkpoint with 'fc' instead of 'head'
    if 'fc.weight' in state_dict and 'head.weight' not in state_dict:
        state_dict['head.weight'] = state_dict.pop('fc.weight')
        state_dict['head.bias'] = state_dict.pop('fc.bias')
    
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully")
    return model

def prepare_features(hrr_data):
    """
    Prepare 6-channel input features from HRR data
    
    Channels:
    0: HRR (normalized)
    1: Q_RADI (zeros - not available)
    2: MLR (zeros - not available)
    3-5: Heskestad physics features (flame height, rate, deviation)
    """
    print("\n🔬 Computing physics features...")
    
    # Compute Heskestad features
    hesk_feats = compute_heskestad_features(hrr_data, fire_diameter=FIRE_DIAMETER)
    
    # Construct full feature matrix [Samples, 6]
    full_data = np.zeros((len(hrr_data), 6), dtype=np.float32)
    full_data[:, 0] = hrr_data
    full_data[:, 3:] = hesk_feats
    
    # Normalize using training stats
    full_data_norm = (full_data - STATS['mean']) / (STATS['std'] + 1e-8)
    
    return full_data_norm

def predict(model, hrr_file, output_plot=None):
    """
    Make prediction from HRR CSV file
    
    Args:
        model: Loaded PyTorch model
        hrr_file: Path to _hrr.csv file from FDS
        output_plot: Optional path to save plot
    """
    print(f"\n📊 Reading data from: {hrr_file}")
    
    # Read CSV
    try:
        df = pd.read_csv(hrr_file, skiprows=1)
        time = df.iloc[:, 0].values
        hrr = df.iloc[:, 1].values
        print(f"✅ Loaded {len(hrr)} time steps")
        print(f"   Time range: {time[0]:.2f}s - {time[-1]:.2f}s")
        print(f"   Peak HRR: {hrr.max():.2f} kW")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    # Prepare features
    full_data_norm = prepare_features(hrr)
    
    # Find peak for prediction point
    peak_idx = np.argmax(hrr)
    start_idx = max(0, peak_idx - INPUT_SEQ_LEN - 5)
    
    # Ensure enough data
    if start_idx + INPUT_SEQ_LEN + PRED_HORIZON > len(full_data_norm):
        start_idx = len(full_data_norm) - INPUT_SEQ_LEN - PRED_HORIZON
    
    if start_idx < 0:
        print(f"❌ Error: Not enough data points. Need at least {INPUT_SEQ_LEN + PRED_HORIZON} points.")
        return None
    
    # Extract sequences
    input_seq = full_data_norm[start_idx : start_idx + INPUT_SEQ_LEN]
    actual_future = full_data_norm[start_idx + INPUT_SEQ_LEN : start_idx + INPUT_SEQ_LEN + PRED_HORIZON, 0]
    input_time = time[start_idx : start_idx + INPUT_SEQ_LEN]
    future_time = time[start_idx + INPUT_SEQ_LEN : start_idx + INPUT_SEQ_LEN + PRED_HORIZON]
    
    # Prepare tensor
    x_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
    
    # Predict
    print(f"\n🔮 Making prediction at t={input_time[-1]:.2f}s...")
    with torch.no_grad():
        y_pred_norm = model(x_tensor)
        y_pred_norm = y_pred_norm[0].numpy()
        # Model outputs 3 channels [HRR, Q_RADI, MLR], extract HRR only
        if y_pred_norm.shape[-1] == 3:
            y_pred_norm = y_pred_norm[:, 0]  # Shape: [10, 3] -> [10]
        elif len(y_pred_norm.shape) == 2 and y_pred_norm.shape[0] == PRED_HORIZON:
            y_pred_norm = y_pred_norm[:, 0]
    
    # Denormalize
    hrr_mean = STATS['mean'][0]
    hrr_std = STATS['std'][0]
    
    y_pred_kw = (y_pred_norm * hrr_std) + hrr_mean
    actual_future_kw = (actual_future * hrr_std) + hrr_mean
    input_seq_kw = (input_seq[:, 0] * hrr_std) + hrr_mean
    
    # Calculate metrics
    mae = np.mean(np.abs(actual_future_kw - y_pred_kw))
    rel_error = (mae / hrr.max()) * 100
    
    # Print results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"MAE: {mae:.4f} kW")
    print(f"Relative Error: {rel_error:.2f}%")
    print(f"Peak HRR: {hrr.max():.2f} kW")
    print("="*70)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Show longer context
    context_start = max(0, start_idx - 50)
    context_hrr = (full_data_norm[context_start:start_idx+INPUT_SEQ_LEN, 0] * hrr_std) + hrr_mean
    context_time = time[context_start:start_idx+INPUT_SEQ_LEN]
    
    ax.plot(context_time, context_hrr, 'b-', label='Past HRR', linewidth=1.5, alpha=0.6)
    ax.plot(input_time, input_seq_kw, 'b-', label='Input Sequence', linewidth=3)
    ax.plot(future_time, actual_future_kw, 'g-', label='Actual Future', linewidth=3, marker='o', markersize=6)
    ax.plot(future_time, y_pred_kw, 'r--', label='Predicted Future', linewidth=3, marker='s', markersize=6)
    
    ax.set_title(f'Fire Prediction Results\\nMAE: {mae:.4f} kW ({rel_error:.2f}% error)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('HRR (kW)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=input_time[-1], color='gray', linestyle=':', linewidth=2, label='Prediction Point')
    
    plt.tight_layout()
    
    # Save plot
    if output_plot is None:
        output_plot = Path(hrr_file).stem + "_prediction.png"
    
    plt.savefig(output_plot, dpi=150)
    print(f"\n✅ Plot saved to: {output_plot}")
    
    plt.show()
    
    return {
        'mae': mae,
        'relative_error': rel_error,
        'peak_hrr': hrr.max(),
        'prediction_time': input_time[-1],
        'predicted_values': y_pred_kw,
        'actual_values': actual_future_kw
    }

def main():
    parser = argparse.ArgumentParser(description='Fire HRR Prediction Tool')
    parser.add_argument('hrr_file', type=str, help='Path to FDS _hrr.csv file')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output plot filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FIRE PREDICTION TOOL")
    print("Physics-Informed LSTM for HRR Forecasting")
    print("="*70)
    
    # Check file exists
    if not Path(args.hrr_file).exists():
        print(f"❌ Error: File not found: {args.hrr_file}")
        sys.exit(1)
    
    # Load model
    model = load_model()
    
    # Make prediction
    results = predict(model, args.hrr_file, args.output)
    
    if results:
        print("\n✅ Prediction complete!")
    else:
        print("\n❌ Prediction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
