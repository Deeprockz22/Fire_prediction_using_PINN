"""
Batch Prediction Script
Automatically processes all HRR files in the Input folder
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# Import required modules
from fire_prediction.models.physics_informed import PhysicsInformedLSTM
from fire_prediction.utils.physics import compute_heskestad_features

# Configuration
MODEL_PATH = "model/best_model.ckpt"
INPUT_DIR = "Input"
OUTPUT_DIR = "Output"
INPUT_SEQ_LEN = 30
PRED_HORIZON = 10
FIRE_DIAMETER = 0.3

# Normalization stats from training
STATS = {
    'mean': np.array([1.6312595e+02, -4.2468037e+01, 3.9271861e-03, 
                      1.2081864e+00, -1.6674624e-08, -1.2529218e-02], dtype=np.float32),
    'std': np.array([8.8223785e+01, 2.5670046e+01, 1.6978320e-03,
                     3.1914881e-01, 3.6280316e-01, 1.2922239e-01], dtype=np.float32)
}

def load_model():
    """Load the pre-trained model"""
    model = PhysicsInformedLSTM(
        input_dim=6,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        lr=0.001,
        pred_horizon=PRED_HORIZON,
        use_physics_loss=True,
        lambda_physics=0.1,
        lambda_monotonic=0.05,
        fire_diameter=FIRE_DIAMETER,
        validate_physics=True
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def prepare_features(hrr_data):
    """Prepare 6-channel input features from HRR data"""
    hesk_feats = compute_heskestad_features(hrr_data, fire_diameter=FIRE_DIAMETER)
    
    full_data = np.zeros((len(hrr_data), 6), dtype=np.float32)
    full_data[:, 0] = hrr_data
    full_data[:, 3:] = hesk_feats
    
    full_data_norm = (full_data - STATS['mean']) / (STATS['std'] + 1e-8)
    return full_data_norm

def predict_single_file(model, hrr_file, output_dir):
    """Make prediction for a single HRR file"""
    try:
        # Read CSV
        df = pd.read_csv(hrr_file, skiprows=1)
        time = df.iloc[:, 0].values
        hrr = df.iloc[:, 1].values
        
        # Prepare features
        full_data_norm = prepare_features(hrr)
        
        # Find peak for prediction point
        peak_idx = np.argmax(hrr)
        start_idx = max(0, peak_idx - INPUT_SEQ_LEN - 5)
        
        if start_idx + INPUT_SEQ_LEN + PRED_HORIZON > len(full_data_norm):
            start_idx = len(full_data_norm) - INPUT_SEQ_LEN - PRED_HORIZON
        
        if start_idx < 0:
            print(f"  ⚠️  Skipped (not enough data): {hrr_file.name}")
            return None
        
        # Extract sequences
        input_seq = full_data_norm[start_idx : start_idx + INPUT_SEQ_LEN]
        actual_future = full_data_norm[start_idx + INPUT_SEQ_LEN : start_idx + INPUT_SEQ_LEN + PRED_HORIZON, 0]
        input_time = time[start_idx : start_idx + INPUT_SEQ_LEN]
        future_time = time[start_idx + INPUT_SEQ_LEN : start_idx + INPUT_SEQ_LEN + PRED_HORIZON]
        
        # Prepare tensor and predict
        x_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            y_pred_norm = model(x_tensor)
            y_pred_norm = y_pred_norm[0].numpy()
        
        # Denormalize
        hrr_mean = STATS['mean'][0]
        hrr_std = STATS['std'][0]
        
        y_pred_kw = (y_pred_norm * hrr_std) + hrr_mean
        actual_future_kw = (actual_future * hrr_std) + hrr_mean
        input_seq_kw = (input_seq[:, 0] * hrr_std) + hrr_mean
        
        # Calculate metrics
        mae = np.mean(np.abs(actual_future_kw - y_pred_kw))
        rel_error = (mae / hrr.max()) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        context_start = max(0, start_idx - 50)
        context_hrr = (full_data_norm[context_start:start_idx+INPUT_SEQ_LEN, 0] * hrr_std) + hrr_mean
        context_time = time[context_start:start_idx+INPUT_SEQ_LEN]
        
        ax.plot(context_time, context_hrr, 'b-', linewidth=1.5, alpha=0.6, label='Past HRR')
        ax.plot(input_time, input_seq_kw, 'b-', linewidth=3, label='Input Sequence')
        ax.plot(future_time, actual_future_kw, 'g-', linewidth=3, marker='o', markersize=6, label='Actual Future')
        ax.plot(future_time, y_pred_kw, 'r--', linewidth=3, marker='s', markersize=6, label='Predicted Future')
        
        ax.set_title(f'{hrr_file.stem}\\nMAE: {mae:.2f} kW ({rel_error:.2f}% error)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('HRR (kW)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=input_time[-1], color='gray', linestyle=':', linewidth=2)
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"{hrr_file.stem}_prediction.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return {
            'file': hrr_file.name,
            'mae': mae,
            'rel_error': rel_error,
            'peak_hrr': hrr.max(),
            'prediction_time': input_time[-1],
            'output_plot': output_path.name
        }
        
    except Exception as e:
        print(f"  ❌ Error processing {hrr_file.name}: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("BATCH FIRE PREDICTION")
    print("Processing all files in Input folder")
    print("="*70)
    
    # Check directories
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    if not input_path.exists():
        print(f"\n❌ Error: Input folder not found!")
        print(f"   Please create: {input_path.absolute()}")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    print(f"\n📂 Input folder: {input_path.absolute()}")
    print(f"📂 Output folder: {output_path.absolute()}")
    
    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in Input folder!")
        print(f"   Please add HRR CSV files to: {input_path.absolute()}")
        sys.exit(1)
    
    print(f"\n✅ Found {len(csv_files)} CSV file(s)")
    
    # Load model
    print("\n🔧 Loading model...")
    model = load_model()
    print("✅ Model loaded")
    
    # Process each file
    print(f"\n{'='*70}")
    print("PROCESSING FILES")
    print("="*70)
    
    results = []
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file.name}")
        result = predict_single_file(model, csv_file, output_path)
        if result:
            results.append(result)
            print(f"  ✅ MAE: {result['mae']:.2f} kW ({result['rel_error']:.2f}%)")
            print(f"  📊 Plot: {result['output_plot']}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(csv_files)}")
    print(f"Successful predictions: {len(results)}")
    print(f"Failed/Skipped: {len(csv_files) - len(results)}")
    
    if results:
        print(f"\nAverage MAE: {np.mean([r['mae'] for r in results]):.2f} kW")
        print(f"Average Relative Error: {np.mean([r['rel_error'] for r in results]):.2f}%")
        
        # Save summary report
        summary_file = output_path / f"prediction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FIRE PREDICTION BATCH PROCESSING SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files: {len(csv_files)}\n")
            f.write(f"Successful: {len(results)}\n\n")
            f.write("="*70 + "\n")
            f.write("INDIVIDUAL RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for result in results:
                f.write(f"File: {result['file']}\n")
                f.write(f"  MAE: {result['mae']:.4f} kW\n")
                f.write(f"  Relative Error: {result['rel_error']:.2f}%\n")
                f.write(f"  Peak HRR: {result['peak_hrr']:.2f} kW\n")
                f.write(f"  Plot: {result['output_plot']}\n\n")
            
            f.write("="*70 + "\n")
            f.write(f"Average MAE: {np.mean([r['mae'] for r in results]):.2f} kW\n")
            f.write(f"Average Relative Error: {np.mean([r['rel_error'] for r in results]):.2f}%\n")
        
        print(f"\n📝 Summary saved to: {summary_file.name}")
    
    print("\n✅ Batch processing complete!")
    print(f"   Check the {OUTPUT_DIR}/ folder for results")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
