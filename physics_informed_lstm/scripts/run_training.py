"""
Training Launcher - Run from physics_informed_lstm/ root directory.
This script ensures correct sys.path before importing the training module.

Usage:
    python run_training.py

Output:
    - Best checkpoint saved to checkpoints/physics_informed_full/
    - Training log saved to training_run_<timestamp>.log
    - Final MAE printed for before/after comparison
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Root = physics_informed_lstm/
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# Patch DATA_DIR to be relative-safe
import fire_prediction.models.train_physics_full as train_module
train_module.DATA_DIR = str(ROOT.parent.parent.parent / "Small_project" / "ml_data")

# Also override checkpoint/log dirs to be relative to ROOT
train_module.EXPERIMENT_NAME = "physics_informed_full_v2"

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ROOT / f"training_run_{timestamp}.log"

    print(f"\n{'='*70}")
    print(f"  FIRE PREDICTION - TRAINING WITH QUICK WINS")
    print(f"{'='*70}")
    print(f"  Start time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data dir   : {train_module.DATA_DIR}")
    print(f"  Log file   : {log_file.name}")
    print(f"  Epochs     : {train_module.MAX_EPOCHS}")
    print(f"  Hidden dim : {train_module.HIDDEN_DIM}")
    print(f"  Batch size : {train_module.BATCH_SIZE}")
    print(f"{'='*70}\n")

    # Change working directory so relative paths in train script resolve correctly
    os.chdir(str(ROOT))

    results = train_module.main()

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*70}")
    print(f"  DONE — {end_time}")
    if results:
        mae = results.get('test_mae', results.get('test_loss', float('nan')))
        baseline = 4.85
        improvement = ((baseline - mae) / baseline) * 100
        print(f"  Test MAE   : {mae:.4f} kW")
        print(f"  Baseline   : {baseline:.2f} kW")
        print(f"  Improvement: {improvement:+.1f}%")
    print(f"{'='*70}\n")
