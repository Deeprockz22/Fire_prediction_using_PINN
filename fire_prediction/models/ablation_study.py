"""
Ablation Study Script - Test Each Physics Component

This script runs 4 training configurations to measure the contribution of each component:
1. Baseline (3 channels, no physics loss)
2. Physics loss only (3 channels + physics loss)
3. Features only (6 channels, no physics loss)
4. Full approach (6 channels + physics loss)

This provides scientific evidence of which components contribute most to accuracy.

Created by: Antigravity AI Assistant
Suggested by: GitHub Copilot CLI
Date: 2026-02-06
"""

import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from fire_prediction.data.physics_dataset import PhysicsInformedDataset
from fire_prediction.models.physics_informed import PhysicsInformedLSTM

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = r"D:\FDS\Small_project\ml_data"
INPUT_SEQ_LEN = 30
PRED_HORIZON = 10
BATCH_SIZE = 32
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.001
FIRE_DIAMETER = 0.3
LAMBDA_PHYSICS = 0.1
LAMBDA_MONOTONIC = 0.05
MAX_EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 0

# Ablation configurations
ABLATION_CONFIGS = [
    {
        "name": "baseline",
        "description": "Baseline (3 channels, no physics)",
        "include_heskestad": False,
        "use_physics_loss": False,
        "validate_physics": False
    },
    {
        "name": "physics_loss_only",
        "description": "Physics Loss Only (3 channels + physics loss)",
        "include_heskestad": False,
        "use_physics_loss": True,
        "validate_physics": True
    },
    {
        "name": "features_only",
        "description": "Features Only (6 channels, no physics loss)",
        "include_heskestad": True,
        "use_physics_loss": False,
        "validate_physics": False
    },
    {
        "name": "full_approach",
        "description": "Full Approach (6 channels + physics loss)",
        "include_heskestad": True,
        "use_physics_loss": True,
        "validate_physics": True
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_datasets_for_config(include_heskestad: bool):
    """Load datasets for specific configuration"""
    
    train_ds = PhysicsInformedDataset(
        DATA_DIR,
        split="train",
        input_seq_len=INPUT_SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        include_heskestad=include_heskestad,
        fire_diameter=FIRE_DIAMETER
    )
    
    val_ds = PhysicsInformedDataset(
        DATA_DIR,
        split="val",
        input_seq_len=INPUT_SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        include_heskestad=include_heskestad,
        fire_diameter=FIRE_DIAMETER,
        train_stats=train_ds.stats if include_heskestad else None
    )
    
    test_ds = PhysicsInformedDataset(
        DATA_DIR,
        split="test",
        input_seq_len=INPUT_SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        include_heskestad=include_heskestad,
        fire_diameter=FIRE_DIAMETER,
        train_stats=train_ds.stats if include_heskestad else None
    )
    
    input_dim = 6 if include_heskestad else 3
    
    return train_ds, val_ds, test_ds, input_dim


def create_model_for_config(input_dim: int, use_physics_loss: bool, validate_physics: bool):
    """Create model for specific configuration"""
    
    model = PhysicsInformedLSTM(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        pred_horizon=PRED_HORIZON,
        use_physics_loss=use_physics_loss,
        lambda_physics=LAMBDA_PHYSICS,
        lambda_monotonic=LAMBDA_MONOTONIC,
        fire_diameter=FIRE_DIAMETER,
        validate_physics=validate_physics
    )
    
    return model


def train_configuration(config: dict):
    """Train a single ablation configuration"""
    
    print("\n" + "="*70)
    print(f"TRAINING: {config['description']}")
    print("="*70)
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    train_ds, val_ds, test_ds, input_dim = load_datasets_for_config(config['include_heskestad'])
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"   Input channels: {input_dim}")
    
    # Create model
    print(f"\n🧠 Creating model...")
    model = create_model_for_config(
        input_dim,
        config['use_physics_loss'],
        config['validate_physics']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Physics loss: {config['use_physics_loss']}")
    print(f"   Physics validation: {config['validate_physics']}")
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Create logger and callbacks
    logger = TensorBoardLogger(
        save_dir="logs",
        name="ablation_study",
        version=config['name']
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/ablation_{config['name']}",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True
    )
    
    # Train
    print(f"\n🚀 Training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model
    best_model = PhysicsInformedLSTM.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    # Evaluate
    print(f"\n🧪 Evaluating...")
    test_results = trainer.test(best_model, test_loader)
    
    # Extract results
    results = {
        "configuration": config['description'],
        "name": config['name'],
        "input_channels": input_dim,
        "physics_loss": config['use_physics_loss'],
        "heskestad_features": config['include_heskestad'],
        "test_loss": test_results[0]['test_loss'],
        "test_mae": test_results[0]['test_mae'],
        "test_mse": test_results[0]['test_mse'],
        "best_val_loss": checkpoint_callback.best_model_score.item(),
        "best_epoch": checkpoint_callback.best_model_path.split("epoch=")[1].split("-")[0] if "epoch=" in checkpoint_callback.best_model_path else "N/A"
    }
    
    if 'test_physics_error' in test_results[0]:
        results['physics_error'] = test_results[0]['test_physics_error']
    
    if 'test_violation_rate' in test_results[0]:
        results['violation_rate'] = test_results[0]['test_violation_rate']
    
    print(f"\n📊 Results:")
    print(f"   Test MAE: {results['test_mae']:.4f} kW")
    print(f"   Test Loss: {results['test_loss']:.6f}")
    print(f"   Best epoch: {results['best_epoch']}")
    
    return results


# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================

def main():
    """Run complete ablation study"""
    
    print("\n" + "="*70)
    print("ABLATION STUDY - HESKESTAD PHYSICS COMPONENTS")
    print("="*70)
    print(f"\nTesting {len(ABLATION_CONFIGS)} configurations:")
    for i, config in enumerate(ABLATION_CONFIGS, 1):
        print(f"  {i}. {config['description']}")
    print("="*70)
    
    # Run all configurations
    all_results = []
    
    for i, config in enumerate(ABLATION_CONFIGS, 1):
        print(f"\n\n{'='*70}")
        print(f"CONFIGURATION {i}/{len(ABLATION_CONFIGS)}")
        print(f"{'='*70}")
        
        results = train_configuration(config)
        all_results.append(results)
        
        print(f"\n✅ Configuration {i} complete!")
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate improvements relative to baseline
    baseline_mae = df[df['name'] == 'baseline']['test_mae'].values[0]
    df['improvement_vs_baseline'] = ((baseline_mae - df['test_mae']) / baseline_mae * 100)
    
    # Save results
    results_file = "ablation_study_results.csv"
    df.to_csv(results_file, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    
    print(f"\n📊 Summary Table:")
    print("\n" + df.to_string(index=False))
    
    print(f"\n\n📈 Key Findings:")
    print(f"   Baseline MAE: {baseline_mae:.4f} kW")
    
    for _, row in df.iterrows():
        if row['name'] != 'baseline':
            print(f"   {row['configuration']}: {row['test_mae']:.4f} kW ({row['improvement_vs_baseline']:+.1f}%)")
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"\n📊 View TensorBoard:")
    print(f"   tensorboard --logdir=logs/ablation_study")
    print("="*70 + "\n")
    
    return df


if __name__ == "__main__":
    results_df = main()
