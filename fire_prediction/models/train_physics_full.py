"""
Full 50-Epoch Training Script with TensorBoard Logging

This script implements comprehensive training for the physics-informed LSTM model:
- 50 epochs with early stopping
- TensorBoard logging for all physics metrics
- Model checkpointing (save best model)
- Comparative evaluation (baseline vs physics-informed)
- Ablation study support

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

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from fire_prediction.data.physics_dataset import PhysicsInformedDataset
from fire_prediction.models.physics_informed import PhysicsInformedLSTM

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data Configuration
DATA_DIR = r"D:\FDS\Small_project\ml_data"
INPUT_SEQ_LEN = 30  # Best from sequence optimization
PRED_HORIZON = 10
BATCH_SIZE = 32

# Model Configuration
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.001

# Physics Configuration
FIRE_DIAMETER = 0.3  # meters
LAMBDA_PHYSICS = 0.1  # Weight for physics consistency loss
LAMBDA_MONOTONIC = 0.05  # Weight for monotonicity loss

# Training Configuration
MAX_EPOCHS = 50
PATIENCE = 10  # Early stopping patience
NUM_WORKERS = 0  # Set to 0 for Windows compatibility

# Experiment Configuration
EXPERIMENT_NAME = "physics_informed_full"
USE_PHYSICS_LOSS = True
INCLUDE_HESKESTAD = True
VALIDATE_PHYSICS = True

# ============================================================================
# TENSORBOARD LOGGER SETUP
# ============================================================================

def create_tensorboard_logger(experiment_name: str, version: str = None):
    """Create TensorBoard logger with custom configuration"""
    logger = TensorBoardLogger(
        save_dir="logs",
        name=experiment_name,
        version=version,
        log_graph=True,  # Log model graph
        default_hp_metric=False
    )
    
    print(f"\n📊 TensorBoard Logging:")
    print(f"   Save dir: logs/{experiment_name}")
    print(f"   Version: {logger.version}")
    print(f"   To view: tensorboard --logdir=logs/{experiment_name}")
    
    return logger

# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets(include_heskestad: bool = True):
    """Load train, val, test datasets with optional Heskestad features"""
    
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    # Load train dataset
    print(f"\n📂 Loading train dataset...")
    print(f"   Heskestad features: {include_heskestad}")
    
    train_ds = PhysicsInformedDataset(
        DATA_DIR,
        split="train",
        input_seq_len=INPUT_SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        include_heskestad=include_heskestad,
        fire_diameter=FIRE_DIAMETER
    )
    
    input_dim = 6 if include_heskestad else 3
    print(f"✅ Train: {len(train_ds)} samples, {input_dim} channels")
    
    # Load val dataset (use train stats)
    print(f"\n📂 Loading val dataset...")
    val_ds = PhysicsInformedDataset(
        DATA_DIR,
        split="val",
        input_seq_len=INPUT_SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        include_heskestad=include_heskestad,
        fire_diameter=FIRE_DIAMETER,
        train_stats=train_ds.stats if include_heskestad else None
    )
    
    print(f"✅ Val: {len(val_ds)} samples")
    
    # Load test dataset (use train stats)
    print(f"\n📂 Loading test dataset...")
    test_ds = PhysicsInformedDataset(
        DATA_DIR,
        split="test",
        input_seq_len=INPUT_SEQ_LEN,
        pred_horizon=PRED_HORIZON,
        include_heskestad=include_heskestad,
        fire_diameter=FIRE_DIAMETER,
        train_stats=train_ds.stats if include_heskestad else None
    )
    
    print(f"✅ Test: {len(test_ds)} samples")
    
    return train_ds, val_ds, test_ds, input_dim

# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(
    input_dim: int,
    use_physics_loss: bool = True,
    validate_physics: bool = True
):
    """Create physics-informed LSTM model"""
    
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n🧠 Model Configuration:")
    print(f"   Input dim: {input_dim} channels")
    print(f"   Hidden dim: {HIDDEN_DIM}")
    print(f"   Num layers: {NUM_LAYERS}")
    print(f"   Total parameters: {total_params:,}")
    print(f"\n⚙️ Physics Configuration:")
    print(f"   Physics loss: {use_physics_loss}")
    print(f"   Physics validation: {validate_physics}")
    print(f"   Lambda physics: {LAMBDA_PHYSICS}")
    print(f"   Lambda monotonic: {LAMBDA_MONOTONIC}")
    
    return model

# ============================================================================
# CALLBACKS
# ============================================================================

def create_callbacks(experiment_name: str):
    """Create training callbacks (checkpointing, early stopping)"""
    
    # Model checkpoint - save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{experiment_name}",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    # Early stopping - stop if validation loss doesn't improve
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=True
    )
    
    print(f"\n📌 Callbacks:")
    print(f"   Checkpoint dir: checkpoints/{experiment_name}")
    print(f"   Early stopping patience: {PATIENCE} epochs")
    
    return [checkpoint_callback, early_stop_callback]

# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model,
    train_ds,
    val_ds,
    experiment_name: str,
    version: str = None
):
    """Train model with TensorBoard logging"""
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False
    )
    
    # Create logger and callbacks
    logger = create_tensorboard_logger(experiment_name, version)
    callbacks = create_callbacks(experiment_name)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True
    )
    
    print(f"\n🚀 Starting training...")
    print(f"   Max epochs: {MAX_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {callbacks[0].best_model_path}")
    print(f"   Best val loss: {callbacks[0].best_model_score:.6f}")
    
    return trainer, callbacks[0].best_model_path

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_ds, experiment_name: str):
    """Evaluate model on test set"""
    
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=False
    )
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=True
    )
    
    print(f"\n🧪 Testing on {len(test_ds)} samples...")
    
    results = trainer.test(model, test_loader)
    
    print(f"\n📊 Test Results:")
    print(f"   Test Loss: {results[0]['test_loss']:.6f}")
    print(f"   Test MAE: {results[0]['test_mae']:.4f} kW")
    print(f"   Test MSE: {results[0]['test_mse']:.6f}")
    
    if 'test_physics_error' in results[0]:
        print(f"   Physics Error: {results[0]['test_physics_error']:.4f}")
    
    if 'test_violation_rate' in results[0]:
        print(f"   Violation Rate: {results[0]['test_violation_rate']:.2%}")
    
    return results[0]

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("PHYSICS-INFORMED LSTM - FULL TRAINING")
    print("="*70)
    print(f"\nExperiment: {EXPERIMENT_NAME}")
    print(f"Configuration:")
    print(f"  - Input sequence: {INPUT_SEQ_LEN} timesteps")
    print(f"  - Prediction horizon: {PRED_HORIZON} timesteps")
    print(f"  - Heskestad features: {INCLUDE_HESKESTAD}")
    print(f"  - Physics loss: {USE_PHYSICS_LOSS}")
    print(f"  - Physics validation: {VALIDATE_PHYSICS}")
    print("="*70)
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Load datasets
    train_ds, val_ds, test_ds, input_dim = load_datasets(INCLUDE_HESKESTAD)
    
    # Create model
    model = create_model(input_dim, USE_PHYSICS_LOSS, VALIDATE_PHYSICS)
    
    # Train model
    trainer, best_model_path = train_model(
        model,
        train_ds,
        val_ds,
        EXPERIMENT_NAME,
        version="full_50epochs"
    )
    
    # Load best model for evaluation
    print(f"\n📥 Loading best model from: {best_model_path}")
    best_model = PhysicsInformedLSTM.load_from_checkpoint(best_model_path)
    
    # Evaluate on test set
    test_results = evaluate_model(best_model, test_ds, EXPERIMENT_NAME)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n✅ Best model saved: {best_model_path}")
    print(f"✅ Test MAE: {test_results['test_mae']:.4f} kW")
    print(f"✅ Baseline MAE: 4.85 kW")
    
    improvement = ((4.85 - test_results['test_mae']) / 4.85) * 100
    print(f"✅ Improvement: {improvement:.1f}%")
    
    print(f"\n📊 View TensorBoard logs:")
    print(f"   tensorboard --logdir=logs/{EXPERIMENT_NAME}")
    print("="*70 + "\n")
    
    return test_results


if __name__ == "__main__":
    results = main()
