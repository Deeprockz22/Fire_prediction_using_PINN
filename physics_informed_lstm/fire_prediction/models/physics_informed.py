"""
Physics-Informed Baseline LSTM with Heskestad Correlation

This model implements THREE approaches to incorporate Heskestad's fire physics:
1. Physics-informed loss function
2. Heskestad-derived features as additional inputs
3. Physics validation layer

Author: Fire Prediction Team
Date: 2026-02-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.physics import (
    physics_consistency_loss,
    monotonicity_loss,
    heskestad_flame_height,
    validate_physics_consistency
)


class PhysicsInformedLSTM(pl.LightningModule):
    """
    LSTM with Heskestad Physics Integration
    
    This model extends the baseline LSTM with three physics-informed components:
    
    **Approach 1: Physics-Informed Loss**
        Loss = MSE + λ₁*physics_penalty + λ₂*monotonicity_penalty
        
    **Approach 2: Heskestad Features**
        Input channels expanded from 3 to 6:
        - Original: [HRR, Q_RADI, MLR]
        - Added: [Flame_Height, dFlame_Height/dt, Flame_Height_Deviation]
        
    **Approach 3: Physics Validation**
        Post-prediction check for physical consistency
    """
    
    def __init__(
        self,
        input_dim: int = 6,              # 3 original + 3 Heskestad features
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        pred_horizon: int = 10,
        # Physics parameters
        use_physics_loss: bool = True,    # Approach 1
        lambda_physics: float = 0.1,      # Weight for physics penalty
        lambda_monotonic: float = 0.05,   # Weight for monotonicity penalty
        fire_diameter: float = 0.3,       # Fire diameter for Heskestad (m)
        validate_physics: bool = True     # Approach 3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # ═══════════════════════════════════════════════════════════
        # LSTM CORE (same as baseline)
        # ═══════════════════════════════════════════════════════════
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output head
        self.head = nn.Linear(hidden_dim, output_dim * pred_horizon)
        
        # ═══════════════════════════════════════════════════════════
        # PHYSICS VALIDATION METRICS (Approach 3)
        # ═══════════════════════════════════════════════════════════
        self.physics_violations = 0
        self.total_predictions = 0
        
    def forward(self, x):
        """
        Forward pass with optional physics features.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
               If input_dim=6: includes Heskestad features (Approach 2)
               If input_dim=3: original features only
        
        Returns:
            predictions: [batch, pred_horizon, 1]
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep
        last_hidden = lstm_out[:, -1, :]
        
        # Generate predictions
        out = self.head(last_hidden)
        
        # Reshape to [batch, pred_horizon, 1]
        batch_size = x.size(0)
        predictions = out.view(batch_size, self.hparams.pred_horizon, self.hparams.output_dim)
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        """
        Training with Physics-Informed Loss (Approach 1)
        """
        (x, _), y = batch
        y_hat = self(x)
        
        if self.hparams.use_physics_loss:
            # ═══════════════════════════════════════════════════════
            # APPROACH 1: Physics-Informed Loss
            # ═══════════════════════════════════════════════════════
            
            # Standard MSE
            mse_loss = F.mse_loss(y_hat, y)
            
            # Physics consistency (Heskestad correlation)
            physics_loss = physics_consistency_loss(
                y_hat, y,
                fire_diameter=self.hparams.fire_diameter,
                lambda_physics=self.hparams.lambda_physics
            )
            
            # Monotonicity (smooth, physical changes)
            mono_loss = monotonicity_loss(
                y_hat,
                lambda_monotonic=self.hparams.lambda_monotonic
            )
            
            # Combined loss
            total_loss = mse_loss + physics_loss + mono_loss
            
            # Log components
            self.log("train_loss", total_loss, prog_bar=True)
            self.log("train_mse", mse_loss)
            self.log("train_physics", physics_loss)
            self.log("train_monotonic", mono_loss)
            
            return total_loss
        else:
            # Standard MSE only
            loss = F.mse_loss(y_hat, y)
            self.log("train_loss", loss, prog_bar=True)
            return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation with physics tracking"""
        (x, _), y = batch
        y_hat = self(x)
        
        # Standard loss
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        
        # ═══════════════════════════════════════════════════════════
        # APPROACH 3: Physics Validation
        # ═══════════════════════════════════════════════════════════
        if self.hparams.validate_physics:
            # Check physics consistency for first sample in batch
            pred_hrr = y_hat[0, :, 0].detach().cpu().numpy()
            true_hrr = y[0, :, 0].detach().cpu().numpy()
            
            is_valid, metrics = validate_physics_consistency(
                pred_hrr, true_hrr,
                fire_diameter=self.hparams.fire_diameter
            )
            
            if not is_valid:
                self.physics_violations += 1
            self.total_predictions += 1
            
            # Log physics metrics
            self.log("val_physics_error", metrics['mean_flame_height_error'])
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test with comprehensive physics analysis"""
        (x, _), y = batch
        y_hat = self(x)
        
        # Standard metrics
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        self.log("test_loss", mse)
        self.log("test_mae", mae)
        
        # ═══════════════════════════════════════════════════════════
        # APPROACH 3: Detailed Physics Validation
        # ═══════════════════════════════════════════════════════════
        if self.hparams.validate_physics:
            # Analyze all samples in batch
            batch_size = y_hat.size(0)
            physics_errors = []
            
            for i in range(batch_size):
                pred_hrr = y_hat[i, :, 0].detach().cpu().numpy()
                true_hrr = y[i, :, 0].detach().cpu().numpy()
                
                is_valid, metrics = validate_physics_consistency(
                    pred_hrr, true_hrr,
                    fire_diameter=self.hparams.fire_diameter
                )
                
                physics_errors.append(metrics['mean_flame_height_error'])
            
            # Log average physics error
            avg_physics_error = sum(physics_errors) / len(physics_errors)
            self.log("test_physics_error", avg_physics_error)
        
        return {"test_loss": mse, "test_mae": mae}
    
    def configure_optimizers(self):
        """Adam optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def on_validation_epoch_end(self):
        """Report physics violation rate"""
        if self.hparams.validate_physics and self.total_predictions > 0:
            violation_rate = self.physics_violations / self.total_predictions
            self.log("physics_violation_rate", violation_rate)
            
            # Reset counters
            self.physics_violations = 0
            self.total_predictions = 0


if __name__ == "__main__":
    # Quick test
    print("Physics-Informed LSTM Model")
    print("="*60)
    
    model = PhysicsInformedLSTM(
        input_dim=6,  # With Heskestad features
        use_physics_loss=True,
        validate_physics=True
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Physics-informed loss: {model.hparams.use_physics_loss}")
    print(f"Physics validation: {model.hparams.validate_physics}")
    print(f"Input dimension: {model.hparams.input_dim} (includes Heskestad features)")
