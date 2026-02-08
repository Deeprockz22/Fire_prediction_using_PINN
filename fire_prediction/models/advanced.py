import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class HybridLSTM(pl.LightningModule):
    """
    Advanced Hybrid LSTM Model for Fire HRR Prediction.
    Integrates static scenario features (Fuel, Room, etc.) with time-series data.
    
    Architecture:
    1. Time Series -> LSTM -> Temporal Embedding
    2. Static Features -> MLP -> Static Embedding
    3. Concat(Temporal, Static) -> MLP Head -> Prediction
    """
    def __init__(
        self, 
        input_dim: int = 3, 
        static_dim: int = 12,
        hidden_dim: int = 64, 
        static_hidden_dim: int = 16,
        num_layers: int = 2, 
        output_dim: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        pred_horizon: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Temporal Encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. Static Feature Encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Fusion Head
        # Concatenate LSTM hidden state + Encoded static features
        fusion_dim = hidden_dim + static_hidden_dim
        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, output_dim * pred_horizon)
        )
        
    def forward(self, x_seq, x_static):
        # x_seq: [batch, seq_len, input_dim]
        # x_static: [batch, static_dim]
        
        # 1. Process time series
        # lstm_out: [batch, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(x_seq)
        
        # Take the output from the last time step
        temporal_embed = lstm_out[:, -1, :] # [batch, hidden_dim]
        
        # 2. Process static features
        static_embed = self.static_encoder(x_static) # [batch, static_hidden]
        
        # 3. Fusion
        combined = torch.cat([temporal_embed, static_embed], dim=1)
        
        # 4. Prediction
        y_hat = self.head(combined)
        
        # Reshape to [batch, pred_horizon, output_dim]
        y_hat = y_hat.view(-1, self.hparams.pred_horizon, self.hparams.output_dim)
        return y_hat

    def training_step(self, batch, batch_idx):
        (x_seq, x_static), y = batch
        
        y_hat = self(x_seq, x_static)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x_seq, x_static), y = batch
        y_hat = self(x_seq, x_static)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        (x_seq, x_static), y = batch
        y_hat = self(x_seq, x_static)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_mae", mae)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


class HybridLSTM_v2(pl.LightningModule):
    """
    ═══════════════════════════════════════════════════════════════════════
    HYBRID LSTM v2 - Physics-Informed Fire Prediction
    ═══════════════════════════════════════════════════════════════════════
    
    PURPOSE:
    This is the MOST ADVANCED model in our ensemble. It combines:
    1. Time-series data (HRR, Q_RADI, MLR over time)
    2. Static scenario features (Fuel type, Room size, Opening factor, etc.)
    
    KEY INNOVATION:
    Instead of just concatenating static features, we use them to INITIALIZE
    the LSTM's memory state. This is like giving the LSTM "prior knowledge"
    about the fire scenario before it starts processing the time series.
    
    WHY THIS WORKS BETTER:
    - A fire in a small room with propane behaves differently than
      a large room with diesel
    - By initializing the LSTM with this context, it can make better
      predictions from the very first timestep
    
    EXAMPLE:
    Static features tell the model: "This is a SMALL room with PROPANE fuel"
    → LSTM starts with memory biased toward "expect rapid growth"
    → As it reads the time series, it refines this initial guess
    """
    
    def __init__(
        self, 
        input_dim: int = 3,        # 3 physics channels (HRR, Q_RADI, MLR)
        static_dim: int = 12,      # 12 static features (see feature_extractor.py)
        hidden_dim: int = 64,      # LSTM internal memory size
        num_layers: int = 2,       # Number of LSTM layers
        output_dim: int = 1,       # Predict HRR only
        dropout: float = 0.1,      # Dropout for regularization
        lr: float = 1e-3,          # Learning rate
        pred_horizon: int = 10     # Predict 10 steps ahead
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # ═══════════════════════════════════════════════════════════════
        # STATIC FEATURE PROJECTORS
        # ═══════════════════════════════════════════════════════════════
        # These layers convert the 12 static features into LSTM initial states
        # 
        # LSTM has TWO internal states:
        # - h_0 (hidden state): Short-term memory
        # - c_0 (cell state): Long-term memory
        #
        # We create separate projections for each, for each layer
        self.static_proj_h = nn.Linear(static_dim, hidden_dim * num_layers)
        self.static_proj_c = nn.Linear(static_dim, hidden_dim * num_layers)
        # Input: [batch, 12] → Output: [batch, 128] (64 × 2 layers)
        
        # ═══════════════════════════════════════════════════════════════
        # LSTM LAYER
        # ═══════════════════════════════════════════════════════════════
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # ═══════════════════════════════════════════════════════════════
        # OUTPUT HEAD - Two-layer MLP for final prediction
        # ═══════════════════════════════════════════════════════════════
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 64 → 64
            nn.ReLU(),                          # Non-linearity
            nn.Linear(hidden_dim, output_dim * pred_horizon)  # 64 → 10
        )
        
    def forward(self, x_seq, x_static):
        """
        FORWARD PASS - How predictions are made
        
        INPUTS:
        x_seq: [batch, 50, 3] - Time series (past 5 seconds)
        x_static: [batch, 12] - Static features (fuel, room, etc.)
        
        OUTPUT:
        y_hat: [batch, 10, 1] - Predicted HRR for next 1 second
        """
        batch_size = x_seq.size(0)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Convert static features to LSTM initial states
        # ═══════════════════════════════════════════════════════════════
        # This is the KEY INNOVATION of this model!
        #
        # Static features (12 numbers) describe the scenario:
        # [1,0,0,0,0,0, 0,1,0, 0.5, 0.0, 0.36]
        #  └─ Fuel ─┘  └Room┘ Open Wind Size
        #
        # We project these to initialize the LSTM's memory:
        h_0 = self.static_proj_h(x_static)  # [batch, 128]
        c_0 = self.static_proj_c(x_static)  # [batch, 128]
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Reshape for LSTM format
        # ═══════════════════════════════════════════════════════════════
        # LSTM expects initial states as: [num_layers, batch, hidden_dim]
        # We have: [batch, hidden_dim * num_layers]
        # Need to reshape and permute:
        h_0 = h_0.view(batch_size, self.hparams.num_layers, self.hparams.hidden_dim)
        h_0 = h_0.permute(1, 0, 2).contiguous()  # [2, batch, 64]
        
        c_0 = c_0.view(batch_size, self.hparams.num_layers, self.hparams.hidden_dim)
        c_0 = c_0.permute(1, 0, 2).contiguous()  # [2, batch, 64]
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Run LSTM with initialized state
        # ═══════════════════════════════════════════════════════════════
        # Now the LSTM starts with "knowledge" about the scenario!
        # As it reads the time series, it refines its predictions
        # based on both the initial context AND the observed dynamics.
        lstm_out, _ = self.lstm(x_seq, (h_0, c_0))
        # lstm_out: [batch, 50, 64]
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Extract final state and predict
        # ═══════════════════════════════════════════════════════════════
        last_out = lstm_out[:, -1, :]  # [batch, 64]
        y_hat = self.head(last_out)     # [batch, 10]
        
        # Reshape to [batch, 10, 1]
        return y_hat.view(-1, self.hparams.pred_horizon, self.hparams.output_dim)

    def training_step(self, batch, batch_idx):
        """TRAINING - Learn from 68 fire scenarios"""
        (x_seq, x_static), y = batch
        y_hat = self(x_seq, x_static)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """VALIDATION - Check performance on 11 unseen scenarios"""
        (x_seq, x_static), y = batch
        y_hat = self(x_seq, x_static)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """TEST - Final evaluation on 14 completely new scenarios"""
        (x_seq, x_static), y = batch
        y_hat = self(x_seq, x_static)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_mae", mae)
        return loss

    def configure_optimizers(self):
        """OPTIMIZER - Adam with default settings"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
