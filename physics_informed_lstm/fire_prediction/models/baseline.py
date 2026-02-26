import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class LSTMRegress(pl.LightningModule):
    """
    ═══════════════════════════════════════════════════════════════════════
    BASELINE LSTM MODEL - Fire HRR Prediction
    ═══════════════════════════════════════════════════════════════════════
    
    PURPOSE:
    This is the simplest model in our ensemble. It uses a Long Short-Term Memory
    (LSTM) network to predict future fire behavior based on past measurements.
    
    WHAT IT DOES:
    - Takes 5 seconds of past fire data (50 timesteps × 3 physics channels)
    - Predicts the next 1 second of HRR (10 timesteps)
    
    INPUT DATA (3 channels):
    1. HRR (Heat Release Rate) - How much energy the fire is releasing
    2. Q_RADI (Radiative Heat) - Heat transferred by radiation
    3. MLR (Mass Loss Rate) - How fast the fuel is burning
    
    ARCHITECTURE:
    Input → LSTM Layers → Fully Connected Layer → HRR Prediction
    """
    
    def __init__(
        self, 
        input_dim: int = 3,        # 3 physics channels (HRR, Q_RADI, MLR)
        hidden_dim: int = 128,      # Size of LSTM's internal memory
        num_layers: int = 2,       # Number of stacked LSTM layers 
        output_dim: int = 1,       # We predict only HRR (1 channel)
        dropout: float = 0.1,      # Dropout rate to prevent overfitting
        lr: float = 1e-3,          # Learning rate for training
        pred_horizon: int = 10,    # How many timesteps ahead to predict (1 second)
        loss_fn: str = 'mse',      # Loss function: 'mse', 'huber', or 'mae'
        huber_delta: float = 1.0   # Delta parameter for Huber loss
    ):
        super().__init__()
        # Save all parameters so we can reload the model later
        self.save_hyperparameters()
        
        # ═══════════════════════════════════════════════════════════════
        # LSTM LAYER - The "Memory" of the Model
        # ═══════════════════════════════════════════════════════════════
        # LSTM is special because it can remember patterns over time.
        # Unlike simple neural networks, it has "gates" that decide:
        # - What to remember from the past
        # - What to forget
        # - What to output
        self.lstm = nn.LSTM(
            input_size=input_dim,      # Expects 3 channels per timestep
            hidden_size=hidden_dim,    # Internal memory size (64 neurons)
            num_layers=num_layers,     # Stack 2 LSTM layers for more capacity
            batch_first=True,          # Input shape: [batch, time, features]
            dropout=dropout if num_layers > 1 else 0  # Dropout between layers
        )
        
        # ═══════════════════════════════════════════════════════════════
        # OUTPUT HEAD - Converts LSTM output to predictions
        # ═══════════════════════════════════════════════════════════════
        # The LSTM outputs a 64-dimensional vector. We need to convert this
        # to 10 HRR predictions (one for each future timestep).
        self.head = nn.Linear(hidden_dim, output_dim * pred_horizon)
        # This creates: 64 → 10 transformation
        
    def forward(self, x):
        """
        FORWARD PASS - How the model makes predictions
        
        INPUT:
        x: [batch_size, 50, 3]
           - batch_size: Number of samples processed together (e.g., 32)
           - 50: Past timesteps (5 seconds at 0.1s resolution)
           - 3: Physics channels (HRR, Q_RADI, MLR)
        
        OUTPUT:
        y_hat: [batch_size, 10, 1]
               - 10: Future timesteps (1 second)
               - 1: Predicted HRR only
        """
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Process the time series through LSTM
        # ═══════════════════════════════════════════════════════════════
        # The LSTM reads the sequence step-by-step, updating its memory
        # at each timestep. It outputs a hidden state for each timestep.
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch, 50, 64]
        #   - For each of the 50 timesteps, we get a 64-dim representation
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Extract the final hidden state
        # ═══════════════════════════════════════════════════════════════
        # We only care about the LAST timestep's output, because it contains
        # the "summary" of everything the LSTM learned from the sequence.
        last_step_out = lstm_out[:, -1, :]  # Shape: [batch, 64]
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Project to future predictions
        # ═══════════════════════════════════════════════════════════════
        y_hat = self.head(last_step_out)  # Shape: [batch, 10]
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Reshape to proper output format
        # ═══════════════════════════════════════════════════════════════
        # Convert [batch, 10] → [batch, 10, 1] for consistency
        y_hat = y_hat.view(-1, self.hparams.pred_horizon, self.hparams.output_dim)
        return y_hat

    def training_step(self, batch, batch_idx):
        """
        TRAINING STEP - Called for each batch during training
        
        This function:
        1. Gets a batch of data
        2. Makes predictions
        3. Computes the error (loss)
        4. PyTorch Lightning automatically does backpropagation
        """
        # Unpack the batch
        (x, _), y = batch
        # x: Input sequence [batch, 50, 3]
        # _: Static features (not used by baseline model)
        # y: Target sequence [batch, 10, 1]
        
        # Make prediction
        y_hat = self(x)
        
        # Calculate loss based on chosen loss function
        # MSE: Standard squared error (sensitive to outliers)
        # Huber: Robust to outliers (quadratic for small errors, linear for large)
        # MAE: Mean absolute error (most robust but slower convergence)
        if self.hparams.loss_fn == 'huber':
            loss = F.huber_loss(y_hat, y, delta=self.hparams.huber_delta)
        elif self.hparams.loss_fn == 'mae':
            loss = F.l1_loss(y_hat, y)
        else:  # default to MSE
            loss = F.mse_loss(y_hat, y)
        
        # Log the loss so we can see it during training
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        VALIDATION STEP - Check performance on unseen data
        
        During training, we periodically check how well the model performs
        on validation data (data it hasn't trained on). This helps us detect
        overfitting (when the model memorizes training data but fails on new data).
        """
        (x, _), y = batch
        y_hat = self(x)
        
        # Use same loss function as training
        if self.hparams.loss_fn == 'huber':
            loss = F.huber_loss(y_hat, y, delta=self.hparams.huber_delta)
        elif self.hparams.loss_fn == 'mae':
            loss = F.l1_loss(y_hat, y)
        else:
            loss = F.mse_loss(y_hat, y)
        
        # Log validation loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        TEST STEP - Final evaluation on test set
        
        After training is complete, we evaluate on the test set to get
        the final accuracy metrics.
        """
        (x, _), y = batch
        y_hat = self(x)
        
        # Always compute MSE and MAE for test metrics (regardless of training loss)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        # Log both metrics
        self.log("test_loss", mse)
        self.log("test_mae", mae)
        
        return {"test_loss": mse, "test_mae": mae}
    
    def configure_optimizers(self):
        """
        OPTIMIZER SETUP - How the model learns
        
        Adam is an adaptive learning algorithm that adjusts the learning
        rate automatically during training. It's the most popular optimizer
        for deep learning.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
