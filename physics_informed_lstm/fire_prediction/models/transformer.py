import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerForecaster(pl.LightningModule):
    """
    Transformer-based Time Series Forecaster for Fire HRR.
    Uses static features integration.
    """
    def __init__(
        self, 
        input_dim: int = 3, 
        static_dim: int = 12,
        d_model: int = 64, 
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-4, # Lower LR for Transformer
        pred_horizon: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Input Projections
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Static Feature Encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. Output Head
        # Concatenate: [Global Avg Pool of Transformer Output] + [Static Embed]
        fusion_dim = d_model * 2
        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, pred_horizon) # Predicting scalar sequence
        )
        
    def forward(self, x_seq, x_static):
        # x_seq: [batch, seq_len, input_dim]
        
        # 1. Embed Input
        x = self.input_proj(x_seq) # [batch, seq_len, d_model]
        x = self.pos_encoder(x)
        
        # 2. Transformer
        # Masking usually not needed for encoder-only regression unless autoregressive training
        memory = self.transformer_encoder(x) # [batch, seq_len, d_model]
        
        # 3. Pooling (Global Average Pooling) -> more robust than just last token
        temporal_embed = memory.mean(dim=1) # [batch, d_model]
        
        # 4. Process Static
        static_embed = self.static_encoder(x_static) # [batch, d_model]
        
        # 5. Fusion
        combined = torch.cat([temporal_embed, static_embed], dim=1) # [batch, d_model*2]
        
        # 6. Predict
        y_hat = self.head(combined) # [batch, pred_horizon]
        
        return y_hat.unsqueeze(-1) # [batch, pred_horizon, 1]

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4) # AdamW often better for Transformers
        # Use a default T_max if trainer is not yet available
        T_max = self.trainer.max_epochs if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'max_epochs') else 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return [optimizer], [scheduler]
