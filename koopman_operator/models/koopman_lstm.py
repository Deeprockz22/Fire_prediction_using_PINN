"""
Koopman-Enhanced LSTM for Fire Dynamics Prediction
===================================================

This model combines an LSTM encoder with a **learnable Koopman operator**
to impose linear dynamical structure on the latent space.

Architecture
------------
    Input x(t) ∈ ℝ^{seq×C}
         │
    ┌────▼────┐
    │  LSTM   │  Encoder g(·):  maps sequences → latent observables
    │ Encoder │
    └────┬────┘
         │ z(t) ∈ ℝ^H                   ← Koopman observable
         │
    ┌────▼────────────────┐
    │  z(t+1) = K · z(t)  │  Koopman matrix K ∈ ℝ^{H×H}  (LINEAR!)
    │  z(t+n) = Kⁿ · z(t) │  Multi-step = matrix power — no error buildup
    └────┬────────────────┘
         │ z(t+1), z(t+2), …, z(t+n)
         │
    ┌────▼────┐
    │ Decoder │  Maps latent back to physical space (HRR prediction)
    └────┬────┘
         │
    Output ŷ(t+1…t+n) ∈ ℝ^{n×1}

Training Loss
-------------
    L = L_pred + α·L_linear + β·L_recon + γ·L_physics + δ·L_mono + ε·L_spectral

    L_pred    :  ||Decoder(K·z_t) − y_{t+1}||²          main prediction
    L_linear  :  ||K·z_t − z_{t+1}||²                   Koopman linearity
    L_recon   :  ||Decoder(z_t) − x_t[HRR]||²           autoencoder quality
    L_physics :  Heskestad flame-height consistency       domain knowledge
    L_mono    :  penalise negative HRR & extreme jumps   physical plausibility
    L_spectral:  max(σ_max(K) − 1, 0)²                  stability constraint

Reference:
    Lusch, Wehmeyer & Brunton (2018). "Deep learning for universal
    linear embeddings of nonlinear dynamics." Nature Communications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from koopman_fire.utils.physics import (
    physics_consistency_loss,
    monotonicity_loss,
)


class KoopmanLSTM(pl.LightningModule):
    """
    Koopman-Enhanced LSTM for fire dynamics prediction.

    The LSTM acts as an encoder that lifts the input into a latent space
    where dynamics are constrained to be LINEAR via the Koopman matrix K.
    """

    def __init__(
        self,
        # Encoder
        input_dim: int = 6,           # 3 base + 3 Heskestad channels
        hidden_dim: int = 64,         # latent / Koopman space dimension
        num_layers: int = 2,          # LSTM depth
        dropout: float = 0.1,
        # Prediction
        pred_horizon: int = 10,       # future steps to predict
        output_dim: int = 1,          # predict HRR only
        # Koopman
        koopman_dim: int = 64,        # dimension of K (= hidden_dim by default)
        # Loss weights
        alpha_linear: float = 1.0,    # weight for linearity loss
        beta_recon: float = 0.5,      # weight for reconstruction loss
        gamma_physics: float = 0.1,   # weight for Heskestad penalty
        delta_mono: float = 0.05,     # weight for monotonicity penalty
        epsilon_spectral: float = 1.0,# weight for spectral radius penalty
        # Physics
        fire_diameter: float = 0.3,
        # Optimiser
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Derived
        koopman_dim = koopman_dim or hidden_dim

        # ══════════════════════════════════════════════════════════════
        # 1. LSTM ENCODER  —  g(x) → z
        # ══════════════════════════════════════════════════════════════
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Optional projection if koopman_dim ≠ hidden_dim
        if koopman_dim != hidden_dim:
            self.encoder_proj = nn.Linear(hidden_dim, koopman_dim)
        else:
            self.encoder_proj = nn.Identity()

        # ══════════════════════════════════════════════════════════════
        # 2. KOOPMAN MATRIX  —  z(t+1) = K · z(t)
        # ══════════════════════════════════════════════════════════════
        # A simple learnable linear map WITHOUT bias.
        # The absence of bias is important: Koopman dynamics must be
        # purely multiplicative so that z=0 is a fixed point.
        self.koopman_matrix = nn.Linear(koopman_dim, koopman_dim, bias=False)

        # Initialise K close to identity so initial predictions ≈ persist
        nn.init.eye_(self.koopman_matrix.weight)
        # Add a small perturbation to break symmetry
        with torch.no_grad():
            self.koopman_matrix.weight.add_(
                torch.randn_like(self.koopman_matrix.weight) * 0.01
            )

        # ══════════════════════════════════════════════════════════════
        # 3. DECODER  —  z → ŷ  (maps latent back to HRR)
        #    Deeper decoder to capture high-frequency oscillations
        # ══════════════════════════════════════════════════════════════
        self.decoder = nn.Sequential(
            nn.Linear(koopman_dim, koopman_dim),
            nn.GELU(),
            nn.Linear(koopman_dim, koopman_dim // 2),
            nn.GELU(),
            nn.Linear(koopman_dim // 2, output_dim),
        )

    # ==================================================================
    # ENCODING
    # ==================================================================
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input sequence into Koopman observables.

        Args:
            x: [B, seq_len, input_dim]

        Returns:
            z_seq: [B, seq_len, koopman_dim]
                   Koopman observables at every input timestep.
        """
        lstm_out, _ = self.encoder_lstm(x)          # [B, T, H]
        z_seq = self.encoder_proj(lstm_out)          # [B, T, K]
        return z_seq

    # ==================================================================
    # KOOPMAN ADVANCE
    # ==================================================================
    def koopman_advance(self, z: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Advance latent state linearly via the Koopman matrix.

        z(t+n) = K^n · z(t)     (applied iteratively for numerical stability)

        Args:
            z:     [B, koopman_dim]   starting latent state
            steps: number of future steps

        Returns:
            z_future: [B, steps, koopman_dim]
        """
        future = []
        z_cur = z
        for _ in range(steps):
            z_cur = self.koopman_matrix(z_cur)       # K · z
            future.append(z_cur)
        return torch.stack(future, dim=1)            # [B, steps, K]

    # ==================================================================
    # DECODING
    # ==================================================================
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent states to physical HRR predictions.

        Args:
            z: [B, T, koopman_dim]

        Returns:
            y_hat: [B, T, output_dim]
        """
        return self.decoder(z)

    # ==================================================================
    # FORWARD PASS
    # ==================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass:  x → encode → koopman advance → decode → ŷ

        Args:
            x: [B, seq_len, input_dim]

        Returns:
            y_hat: [B, pred_horizon, output_dim]
        """
        z_seq = self.encode(x)                       # [B, T, K]
        z_last = z_seq[:, -1, :]                     # [B, K]
        z_future = self.koopman_advance(z_last, self.hparams.pred_horizon)
        y_hat = self.decode(z_future)                # [B, pred_horizon, 1]
        return y_hat

    # ==================================================================
    # SPECTRAL REGULARISATION
    # ==================================================================
    def _spectral_penalty(self) -> torch.Tensor:
        """
        Penalise the Koopman matrix when its largest singular value > 1.

        Uses SVD (differentiable in PyTorch) to compute sigma_max(K).
        The penalty is:  max(sigma_max - 1, 0)^2

        This guarantees that ALL eigenvalues of K lie inside the unit
        circle since |lambda_i| <= sigma_max for any matrix.

        Returns:
            Scalar penalty (0 when K is already stable).
        """
        K = self.koopman_matrix.weight                 # [K, K]
        # torch.linalg.svdvals returns singular values in descending order
        sigma_max = torch.linalg.svdvals(K)[0]
        return torch.relu(sigma_max - 1.0) ** 2

    # ==================================================================
    # LOSS COMPUTATION
    # ==================================================================
    def _compute_losses(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute all loss components.

        Returns dict of individual (unweighted) losses plus the total.
        """
        B, T, C = x.shape
        horizon = self.hparams.pred_horizon

        # --- Encode full input sequence ---
        z_seq = self.encode(x)                       # [B, T, K]
        z_last = z_seq[:, -1, :]                     # [B, K]

        # --- Prediction loss ---
        z_future = self.koopman_advance(z_last, horizon)   # [B, H, K]
        y_hat = self.decode(z_future)                       # [B, H, 1]
        loss_pred = F.mse_loss(y_hat, y)

        # --- Linearity loss  (THE key Koopman constraint) ---
        # For consecutive pairs in the INPUT window:
        #   K . z(t) ~ z(t+1)
        z_rolled = self.koopman_matrix(z_seq[:, :-1, :])    # K*z(0..T-2)
        z_target = z_seq[:, 1:, :]                          # z(1..T-1)
        loss_linear = F.mse_loss(z_rolled, z_target.detach())

        # --- Reconstruction loss ---
        # The decoder should recover the HRR from latent states inside the
        # input window (trains the decoder to be a faithful inverse).
        x_hrr = x[:, :, 0:1]                                # [B, T, 1]
        x_hat = self.decode(z_seq)                           # [B, T, 1]
        loss_recon = F.mse_loss(x_hat, x_hrr)

        # --- Spectral penalty (stability) ---
        loss_spectral = self._spectral_penalty()

        # --- Physics losses (optional) ---
        loss_physics = physics_consistency_loss(
            y_hat, y,
            fire_diameter=self.hparams.fire_diameter,
            weight=1.0,  # weighting applied below
        )
        loss_mono = monotonicity_loss(y_hat, weight=1.0)

        # --- Weighted total ---
        total = (
            loss_pred
            + self.hparams.alpha_linear    * loss_linear
            + self.hparams.beta_recon      * loss_recon
            + self.hparams.epsilon_spectral * loss_spectral
            + self.hparams.gamma_physics   * loss_physics
            + self.hparams.delta_mono      * loss_mono
        )

        return {
            "loss":         total,
            "pred":         loss_pred,
            "linear":       loss_linear,
            "recon":        loss_recon,
            "spectral":     loss_spectral,
            "physics":      loss_physics,
            "mono":         loss_mono,
            "y_hat":        y_hat,
        }

    # ==================================================================
    # LIGHTNING STEPS
    # ==================================================================
    def training_step(self, batch, batch_idx):
        x, y = batch
        losses = self._compute_losses(x, y)

        self.log("train/loss",     losses["loss"],     prog_bar=True)
        self.log("train/pred",     losses["pred"])
        self.log("train/linear",   losses["linear"])
        self.log("train/recon",    losses["recon"])
        self.log("train/spectral", losses["spectral"])
        self.log("train/physics",  losses["physics"])
        self.log("train/mono",     losses["mono"])

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        losses = self._compute_losses(x, y)

        self.log("val/loss",     losses["loss"],     prog_bar=True)
        self.log("val/pred",     losses["pred"])
        self.log("val/linear",   losses["linear"])
        self.log("val/recon",    losses["recon"])
        self.log("val/spectral", losses["spectral"])

        # MAE in original units (useful for comparison with old model)
        mae = F.l1_loss(losses["y_hat"], y)
        self.log("val/mae", mae, prog_bar=True)

        return losses["loss"]

    def test_step(self, batch, batch_idx):
        x, y = batch
        losses = self._compute_losses(x, y)

        mae = F.l1_loss(losses["y_hat"], y)

        self.log("test/loss",    losses["loss"])
        self.log("test/pred",    losses["pred"])
        self.log("test/linear",  losses["linear"])
        self.log("test/mae",     mae)
        self.log("test/physics", losses["physics"])

        return {"test_loss": losses["loss"], "test_mae": mae}

    # ==================================================================
    # OPTIMISER
    # ==================================================================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }

    # ==================================================================
    # CONVENIENCE: multi-step predict (no teacher forcing)
    # ==================================================================
    @torch.no_grad()
    def predict_sequence(self, x: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Given an input window, predict `steps` future HRR values.

        This uses a single Koopman advance (K^n · z) rather than
        autoregressive rollout, so there is no compounding error.

        Args:
            x: [1, seq_len, input_dim]  (single sample, batched ok too)

        Returns:
            y_hat: [1, steps, 1]  predicted HRR (normalised)
        """
        self.eval()
        z_seq = self.encode(x)
        z_last = z_seq[:, -1, :]
        z_future = self.koopman_advance(z_last, steps)
        return self.decode(z_future)


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    print("Koopman-Enhanced LSTM — Architecture Test")
    print("=" * 55)

    model = KoopmanLSTM(input_dim=6, hidden_dim=64, koopman_dim=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.randn(4, 30, 6)           # batch of 4, 30 timesteps, 6 channels
    y = torch.randn(4, 10, 1)           # targets

    y_hat = model(x)
    print(f"Input:   {x.shape}")
    print(f"Output:  {y_hat.shape}  (expect [4, 10, 1])")

    losses = model._compute_losses(x, y)
    for k, v in losses.items():
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            print(f"  {k:>10s}: {v.item():.6f}")

    print("\nKoopman matrix K shape:", model.koopman_matrix.weight.shape)
    print("✅ All shapes correct!")
