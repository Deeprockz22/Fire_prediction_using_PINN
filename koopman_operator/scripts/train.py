"""
Training Script — Koopman-Enhanced LSTM for Fire Dynamics
==========================================================

Usage:
    cd D:\\FDS\\Small_project
    python -m koopman_fire.train

TensorBoard:
    tensorboard --logdir=koopman_fire/runs
"""

import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from koopman_fire.data.dataset import FireTimeSeriesDataset
from koopman_fire.models.koopman_lstm import KoopmanLSTM
from koopman_fire.utils.koopman_analysis import KoopmanAnalyzer


# ═════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ──
    "data_dir":        r"D:\FDS\Small_project\ml_data",
    "input_seq_len":   30,
    "pred_horizon":    10,
    "fire_diameter":   0.3,
    "include_heskestad": True,
    "batch_size":      64,
    "num_workers":     0,        # 0 for Windows compatibility

    # ── Model (v2: bigger latent space for oscillatory modes) ──
    "input_dim":       6,        # 3 base + 3 Heskestad
    "hidden_dim":      256,      # was 64 → more representational power
    "koopman_dim":     256,      # was 64 → more Koopman modes for oscillations
    "num_layers":      2,
    "dropout":         0.1,

    # ── Loss weights (v2: prioritise prediction, relax constraints) ──
    "alpha_linear":    0.03,      # was 1.0 → less rigid linearity = sharper preds
    "beta_recon":      0.02,      # was 0.5 → less autoencoder emphasis
    "gamma_physics":   0.01,      # unchanged
    "delta_mono":      0.005,     # unchanged
    "epsilon_spectral": 0.01,     # was 1.0 → allow eigenvalues closer to 1 (less damping)

    # ── Optimiser ──
    "lr":              1e-3,
    "weight_decay":    1e-5,

    # ── Training ──
    "max_epochs":      50,
    "patience":        12,       # Early stopping
    "seed":            42,
}


# ═════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════

def load_data(cfg: dict):
    """Load train / val / test splits."""

    train_ds = FireTimeSeriesDataset(
        cfg["data_dir"],
        split="train",
        input_seq_len=cfg["input_seq_len"],
        pred_horizon=cfg["pred_horizon"],
        fire_diameter=cfg["fire_diameter"],
        include_heskestad=cfg["include_heskestad"],
    )

    val_ds = FireTimeSeriesDataset(
        cfg["data_dir"],
        split="val",
        input_seq_len=cfg["input_seq_len"],
        pred_horizon=cfg["pred_horizon"],
        fire_diameter=cfg["fire_diameter"],
        include_heskestad=cfg["include_heskestad"],
        train_stats=train_ds.stats,
    )

    test_ds = FireTimeSeriesDataset(
        cfg["data_dir"],
        split="test",
        input_seq_len=cfg["input_seq_len"],
        pred_horizon=cfg["pred_horizon"],
        fire_diameter=cfg["fire_diameter"],
        include_heskestad=cfg["include_heskestad"],
        train_stats=train_ds.stats,
    )

    return train_ds, val_ds, test_ds


# ═════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════

def create_model(cfg: dict) -> KoopmanLSTM:
    model = KoopmanLSTM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        koopman_dim=cfg["koopman_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pred_horizon=cfg["pred_horizon"],
        alpha_linear=cfg["alpha_linear"],
        beta_recon=cfg["beta_recon"],
        gamma_physics=cfg["gamma_physics"],
        delta_mono=cfg["delta_mono"],
        epsilon_spectral=cfg["epsilon_spectral"],
        fire_diameter=cfg["fire_diameter"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    return model


# ═════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════

def train(cfg: dict):
    """Full training pipeline."""

    pl.seed_everything(cfg["seed"])

    # ── Data ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  KOOPMAN-ENHANCED LSTM  —  FIRE DYNAMICS PREDICTION")
    print("=" * 65)

    train_ds, val_ds, test_ds = load_data(cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], persistent_workers=False,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = create_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n🧠 Model")
    print(f"   Parameters:      {n_params:,}")
    print(f"   Koopman dim:     {cfg['koopman_dim']}")
    print(f"   LSTM hidden:     {cfg['hidden_dim']}")
    print(f"   LSTM layers:     {cfg['num_layers']}")
    print(f"\n⚖️  Loss weights")
    print(f"   α (linearity):   {cfg['alpha_linear']}")
    print(f"   β (recon):       {cfg['beta_recon']}")
    print(f"   γ (physics):     {cfg['gamma_physics']}")
    print(f"   δ (monotonic):   {cfg['delta_mono']}")

    # ── Callbacks ─────────────────────────────────────────────────
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="koopman-best-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=cfg["patience"],
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Logger ────────────────────────────────────────────────────
    log_dir = Path(__file__).parent.parent / "runs"
    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name="koopman_lstm",
        log_graph=False,
    )

    print(f"\n📊 Logging → {log_dir / 'koopman_lstm'}")
    print(f"   tensorboard --logdir={log_dir / 'koopman_lstm'}")

    # ── Trainer ───────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True,
    )

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n🚀 Training for up to {cfg['max_epochs']} epochs "
          f"(early stop patience = {cfg['patience']})")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print("-" * 65)

    trainer.fit(model, train_loader, val_loader)

    print(f"\n✅ Training complete!")
    print(f"   Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"   Best val loss:   {checkpoint_cb.best_model_score:.6f}")

    # ── Test ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TEST EVALUATION")
    print("=" * 65)

    best_model = KoopmanLSTM.load_from_checkpoint(checkpoint_cb.best_model_path)
    results = trainer.test(best_model, test_loader)

    print(f"\n📊 Test Results:")
    for k, v in results[0].items():
        print(f"   {k}: {v:.6f}")

    # ── Koopman Analysis ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  KOOPMAN SPECTRAL ANALYSIS")
    print("=" * 65)

    analyser = KoopmanAnalyzer(best_model)
    analyser.print_spectrum(n=10)

    plot_path = str(ckpt_dir / "koopman_eigenvalues.png")
    analyser.plot_eigenvalues(save_path=plot_path)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Checkpoint:         {checkpoint_cb.best_model_path}")
    print(f"  Eigenvalue plot:    {plot_path}")
    print(f"  Koopman stable:     {analyser.is_stable}")
    print(f"  Spectral radius:    {analyser.magnitudes[0]:.4f}")
    print(f"  Test MAE:           {results[0].get('test/mae', 'N/A')}")
    print("=" * 65 + "\n")

    return best_model, results[0], analyser


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train(CONFIG)
