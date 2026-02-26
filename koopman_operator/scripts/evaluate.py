"""Quick evaluation - ASCII only for Windows compatibility."""
import sys, os
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from koopman_fire.models.koopman_lstm import KoopmanLSTM
from koopman_fire.data.dataset import FireTimeSeriesDataset
import numpy as np

# Load model
print("Loading model...")
model = KoopmanLSTM.load_from_checkpoint("koopman_fire/checkpoints/last.ckpt")
model.eval()
model.cpu()

# Load data
print("Loading data...")
train_ds = FireTimeSeriesDataset(r"D:\FDS\Small_project\ml_data", split="train")
test_ds = FireTimeSeriesDataset(
    r"D:\FDS\Small_project\ml_data", split="test", train_stats=train_ds.stats
)

# Manual test loop
print("Evaluating on %d test samples..." % len(test_ds))
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

total_mse = 0.0
total_mae = 0.0
n = 0

with torch.no_grad():
    for x, y in test_loader:
        y_hat = model(x)
        total_mse += F.mse_loss(y_hat, y).item()
        total_mae += F.l1_loss(y_hat, y).item()
        n += 1

avg_mse = total_mse / n
avg_mae = total_mae / n

print("")
print("=" * 55)
print("  TEST RESULTS")
print("=" * 55)
print("  Test MSE:  %.6f" % avg_mse)
print("  Test MAE:  %.4f" % avg_mae)
print("  Baseline MAE (old LSTM): ~4.85")
print("=" * 55)

# Koopman eigenvalue analysis
K = model.koopman_matrix.weight.detach().cpu().numpy()
eigenvalues, _ = np.linalg.eig(K)
mags = np.abs(eigenvalues)
order = np.argsort(-mags)
eigenvalues = eigenvalues[order]
mags = mags[order]

print("")
print("=" * 55)
print("  KOOPMAN SPECTRAL ANALYSIS")
print("=" * 55)
print("  Matrix size: %d x %d" % K.shape)
print("  Spectral radius: %.4f" % mags[0])
print("  Stable (all |lam| <= 1): %s" % str(bool(np.all(mags <= 1.0 + 1e-6))))
print("-" * 55)
print("  %3s  %8s  %10s  %10s  %10s  %s" % ("#", "|lam|", "angle(rad)", "Re", "Im", "Type"))
print("-" * 55)
for i in range(min(10, len(eigenvalues))):
    ev = eigenvalues[i]
    mag = mags[i]
    ang = np.angle(ev)
    label = "GROWING" if mag > 1.01 else ("DECAYING" if mag < 0.99 else "STEADY")
    if abs(np.imag(ev)) > 1e-4:
        label += "+OSC"
    print("  %3d  %8.4f  %10.4f  %10.4f  %10.4f  %s" % (i+1, mag, ang, np.real(ev), np.imag(ev), label))
print("=" * 55)
