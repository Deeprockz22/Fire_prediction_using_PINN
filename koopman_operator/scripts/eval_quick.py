import sys, json
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from koopman_fire.models.koopman_lstm import KoopmanLSTM
from koopman_fire.data.dataset import FireTimeSeriesDataset

# Try best checkpoint first, fall back to last
import glob
ckpts = glob.glob("koopman_fire/checkpoints/koopman-best-*.ckpt")
ckpt = ckpts[0] if ckpts else "koopman_fire/checkpoints/last.ckpt"

model = KoopmanLSTM.load_from_checkpoint(ckpt)
model.eval()
model.cpu()

train_ds = FireTimeSeriesDataset(r"D:\FDS\Small_project\ml_data", split="train")
test_ds = FireTimeSeriesDataset(r"D:\FDS\Small_project\ml_data", split="test", train_stats=train_ds.stats)
loader = DataLoader(test_ds, batch_size=64, shuffle=False)

mse_t = 0.0
mae_t = 0.0
n = 0
with torch.no_grad():
    for x, y in loader:
        yh = model(x)
        mse_t += F.mse_loss(yh, y).item()
        mae_t += F.l1_loss(yh, y).item()
        n += 1

K = model.koopman_matrix.weight.detach().cpu().numpy()
ev, _ = np.linalg.eig(K)
mags = np.abs(ev)
order = np.argsort(-mags)
mags = mags[order]

res = {
    "checkpoint": ckpt,
    "test_mse": round(mse_t / n, 6),
    "test_mae": round(mae_t / n, 6),
    "spectral_radius": round(float(mags[0]), 4),
    "stable": bool(np.all(mags <= 1.001)),
    "top5_magnitudes": [round(float(m), 4) for m in mags[:5]],
    "epsilon_spectral": float(model.hparams.get("epsilon_spectral", 0)),
}
with open("koopman_fire/results_v2.json", "w") as f:
    json.dump(res, f, indent=2)
