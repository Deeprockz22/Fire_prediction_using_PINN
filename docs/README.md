# 🔥 Fire HRR Prediction Tool

**Physics-Informed LSTM for Fire Dynamics Forecasting**  
Model version: 2.0.0 | Training: 222 FDS scenarios

---

## ⚡ Quick Start

```bash
python fire_predict.py
```

That's it — an interactive menu guides you through everything.

---

## 📖 Menu Options

| # | Option | Description |
|---|--------|-------------|
| 1 | Quick Predict | Enter a file path, get a prediction |
| 2 | Run Example | See the model work on sample data |
| 3 | Batch Process | Process all CSVs from `Input/` folder |
| 4 | Generate FDS File | Create random FDS scenarios for testing |
| 5 | Train Model | Retrain from scratch (advanced) |
| 6 | Manage Files | List, open, clean folders |
| 7 | Setup & Diagnostics | Install, verify, troubleshoot |
| 9 | Exit | |

---

## 📂 Folder Structure

```
fire_prediction_deployment/
├── fire_predict.py          ⭐ Run this
├── sync_training_data.py    🔄 Sync new FDS scenarios into training_data/
├── requirements.txt         📦 Python dependencies
│
├── model/
│   └── best_model.ckpt      💾 Deployed Physics-Informed LSTM
│
├── fire_prediction/         🧠 Model library
│   ├── models/physics_informed.py
│   └── utils/physics.py
│
├── training_data/           📊 HRR CSVs + ml_dataset.json (222 scenarios)
├── Input/                   📥 Drop your *_hrr.csv files here
└── Output/                  📤 Prediction plots saved here
```

---

## 🔄 Adding New FDS Scenarios

When you run new FDS simulations, sync them into the deployment folder with:

```bash
python sync_training_data.py          # copy CSVs + regenerate ml_dataset.json
python sync_training_data.py --dry-run  # preview without changes
```

All data originates from `fds_scenarios/` — this is the single source of truth.

---

## 🧠 Training Your Own Model

```bash
python fire_predict.py
# Choose: 5 (Train Model) and follow the prompts
```

**Requirements:**
- 100+ FDS scenarios in `training_data/` (run sync first)
- GPU recommended (10–30× faster than CPU)
- 8 GB+ RAM, ~5 GB disk

**Default hyperparameters:**
```
Epochs:      50 (early stopping, patience=10)
Batch size:  32
LR:          0.001
Sequence:    30 timesteps input → 10 timesteps output
LSTM:        2 layers × 128 units
```

**Target performance:**
- val_mae < 10 kW → Good
- val_mae < 5 kW  → Excellent

**Output files:**
```
checkpoints/   ← epoch checkpoints during training
model/best_model.ckpt  ← best checkpoint (auto-deployed)
logs/          ← TensorBoard logs (tensorboard --logdir=logs)
```

---

## 🔬 Physics Architecture

The model is a **Physics-Informed LSTM** with 6 input channels:

| Ch | Feature | Source |
|----|---------|--------|
| 0 | HRR (kW) | Raw FDS output |
| 1 | Q_RADI (kW/m²) | Raw FDS output |
| 2 | MLR (kg/s) | Raw FDS output |
| 3 | Flame Height | **Heskestad (1984)** — `L_f = -1.02D + 0.235·Q_c^(2/5)` |
| 4 | Flame Growth Rate | Heskestad derivative |
| 5 | Flame Deviation | Heskestad deviation from mean |

**Physics loss** during training penalises predictions that violate the Heskestad correlation, enforcing physical consistency.

**Performance vs baseline (5.18 kW MAE):**

| Model | MAE | Δ |
|-------|-----|---|
| Baseline LSTM | 5.18 kW | — |
| + Physics (Heskestad) | 4.75 kW | **+8.3%** |
| Koopman (full dataset) | 2.72 kW | **+47.5%** |

### Supported correlations (for future retraining with 9 channels):
- **McCaffrey (1979)** — plume region classification (continuous / intermittent / far-field)
- **Thomas (1963)** — ventilation mass flow: `m_dot ≈ 0.5 × A_w × √H_w`
- **Buoyancy scaling** — Q^(2/5) universal power law

To enable 9-channel mode, set `INCLUDE_ENHANCED_FEATURES = True` in `fire_prediction/models/train_physics_full.py` and retrain.

---

## 📚 References

1. Heskestad, G. (1984). *Engineering relations for fire plumes.* Fire Safety Journal, 7(1), 25–32.
2. McCaffrey, B. J. (1979). *Purely buoyant diffusion flames.* NBSIR 79-1910.
3. Thomas, P. H. (1963). *The size of flames from natural fires.* Symp. (Int.) Combustion, 9, 844–859.

---

## 🆘 Troubleshooting

| Problem | Fix |
|---------|-----|
| Missing packages | `pip install -r requirements.txt` |
| "Model file missing" | Check `model/best_model.ckpt` exists |
| Out of memory | Reduce `BATCH_SIZE = 16` in train script |
| High validation loss | Add more diverse scenarios, increase epochs |
| Slow training | Install CUDA + PyTorch GPU build |

Run diagnostics anytime:
```bash
python fire_predict.py check
```
