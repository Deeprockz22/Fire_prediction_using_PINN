# Koopman-Enhanced LSTM for Fire Dynamics Prediction

## Overview

This module implements a **Koopman-Enhanced LSTM** that integrates Koopman operator 
theory with deep learning for predicting fire Heat Release Rate (HRR).

### Key Innovation

Instead of a purely data-driven LSTM, we impose a **linear structure** on the latent 
dynamics via a learnable **Koopman matrix K**:

```
x(t) → LSTM Encoder → z(t)     [latent Koopman observable]
                        ↓
                   z(t+1) = K · z(t)    [LINEAR dynamics!]
                        ↓
              Decoder → ŷ(t+1)   [HRR prediction]
```

**Why this matters:**
- **Multi-step prediction** without error accumulation: `z(t+n) = K^n · z(t)`
- **Interpretability**: eigenvalues of K reveal fire dynamics modes
- **Stability**: linear structure prevents wild divergence

## Folder Structure

```
koopman_fire/
├── __init__.py
├── train.py                      # Training script (run this!)
├── README.md                     # You are here
├── data/
│   ├── __init__.py
│   └── dataset.py                # Fire time-series dataset
├── models/
│   ├── __init__.py
│   └── koopman_lstm.py           # Koopman-Enhanced LSTM model
└── utils/
    ├── __init__.py
    ├── physics.py                # Heskestad correlations & physics losses
    └── koopman_analysis.py       # Eigenvalue analysis of trained K
```

## Quick Start

```bash
# From D:\FDS\Small_project
python -m koopman_fire.train
```

## Training Loss Components

| Loss | Symbol | Purpose |
|------|--------|---------|
| Prediction | `L_pred` | Main MSE between predicted and actual future HRR |
| Linearity | `α · L_linear` | **Koopman constraint**: `K·z(t) ≈ z(t+1)` |
| Reconstruction | `β · L_recon` | Decoder recovers HRR from latent states |
| Physics | `γ · L_physics` | Heskestad flame-height consistency |
| Monotonicity | `δ · L_mono` | Penalise negative HRR / extreme jumps |

## Koopman Spectral Analysis

After training, the eigenvalues of K reveal the dynamical structure:

| Eigenvalue | Interpretation |
|------------|----------------|
| \|λ\| < 1 | Decaying mode (fire dying down) |
| \|λ\| = 1 | Neutral mode (steady-state) |
| \|λ\| > 1 | Growing mode (fire growth phase) |
| Im(λ) ≠ 0 | Oscillatory mode (pulsating fire) |

## References

- Lusch, Wehmeyer & Brunton (2018). *"Deep learning for universal linear 
  embeddings of nonlinear dynamics."* Nature Communications.
- Heskestad, G. (1984). *"Engineering relations for fire plumes."*
  Fire Safety Journal, 7(1), 25-32.
