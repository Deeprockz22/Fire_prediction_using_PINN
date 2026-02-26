"""
Koopman-Enhanced LSTM for Fire Dynamics Prediction
===================================================

A physics-informed deep learning framework that integrates Koopman operator
theory with LSTM networks for predicting fire Heat Release Rate (HRR).

Architecture:
    1. LSTM Encoder:    Maps input sequences to latent Koopman observables
    2. Koopman Matrix:  Advances latent state via LINEAR dynamics z(t+1) = K·z(t)
    3. Decoder:         Maps latent states back to physical predictions

Key Innovation:
    The linearity constraint on the latent space gives us:
    - Multi-step prediction without error accumulation (z(t+n) = K^n · z(t))
    - Spectral analysis of fire dynamics via eigenvalues of K
    - Better generalization through structured dynamics

Author: Fire Prediction Team
Date: 2026-02-18
"""

__version__ = "1.0.0"
