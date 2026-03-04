# 🔥 Fire Prediction Model Improvement TODO List

**Created:** 2026-03-04  
**Project:** Physics-Informed LSTM for Fire HRR Prediction  
**Goal:** Improve model performance beyond dataset expansion

---

## 📋 Priority Levels
- 🔴 **HIGH** - Quick wins, high impact
- 🟡 **MEDIUM** - Moderate effort, good impact
- 🟢 **LOW** - Research/experimental, long-term

---

## 1. HYPERPARAMETER OPTIMIZATION 🎯

### 🔴 HIGH PRIORITY
- [ ] Implement learning rate scheduling
  - [ ] Add `ReduceLROnPlateau` to training loop
  - [ ] Test `CosineAnnealingLR` scheduler
  - [ ] Compare validation loss curves
  - **Files:** `fire_prediction/models/physics_informed.py`, `train_physics_full.py`

- [ ] Gradient clipping implementation
  - [ ] Add `torch.nn.utils.clip_grad_norm_()` 
  - [ ] Test max_norm values: [0.5, 1.0, 2.0]
  - [ ] Monitor gradient statistics
  - **Files:** `train_physics_full.py`

### 🟡 MEDIUM PRIORITY
- [ ] Automated hyperparameter search
  - [ ] Install Optuna: `pip install optuna`
  - [ ] Create HP search script: `hp_search.py`
  - [ ] Define search space:
    - Hidden dim: [64, 128, 256, 512]
    - Num layers: [2, 3, 4]
    - Dropout: [0.1, 0.2, 0.3, 0.4]
    - Learning rate: [1e-4, 5e-4, 1e-3]
  - [ ] Run 50-100 trials
  - [ ] Save best config to `config/best_hp.yaml`

- [ ] Weight decay experimentation
  - [ ] Switch from Adam to AdamW
  - [ ] Test weight_decay: [1e-5, 1e-4, 1e-3]
  - [ ] Compare validation performance

---

## 2. MODEL ARCHITECTURE ENHANCEMENTS 🏗️

### 🔴 HIGH PRIORITY
- [ ] Implement Ensemble Model
  - [ ] Create `models/ensemble.py`
  - [ ] Load Physics-LSTM checkpoint
  - [ ] Load Transformer checkpoint
  - [ ] Load Hybrid-LSTM checkpoint
  - [ ] Implement weighted averaging
  - [ ] Test ensemble on validation set
  - [ ] Compare to individual models
  - **Expected improvement:** 10-15% MAE reduction

### 🟡 MEDIUM PRIORITY
- [ ] Add Attention Mechanism to LSTM
  - [ ] Create `models/attention_lstm.py`
  - [ ] Implement multi-head attention layer
  - [ ] Add attention weights visualization
  - [ ] Compare to baseline LSTM
  - **Inspiration:** Transformer attention

- [ ] Residual connections
  - [ ] Add skip connections in LSTM
  - [ ] Test residual + LSTM combination
  - [ ] Compare training stability
  - **Files:** `models/physics_informed.py`

### 🟢 LOW PRIORITY
- [ ] Graph Neural Network approach
  - [ ] Model fire spread as graph
  - [ ] Implement GNN for spatial-temporal prediction
  - [ ] Research paper implementation
  - **Status:** Experimental

---

## 3. ADVANCED LOSS FUNCTIONS 📊

### 🔴 HIGH PRIORITY
- [ ] Peak Detection Loss
  - [ ] Create `utils/losses.py`
  - [ ] Implement peak-weighted MSE
  - [ ] Add to physics loss combination
  - [ ] Test weight factors: [2, 5, 10]
  - **Impact:** Better peak HRR prediction

```python
def peak_penalty_loss(pred, target):
    """Heavily penalize errors at peak HRR"""
    peak_mask = target > (target.mean() + target.std())
    weights = torch.where(peak_mask, 5.0, 1.0)
    return (weights * (pred - target)**2).mean()
```

### 🟡 MEDIUM PRIORITY
- [ ] Huber Loss (robust to outliers)
  - [ ] Replace MSE with Huber loss
  - [ ] Test delta parameter: [0.5, 1.0, 2.0]
  - [ ] Compare robustness on extreme scenarios

- [ ] Quantile Loss for uncertainty
  - [ ] Predict [10th, 50th, 90th] percentiles
  - [ ] Implement pinball loss
  - [ ] Visualize prediction intervals

### 🟢 LOW PRIORITY
- [ ] Dynamic Time Warping (DTW) Loss
  - [ ] Install `tslearn`: `pip install tslearn`
  - [ ] Implement DTW-based loss
  - [ ] Test on misaligned sequences
  - **Use case:** Better temporal alignment

---

## 4. FEATURE ENGINEERING 🔧

### 🔴 HIGH PRIORITY
- [ ] Add time derivatives
  - [ ] Compute dHRR/dt (first derivative)
  - [ ] Compute d²HRR/dt² (second derivative)
  - [ ] Add to input features
  - [ ] Test impact on prediction accuracy
  - **Files:** `data/feature_extractor.py`

### 🟡 MEDIUM PRIORITY
- [ ] Additional physics features
  - [ ] Ventilation factor: A*√H
  - [ ] Fuel load density (MJ/m²)
  - [ ] Room aspect ratio
  - [ ] Ceiling height effect
  - [ ] Add to static features
  - **Files:** `data/feature_extractor.py`, `utils/physics.py`

- [ ] Combustion features
  - [ ] O₂ consumption rate
  - [ ] CO₂ production rate
  - [ ] Smoke production rate
  - [ ] Extract from FDS output if available

### 🟢 LOW PRIORITY
- [ ] Domain-specific feature selection
  - [ ] Use mutual information
  - [ ] Feature importance from Random Forest
  - [ ] Remove redundant features

---

## 5. DATA AUGMENTATION 📈

### 🔴 HIGH PRIORITY
- [ ] Time-series noise injection
  - [ ] Add Gaussian noise: σ = [0.01, 0.05, 0.1] * std(HRR)
  - [ ] Physics-consistent noise bounds
  - [ ] Augment during training only
  - [ ] Test impact on generalization
  - **Files:** `data/dataset.py`

```python
def add_noise(hrr, noise_level=0.05):
    """Add physics-consistent Gaussian noise"""
    std = hrr.std()
    noise = torch.randn_like(hrr) * noise_level * std
    return torch.clamp(hrr + noise, min=0)  # HRR can't be negative
```

### 🟡 MEDIUM PRIORITY
- [ ] Temporal augmentation
  - [ ] Time stretching: 0.8x - 1.2x speed
  - [ ] Random time shifts: ±5 timesteps
  - [ ] Test on validation set
  - **Files:** `data/dataset.py`

- [ ] Rolling window variations
  - [ ] Vary sequence length: [20, 30, 40, 50]
  - [ ] Random window sampling
  - [ ] Helps generalization

### 🟢 LOW PRIORITY
- [ ] SMOTE for time series
  - [ ] Synthetic Minority Over-sampling
  - [ ] Create intermediate scenarios
  - [ ] Balance dataset

---

## 6. REGULARIZATION TECHNIQUES 🛡️

### 🔴 HIGH PRIORITY
- [ ] Layer Normalization
  - [ ] Add `nn.LayerNorm` after LSTM
  - [ ] Stabilizes training
  - [ ] Compare to BatchNorm
  - **Files:** `models/physics_informed.py`

### 🟡 MEDIUM PRIORITY
- [ ] Dropout tuning
  - [ ] Test dropout rates: [0.1, 0.2, 0.3, 0.4, 0.5]
  - [ ] Use different dropout for different layers
  - [ ] Monitor overfitting

- [ ] Early stopping refinement
  - [ ] Increase patience: 20 → 30 epochs
  - [ ] Monitor multiple metrics (MAE + R²)
  - [ ] Save top-3 checkpoints

### 🟢 LOW PRIORITY
- [ ] Mixup training
  - [ ] Mix two training samples
  - [ ] Test mix ratios: [0.2, 0.4]
  - [ ] Research paper implementation

---

## 7. CURRICULUM LEARNING 📚

### 🟡 MEDIUM PRIORITY
- [ ] Progressive difficulty training
  - [ ] Phase 1 (Epochs 1-10): Short horizon (5 steps)
  - [ ] Phase 2 (Epochs 11-20): Medium horizon (10 steps)
  - [ ] Phase 3 (Epochs 21-30): Long horizon (20 steps)
  - [ ] Phase 4 (Epochs 31+): Extreme scenarios
  - [ ] Implement curriculum scheduler
  - **Files:** Create `training/curriculum.py`

### 🟢 LOW PRIORITY
- [ ] Difficulty scoring
  - [ ] Score scenarios by prediction difficulty
  - [ ] Train on easy → hard
  - [ ] Adaptive curriculum

---

## 8. MULTI-TASK LEARNING 🎯

### 🟡 MEDIUM PRIORITY
- [ ] Multi-output prediction
  - [ ] Primary: HRR prediction
  - [ ] Auxiliary: Flame height prediction
  - [ ] Auxiliary: Peak HRR time prediction
  - [ ] Auxiliary: Fire growth rate
  - [ ] Implement multi-task loss
  - [ ] Test auxiliary task weights
  - **Files:** Create `models/multitask.py`

```python
class MultiTaskFirePredictor(pl.LightningModule):
    def __init__(self):
        self.hrr_head = nn.Linear(hidden, 1)
        self.flame_height_head = nn.Linear(hidden, 1)
        self.peak_time_head = nn.Linear(hidden, 1)
    
    def loss(self, pred, target):
        loss_hrr = F.mse_loss(pred['hrr'], target['hrr'])
        loss_flame = F.mse_loss(pred['flame'], target['flame'])
        return loss_hrr + 0.3*loss_flame  # Weight auxiliary
```

---

## 9. TRANSFER LEARNING 🔄

### 🟢 LOW PRIORITY
- [ ] Pre-training strategy
  - [ ] Pre-train on temperature prediction
  - [ ] Pre-train on smoke production
  - [ ] Fine-tune on HRR prediction
  - [ ] Compare to training from scratch
  - **Files:** Create `training/pretrain.py`

- [ ] Domain adaptation
  - [ ] Train on synthetic FDS data
  - [ ] Adapt to real fire data
  - [ ] Domain adversarial training

---

## 10. UNCERTAINTY QUANTIFICATION 📊

### 🟡 MEDIUM PRIORITY
- [ ] MC Dropout for uncertainty
  - [ ] Implement `predict_with_uncertainty()`
  - [ ] Run 100 forward passes with dropout ON
  - [ ] Compute mean and std
  - [ ] Visualize prediction intervals
  - **Files:** `fire_predict.py`

```python
def predict_with_uncertainty(model, x, n_samples=100):
    model.train()  # Keep dropout active
    predictions = [model(x) for _ in range(n_samples)]
    mean = torch.stack(predictions).mean(0)
    std = torch.stack(predictions).std(0)
    return mean, std
```

### 🟢 LOW PRIORITY
- [ ] Quantile Regression
  - [ ] Predict 10th, 50th, 90th percentiles
  - [ ] Implement quantile loss
  - [ ] Calibrate prediction intervals

- [ ] Bayesian LSTM
  - [ ] Implement variational inference
  - [ ] Sample from weight posterior
  - [ ] Proper uncertainty estimates

---

## 11. EVALUATION & ANALYSIS 📈

### 🔴 HIGH PRIORITY
- [ ] Comprehensive benchmarking
  - [ ] Create `evaluate_all_models.py`
  - [ ] Test all models on same test set
  - [ ] Metrics: MAE, RMSE, MAPE, R²
  - [ ] Generate comparison table
  - [ ] Save to `results/benchmark.csv`

- [ ] Error analysis
  - [ ] Identify failure modes
  - [ ] Plot worst predictions
  - [ ] Categorize by scenario type
  - [ ] Guide next improvements

### 🟡 MEDIUM PRIORITY
- [ ] Ablation study automation
  - [ ] Test each component individually
  - [ ] Physics loss: ON/OFF
  - [ ] Heskestad features: ON/OFF
  - [ ] Attention: ON/OFF
  - [ ] Generate ablation table

- [ ] Visualization dashboard
  - [ ] Real-time prediction plots
  - [ ] Attention weight heatmaps
  - [ ] Feature importance
  - [ ] Training curves

---

## 12. DEPLOYMENT & OPTIMIZATION ⚡

### 🟡 MEDIUM PRIORITY
- [ ] Model compression
  - [ ] Quantization (FP32 → FP16)
  - [ ] Pruning (remove small weights)
  - [ ] Knowledge distillation
  - [ ] Test inference speed

- [ ] ONNX export
  - [ ] Export to ONNX format
  - [ ] Test with ONNX Runtime
  - [ ] Faster inference

### 🟢 LOW PRIORITY
- [ ] Edge deployment
  - [ ] Optimize for CPU inference
  - [ ] TensorRT optimization
  - [ ] Mobile deployment (TFLite)

---

## 📊 PROGRESS TRACKING

### Quick Wins (Do First) 🚀
- [ ] Learning rate scheduling
- [ ] Gradient clipping
- [ ] Peak detection loss
- [ ] Time derivative features
- [ ] Noise injection augmentation
- [ ] Ensemble model
- [ ] Layer normalization

### Medium-term Goals (1-2 weeks) 🎯
- [ ] Hyperparameter search
- [ ] Attention mechanism
- [ ] Multi-task learning
- [ ] MC Dropout uncertainty
- [ ] Comprehensive benchmarking

### Long-term Research (1+ month) 🔬
- [ ] GNN approach
- [ ] DTW loss
- [ ] Bayesian LSTM
- [ ] Domain adaptation
- [ ] Transfer learning

---

## 📝 NOTES & IDEAS

### Code Organization
```
fire_prediction_deployment/
├── physics_informed_lstm/
│   ├── improvements/              # NEW: All improvements here
│   │   ├── losses.py             # Custom loss functions
│   │   ├── ensemble.py           # Ensemble model
│   │   ├── augmentation.py       # Data augmentation
│   │   ├── curriculum.py         # Curriculum learning
│   │   ├── uncertainty.py        # Uncertainty quantification
│   │   └── hp_search.py          # Hyperparameter optimization
│   ├── results/                  # Benchmark results
│   └── experiments/              # Experiment logs
```

### Expected Improvements (Cumulative)
- Baseline: MAE = X
- + LR scheduling: -5%
- + Ensemble: -10%
- + Augmentation: -8%
- + Peak loss: -7%
- + Attention: -12%
- **Total potential: ~35-40% improvement**

---

## 🎯 IMMEDIATE NEXT STEPS

1. **This Week:**
   - [ ] Implement learning rate scheduling
   - [ ] Add gradient clipping
   - [ ] Create peak detection loss
   - [ ] Test ensemble model

2. **Next Week:**
   - [ ] Run hyperparameter search
   - [ ] Implement data augmentation
   - [ ] Add MC Dropout uncertainty
   - [ ] Comprehensive benchmark

3. **End of Month:**
   - [ ] Complete all HIGH priority items
   - [ ] Start MEDIUM priority items
   - [ ] Document all improvements
   - [ ] Publish results comparison

---

## 📚 REFERENCES & RESOURCES

### Papers to Read
- [ ] "Attention is All You Need" (Transformer)
- [ ] "Deep Learning for Time Series Forecasting"
- [ ] "Physics-Informed Neural Networks"
- [ ] "Uncertainty in Deep Learning"

### Libraries to Explore
- [ ] Optuna (HP optimization)
- [ ] tslearn (Time series ML)
- [ ] sktime (Time series toolkit)
- [ ] PyTorch Lightning (already using)

---

**Last Updated:** 2026-03-04  
**Status:** Ready to implement  
**Next Review:** Weekly updates

---

*Remember: Measure everything. Every change should be validated on validation set before merging!* 📊✅
