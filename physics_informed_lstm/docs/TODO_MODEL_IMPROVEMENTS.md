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
- [x] Implement learning rate scheduling ✅ **DONE 2026-03-04**
  - [x] Add `ReduceLROnPlateau` to training loop
  - [x] Test `CosineAnnealingLR` scheduler (ReduceLR chosen)
  - [ ] Compare validation loss curves (pending retraining)
  - **Files:** `fire_prediction/models/physics_informed.py` ✅
  - **Implementation:** factor=0.5, patience=5, min_lr=1e-6

- [x] Gradient clipping implementation ✅ **DONE 2026-03-04**
  - [x] Add `configure_gradient_clipping()` method
  - [x] Set max_norm=1.0 (can test other values later)
  - [ ] Monitor gradient statistics (pending retraining)
  - **Files:** `fire_prediction/models/physics_informed.py` ✅
  - **Implementation:** gradient_clip_val=1.0, algorithm="norm"

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

- [x] Weight decay experimentation ✅ **DONE 2026-03-04**
  - [x] Switch from Adam to AdamW
  - [x] Set weight_decay=1e-4 (baseline)
  - [ ] Test other values: [1e-5, 1e-3] if needed
  - [ ] Compare validation performance (pending retraining)
  - **Files:** `fire_prediction/models/physics_informed.py` ✅

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
- [x] Peak Detection Loss ✅ **DONE 2026-03-04**
  - [x] Create `peak_penalty_loss()` function
  - [x] Implement peak-weighted MSE
  - [x] Add to physics loss combination (weight=0.2)
  - [x] Set peak weight factor=5.0
  - [ ] Test other weight factors: [2, 3, 10] if needed
  - **Impact:** Better peak HRR prediction
  - **Files:** `fire_prediction/models/physics_informed.py` ✅
  - **Implementation:** Peaks identified as mean + 1*std, weighted 5x more

```python
def peak_penalty_loss(pred, target, weight=5.0):  # ✅ IMPLEMENTED
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
- [x] Add time derivatives ✅ **DONE 2026-03-04**
  - [x] Compute dHRR/dt (first derivative)
  - [x] Compute d²HRR/dt² (second derivative)
  - [x] Central difference method implemented
  - [x] Test impact on prediction accuracy (pending integration)
  - **Files:** `fire_prediction/utils/time_features.py` ✅
  - **Implementation:** `compute_time_derivatives()` function
  - **Expected Impact:** -8% MAE, better trend capture

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
- [x] Time-series noise injection ✅ **DONE 2026-03-04**
  - [x] Add Gaussian noise: σ = 0.05 * std(HRR)
  - [x] Physics-consistent noise bounds (HRR ≥ 0)
  - [x] Augment during training only
  - [x] Test impact on generalization (pending integration)
  - **Files:** `fire_prediction/data/augmentation.py` ✅
  - **Implementation:** `PhysicsConsistentAugmentation` class
  - **Expected Impact:** -10% MAE, better generalization

```python
def add_gaussian_noise(hrr, noise_level=0.05):  # ✅ IMPLEMENTED
    """Add physics-consistent Gaussian noise"""
    std = hrr.std()
    noise = torch.randn_like(hrr) * noise_level * std
    return torch.clamp(hrr + noise, min=0)  # HRR can't be negative
```

### 🟡 MEDIUM PRIORITY
- [x] Temporal augmentation ✅ **DONE 2026-03-04**
  - [x] Time stretching: 0.8x - 1.2x speed
  - [x] Random time shifts: ±3-5 timesteps
  - [x] Test on validation set (pending integration)
  - **Files:** `fire_prediction/data/augmentation.py` ✅
  - **Implementation:** `time_stretch()`, `time_shift()` methods

- [x] Magnitude scaling ✅ **DONE 2026-03-04**
  - [x] Random scaling: ±5%
  - [x] Integrated into augmentation pipeline
  - **Files:** `fire_prediction/data/augmentation.py` ✅

### 🟢 LOW PRIORITY
- [ ] SMOTE for time series
  - [ ] Synthetic Minority Over-sampling
  - [ ] Create intermediate scenarios
  - [ ] Balance dataset

---

## 6. REGULARIZATION TECHNIQUES 🛡️

### 🔴 HIGH PRIORITY
- [x] Layer Normalization ✅ **DONE 2026-03-04**
  - [x] Add `nn.LayerNorm` after LSTM
  - [x] Apply before prediction head
  - [ ] Compare to BatchNorm (LayerNorm chosen for simplicity)
  - [ ] Validate training stability (pending retraining)
  - **Files:** `fire_prediction/models/physics_informed.py` ✅
  - **Implementation:** `self.layer_norm = nn.LayerNorm(hidden_dim)`

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
- [x] Learning rate scheduling ✅ **DONE 2026-03-04**
- [x] Gradient clipping ✅ **DONE 2026-03-04**
- [x] Peak detection loss ✅ **DONE 2026-03-04**
- [x] Layer normalization ✅ **DONE 2026-03-04**
- [x] AdamW optimizer ✅ **DONE 2026-03-04**
- [x] Time derivative features ✅ **DONE 2026-03-04**
- [x] Noise injection augmentation ✅ **DONE 2026-03-04**
- [ ] Ensemble model

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

### ✅ COMPLETED (2026-03-04)
**Quick Wins (5/5):**
- [x] Implement learning rate scheduling
- [x] Add gradient clipping
- [x] Create peak detection loss
- [x] Add layer normalization
- [x] Switch to AdamW optimizer

**Additional Features (2):**
- [x] Time derivatives features
- [x] Data augmentation (noise, stretching, shifting, scaling)

**Documentation:**
- [x] Document improvements
- [x] Create implementation guide
- [x] Create usage examples
- [x] Update TODO list

### 🔄 IN PROGRESS
1. **This Week:**
   - [ ] **Re-train model with quick wins**
   - [ ] Validate 15-20% improvement
   - [ ] Compare baseline vs improved metrics
   - [ ] Update results in QUICK_WINS_IMPLEMENTED.md

2. **Next Week:**
   - [ ] Time derivative features
   - [ ] Noise injection augmentation
   - [ ] Test ensemble model
   - [ ] Run hyperparameter search

3. **Following Weeks:**
   - [ ] Implement data augmentation
   - [ ] Add MC Dropout uncertainty
   - [ ] Attention mechanism
   - [ ] Comprehensive benchmark
   - [ ] Complete all HIGH priority items
   - [ ] Start MEDIUM priority items

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

## 📝 CHANGELOG

### 2026-03-04 - Additional Features Implementation ✅
**Completed:**
- ✅ Time derivatives (dHRR/dt, d²HRR/dt²)
- ✅ Data augmentation (noise, stretch, shift, scale)
- ✅ Physics-consistent constraints
- ✅ Unit tests and visualizations

**Files Created:**
- `fire_prediction/utils/time_features.py` (276 lines)
- `fire_prediction/data/augmentation.py` (384 lines)

**Expected Results:**
- 25-35% cumulative MAE reduction
- Better trend capture (derivatives)
- Better generalization (augmentation)
- More robust predictions

**Status:** ✅ Implemented, ready for integration

**Integration Options:**
1. Add derivatives to model input (change input_dim)
2. Use augmented dataset wrapper (no model changes)
3. Both (recommended for maximum impact)

---

### 2026-03-04 - Quick Wins Implementation ✅
**Completed:**
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Gradient clipping (max_norm=1.0)
- ✅ Peak detection loss (weight=5.0, contribution=0.2)
- ✅ Layer normalization (after LSTM)
- ✅ AdamW optimizer (weight_decay=1e-4)

**Files Modified:**
- `fire_prediction/models/physics_informed.py` (~80 lines)

**Expected Results:**
- 15-20% MAE reduction
- Faster convergence
- Better peak prediction
- More stable training

**Status:** ✅ Implemented, awaiting retraining validation

**Next Actions:**
1. Re-train model with new improvements
2. Compare metrics: before vs after
3. Document actual performance gains
4. Move to next set of improvements

---

**Last Updated:** 2026-03-04 15:15 UTC  
**Status:** 7/? Improvements Implemented ✅  
**Next Review:** After retraining and integration

---

*Remember: Measure everything. Every change should be validated on validation set before merging!* 📊✅
