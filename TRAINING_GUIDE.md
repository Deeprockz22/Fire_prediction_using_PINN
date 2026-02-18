# 🧠 Model Training Guide

## Overview

The Fire HRR Prediction Tool now includes an **integrated training feature** that allows you to train your own model from scratch using your custom FDS simulation data.

---

## 🚀 Quick Start

### Option 1: Interactive Menu
```bash
python fire_predict.py
# Choose: 5 (Train Model)
# Follow the interactive prompts
```

### Option 2: Direct Training (Advanced)
```bash
cd fire_prediction/models
python train_physics_full.py
```

---

## 📋 Prerequisites

### Required:
- ✅ Python 3.8+
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Training data in proper format

### Recommended:
- 🎯 **100+ diverse FDS scenarios** (more is better!)
- 💻 **GPU with CUDA** (10-30x faster than CPU)
- 🧠 **8GB+ RAM** (16GB for large datasets)
- 💾 **5-10GB free disk space**

---

## 📂 Data Preparation

### 1. Folder Structure
Create the following structure:
```
training_data/
├── train/          # 70% of your data
│   ├── scenario_001_hrr.csv
│   ├── scenario_002_hrr.csv
│   └── ...
├── val/            # 15% of your data
│   ├── scenario_101_hrr.csv
│   └── ...
└── test/           # 15% of your data
    ├── scenario_201_hrr.csv
    └── ...
```

### 2. CSV Format
Each CSV file must have:
- **Time** column (in seconds)
- **HRR** column (Heat Release Rate in kW)
- At least 40 time steps
- Regular time intervals (e.g., 0.1s steps)

**Example:**
```csv
Time,HRR
0.0,0.5
0.1,2.3
0.2,5.8
0.3,12.4
...
```

### 3. Data Diversity
For best results, include scenarios with:
- ✅ Different fire sizes (10 kW - 500 kW)
- ✅ Various growth rates (slow, medium, fast)
- ✅ Multiple room geometries
- ✅ Different ventilation conditions
- ✅ Various fuel types

---

## ⚙️ Training Configuration

### Default Settings:
```python
Epochs: 50 (with early stopping)
Batch Size: 32
Learning Rate: 0.001
Sequence Length: 30 steps
Prediction Horizon: 10 steps
Architecture: 2-layer LSTM (128 units)
```

### Physics Features:
- ✅ Heskestad flame height correlation
- ✅ McCaffrey plume regions
- ✅ Ventilation flow effects
- ✅ Buoyancy scaling laws

---

## 🎯 Training Process

### What Happens:
1. **Load Data** - Reads and validates training/validation datasets
2. **Initialize Model** - Creates physics-informed LSTM architecture
3. **Training Loop** - 50 epochs with early stopping (patience=10)
4. **Checkpointing** - Saves best models during training
5. **Final Save** - Copies best model to `model/best_model.ckpt`

### Expected Time:
- **With GPU**: 20-60 minutes (depends on data size)
- **With CPU**: 1-3 hours (or more for large datasets)

### Monitor Progress:
```bash
# In another terminal:
tensorboard --logdir=logs

# Open browser to: http://localhost:6006
```

---

## 📊 Training Metrics

### During Training, You'll See:
- **train_loss** - Training loss (lower is better)
- **val_loss** - Validation loss (main metric)
- **physics_loss** - Physics consistency penalty
- **monotonic_loss** - Monotonicity constraint
- **val_mae** - Mean Absolute Error on validation

### Target Performance:
- ✅ **val_mae < 10 kW** - Good
- ✅ **val_mae < 5 kW** - Excellent
- ⚠️ **val_mae > 20 kW** - May need more/better data

---

## 🔧 Troubleshooting

### "Not enough training data"
**Solution:** Need at least 50 scenarios in train/ folder
- Generate more FDS simulations
- Use the built-in FDS generator (Menu Option 4)

### "Out of memory" error
**Solutions:**
- Reduce batch size (edit script: `BATCH_SIZE = 16`)
- Use fewer scenarios temporarily
- Close other applications
- Upgrade RAM

### Training is very slow
**Solutions:**
- Install CUDA + PyTorch GPU version
- Reduce number of epochs
- Use fewer data files for testing
- Run overnight

### High validation loss
**Solutions:**
- Add more diverse scenarios
- Check data quality (outliers, errors)
- Increase training epochs
- Adjust learning rate

### "Module not found" errors
**Solution:** 
```bash
pip install -r requirements.txt
```

---

## 💾 Output Files

### After Training:
```
checkpoints/                    # Training checkpoints
├── model-epoch=05-val_loss=0.0234.ckpt
├── model-epoch=12-val_loss=0.0198.ckpt
└── last.ckpt

model/
└── best_model.ckpt            # BEST model (auto-deployed!)

logs/
└── training/                   # TensorBoard logs
    └── version_X/
```

---

## 🎓 Advanced Tips

### 1. Hyperparameter Tuning
Edit `fire_prediction/models/train_physics_full.py`:
```python
HIDDEN_DIM = 128          # Try 64, 256
NUM_LAYERS = 2            # Try 1, 3
LEARNING_RATE = 0.001     # Try 0.0001, 0.01
LAMBDA_PHYSICS = 0.1      # Try 0.05, 0.2
```

### 2. Transfer Learning
Start from existing model:
```python
# In train script, add:
checkpoint = torch.load('model/best_model.ckpt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 3. Data Augmentation
- Add Gaussian noise to HRR data
- Time-shift sequences
- Scale HRR values (±10%)

### 4. Ensemble Models
- Train 3-5 models with different seeds
- Average predictions for better accuracy

---

## 📈 Validation

### After Training:
```bash
python fire_predict.py
# Choose: 2 (Run Example)
# Check prediction accuracy
```

### Benchmark on Test Set:
```bash
python fire_predict.py --batch
# Put test files in Input/
# Check prediction errors
```

---

## 🔄 Retraining Schedule

### When to Retrain:
- ✅ Every 50-100 new scenarios
- ✅ When prediction errors increase
- ✅ For new fire types or geometries
- ✅ When adding new physics features

### Incremental Training:
1. Keep existing `training_data/` structure
2. Add new scenarios to appropriate folders
3. Run training again
4. Compare old vs new model performance

---

## 📚 Additional Resources

- **Physics Correlations**: See `PHYSICS_CORRELATIONS.md`
- **Model Architecture**: See `fire_prediction/models/physics_informed.py`
- **Data Processing**: See `fire_prediction/data/physics_dataset.py`
- **Full Training Script**: See `fire_prediction/models/train_physics_full.py`

---

## ✅ Success Checklist

Before deploying your trained model:
- [ ] Validation MAE < 10 kW
- [ ] No overfitting (train/val loss similar)
- [ ] Tested on 10+ unseen scenarios
- [ ] Physics constraints satisfied
- [ ] Predictions are monotonic (for growth phase)
- [ ] TensorBoard shows convergence
- [ ] Model file saved to `model/best_model.ckpt`

---

## 🆘 Need Help?

1. **Check diagnostics**: 
   ```bash
   python fire_predict.py check
   ```

2. **Review training logs**: Open TensorBoard

3. **Validate data format**: Ensure CSV structure is correct

4. **Test with small dataset**: Use 10-20 files first

5. **Community**: Check documentation in `docs/` folder

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Train model | `python fire_predict.py` → Option 5 |
| View logs | `tensorboard --logdir=logs` |
| Check status | `python fire_predict.py check` |
| Test model | `python fire_predict.py --example` |
| Full training | `python fire_prediction/models/train_physics_full.py` |

---

**Happy Training! 🔥🧠**
