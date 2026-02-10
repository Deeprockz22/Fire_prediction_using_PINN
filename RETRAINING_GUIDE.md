# Retraining Guide - Avoiding Checkpoint Issues

## Common Problem

Every time you retrain the model, you might face checkpoint compatibility issues. This happens because of **inconsistencies between training and prediction code**.

## Root Causes

### 1. **Layer Name Mismatch**
- **Training script** might use: `self.fc` 
- **Prediction scripts** expect: `self.head`
- **Result**: `KeyError` or layer mismatch errors

### 2. **Checkpoint Key Differences**
Different training frameworks save checkpoints differently:
- PyTorch Lightning: `'state_dict'`
- Manual training: `'model_state_dict'`
- Direct save: No wrapper key

## ✅ FIXED Solution (As of 2026-02-10)

All scripts now use **consistent naming**:

### **Training Script** (`retrain_model.py`)
```python
# Line 62 - Uses 'head' (not 'fc')
self.head = nn.Linear(hidden_dim, output_dim * pred_horizon)

# Line 312 - Saves with 'model_state_dict'
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'val_loss': val_loss,
    ...
}, MODEL_DIR / "best_model.ckpt")
```

### **Prediction Scripts** (`predict.py`, `batch_predict.py`)
```python
# Handles multiple checkpoint formats automatically
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Handles legacy naming (fc → head)
if 'fc.weight' in state_dict and 'head.weight' not in state_dict:
    state_dict['head.weight'] = state_dict.pop('fc.weight')
    state_dict['head.bias'] = state_dict.pop('fc.bias')
```

## How to Retrain Without Issues

### **Step 1: Prepare Training Data**
```bash
# Add your FDS simulation CSV files to training_data/
cp your_simulation_hrr.csv training_data/

# Update manifest
python add_training_data.py
```

### **Step 2: Train the Model**
```bash
# Default: 30 epochs
python retrain_model.py

# Custom epochs
python retrain_model.py --epochs 50

# Custom batch size and learning rate
python retrain_model.py --epochs 100 --batch-size 64 --lr 0.0005
```

### **Step 3: Verify Checkpoint**
```bash
# Check checkpoint contents
python checkpoint_info.py

# You should see:
# - Epoch number
# - Training/validation loss
# - Configuration (input_dim=6, hidden_dim=128, etc.)
# - Timestamp
```

### **Step 4: Test Predictions**
```bash
# Test single prediction
python predict.py Input/test_11cm_mesh_hrr.csv

# Test batch prediction
python batch_predict.py
```

## Checkpoint Compatibility Checklist

Before retraining, ensure:

- [ ] `retrain_model.py` uses `self.head` (not `self.fc`)
- [ ] Checkpoint saves with `'model_state_dict'` key
- [ ] `predict.py` and `batch_predict.py` handle multiple formats
- [ ] Model architecture matches:
  - `input_dim=6`
  - `hidden_dim=128`
  - `num_layers=2`
  - `output_dim=3`
  - `pred_horizon=10`

## If You Still Get Errors

### **Error: "KeyError: 'state_dict'" or "KeyError: 'model_state_dict'"**

**Solution:** Update `predict.py` and `batch_predict.py` with robust loading:

```python
checkpoint = torch.load(str(MODEL_PATH), map_location='cpu')

# Handle different formats
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Handle legacy naming
if 'fc.weight' in state_dict and 'head.weight' not in state_dict:
    state_dict['head.weight'] = state_dict.pop('fc.weight')
    state_dict['head.bias'] = state_dict.pop('fc.bias')

model.load_state_dict(state_dict)
```

### **Error: "size mismatch for head.weight"**

**Cause:** Model architecture mismatch between training and prediction.

**Solution:** Ensure consistent configuration:

1. Check `retrain_model.py` (lines 259-265):
```python
model = PhysicsLSTM(
    input_dim=6,
    hidden_dim=128,
    num_layers=2,
    output_dim=3,      # Must match!
    pred_horizon=10
)
```

2. Check `predict.py` (lines 40-52):
```python
model = PhysicsInformedLSTM(
    input_dim=6,
    hidden_dim=128,
    num_layers=2,
    output_dim=3,      # Must match!
    pred_horizon=PRED_HORIZON,
    ...
)
```

### **Error: "missing keys" or "unexpected keys"**

**Cause:** Different model class between training and prediction.

**Solution:** Use the same model architecture. The current setup uses:
- **Training:** `PhysicsLSTM` in `retrain_model.py`
- **Prediction:** `PhysicsInformedLSTM` from `fire_prediction.models.physics_informed`

Both should have identical layer names and structure.

## Advanced: Training with PyTorch Lightning

If you use PyTorch Lightning for training (like the original model), save checkpoint as:

```python
import pytorch_lightning as pl

# During training
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='model',
    filename='best_model',
    save_top_k=1,
    monitor='val_loss',
    mode='min'
)

trainer = pl.Trainer(
    max_epochs=50,
    callbacks=[checkpoint_callback]
)

trainer.fit(model, train_loader, val_loader)

# After training, convert to compatible format
ckpt = torch.load('model/best_model.ckpt')
torch.save({
    'model_state_dict': ckpt['state_dict'],  # Note: Lightning uses 'state_dict'
    'epoch': ckpt['epoch'],
    'val_loss': ckpt['checkpoint_callback_best_model_score'],
    'timestamp': datetime.now().isoformat(),
    'config': {
        'input_dim': 6,
        'hidden_dim': 128,
        'num_layers': 2,
        'output_dim': 3,
        'pred_horizon': 10
    }
}, 'model/best_model.ckpt')
```

## Quick Reference

### **Files to Check When Retraining:**

| File | Purpose | Key Check |
|------|---------|-----------|
| `retrain_model.py` | Training script | Uses `self.head`, saves `model_state_dict` |
| `predict.py` | Single prediction | Handles multiple checkpoint formats |
| `batch_predict.py` | Batch prediction | Handles multiple checkpoint formats |
| `fire_prediction/models/physics_informed.py` | Model architecture | Layer named `self.head` |

### **Checkpoint Structure:**
```python
{
    'model_state_dict': OrderedDict(...),  # The actual weights
    'epoch': 15,
    'val_loss': 0.0819,
    'train_loss': 0.0805,
    'timestamp': '2026-02-10T11:44:31.981789',
    'config': {
        'input_dim': 6,
        'hidden_dim': 128,
        'num_layers': 2,
        'output_dim': 3,
        'pred_horizon': 10
    }
}
```

## Summary

✅ **Current Status (After Fix):**
- `retrain_model.py` uses `self.head` consistently
- Checkpoint saves with `'model_state_dict'` key
- `predict.py` and `batch_predict.py` handle any checkpoint format
- **No more checkpoint issues after retraining!**

⚠️ **If You Still Have Issues:**
1. Check layer names match (`head` not `fc`)
2. Verify checkpoint format with `checkpoint_info.py`
3. Ensure model architecture matches exactly
4. Use the robust loading code shown above

---

**Last Updated:** 2026-02-10  
**Status:** ✅ All checkpoint compatibility issues resolved  
**Tested:** 4 epochs → 15 epochs retraining (working perfectly)
