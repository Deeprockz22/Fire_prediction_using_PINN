# Model Checkpoint Documentation

## Checkpoint Status: ✅ READY TO USE

The trained model checkpoint is **already included** in the repository and ready for predictions.

## Checkpoint Details

### Location
```
model/best_model.ckpt
```

### Checkpoint Information
- **File Size**: 806.91 KB (826,274 bytes)
- **Total Parameters**: 205,598 parameters
- **Training Epoch**: 4
- **Training Loss**: 0.0711
- **Validation Loss**: 0.0708
- **Created**: 2026-02-08 06:30:00

### Model Configuration
```python
{
  "input_dim": 6,        # 6 input channels (HRR, Q_RADI, MLR + 3 physics features)
  "hidden_dim": 128,     # LSTM hidden dimension
  "num_layers": 2,       # 2 LSTM layers
  "output_dim": 3,       # 3 output channels (HRR, Q_RADI, MLR)
  "pred_horizon": 10     # Predict 10 time steps ahead
}
```

### Model Architecture
- **Type**: Physics-Informed LSTM
- **Input Features**: 
  - Channel 1-3: HRR, Q_RADI, MLR (raw FDS outputs)
  - Channel 4-6: Flame Height, dFlame_Height/dt, Flame_Height_Deviation (Heskestad physics)
- **Output**: 30 values (3 channels × 10 time steps)
- **Physics Constraints**: Heskestad correlation, monotonicity validation

## Usage

### Quick Start
```bash
# Run prediction on any HRR data file
python predict.py Input/your_data_hrr.csv

# Specify custom output location
python predict.py Input/data.csv --output Output/result.png
```

### Batch Processing
```bash
# Process multiple files
python batch_predict.py
```

## Verification

### Test the Checkpoint
```bash
# Inspect checkpoint contents
python inspect_ckpt.py

# View detailed checkpoint info
python checkpoint_info.py

# Run test prediction
python predict.py Input/EXTREME_TEST_5719_hrr.csv
```

### Expected Results
With the included test data, you should see:
- Model loads successfully ✅
- MAE (Mean Absolute Error): ~20-70 kW depending on fire scenario
- Relative Error: ~10-30%
- Prediction plot generated

## Checkpoint Contents

The checkpoint file contains:
1. **model_state_dict**: Trained neural network weights
2. **epoch**: Training epoch when saved
3. **train_loss**: Final training loss
4. **val_loss**: Final validation loss
5. **timestamp**: When the model was saved
6. **config**: Model hyperparameters

## Troubleshooting

### If checkpoint is missing or corrupted:
```bash
# Check if checkpoint exists
ls model/best_model.ckpt

# Verify checkpoint integrity
python inspect_ckpt.py
```

### If you need to retrain:
```bash
# Retrain from scratch (requires training data)
python retrain_model.py
```

### If you see "FileNotFoundError":
The checkpoint must be located at `model/best_model.ckpt`. Ensure:
1. The `model/` directory exists
2. The checkpoint file is named exactly `best_model.ckpt`
3. You're running from the project root directory

## Performance Benchmarks

### Test Results
| Test Case | MAE (kW) | Relative Error | Peak HRR (kW) |
|-----------|----------|----------------|---------------|
| 11cm Mesh Simulation | 22.87 | 10.73% | 213.16 |
| Extreme Test 5719 | 64.76 | 28.72% | 225.48 |

### Hardware Requirements
- **CPU**: Any modern CPU (tested on Intel/AMD)
- **RAM**: ~2 GB minimum
- **Storage**: 1 MB for checkpoint
- **GPU**: Optional (CPU inference is fast enough)

## Notes

✅ **The checkpoint is already included** - no additional setup needed
✅ **Trained on FDS fire simulation data** - optimized for fire prediction
✅ **Physics-informed** - includes Heskestad flame height correlation
✅ **Production-ready** - validated on multiple test cases

## Related Files
- `predict.py` - Main prediction script
- `inspect_ckpt.py` - Checkpoint inspection tool
- `checkpoint_info.py` - Detailed checkpoint information
- `retrain_model.py` - Retrain the model (optional)
- `11CM_MESH_TEST_RESULTS.md` - Test results documentation
