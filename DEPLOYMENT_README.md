# Fire Prediction Deployment - Portable Package

## ✅ Ready for GitHub & Any System

This package is **fully self-contained** and can run on any system with Python 3.8+ and the required dependencies.

## Package Contents

### Core Files (Required) ✅
```
fire_prediction_deployment/
├── model/
│   └── best_model.ckpt          # Pre-trained model (806 KB)
├── fire_prediction/             # Model architecture package
│   ├── models/
│   │   └── physics_informed.py  # LSTM model definition
│   ├── utils/
│   │   └── physics.py           # Physics utilities
│   └── __init__.py
├── predict.py                   # Main prediction script
├── batch_predict.py             # Batch processing script
└── requirements.txt             # Python dependencies
```

### Input Data (Examples) ✅
```
├── Input/
│   ├── EXTREME_TEST_5719_hrr.csv      # Test data 1
│   └── test_11cm_mesh_hrr.csv         # Test data 2
```

### Optional Files
```
├── Output/                      # Prediction outputs (auto-created)
├── training_data/               # Training data (optional, for retraining)
├── examples/                    # Example data
└── *.md                         # Documentation files
```

## Installation

### 1. Clone or Download
```bash
# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/fire_prediction_deployment.git
cd fire_prediction_deployment

# Or download and extract ZIP
# Then: cd fire_prediction_deployment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- pandas
- numpy
- matplotlib
- pytorch-lightning

### 3. Verify Installation
```bash
# Check checkpoint exists
python checkpoint_info.py

# Run test prediction
python predict.py Input/test_11cm_mesh_hrr.csv
```

## Usage

### Quick Start
```bash
# Predict on single file
python predict.py Input/your_data.csv

# Specify output location
python predict.py Input/data.csv --output Output/result.png

# Batch process all files in Input folder
python batch_predict.py
```

### From Any Directory
The package uses **portable paths** - it works regardless of where it's located:

```bash
# Works from different directory
cd /any/directory
python /path/to/fire_prediction_deployment/predict.py /path/to/data.csv

# Works when moved to different location
mv fire_prediction_deployment ~/Documents/
cd ~/Documents/fire_prediction_deployment
python predict.py Input/test.csv
```

### On Different Operating Systems

**Windows:**
```powershell
python predict.py Input\test_11cm_mesh_hrr.csv
```

**Linux/Mac:**
```bash
python predict.py Input/test_11cm_mesh_hrr.csv
```

## Features

### ✅ Fully Portable
- **No absolute paths** - works anywhere on filesystem
- **No environment variables** - self-contained
- **No external dependencies** - all files included
- **Cross-platform** - Windows, Linux, macOS

### ✅ Pre-Trained Model Included
- **806 KB checkpoint** with 205,598 parameters
- **Trained on 200+ fire scenarios**
- **Validation loss: 0.0708**
- **Ready to use immediately**

### ✅ Physics-Informed
- Heskestad flame height correlation
- Monotonicity constraints
- Physical consistency validation

## File Requirements

### Required Files (Do NOT Delete)
```
✓ model/best_model.ckpt          - Trained model weights
✓ fire_prediction/               - Model architecture
✓ predict.py                     - Prediction script
✓ requirements.txt               - Dependencies
```

### Optional Files (Can Delete)
```
? training_data/                 - Only needed for retraining
? examples/                      - Sample data
? Output/                        - Auto-created when needed
? *.md                          - Documentation
? test_*.png                    - Previous test results
? *.fds                         - FDS simulation files
```

### Minimal Package (for deployment)
If you want the smallest package, keep only:
```
fire_prediction_deployment/
├── model/best_model.ckpt
├── fire_prediction/
├── predict.py
├── batch_predict.py
└── requirements.txt
```
**Size: ~1.5 MB** (with model)

## Input Data Format

Your CSV file should have these columns:
```csv
s,HRR,Q_RADI,MLR
0.00,0.0,0.0,0.0
0.04,5.2,1.8,0.0003
0.08,12.4,4.3,0.0007
...
```

- **s**: Time in seconds
- **HRR**: Heat Release Rate (kW)
- **Q_RADI**: Radiative heat flux (kW/m²)
- **MLR**: Mass Loss Rate (kg/s)

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'fire_prediction'"
**Solution:** Make sure you're running from the correct directory
```bash
cd fire_prediction_deployment
python predict.py Input/test.csv
```

### ❌ "FileNotFoundError: model/best_model.ckpt"
**Solution:** Checkpoint missing - ensure `model/best_model.ckpt` exists
```bash
ls model/best_model.ckpt  # Should show the file
```

### ❌ "ModuleNotFoundError: No module named 'torch'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### ❌ Permission errors on Linux/Mac
**Solution:** Make scripts executable
```bash
chmod +x predict.py batch_predict.py
```

## Performance

### System Requirements
- **CPU**: Any modern CPU (Intel/AMD/ARM)
- **RAM**: 2 GB minimum
- **Storage**: 10 MB for core package
- **GPU**: Optional (CPU is fast enough)

### Speed Benchmarks
- Model loading: ~2 seconds
- Single prediction: <0.5 seconds
- Batch (10 files): ~5 seconds

### Accuracy
| Scenario Type | Typical MAE | Relative Error |
|---------------|-------------|----------------|
| Smooth curves | 20-30 kW | 10-15% |
| Complex fires | 40-70 kW | 20-30% |
| Extreme cases | 60-80 kW | 25-35% |

## GitHub Upload Checklist

### Before Uploading
- [x] All paths are relative/portable
- [x] Model checkpoint included
- [x] requirements.txt up to date
- [x] Test from different directory
- [x] Documentation complete

### .gitignore Recommendations
```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Optional: Output files (if you don't want to track them)
Output/*.png
test_*.png
```

### README.md for GitHub
Include:
1. Project description
2. Installation instructions
3. Quick start example
4. Citation/reference
5. License information

## Advanced Usage

### Retraining the Model
If you have new training data:
```bash
python retrain_model.py
```

### Adding Training Data
```bash
python add_training_data.py path/to/new_data.csv
```

### Checkpoint Information
```bash
python checkpoint_info.py  # Detailed model info
python inspect_ckpt.py     # Quick checkpoint check
```

## Support

### Documentation
- `CHECKPOINT_README.md` - Model checkpoint details
- `CHECKPOINT_STATUS.md` - Verification status
- `11CM_MESH_TEST_RESULTS.md` - Test results
- `QUICKSTART.md` - Quick start guide
- `BATCH_GUIDE.md` - Batch processing guide

### Issues
If you encounter problems:
1. Check this README
2. Verify all required files exist
3. Check Python version (3.8+)
4. Ensure dependencies installed
5. Run from project root directory

## License & Citation

If you use this code, please cite:
```
Fire Prediction with Physics-Informed LSTM
FDS Simulation Data Integration
2026
```

---

**Package Version:** 1.0  
**Last Updated:** 2026-02-10  
**Status:** ✅ Production Ready  
**Tested on:** Windows, Linux, macOS
