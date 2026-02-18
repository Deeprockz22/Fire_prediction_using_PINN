# 🔥 Fire HRR Prediction Tool

## ⚡ Quick Start

```bash
python fire_predict.py
```

**That's it!** An interactive menu guides you through everything.

---

## 🎯 First Time?

1. Run: `python fire_predict.py`
2. Choose: **5** (Setup & Diagnostics)
3. Choose: **1** (Run Full Setup Wizard)
4. Wait ~3 minutes for installation
5. Choose: **2** (Run Example) from main menu
6. See prediction plot!

**You're ready!** 🎉

---

## 📖 Main Menu Options

1. **Quick Predict** - Enter your file path, get prediction
2. **Run Example** - See how it works (no setup needed!)
3. **Batch Process** - Process multiple files from Input/ folder
4. **Generate FDS File** - Create random FDS scenarios for testing
5. **Train Model** - Train new model from scratch (advanced)
6. **Manage Files** - List, open, clean folders
7. **Setup & Diagnostics** - Install, verify, troubleshoot
8. **Help & Information** - Guides, FAQ, tips
9. **Exit**

---

## 💡 Common Tasks

### Predict One File
```bash
python fire_predict.py your_simulation_hrr.csv
```

### Predict Many Files
```bash
# 1. Put CSV files in Input/ folder (use menu Option 4 → 3)
# 2. Run batch:
python fire_predict.py --batch
# 3. After processing, choose how to view results:
#    • Option 1: Open Output folder (see all plots at once)
#    • Option 2: Display plots in Python windows (one by one)
#    • Option 3: Skip viewing (view later)
```

### Check if Working
```bash
python fire_predict.py check
```

### Generate FDS Test Scenarios
```bash
python fire_predict.py
# Choose: 4 (Generate FDS File)
# Choose: 1 (Fully Random) or 2 (Custom Parameters)
# Generated .fds file saved to Input/
# Run in FDS to get CSV output for predictions!
```

### Train Your Own Model (Advanced)
```bash
python fire_predict.py
# Choose: 5 (Train Model)
# Follow the prompts
# Requirements:
#   - Training data in training_data/ folder
#   - 100+ FDS scenarios recommended
#   - GPU recommended (or 1-3 hours on CPU)
#   - 4-8GB RAM
```

**Training Data Structure:**
```
training_data/
├── train/  (CSV files for training)
├── val/    (CSV files for validation)
└── test/   (CSV files for testing)
```

---

## 🆘 Having Issues?

```bash
python fire_predict.py
# Choose: 6 (Help & Information)
# Choose: 4 (Troubleshooting Tips)
```

Or run diagnostics:
```bash
python fire_predict.py check
```

---

## 🔬 Physics-Informed Architecture

This tool uses a **Physics-Informed LSTM** that integrates multiple fire science correlations:

### Embedded Correlations:
- **Heskestad (1984)**: Flame height and growth dynamics
- **McCaffrey (1979)**: Plume region characterization  
- **Thomas (1963)**: Window/ventilation flow effects
- **Buoyancy Scaling**: Fundamental Q^(2/5) power law

### Why Physics Matters:
✅ **8.3% accuracy improvement** over baseline  
✅ **Physical consistency** - predictions obey fire laws  
✅ **Better generalization** on unseen scenarios  
✅ **Interpretable predictions** with confidence bounds  

📖 See [PHYSICS_CORRELATIONS.md](PHYSICS_CORRELATIONS.md) for technical details.

---

## 📦 What You Need

- Python 3.8 or higher
- Internet (for first-time setup only)
- FDS simulation CSV files (*_hrr.csv format)

**Everything else is automatic!**

---

## ✨ Features

✅ Interactive menu (no commands to memorize)  
✅ Automatic file management (script handles folders)  
✅ FDS scenario generator (test case creation)  
✅ Setup wizard (installs everything)  
✅ Built-in help (no external docs needed)  
✅ Batch processing (many files at once)  
✅ Self-diagnostic (checks health)  
✅ Works offline (after setup)

---

## 🎓 Learning Path

- **2 minutes:** Try the example (menu Option 2)
- **5 minutes:** Predict your first file (menu Option 1)
- **10 minutes:** Batch process multiple files (menu Option 3)

---

## 📁 File Structure

```
fire_prediction_deployment/
├── fire_predict.py       ⭐ Run this file
├── requirements.txt      📦 Dependencies
├── model/                💾 Trained model
├── fire_prediction/      🧠 Core code
├── examples/             📚 Sample data
├── Input/                📥 Put your files here
└── Output/               📤 Results appear here
```

---

## 🚀 That's All You Need to Know!

Run the script, use the menu, get predictions. Simple! 🎉

For command-line reference: `python fire_predict.py --help`
