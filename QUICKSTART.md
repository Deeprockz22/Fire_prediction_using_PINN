# Quick Start Guide

## Installation (One-time setup)
```bash
pip install -r requirements.txt
```

## Basic Usage
```bash
python predict.py examples/sample_scenario_hrr.csv
```

## Expected Output
- Console shows prediction accuracy (MAE, relative error)
- PNG plot is saved automatically
- Plot window appears showing prediction vs actual

## Your First Prediction
1. Open terminal/command prompt
2. Navigate to this folder
3. Run: `python predict.py examples/sample_scenario_hrr.csv`
4. View the generated plot!

## Using Your Own Data
```bash
python predict.py path/to/your_FDS_scenario_hrr.csv
```

The model predicts the next 10 time steps given 30 time steps of input.

---

**Need help?** See README.md for full documentation.
