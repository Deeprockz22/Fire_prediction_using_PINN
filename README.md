# Fire Prediction Tool - User Guide

## Overview

This package contains a trained **Physics-Informed LSTM model** that predicts future Heat Release Rate (HRR) values from Fire Dynamics Simulator (FDS) output data.

**Capabilities:**
- Predicts next 10 time steps of HRR given 30 time steps of input
- Works with various fuels (Propane, Methane, Diesel, n-Heptane, Dodecane)
- Handles different room sizes and ventilation conditions
- Achieves 2-4% prediction error on unseen scenarios

**Model Performance:**
- Validation Test 1 (Propane, small room): 3.8% error
- Validation Test 2 (Methane, medium room): 2.31% error
- Test set average: 0.05 kW MAE

---

## Quick Start

### 1. Installation

**Requirements:**
- Python 3.8 or higher
- pip package manager

**Install dependencies:**
```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- PyTorch Lightning (training framework)
- NumPy, Pandas, Matplotlib (data processing and visualization)

---

### 2. Basic Usage

**Command:**
```bash
python predict.py path/to/your_scenario_hrr.csv
```

**Example:**
```bash
python predict.py examples/sample_scenario_hrr.csv
```

**Output:**
- Console output showing prediction accuracy
- PNG plot file saved as `your_scenario_prediction.png`
- Interactive matplotlib window with results

---

## Input File Format

The tool requires FDS output files ending with `_hrr.csv`.

**Expected format:**
```
Time,HRR
0.0,0.0
0.1,5.2
0.2,12.8
...
```

**How to generate from FDS:**
1. Run your FDS simulation
2. Look for file named `{CHID}_hrr.csv` in output directory
3. Use this file as input to the prediction tool

---

## Understanding the Output

### Console Output

```
======================================================================
PREDICTION RESULTS
======================================================================
MAE: 3.3036 kW
Relative Error: 2.31%
Peak HRR: 142.87 kW
======================================================================
```

- **MAE (Mean Absolute Error):** Average difference between prediction and actual values
- **Relative Error:** MAE as percentage of peak HRR
- **Peak HRR:** Maximum heat release rate in the scenario

### Plot Visualization

The generated plot shows:
- **Blue line (Past HRR):** Historical data for context
- **Blue thick line (Input Sequence):** Last 30 time steps used for prediction
- **Green line with circles (Actual Future):** Ground truth (if available)
- **Red dashed line with squares (Predicted Future):** Model's 10-step prediction

---

## Advanced Usage

### Save plot to specific location:
```bash
python predict.py scenario_hrr.csv --output results/my_prediction.png
```

### Batch processing multiple scenarios:

**The Easy Way:**
Use the dedicated batch script to process all files in the `Input` folder:
```bash
python batch_predict.py
```
1. Place your `_hrr.csv` files in the `Input` folder.
2. Run the script.
3. Results will be saved in the `Output` folder.

**The Manual Way (Loop):**
```bash
for file in data/*.hrr.csv; do
    python predict.py "$file"
done
```

---

## Model Details

### Architecture
- **Type:** Physics-Informed LSTM (Long Short-Term Memory)
- **Input channels:** 6
  - HRR (Heat Release Rate)
  - Q_RADI (Radiative Heat Flux) - placeholder
  - MLR (Mass Loss Rate) - placeholder
  - Flame Height (Heskestad correlation)
  - Flame Height Rate
  - Flame Height Deviation

### Training Dataset
- **Scenarios:** 221 FDS simulations
- **Fuel types:** Propane, Methane, Diesel, n-Heptane, Dodecane
- **Room sizes:** Small (2m), Medium (3m), Large (4m)
- **Behaviors:** Constant, Growth, Decay, Pulsating, Wind-affected

### Performance
- **Test MAE:** 0.05 kW
- **Validation MAE:** 3-7 kW (~2-4% relative error)
- **Inference time:** < 1 second per prediction

---

## Troubleshooting

### Error: "File not found"
- Check that the CSV file path is correct
- Ensure the file ends with `_hrr.csv`

### Error: "Not enough data points"
- The model requires at least 40 time steps (30 input + 10 prediction)
- Check your FDS simulation duration

### Error: "Module not found"
- Run `pip install -r requirements.txt`
- Ensure you're in the correct directory

### Poor prediction accuracy
- Model works best on scenarios similar to training data
- Very unusual conditions (extreme wind, very large rooms) may reduce accuracy
- Check that input data is from a valid FDS simulation

---

## File Structure

```
fire_prediction_deployment/
├── predict.py              # Main prediction script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── model/
│   └── best_model.ckpt    # Trained model weights
├── fire_prediction/       # Python package
│   ├── models/            # Model architecture
│   ├── utils/             # Physics utilities
│   └── data/              # Data processing
└── examples/
    └── sample_scenario_hrr.csv  # Example input file
```

---

## Tips for Best Results

1. **Use clean FDS data:** Ensure your simulation ran successfully
2. **Adequate timesteps:** More data = better context for prediction
3. **Similar scenarios:** Predictions work best on fires similar to training data
4. **Validate predictions:** Compare with actual data if available

---

## Citation

If you use this tool in research, please cite:
```
Physics-Informed LSTM for Fire Dynamics Prediction
Trained on 221 FDS scenarios with Heskestad flame height correlation
Validation error: 2-4% on unseen scenarios
```

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your FDS output files are formatted correctly
3. Ensure all dependencies are installed

---

## License

This tool is provided for research and educational purposes.

---

## Version History

- **v1.0** (February 2026)
  - Initial release
  - 221 training scenarios
  - 6-channel physics-informed architecture
  - MAE: 0.05 kW on test set
