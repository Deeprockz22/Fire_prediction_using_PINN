# Batch Prediction Guide

## Quick Start

1. **Place your HRR files in the Input folder:**
   ```
   Input/
   ├── scenario1_hrr.csv
   ├── scenario2_hrr.csv
   └── scenario3_hrr.csv
   ```

2. **Run the batch script:**
   ```bash
   python batch_predict.py
   ```

3. **Check the Output folder for results:**
   ```
   Output/
   ├── scenario1_hrr_prediction.png
   ├── scenario2_hrr_prediction.png
   ├── scenario3_hrr_prediction.png
   └── prediction_summary_20260208_053000.txt
   ```

## What It Does

- Automatically finds all `.csv` files in `Input/` folder
- Runs predictions on each file
- Saves plots to `Output/` folder
- Generates a summary report with all results

## Output Files

**For each input:**
- `{filename}_prediction.png` - Visualization plot

**Summary report:**
- `prediction_summary_{timestamp}.txt` - Contains:
  - MAE for each file
  - Relative error percentages
  - Average statistics
  - Peak HRR values

## Error Handling

- **No files found:** Add CSV files to Input/ folder
- **Insufficient data:** File needs at least 40 time steps
- **Processing errors:** Skips problematic files and continues

## Example Output

```
======================================================================
BATCH FIRE PREDICTION
Processing all files in Input folder
======================================================================

✅ Found 3 CSV file(s)

[1/3] Processing: test1_hrr.csv
  ✅ MAE: 3.30 kW (2.31%)
  📊 Plot: test1_hrr_prediction.png

[2/3] Processing: test2_hrr.csv
  ✅ MAE: 7.16 kW (3.82%)
  📊 Plot: test2_hrr_prediction.png

======================================================================
SUMMARY
======================================================================
Total files processed: 2
Successful predictions: 2
Average MAE: 5.23 kW
Average Relative Error: 3.07%
```
