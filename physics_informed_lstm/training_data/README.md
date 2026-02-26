# Training Data Folder

Place your FDS `_hrr.csv` files in this folder to add them to the training dataset.

## Expected File Format

Files should be named `{scenario_name}_hrr.csv` and contain:
```csv
Time,HRR
0.0,0.0
0.1,5.2
0.2,12.8
...
```

## How to Add New Training Data

1. Copy your `_hrr.csv` files into this folder.
2. Run: `python add_training_data.py`
3. The script will process and integrate the new data.

## How to Retrain the Model

After adding new data:
```bash
python retrain_model.py
```

This will create a new model in the `model/` folder.
