"""
Add Training Data Script
========================
Processes new HRR CSV files and adds them to the training dataset.

Usage:
    python add_training_data.py [path_to_csv_or_folder]

If no path is provided, processes all CSVs in the training_data/ folder.
"""

import os
import sys
import json
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
TRAINING_DATA_DIR = SCRIPT_DIR / "training_data"
MANIFEST_FILE = TRAINING_DATA_DIR / "manifest.json"


def load_manifest():
    """Load or create the training data manifest."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {"files": [], "last_updated": None, "total_scenarios": 0}


def save_manifest(manifest):
    """Save the manifest file."""
    manifest["last_updated"] = datetime.now().isoformat()
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def validate_csv(filepath):
    """Validate that a CSV file has the expected format."""
    try:
        df = pd.read_csv(filepath)
        
        # Check for required columns (case-insensitive)
        columns_lower = [c.lower() for c in df.columns]
        
        has_time = 'time' in columns_lower or 's' in columns_lower
        has_hrr = 'hrr' in columns_lower or 'kw' in columns_lower
        
        if not has_time:
            return False, "Missing 'Time' column"
        if not has_hrr:
            return False, "Missing 'HRR' column"
        if len(df) < 40:
            return False, f"Not enough data points ({len(df)} < 40 required)"
            
        return True, f"Valid: {len(df)} time steps"
    except Exception as e:
        return False, str(e)


def add_file(source_path, manifest):
    """Add a single CSV file to the training data."""
    source = Path(source_path)
    
    if not source.exists():
        print(f"  ❌ File not found: {source}")
        return False
        
    if not source.name.endswith("_hrr.csv"):
        print(f"  ⚠️  Skipping (not _hrr.csv): {source.name}")
        return False
    
    # Validate
    valid, message = validate_csv(source)
    if not valid:
        print(f"  ❌ Invalid: {source.name} - {message}")
        return False
    
    # Copy to training_data folder if not already there
    dest = TRAINING_DATA_DIR / source.name
    if source.parent != TRAINING_DATA_DIR:
        shutil.copy2(source, dest)
        print(f"  📁 Copied: {source.name}")
    
    # Add to manifest if not already present
    if source.name not in manifest["files"]:
        manifest["files"].append(source.name)
        manifest["total_scenarios"] = len(manifest["files"])
        print(f"  ✅ Added: {source.name} ({message})")
        return True
    else:
        print(f"  ℹ️  Already in manifest: {source.name}")
        return False


def process_folder(folder_path, manifest):
    """Process all CSV files in a folder."""
    folder = Path(folder_path)
    csv_files = list(folder.glob("*_hrr.csv"))
    
    if not csv_files:
        print(f"  ⚠️  No _hrr.csv files found in {folder}")
        return 0
    
    added = 0
    for csv_file in csv_files:
        if add_file(csv_file, manifest):
            added += 1
    
    return added


def main():
    print("=" * 60)
    print("ADD TRAINING DATA")
    print("=" * 60)
    
    # Ensure training_data folder exists
    TRAINING_DATA_DIR.mkdir(exist_ok=True)
    
    # Load manifest
    manifest = load_manifest()
    print(f"\n📊 Current training data: {manifest['total_scenarios']} scenarios")
    
    # Determine source
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
        if source.is_file():
            print(f"\n📂 Processing file: {source}")
            add_file(source, manifest)
        elif source.is_dir():
            print(f"\n📂 Processing folder: {source}")
            added = process_folder(source, manifest)
            print(f"\n✅ Added {added} new file(s)")
    else:
        # Process training_data folder
        print(f"\n📂 Processing: {TRAINING_DATA_DIR}")
        added = process_folder(TRAINING_DATA_DIR, manifest)
        print(f"\n✅ Processed {added} new file(s)")
    
    # Save manifest
    save_manifest(manifest)
    print(f"\n📊 Total training scenarios: {manifest['total_scenarios']}")
    print(f"📝 Manifest saved: {MANIFEST_FILE}")
    
    print("\n" + "=" * 60)
    print("Next step: Run 'python retrain_model.py' to retrain the model")
    print("=" * 60)


if __name__ == "__main__":
    main()
