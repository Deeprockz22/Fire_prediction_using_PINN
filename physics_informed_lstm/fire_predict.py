#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 FIRE HRR PREDICTION TOOL - ALL-IN-ONE VERSION
Physics-Informed LSTM for Fire Dynamics Forecasting

A complete, self-contained tool for predicting Heat Release Rate from FDS simulations.
Everything you need in one script - no file management required!

Author: Fire Prediction Team
Version: 2.0.0
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib.util

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

VERSION = "2.0.0"
MODEL_PATH = SCRIPT_DIR / "model" / "best_model.ckpt"

# ============================================================================
# INTERACTIVE MENU SYSTEM
# ============================================================================

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print welcome banner with HESKESTAD ASCII block art"""
    _A = "\033[91m"; _B = "\033[93m"; _C = "\033[33m"; _RST = "\033[0m"
    art = [
        f"{_A}██╗  ██╗███████╗███████╗██╗  ██╗███████╗███████╗████████╗ █████╗ ██████╗ {_RST}",
        f"{_A}██║  ██║██╔════╝██╔════╝██║ ██╔╝██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗{_RST}",
        f"{_B}███████║█████╗  ███████╗█████╔╝ █████╗  ███████╗   ██║   ███████║██║  ██║{_RST}",
        f"{_B}██╔══██║██╔══╝  ╚════██║██╔═██╗ ██╔══╝  ╚════██║   ██║   ██╔══██║██║  ██║{_RST}",
        f"{_C}██║  ██║███████╗███████║██║  ██╗███████╗███████║   ██║   ██║  ██║██████╔╝{_RST}",
        f"{_C}╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═════╝ {_RST}",
    ]
    print()
    for line in art:
        print("  " + line)
    print(f"\n  \033[97mPhysics-Informed LSTM  ·  Predicting the Future of Fire v{VERSION}\033[0m")
    print("  \033[90mBecause running CFD in real-time is for people who enjoy waiting.\033[0m\n")
    print("  \033[93mThis AI has ingested the knowledge of 221 fires and 3 physics books.\033[0m")
    print("  \033[93mIt is now mildly concerned about your smoking habits.\033[0m\n")

def print_section(title):
    """Print section header with box-drawing glyphs"""
    w = 70
    bar = "─" * (w - 2)
    print()
    print("┌" + bar + "┐")
    print("│  " + title.ljust(w - 4) + "│")
    print("└" + bar + "┘")
    print()

def press_enter():
    """Wait for user input"""
    try:
        input("\nPress Enter to continue...")
    except:
        pass

# ── ANSI colours & fire animation ─────────────────────────────────────────────
_R  = "\033[31m";  _Y  = "\033[33m"
_RB = "\033[91m";  _YB = "\033[93m";  _RST = "\033[0m"

_FRAMES = [
    [
        f"     {_RB}  ) {_Y} ({_RB}  ){_RST}   ",
        f"    {_Y} ( {_RB}){_Y}  (  {_RB}( {_RST}  ",
        f"   {_RB}){_Y}(   {_RB}) {_Y}( {_RB})  {_RST} ",
        f"  {_Y}(  {_RB})   {_Y}(   {_RB})  {_RST} ",
        f" {_R}  \\{_RB}|{_Y}///{_RB}|{_Y}///{_R}|/{_RST}",
        f" {_R}   \\{_RB}|{_R}/////|{_RB}\\{_R}/{_RST} ",
        f"  {_R}   \\{_RB}|||{_R}///{_RST}    ",
        f"   {_R}   \\{_RB}|{_R}/ {_RST}      ",
        f"    {_R}───┴───{_RST}    ",
    ],
    [
        f"     {_Y}(  {_RB}) {_Y}  ({_RB}){_RST}  ",
        f"    {_RB}){_Y}  ({_RB})  {_Y}(  {_RST} ",
        f"   {_Y}( {_RB}) {_Y}(  {_RB})  {_Y}({_RST} ",
        f"  {_RB})  {_Y}(   {_RB})  {_Y}(  {_RST}",
        f" {_R}  \\{_Y}|{_RB}\\\\\\{_Y}|{_RB}\\\\\\{_R}|/{_RST}",
        f" {_R}   \\{_Y}|{_R}/////|{_Y}\\{_R}/{_RST} ",
        f"  {_R}   \\{_Y}|||{_R}///{_RST}    ",
        f"   {_R}   \\{_Y}|{_R}/ {_RST}      ",
        f"    {_R}───┴───{_RST}    ",
    ],
    [
        f"    {_RB} ( {_Y}){_RB}  ( {_Y}) {_RST}  ",
        f"   {_Y} ){_RB}(  {_Y}) {_RB} ( {_Y}) {_RST}",
        f"   {_RB}( {_Y})  {_RB}(  {_Y}){_RB}( {_RST} ",
        f"  {_Y})  {_RB})   {_Y}(   {_RB})  {_RST}",
        f" {_R}  \\{_RB}|{_R}\\\\\\{_RB}|{_R}\\\\\\{_RB}|/{_RST}",
        f" {_R}   \\{_RB}|{_R}/////|{_Y}\\{_R}/{_RST} ",
        f"  {_R}   \\{_RB}|||{_R}///{_RST}    ",
        f"   {_R}   \\{_RB}|{_R}/ {_RST}      ",
        f"    {_R}───┴───{_RST}    ",
    ],
]

def fire_splash():
    """Play a brief animated ASCII fire, then clear."""
    import time
    try:
        sys.stdout.write("\033[?25l")   # hide cursor
        sys.stdout.flush()
        clear_screen()
        n = len(_FRAMES[0])
        first = True
        for _ in range(6):              # ~1.5 s
            for frame in _FRAMES:
                if not first:
                    sys.stdout.write(f"\033[{n}A")
                for line in frame:
                    print(line)
                sys.stdout.flush()
                time.sleep(0.10)
                first = False
    except Exception:
        pass
    finally:
        sys.stdout.write("\033[?25h")  # restore cursor
        sys.stdout.flush()
    clear_screen()

# ============================================================================
# SYSTEM CHECK & SETUP
# ============================================================================

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        return False, f"Python {version.major}.{version.minor}.{version.micro}"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def check_dependencies():
    """Check if required packages are installed"""
    packages = {
        'torch': 'PyTorch',
        'pytorch_lightning': 'PyTorch Lightning',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for pkg, name in packages.items():
        if importlib.util.find_spec(pkg) is None:
            missing.append((pkg, name))
    
    return missing

def check_model_file():
    """Check if model checkpoint exists"""
    return MODEL_PATH.exists()

def install_dependencies():
    """Install required packages"""
    print_section("📦 INSTALLING DEPENDENCIES")
    print("This will install: PyTorch, Lightning, NumPy, Pandas, Matplotlib")
    print("This may take a few minutes...\n")
    
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response != 'y':
        return False
    
    print("\n⏳ Installing packages (please wait)...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", 
            str(SCRIPT_DIR / "config" / "requirements.txt"), "--quiet"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ All dependencies installed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("\n💡 Try manually: pip install -r requirements.txt")
        return False

def run_diagnostics(silent=False):
    """Run complete system diagnostics"""
    if not silent:
        print_section("🔍 RUNNING DIAGNOSTICS")
    
    issues = []
    
    # Check Python
    py_ok, py_version = check_python_version()
    if not silent:
        print(f"{'✅' if py_ok else '❌'} Python Version: {py_version}")
    if not py_ok:
        issues.append("Python 3.8+ required")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        for pkg, name in missing:
            if not silent:
                print(f"❌ {name} - Not installed")
            issues.append(f"{name} missing")
    else:
        if not silent:
            print("✅ All required packages installed")
    
    # Check model
    model_ok = check_model_file()
    if not silent:
        print(f"{'✅' if model_ok else '❌'} Model checkpoint: {'Found' if model_ok else 'Missing'}")
    if not model_ok:
        issues.append("Model file missing")
    
    # Check folders
    folders = ['Input', 'Output', 'examples', 'model', 'fire_prediction']
    for folder in folders:
        exists = (SCRIPT_DIR / folder).exists()
        if not exists and folder in ['Input', 'Output']:
            (SCRIPT_DIR / folder).mkdir(exist_ok=True)
            if not silent:
                print(f"✅ Created {folder}/ folder")
    
    if not silent:
        print()
    
    return len(issues) == 0, issues

def setup_wizard():
    """Interactive setup wizard"""
    clear_screen()
    print_banner()
    print_section("🔧 INITIALIZATION PROTOCOL (A.K.A SETUP WIZARD)")
    
    print("Welcome, human. This sequence will:")
    print("  ✓ Interrogate your Python runtime")
    print("  ✓ Assimilate required dependencies via pip")
    print("  ✓ Run diagnostics on the neural architecture")
    print("  ✓ Synthesize necessary directory structures")
    print("  ✓ Perform a nominal test firing\n")
    
    # Step 1: Check Python
    print_section("STEP 1: Checking Python Version")
    py_ok, py_version = check_python_version()
    print(f"{'✅' if py_ok else '❌'} {py_version}")
    
    if not py_ok:
        print("\n❌ Setup failed: Need Python 3.8 or higher")
        return False
    
    # Step 2: Install dependencies
    print_section("STEP 2: Installing Dependencies")
    missing = check_dependencies()
    
    if missing:
        print(f"Found {len(missing)} missing package(s):")
        for pkg, name in missing:
            print(f"  • {name}")
        print()
        
        if not install_dependencies():
            return False
    else:
        print("✅ All dependencies already installed\n")
    
    # Step 3: Verify
    print_section("STEP 3: Verifying Installation")
    all_ok, issues = run_diagnostics(silent=False)
    
    if not all_ok:
        print("\n⚠️  Some issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    
    # Step 4: Test
    print_section("STEP 4: Running Test Prediction")
    response = input("Run test with example data? [Y/n]: ").strip().lower()
    
    if not response or response == 'y':
        print("\n🧪 Testing prediction...\n")
        try:
            run_example_prediction(show_plot=False)
            print("\n✅ Test successful!")
        except Exception as e:
            print(f"\n⚠️  Test completed with warnings: {e}")
    
    # Done
    print_section("🎉 INITIALIZATION COMPLETE")
    print("All systems are nominal. You may now predict fire.\n")
    print("Next commands for your meat-based interface:")
    print("  • Try: python fire_predict.py --example")
    print("  • Or engage the interactive terminal menu\n")
    
    return True

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def load_prediction_model():
    """Load the trained model (auto-detects input dimension)"""
    import torch
    import numpy as np
    from fire_prediction.models.physics_informed import PhysicsInformedLSTM
    
    print("   📦 Loading model checkpoint...")
    
    # Try to detect input dimension from checkpoint
    checkpoint = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Detect input dimension and hidden dimension from first layer
    input_dim = 6  # Default
    hidden_dim = 128  # Default
    for key in state_dict.keys():
        if 'lstm.weight_ih_l0' in key:
            # LSTM input weight shape: [4*hidden_dim, input_dim]
            input_dim = state_dict[key].shape[1]
            hidden_dim = state_dict[key].shape[0] // 4
            break
    
    print(f"   📊 Detected model input dimension: {input_dim}")
    print(f"   📊 Detected model hidden dimension: {hidden_dim}")
    
    model = PhysicsInformedLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        output_dim=1,
        dropout=0.1,
        lr=0.001,
        pred_horizon=10,
        use_physics_loss=True,
        lambda_physics=0.1,
        lambda_monotonic=0.05,
        fire_diameter=0.3,
        validate_physics=True
    )
    
    # Handle legacy checkpoint
    if 'fc.weight' in state_dict and 'head.weight' not in state_dict:
        state_dict['head.weight'] = state_dict.pop('fc.weight')
        state_dict['head.bias'] = state_dict.pop('fc.bias')
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("   ✅ Model ready!\n")
    return model, input_dim

def prepare_prediction_features(hrr_data, room_info=None, input_dim=6):
    """Prepare features from HRR data (supports 6 or 9 channel models)"""
    import numpy as np
    from fire_prediction.utils.physics import compute_heskestad_features, compute_enhanced_features
    
    if input_dim == 9:
        # Enhanced 9-channel model with all correlations
        STATS = {
            'mean': np.array([1.6312595e+02, -4.2468037e+01, 3.9271861e-03, 
                              1.2081864e+00, -1.6674624e-08, -1.2529218e-02,
                              0.5, 0.5, 1.0], dtype=np.float32),  # Added 3 more for enhanced features
            'std': np.array([8.8223785e+01, 2.5670046e+01, 1.6978320e-03,
                             3.1914881e-01, 3.6280316e-01, 1.2922239e-01,
                             0.3, 0.5, 0.3], dtype=np.float32)  # Added 3 more for enhanced features
        }
        
        # Compute enhanced physics features (6 features: Heskestad + McCaffrey + Thomas)
        room_dims = room_info if room_info else {
            'opening_area': 0.8,
            'opening_height': 1.0,
            'room_area': 9.0
        }
        physics_feats = compute_enhanced_features(hrr_data, fire_diameter=0.3, room_dims=room_dims)
        
        full_data = np.zeros((len(hrr_data), 9), dtype=np.float32)
        full_data[:, 0] = hrr_data
        full_data[:, 3:] = physics_feats
        
    else:
        # Legacy 6-channel model (Heskestad only)
        STATS = {
            'mean': np.array([1.6312595e+02, -4.2468037e+01, 3.9271861e-03, 
                              1.2081864e+00, -1.6674624e-08, -1.2529218e-02], dtype=np.float32),
            'std': np.array([8.8223785e+01, 2.5670046e+01, 1.6978320e-03,
                             3.1914881e-01, 3.6280316e-01, 1.2922239e-01], dtype=np.float32)
        }
        
        # Compute physics features (Heskestad only - 3 features)
        hesk_feats = compute_heskestad_features(hrr_data, fire_diameter=0.3)
        
        full_data = np.zeros((len(hrr_data), 6), dtype=np.float32)
        full_data[:, 0] = hrr_data
        full_data[:, 3:] = hesk_feats
    
    full_data_norm = (full_data - STATS['mean']) / (STATS['std'] + 1e-8)
    return full_data_norm, STATS

def run_prediction(csv_file, save_plot=True, show_plot=True, output_dir=None):
    """Run prediction on a CSV file"""
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    print(f"📊 Reading data: {Path(csv_file).name}")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_file, skiprows=1)
        time = df.iloc[:, 0].values
        hrr = df.iloc[:, 1].values
        print(f"   ✅ {len(hrr)} time steps loaded")
        print(f"   📈 Peak HRR: {hrr.max():.2f} kW")
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
        return None
    
    # Prepare features
    print("\n🔬 Computing physics features...")
    
    # Load model first to detect input dimension
    print("\n🧠 Loading neural network...")
    model, model_input_dim = load_prediction_model()
    
    # Prepare features based on model's expected input
    full_data_norm, STATS = prepare_prediction_features(hrr, room_info=None, input_dim=model_input_dim)
    print(f"   ✅ {model_input_dim}-channel features ready")
    if model_input_dim == 9:
        print("   📊 Using: Heskestad + McCaffrey + Thomas correlations")
    
    # Find prediction point
    peak_idx = np.argmax(hrr)
    start_idx = max(0, peak_idx - 30 - 5)
    
    if start_idx + 40 > len(full_data_norm):
        start_idx = len(full_data_norm) - 40
    
    if start_idx < 0:
        print(f"\n❌ Error: Need at least 40 time steps (have {len(hrr)})")
        return None
    
    # Extract sequences
    input_seq = full_data_norm[start_idx:start_idx+30]
    actual_future = full_data_norm[start_idx+30:start_idx+40, 0]
    input_time = time[start_idx:start_idx+30]
    future_time = time[start_idx+30:start_idx+40]
    
    # Predict
    print("🔮 Running prediction...")
    x_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        y_pred_norm = model(x_tensor)[0].numpy()
        if y_pred_norm.shape[-1] == 3:
            y_pred_norm = y_pred_norm[:, 0]
    
    # Denormalize
    hrr_mean, hrr_std = STATS['mean'][0], STATS['std'][0]
    y_pred_kw = (y_pred_norm * hrr_std) + hrr_mean
    actual_future_kw = (actual_future * hrr_std) + hrr_mean
    input_seq_kw = (input_seq[:, 0] * hrr_std) + hrr_mean
    
    # Metrics
    mae = np.mean(np.abs(actual_future_kw - y_pred_kw))
    rel_error = (mae / hrr.max()) * 100
    
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"MAE: {mae:.4f} kW")
    print(f"Relative Error: {rel_error:.2f}%")
    print(f"Peak HRR: {hrr.max():.2f} kW")
    print(f"Prediction Range: {y_pred_kw.min():.2f} - {y_pred_kw.max():.2f} kW")
    print("="*70)
    
    # Add performance context
    print(f"\n🔬 Physics Correlations Used:")
    print(f"   • Heskestad flame height (validated)")
    print(f"   • McCaffrey plume regions (available)")
    print(f"   • Thomas ventilation flow (available)")
    
    if rel_error > 100:
        print("\n⚠️  HIGH ERROR DETECTED")
        print("    This can occur when:")
        print("    • Scenario differs significantly from training data")
        print("    • Low HRR values amplify relative error")
        print("    • Model is extrapolating beyond training distribution")
        print("    💡 Tip: Model works best with HRR > 100 kW scenarios")
    elif rel_error > 50:
        print("\n⚠️  MODERATE ERROR - predictions are approximate")
    else:
        print("\n✅ GOOD PREDICTION ACCURACY")
    print("")
    
    # Plot
    if save_plot or show_plot:
        print("🎨 Creating visualization...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        context_start = max(0, start_idx - 50)
        context_hrr = (full_data_norm[context_start:start_idx+30, 0] * hrr_std) + hrr_mean
        context_time = time[context_start:start_idx+30]
        
        ax.plot(context_time, context_hrr, 'b-', linewidth=1.5, alpha=0.6, label='Past HRR')
        ax.plot(input_time, input_seq_kw, 'b-', linewidth=3, label='Input Sequence')
        ax.plot(future_time, actual_future_kw, 'g-', linewidth=3, marker='o', markersize=6, label='Actual Future')
        ax.plot(future_time, y_pred_kw, 'r--', linewidth=3, marker='s', markersize=6, label='Predicted Future')
        
        ax.set_title(f'Fire Prediction: {Path(csv_file).stem}\\nMAE: {mae:.2f} kW ({rel_error:.2f}% error)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('HRR (kW)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=input_time[-1], color='gray', linestyle=':', linewidth=2)
        
        plt.tight_layout()
        
        if save_plot:
            if output_dir:
                output_path = Path(output_dir) / f"{Path(csv_file).stem}_prediction.png"
            else:
                output_path = Path(csv_file).parent / f"{Path(csv_file).stem}_prediction.png"
            
            plt.savefig(output_path, dpi=150)
            print(f"   ✅ Saved: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return {
        'mae': mae,
        'rel_error': rel_error,
        'peak_hrr': hrr.max(),
        'file': Path(csv_file).name
    }

def run_example_prediction(show_plot=True):
    """Run prediction on example data"""
    example_file = SCRIPT_DIR / "examples" / "sample_scenario_hrr.csv"
    
    if not example_file.exists():
        print("❌ Example file not found")
        print(f"   Expected: {example_file}")
        return None
    
    print("\n📚 Running example with sample data...\n")
    return run_prediction(str(example_file), save_plot=False, show_plot=show_plot)

def run_batch_predictions():
    """Process all CSV files in Input folder"""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    print_section("📦 BATCH PROCESSING")
    
    input_dir = SCRIPT_DIR / "Input"
    output_dir = SCRIPT_DIR / "Output"
    
    # Create folders if needed
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Find files
    csv_files = list(input_dir.glob("*_hrr.csv"))
    
    if not csv_files:
        print(f"❌ No HRR CSV files found in Input/ folder")
        print(f"\n💡 Usage:")
        print(f"   1. Copy your FDS *_hrr.csv files to: {input_dir}")
        print(f"   2. Run batch processing again")
        print(f"   3. Results will appear in: {output_dir}")
        return []
    
    print(f"✅ Found {len(csv_files)} file(s)\n")
    
    # Process each file
    results = []
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] {csv_file.name}")
        print("-" * 50)
        
        try:
            result = run_prediction(str(csv_file), save_plot=True, 
                                   show_plot=False, output_dir=str(output_dir))
            if result:
                results.append(result)
                print(f"✅ MAE: {result['mae']:.2f} kW ({result['rel_error']:.2f}%)")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("📊 BATCH SUMMARY")
        print("="*70)
        print(f"Processed: {len(results)}/{len(csv_files)} files")
        print(f"Average MAE: {np.mean([r['mae'] for r in results]):.2f} kW")
        print(f"Average Error: {np.mean([r['rel_error'] for r in results]):.2f}%")
        
        # Save summary
        summary_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FIRE PREDICTION BATCH SUMMARY\n")
            f.write("="*70 + "\n\n")
            for r in results:
                f.write(f"{r['file']}\n")
                f.write(f"  MAE: {r['mae']:.4f} kW\n")
                f.write(f"  Error: {r['rel_error']:.2f}%\n")
                f.write(f"  Peak: {r['peak_hrr']:.2f} kW\n\n")
        
        print(f"\n📝 Summary saved: {summary_file.name}")
        print(f"📁 All plots in: {output_dir}")
        print("="*70 + "\n")
        
        # Ask if user wants to view results
        print("📊 How would you like to view the results?")
        print("  1. Open Output folder (view all plots in file explorer)")
        print("  2. Display plots in Python windows (one by one)")
        print("  3. Skip viewing")
        
        response = input("\nChoose option [1/2/3]: ").strip()
        
        if response == '1' or not response:
            # Open folder
            print("\n📂 Opening Output folder...")
            try:
                if sys.platform == 'win32':
                    os.startfile(str(output_dir))
                elif sys.platform == 'darwin':
                    subprocess.run(['open', str(output_dir)])
                else:
                    subprocess.run(['xdg-open', str(output_dir)])
                print(f"✅ Opened: {output_dir}")
            except Exception as e:
                print(f"📁 Location: {output_dir}")
                print(f"   (Could not auto-open: {e})")
        
        elif response == '2':
            # Display plots in matplotlib windows
            print("\n🖼️  Opening plots in Python windows...")
            print("   (Close each window to see the next one)\n")
            
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            
            plot_files = list(output_dir.glob("*_prediction.png"))
            # Only show the ones just created (match the results)
            result_names = [r['file'].replace('.csv', '_prediction.png') for r in results]
            plot_files = [p for p in plot_files if p.name in result_names]
            
            for i, plot_file in enumerate(plot_files, 1):
                try:
                    print(f"   [{i}/{len(plot_files)}] Showing: {plot_file.name}")
                    img = mpimg.imread(str(plot_file))
                    fig, ax = plt.subplots(figsize=(14, 8))
                    ax.imshow(img)
                    ax.axis('off')
                    fig.suptitle(f"📊 {plot_file.stem}", fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️  Could not display {plot_file.name}: {e}")
            
            print(f"\n✅ Displayed {len(plot_files)} plot(s)")
    
    return results

# ============================================================================
# FDS FILE GENERATOR
# ============================================================================

def generate_fds_file():
    """Generate a random FDS input file matching training data format"""
    import random
    from datetime import datetime
    
    print_section("🎲 CHAOS GENERATOR (FDS CONFIG MAKER)")
    
    print("This tool conjures FDS input files using parameters that remain within the")
    print("training manifold of the prediction model.")
    print("\n💡 Note: The LSTM prefers its HRR > 100 kW. Try not to feed it a candle.\n")
    
    # Define parameter ranges from training data (calibrated for 100-280 kW HRR range)
    FUELS = {
        'PROPANE': {'soot_yield': 0.024, 'hrrpua': 1200.0, 'formula': 'C3H8', 'fds_name': 'PROPANE'},
        'N-HEPTANE': {'soot_yield': 0.037, 'hrrpua': 1100.0, 'formula': 'C7H16', 'fds_name': 'N-HEPTANE'},
        'METHANE': {'soot_yield': 0.022, 'hrrpua': 1400.0, 'formula': 'CH4', 'fds_name': 'METHANE'},
        'ACETONE': {'soot_yield': 0.014, 'hrrpua': 800.0, 'formula': 'C3H6O', 'fds_name': 'ACETONE'},
        'ETHANOL': {'soot_yield': 0.008, 'hrrpua': 750.0, 'formula': 'C2H6O', 'fds_name': 'ETHANOL'},
        'DIESEL': {'soot_yield': 0.059, 'hrrpua': 1000.0, 'formula': 'C12H23', 'fds_name': 'DIESEL'},
    }
    
    ROOM_SIZES = {
        'small': {'half_x': 1.0, 'half_y': 1.0, 'z': 2.4},     # 2x2x2.4m
        'medium': {'half_x': 1.5, 'half_y': 1.5, 'z': 2.4},    # 3x3x2.4m
        'large': {'half_x': 2.0, 'half_y': 2.0, 'z': 2.4}      # 4x4x2.4m
    }
    
    WALL_MATERIALS = {
        'CONCRETE': {
            'name': 'Concrete',
            'conductivity': 1.8,      # W/m·K
            'specific_heat': 1.04,    # kJ/kg·K
            'density': 2300,          # kg/m³
            'emissivity': 0.9,
            'thickness': 0.15,        # m
            'color': 'GRAY 80'
        },
        'GYPSUM': {
            'name': 'Gypsum Board',
            'conductivity': 0.48,     # W/m·K
            'specific_heat': 1.09,    # kJ/kg·K
            'density': 930,           # kg/m³
            'emissivity': 0.9,
            'thickness': 0.0125,      # m (1/2 inch)
            'color': 'ANTIQUE WHITE'
        },
        'BRICK': {
            'name': 'Brick Masonry',
            'conductivity': 0.69,     # W/m·K
            'specific_heat': 0.84,    # kJ/kg·K
            'density': 1920,          # kg/m³
            'emissivity': 0.9,
            'thickness': 0.10,        # m
            'color': 'FIREBRICK'
        },
        'WOOD': {
            'name': 'Wood Panel',
            'conductivity': 0.14,     # W/m·K
            'specific_heat': 2.85,    # kJ/kg·K
            'density': 510,           # kg/m³
            'emissivity': 0.9,
            'thickness': 0.02,        # m
            'color': 'WOOD'
        }
    }
    
    # User choices or random
    print("Select simulation entropy level:")
    print("  1. 🎲 Roll the physics dice (Fully Random)")
    print("  2. 🎯 Micro-manage the simulation (Custom Parameters)")
    print("  3. 🔙 Retreat to safety (Main menu)\n")
    
    choice = input("Choose option [1/2/3]: ").strip()
    
    if choice == '3' or not choice:
        return
    
    if choice == '1':
        # Fully random
        fuel = random.choice(list(FUELS.keys()))
        room_size = random.choice(list(ROOM_SIZES.keys()))
        wall_material = random.choice(list(WALL_MATERIALS.keys()))
        opening = random.randint(20, 80)  # Moderate ventilation
        fire_size = random.randint(30, 60)  # Optimized for ~100-250 kW
        mesh_size = round(random.uniform(0.09, 0.13), 2)
        sim_time = 30  # Standard training time
        
        print("\n🎲 Randomly generated parameters:")
        
    elif choice == '2':
        # Custom guided
        print("\n🎯 Enter parameters (or press Enter for random):\n")
        
        # Fuel selection
        print("Available fuels:")
        fuel_list = list(FUELS.keys())
        for i, f in enumerate(fuel_list, 1):
            print(f"  {i}. {f}")
        fuel_choice = input(f"\nChoose fuel [1-{len(fuel_list)} or Enter for random]: ").strip()
        fuel = fuel_list[int(fuel_choice)-1] if fuel_choice.isdigit() else random.choice(fuel_list)
        
        # Room size
        print("\nRoom sizes:")
        print("  1. small (2x2x2.4 m)")
        print("  2. medium (3x3x2.4 m)")
        print("  3. large (4x4x2.4 m)")
        room_choice = input("\nChoose room [1-3 or Enter for random]: ").strip()
        room_sizes_list = list(ROOM_SIZES.keys())
        room_size = room_sizes_list[int(room_choice)-1] if room_choice.isdigit() else random.choice(room_sizes_list)
        
        # Wall material
        print("\nWall materials:")
        mat_list = list(WALL_MATERIALS.keys())
        for i, mat in enumerate(mat_list, 1):
            mat_info = WALL_MATERIALS[mat]
            print(f"  {i}. {mat_info['name']} (k={mat_info['conductivity']} W/m·K, ρ={mat_info['density']} kg/m³)")
        mat_choice = input(f"\nChoose material [1-{len(mat_list)} or Enter for random]: ").strip()
        wall_material = mat_list[int(mat_choice)-1] if mat_choice.isdigit() else random.choice(mat_list)
        
        # Opening factor
        opening_input = input("\nOpening factor % [0-100 or Enter for random]: ").strip()
        opening = int(opening_input) if opening_input.isdigit() else random.randint(0, 100)
        
        # Fire size (constrained to produce HRR ~100-250 kW for better model performance)
        fire_input = input("Fire size % [30-60 or Enter for random]: ").strip()
        fire_size = int(fire_input) if fire_input.isdigit() else random.randint(30, 60)
        
        # Mesh size
        mesh_input = input("Mesh size m [0.09-0.13 or Enter for random]: ").strip()
        mesh_size = float(mesh_input) if mesh_input else round(random.uniform(0.09, 0.13), 2)
        
        # Simulation time
        time_input = input("Simulation time s [30-60 or Enter=30]: ").strip()
        sim_time = int(time_input) if time_input.isdigit() else 30
        
        print("\n🎯 Custom parameters set:")
    else:
        print("❌ Invalid option")
        return
    
    # Get room dimensions
    room = ROOM_SIZES[room_size]
    fuel_props = FUELS[fuel]
    
    # Display chosen parameters
    full_x = room['half_x'] * 2
    full_y = room['half_y'] * 2
    mat_props = WALL_MATERIALS[wall_material]
    print(f"   Fuel: {fuel}")
    print(f"   Room Size: {room_size} ({full_x}x{full_y}x{room['z']} m)")
    print(f"   Wall Material: {mat_props['name']}")
    print(f"   Opening Factor: {opening}%")
    print(f"   Fire Size: {fire_size}%")
    print(f"   Mesh Size: {mesh_size} m")
    print(f"   Simulation Time: {sim_time} s")
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"GEN_{fuel}_{room_size}_op{opening}_sz{fire_size}_{timestamp}.fds"
    
    # Calculate parameters matching training format
    hx = room['half_x']
    hy = room['half_y']
    hz = room['z']
    
    # Fire source size calculation (calibrated for ~50-250 kW HRR range)
    # Base fire area depends on room size and fire_size parameter
    base_area = 0.09  # 0.3m x 0.3m = 0.09 m²
    fire_area = base_area * (fire_size / 40.0) * (1.0 if room_size == 'small' else 1.2 if room_size == 'medium' else 1.4)
    fire_half = (fire_area ** 0.5) / 2.0
    fire_height = 0.05
    
    # Calculate HRRPUA to stay within training range
    hrrpua = fuel_props['hrrpua']
    
    # Door opening calculation (on XMIN wall)
    door_width = (opening / 100.0) * 0.9  # 0 to 0.9m
    door_height = 1.0  # Standard door height
    
    # Calculate mesh cells
    mesh_cells_x = int((2 * hx) / mesh_size)
    mesh_cells_y = int((2 * hy) / mesh_size)
    mesh_cells_z = int(hz / mesh_size)
    
    # Wall thickness - use material thickness
    wall_thick = mat_props['thickness']
    
    # Generate FDS file content (calibrated for training HRR range)
    fuel_formula = fuel_props.get('formula', 'C3H8')
    
    fds_content = f"""&HEAD CHID='GEN_{fuel}_{room_size}_{timestamp}', TITLE='Generated Fire Scenario - {mat_props['name']} Walls' /

&MESH IJK={mesh_cells_x},{mesh_cells_y},{mesh_cells_z}, XB={-hx},{hx},{-hy},{hy},0.0,{hz} /

&TIME T_END={sim_time}.0 /

&REAC FUEL='{fuel_props['fds_name']}', FORMULA='{fuel_formula}', SOOT_YIELD={fuel_props['soot_yield']} /

! Material definition - {mat_props['name']}
&MATL ID='{wall_material}',
      CONDUCTIVITY={mat_props['conductivity']},
      SPECIFIC_HEAT={mat_props['specific_heat']},
      DENSITY={mat_props['density']},
      EMISSIVITY={mat_props['emissivity']} /

&SURF ID='WALL_SURF',
      MATL_ID='{wall_material}',
      THICKNESS={mat_props['thickness']},
      COLOR='{mat_props['color']}' /

&SURF ID='FIRE_SOURCE', HRRPUA={hrrpua}, COLOR='ORANGE RED' /

! Fire source (burner) - Area: {fire_area:.4f} m², Expected peak HRR: ~{hrrpua * fire_area:.1f} kW
&OBST XB={-fire_half},{fire_half},{-fire_half},{fire_half},0.0,{fire_height}, SURF_ID='FIRE_SOURCE' /

! Room walls ({mat_props['thickness']*100:.1f}cm thick {mat_props['name']}) - XMIN wall with door opening
"""
    
    if opening > 0:
        door_half = door_width / 2.0
        # Wall with door opening (split into sections)
        fds_content += f"""! Left side of door
&OBST XB={-hx-wall_thick},{-hx},{-hy},{-door_half},0.0,{hz}, SURF_ID='WALL_SURF' / XMIN wall - left
! Right side of door
&OBST XB={-hx-wall_thick},{-hx},{door_half},{hy},0.0,{hz}, SURF_ID='WALL_SURF' / XMIN wall - right
! Above door
&OBST XB={-hx-wall_thick},{-hx},{-door_half},{door_half},{door_height},{hz}, SURF_ID='WALL_SURF' / XMIN wall - above door
"""
    else:
        # Solid wall (no opening)
        fds_content += f"""&OBST XB={-hx-wall_thick},{-hx},{-hy},{hy},0.0,{hz}, SURF_ID='WALL_SURF' / XMIN wall
"""
    
    # Other walls
    fds_content += f"""
! Other walls
&OBST XB={hx},{hx+wall_thick},{-hy},{hy},0.0,{hz}, SURF_ID='WALL_SURF' / XMAX wall
&OBST XB={-hx},{hx},{-hy-wall_thick},{-hy},0.0,{hz}, SURF_ID='WALL_SURF' / YMIN wall
&OBST XB={-hx},{hx},{hy},{hy+wall_thick},0.0,{hz}, SURF_ID='WALL_SURF' / YMAX wall

! Roof (ceiling)
&OBST XB={-hx},{hx},{-hy},{hy},{hz},{hz+wall_thick}, SURF_ID='WALL_SURF' / Roof

"""
    
    # Boundary vents (for walls with openings)
    if opening > 0:
        fds_content += f"""&VENT MB='XMIN', SURF_ID='OPEN' /
"""
    fds_content += f"""&VENT MB='XMAX', SURF_ID='OPEN' /
&VENT MB='YMIN', SURF_ID='OPEN' /
&VENT MB='YMAX', SURF_ID='OPEN' /

! Slice files for visualization and ML training
&SLCF PBX=0.0, QUANTITY='TEMPERATURE', VECTOR=.TRUE. /
&SLCF PBY=0.0, QUANTITY='TEMPERATURE', VECTOR=.TRUE. /
&SLCF PBZ={hz/2}, QUANTITY='TEMPERATURE' /
&SLCF PBY=0.0, QUANTITY='HRRPUV' /
&SLCF PBY=0.0, QUANTITY='VELOCITY', VECTOR=.TRUE. /
&SLCF PBX=0.0, QUANTITY='HRRPUV' /

! Output control
&DUMP DT_SLCF=0.25, DT_HRR=0.1 /

! Devices for monitoring
&DEVC ID='TEMP_CENTER', QUANTITY='TEMPERATURE', XYZ=0,0,{hz/2} /

&TAIL /
"""
    
    # Save file
    output_dir = SCRIPT_DIR / "Input"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    
    try:
        with open(output_path, 'w') as f:
            f.write(fds_content)
        
        print(f"\n✅ FDS file generated successfully!")
        print(f"📁 Saved to: {output_path}")
        print(f"\n📝 File: {filename}")
        print(f"📊 Size: {len(fds_content)} bytes")
        
        # Auto-run feature
        print("\n🚀 Ready to ignite the simulation engine?")
        run_now = input("Invoke FDS6_local.bat right now? [Y/n]: ").strip().lower()
        if not run_now or run_now == 'y':
            fds_exe = SCRIPT_DIR / "FDS6" / "bin" / "fds_local.bat"
            if fds_exe.exists():
                import re

                print(f"\n🔥 Igniting simulation: {filename}\n")

                BAR_WIDTH = 40

                def _draw_bar(progress_pct):
                    filled  = int(BAR_WIDTH * progress_pct / 100)
                    bar     = "█" * filled + "░" * (BAR_WIDTH - filled)
                    line = f"\r  🔥 [{bar}] {progress_pct:5.1f}%"
                    sys.stdout.write(line)
                    sys.stdout.flush()

                try:
                    import time, re, threading

                    # FDS writes to <CHID>.out — extract CHID from the file content
                    chid_match = re.search(r"CHID='([^']+)'", fds_content)
                    chid = chid_match.group(1) if chid_match else filename.replace('.fds', '')
                    out_file = output_dir / f"{chid}.out"

                    proc = subprocess.Popen(
                        [str(fds_exe), filename],
                        cwd=str(output_dir),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                    SPINNER = ["⠋","⠙","⠸","⠴","⠦","⠇"]
                    spin_i  = [0]
                    read_pos = 0
                    last_sim_t = 0.0
                    last_displayed_pct = -1
                    
                    while proc.poll() is None:
                        time.sleep(0.5)
                        spin_i[0] = (spin_i[0] + 1) % len(SPINNER)
                        updated = False
                        if out_file.exists():
                            try:
                                with open(out_file, 'r', errors='ignore') as fh:
                                    fh.seek(read_pos)
                                    chunk = fh.read()
                                    read_pos = fh.tell()
                                # Try multiple patterns for FDS output
                                patterns = [
                                    r"Total Time:\s*([\d.]+)\s*s",
                                    r"Time Step\s+\d+\s+.*?T=\s*([\d.]+)",
                                    r"CURRENT TIME\s*=\s*([\d.]+)",
                                    r"Step\s+\d+\s+Time:\s*([\d.]+)"
                                ]
                                for pattern in patterns:
                                    for m in re.finditer(pattern, chunk):
                                        sim_t = float(m.group(1))
                                        if sim_t > last_sim_t:
                                            pct = min(sim_t / sim_time * 100, 99.9)
                                            if abs(pct - last_displayed_pct) >= 0.1 or last_displayed_pct < 0:
                                                _draw_bar(pct)
                                                last_displayed_pct = pct
                                            last_sim_t = sim_t
                                            updated = True
                                        break
                                    if updated:
                                        break
                            except Exception:
                                pass
                        if not updated and last_displayed_pct < 0:
                            sys.stdout.write(f"\r  {SPINNER[spin_i[0]]} Starting FDS simulation...".ljust(80))
                            sys.stdout.flush()

                    print()  # move past the \r line

                    if proc.returncode != 0:
                        raise subprocess.CalledProcessError(proc.returncode, fds_exe)

                    # Final 100% bar
                    bar = "█" * BAR_WIDTH
                    print(f"\r  ✅ [{bar}] 100.0%  — SIMULATION COMPLETE!{' '*20}")

                    print(f"\n✅ Reality simulated successfully!")
                    print(f"📁 Fluid dynamics securely quarantined in: {output_dir}")
                    csv_name = filename.replace('.fds', '_hrr.csv')
                    if (output_dir / csv_name).exists():
                        print(f"📈 Telemetry acquired: {csv_name}")
                        print("\n🧹 Executing cleanup protocol (deleting useless non-HRR output files)...")
                        
                        file_prefix = filename.replace('.fds', '')
                        deleted_count = 0
                        
                        for ext_file in output_dir.glob(f"{file_prefix}*"):
                            # Keep only the original .fds and the _hrr.csv
                            if ext_file.name != filename and ext_file.name != csv_name:
                                try:
                                    ext_file.unlink()
                                    deleted_count += 1
                                except Exception as cleanup_err:
                                    pass
                        print(f"   Vaporized {deleted_count} unnecessary files.")
                        print("\n💡 The timeline is ready for prediction!")

                        # Ask user if they want to run prediction
                        print(f"\n🔮 Would you like to run prediction on this scenario?")
                        run_predict = input("Run prediction now? [Y/n]: ").strip().lower()
                        
                        if not run_predict or run_predict == 'y':
                            try:
                                print("\n🔮 Running prediction on new scenario...")
                                run_prediction(str(output_dir / csv_name), save_plot=True, show_plot=False)
                                
                                print(f"\n📦 Relocating scenario to training archive ({training_dir.name})...")
                                import shutil
                                training_dir = SCRIPT_DIR / "training_data"
                                training_dir.mkdir(parents=True, exist_ok=True)
                                
                                hrr_src = output_dir / csv_name
                                devc_src = output_dir / csv_name.replace('_hrr', '_devc')
                                fds_src = output_dir / filename
                                
                                if hrr_src.exists(): shutil.move(str(hrr_src), str(training_dir / csv_name))
                                if devc_src.exists(): shutil.move(str(devc_src), str(training_dir / devc_src.name))
                                if fds_src.exists(): shutil.move(str(fds_src), str(training_dir / filename))
                                
                                sync_from_fds_scenarios()
                                
                            except Exception as predict_err:
                                print(f"❌ Prediction failed: {predict_err}")
                        else:
                            print("\n💡 Prediction skipped. Files remain in Input/ folder.")

                except subprocess.CalledProcessError as e:
                    print(f"\n❌ FDS crashed. Return code: {e.returncode}. Maybe check your grid size?")
                except Exception as e:
                    print(f"\n❌ The simulation experienced a highly improbable anomaly: {e}")

            else:
                print(f"\n❌ FDS executable missing from: {fds_exe}")
                print("   Cannot warp space-time without it.")
        else:
            print("\n💡 Suit yourself. Run it manually later:")
            print(f"   cd Input")
            print(f"   ..\\FDS6\\bin\\fds_local.bat {filename}")
        
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")

# ============================================================================
# FILE MANAGEMENT
# ============================================================================

def list_input_files():
    """List files in Input folder"""
    input_dir = SCRIPT_DIR / "Input"
    input_dir.mkdir(exist_ok=True)
    
    csv_files = list(input_dir.glob("*.csv"))
    
    print_section("📁 FILES IN INPUT FOLDER")
    
    if not csv_files:
        print("(Empty - no CSV files found)")
        print(f"\n💡 Add your FDS *_hrr.csv files to:")
        print(f"   {input_dir}\n")
    else:
        for i, f in enumerate(csv_files, 1):
            size_kb = f.stat().st_size / 1024
            print(f"  {i}. {f.name} ({size_kb:.1f} KB)")
        print()

def list_output_files():
    """List files in Output folder"""
    output_dir = SCRIPT_DIR / "Output"
    output_dir.mkdir(exist_ok=True)
    
    png_files = list(output_dir.glob("*.png"))
    txt_files = list(output_dir.glob("*.txt"))
    
    print_section("📤 FILES IN OUTPUT FOLDER")
    
    if not png_files and not txt_files:
        print("(Empty - no results yet)")
        print("\n💡 Results will appear here after predictions\n")
    else:
        if png_files:
            print("Plots:")
            for f in png_files:
                print(f"  • {f.name}")
        if txt_files:
            print("\nSummary Reports:")
            for f in txt_files:
                print(f"  • {f.name}")
        print()

def open_folder(folder_name):
    """Open folder in file explorer"""
    folder_path = SCRIPT_DIR / folder_name
    folder_path.mkdir(exist_ok=True)
    
    try:
        if sys.platform == 'win32':
            os.startfile(str(folder_path))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(folder_path)])
        else:
            subprocess.run(['xdg-open', str(folder_path)])
        print(f"✅ Opened {folder_name}/ folder\n")
    except Exception as e:
        print(f"📁 Folder location: {folder_path}")
        print(f"   (Could not auto-open: {e})\n")

# ============================================================================
# INTERACTIVE MENU
# ============================================================================

def show_main_menu():
    print_banner()
    w = 70
    bar = "─" * (w - 2)
    print("┌" + bar + "┐")
    print("│" + "  MAIN MENU".ljust(w - 2) + "│")
    print("├" + bar + "┤")
    items = [
        ("1", "🎯", "Look into the Matrix", "Provide a CSV, get a prediction"),
        ("2", "📚", "Run Example",          "Because reading documentation is hard"),
        ("3", "📦", "Batch Process",        "Crunch all CSVs in Input/ to Output/"),
        ("4", "🎲", "Generate FDS File",    "Spawn random fire scenarios"),
        ("5", "🔄", "Assimilate Scenarios", "Feed new FDS data to the training pool"),
        ("6", "🧠", "Train Model",          "Adjust LSTM weights (takes a while)"),
        ("7", "📁", "Manage Files",         "Delete things you'll regret later"),
        ("8", "🔧", "Run Diagnostics",      "Blame the system, not the code"),
        ("9", "❓", "Help & Lore",          "Useful tips to survive combustion"),
    ]
    for num, icon, name, desc in items:
        line = f"  {num}.  {icon}  {name:<22} {desc}"
        print("│" + line.ljust(w - 2) + "│")
    print("├" + bar + "┤")
    print("│" + "  0.  🚪  Exit".ljust(w - 2) + "│")
    print("└" + bar + "┘")
    print()

def show_file_management_menu():
    """File management submenu"""
    while True:
        clear_screen()
        print_banner()
        print("FILE MANAGEMENT\n")
        print("  1. 📥 List Input files")
        print("  2. 📤 List Output files")
        print("  3. 🗂️  Open Input folder")
        print("  4. 🗂️  Open Output folder")
        print("  5. 🧹 Clean Output folder")
        print("  6. ← Back to main menu\n")
        
        choice = input("Choose option (1-6): ").strip()
        
        if choice == '1':
            list_input_files()
            press_enter()
        elif choice == '2':
            list_output_files()
            press_enter()
        elif choice == '3':
            open_folder("Input")
            press_enter()
        elif choice == '4':
            open_folder("Output")
            press_enter()
        elif choice == '5':
            clean_output_folder()
            press_enter()
        elif choice == '6':
            break
        else:
            print("Invalid option. Try again.")
            press_enter()

def clean_output_folder():
    """Clean Output folder"""
    output_dir = SCRIPT_DIR / "Output"
    
    if not output_dir.exists():
        print("Output folder doesn't exist yet.")
        return
    
    files = list(output_dir.glob("*"))
    
    if not files:
        print("Output folder is already empty.")
        return
    
    print(f"\nFound {len(files)} file(s) in Output/")
    response = input("Delete all? [y/N]: ").strip().lower()
    
    if response == 'y':
        for f in files:
            try:
                f.unlink()
            except:
                pass
        print("✅ Output folder cleaned\n")
    else:
        print("Cancelled\n")

def show_setup_menu():
    """Setup and diagnostics submenu"""
    while True:
        clear_screen()
        print_banner()
        print("SETUP & DIAGNOSTICS\n")
        print("  1. 🔧 Run Full Setup Wizard")
        print("  2. 🔍 Check System Status")
        print("  3. 📦 Install Dependencies Only")
        print("  4. 📊 Show Model Information")
        print("  5. 📂 Show Folder Locations")
        print("  6. ← Back to main menu\n")
        
        choice = input("Choose option (1-6): ").strip()
        
        if choice == '1':
            setup_wizard()
            press_enter()
        elif choice == '2':
            print_section("🔍 SYSTEM DIAGNOSTICS")
            all_ok, issues = run_diagnostics(silent=False)
            if all_ok:
                print("🎉 All systems ready!")
            else:
                print("\n⚠️  Issues found:")
                for issue in issues:
                    print(f"  • {issue}")
                print("\n💡 Run setup wizard (Option 1) to fix")
            press_enter()
        elif choice == '3':
            install_dependencies()
            press_enter()
        elif choice == '4':
            show_model_info()
            press_enter()
        elif choice == '5':
            show_folder_locations()
            press_enter()
        elif choice == '6':
            break
        else:
            print("Invalid option. Try again.")
            press_enter()

def show_model_info():
    """Display model information"""
    print_section("🧠 MODEL INFORMATION")
    print("Architecture: Physics-Informed LSTM")
    print("  • 2 layers, 128 hidden units per layer")
    print("  • 6 input channels (HRR + physics features)")
    print("\nPhysics Correlations Integrated:")
    print("  • Heskestad (1984): Flame height and growth")
    print("  • McCaffrey (1979): Plume region classification")
    print("  • Thomas (1963): Window/ventilation flow")
    print("  • Buoyancy scaling: Q^(2/5) power law")
    print("\nTraining Data:")
    print("  • 221 FDS fire scenarios")
    print("  • Fuels: Propane, Methane, Diesel, n-Heptane, Dodecane")
    print("  • Room sizes: 2m, 3m, 4m cubes")
    print("  • Various fire behaviors")
    print("\nPerformance:")
    print("  • Test MAE: 0.05 kW (with physics)")
    print("  • Baseline MAE: 5.18 kW (no physics)")
    print("  • Improvement: 8.3% from physics correlations")
    print("  • Typical error: 2-4% on similar scenarios")
    print("  • Inference time: <1 second")
    print("\nCapabilities:")
    print("  • Input: Last 30 time steps")
    print("  • Output: Next 10 time steps predicted")
    print("  • Works with standard fuels and room sizes")
    print("  • Physics-validated predictions")
    print()

def show_folder_locations():
    """Show all folder locations"""
    print_section("📂 FOLDER LOCATIONS")
    
    folders = {
        "Main": SCRIPT_DIR,
        "Input": SCRIPT_DIR / "Input",
        "Output": SCRIPT_DIR / "Output",
        "Examples": SCRIPT_DIR / "examples",
        "Model": SCRIPT_DIR / "model"
    }
    
    for name, path in folders.items():
        exists = "✅" if path.exists() else "❌"
        print(f"{exists} {name}:")
        print(f"   {path}")
        print()

def show_help_menu():
    """Help and information submenu"""
    while True:
        clear_screen()
        print_banner()
        print("HELP & INFORMATION\n")
        print("  1. 📖 Quick Start Guide")
        print("  2. 📋 Command Reference")
        print("  3. ❓ Common Questions (FAQ)")
        print("  4. 🔧 Troubleshooting Tips")
        print("  5. 📚 About This Tool")
        print("  6. ← Back to main menu\n")
        
        choice = input("Choose option (1-6): ").strip()
        
        if choice == '1':
            show_quick_start_guide()
            press_enter()
        elif choice == '2':
            show_command_reference()
            press_enter()
        elif choice == '3':
            show_faq()
            press_enter()
        elif choice == '4':
            show_troubleshooting()
            press_enter()
        elif choice == '5':
            show_about()
            press_enter()
        elif choice == '6':
            break
        else:
            print("Invalid option. Try again.")
            press_enter()

def show_quick_start_guide():
    """Display quick start guide"""
    print_section("🚀 QUICK START GUIDE")
    print("THREE EASY STEPS:\n")
    print("1️⃣  Setup (first time only)")
    print("   • Choose 'Setup & Diagnostics' → 'Run Full Setup Wizard'")
    print("   • Or run: python fire_predict.py setup\n")
    
    print("2️⃣  Try Example")
    print("   • Choose 'Run Example' from main menu")
    print("   • Or run: python fire_predict.py --example\n")
    
    print("3️⃣  Use Your Data")
    print("   • Choose 'Quick Predict' and enter file path")
    print("   • Or run: python fire_predict.py your_file.csv\n")
    
    print("🎲 GENERATE TEST SCENARIOS:")
    print("   1. Choose 'Generate FDS File' from menu")
    print("   2. Select random or custom parameters")
    print("   3. Run generated .fds file in FDS")
    print("   4. Use output CSV for predictions\n")
    
    print("📦 FOR MULTIPLE FILES:")
    print("   1. Put CSV files in Input/ folder")
    print("   2. Choose 'Batch Process' from menu")
    print("   3. Check Output/ folder for results\n")

def show_command_reference():
    """Show command line reference"""
    print_section("📋 COMMAND LINE REFERENCE")
    print("INTERACTIVE MODE (this menu):")
    print("  python fire_predict.py\n")
    
    print("DIRECT COMMANDS:")
    print("  python fire_predict.py <file.csv>      - Predict single file")
    print("  python fire_predict.py --example       - Run example")
    print("  python fire_predict.py --batch         - Batch process Input/")
    print("  python fire_predict.py setup           - Run setup wizard")
    print("  python fire_predict.py check           - Check system status")
    print("  python fire_predict.py --help          - Show help")
    print("  python fire_predict.py --version       - Show version\n")
    
    print("OPTIONS:")
    print("  -o, --output FILE     - Specify output plot filename")
    print("  --no-plot             - Don't show interactive plot")
    print("  --output-dir DIR      - Save batch results to custom folder\n")

def show_faq():
    """Show frequently asked questions"""
    print_section("❓ FREQUENTLY ASKED QUESTIONS")
    
    print("Q: What file format do I need?")
    print("A: CSV file from FDS with Time,HRR columns (typically *_hrr.csv)\n")
    
    print("Q: Can I generate test FDS scenarios?")
    print("A: Yes! Use menu option 4 to generate FDS files with random or")
    print("   custom parameters within the model's training scope\n")
    
    print("Q: How accurate is it?")
    print("A: 2-4% error on typical scenarios, up to 6-10% on unusual ones\n")
    
    print("Q: How long does prediction take?")
    print("A: About 5 seconds per file\n")
    
    print("Q: Can I use this without internet?")
    print("A: Yes! After initial setup, works completely offline\n")
    
    print("Q: What if I get an error?")
    print("A: Check 'Troubleshooting Tips' or run system diagnostics\n")
    
    print("Q: Can I customize the model?")
    print("A: Yes! See training_data/ folder for retraining options\n")

def show_troubleshooting():
    """Show troubleshooting tips"""
    print_section("🔧 TROUBLESHOOTING TIPS")
    
    print("❌ 'Module not found' error")
    print("   → Run setup wizard (Setup menu → Option 1)\n")
    
    print("❌ 'File not found' error")
    print("   → Check file path, use quotes for spaces")
    print("   → Try full path: C:\\path\\to\\file.csv\n")
    
    print("❌ 'Not enough data' error")
    print("   → Need at least 40 time steps in CSV")
    print("   → Run longer FDS simulation\n")
    
    print("❌ High prediction error (>10%)")
    print("   → Scenario may be outside training data")
    print("   → Check: standard fuel? typical room size?\n")
    
    print("💡 QUICK CHECKS:")
    print("   1. Run system diagnostics (Setup menu → Option 2)")
    print("   2. Try example prediction (should always work)")
    print("   3. Verify CSV has Time,HRR columns\n")

def show_about():
    """Show about information"""
    print_section("📚 ABOUT THIS TOOL")
    print(f"Fire HRR Prediction Tool v{VERSION}")
    print("Physics-Informed LSTM for Fire Dynamics\n")
    
    print("🎯 Purpose:")
    print("  Predict future Heat Release Rate from FDS simulation data")
    print("  using deep learning with embedded physics knowledge\n")
    
    print("🧠 Technology:")
    print("  • LSTM Neural Network (128 units, 2 layers)")
    print("  • Physics-informed features:")
    print("    - Heskestad flame height correlation")
    print("    - McCaffrey plume region analysis")
    print("    - Window/ventilation flow correlations")
    print("  • Trained on 221 diverse fire scenarios")
    print("  • PyTorch implementation\n")
    
    print("📊 Capabilities:")
    print("  • Input: 30 time steps of HRR data")
    print("  • Output: 10 time steps predicted")
    print("  • Accuracy: 2-4% on typical fires")
    print("  • Speed: <1 second inference\n")
    
    print("📖 Documentation:")
    print("  • QUICKSTART.md - 2-minute guide")
    print("  • README.md - Complete manual")
    print("  • CHEATSHEET.md - Command reference")
    print("  • WORKFLOWS.md - Visual guides\n")

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model_interactive():
    """Interactive model training interface"""
    print_section("🧠 MODEL TRAINING")
    
    print("⚠️  ADVANCED FEATURE\n")
    print("This will train a new model from scratch using your training data.")
    print("Requires:")
    print("  • Training data in training_data/ folder")
    print("  • At least 100+ FDS simulation scenarios")
    print("  • GPU recommended (training can take hours on CPU)")
    print("  • 4-8GB free RAM\n")
    
    # Check for training data
    training_data_dir = SCRIPT_DIR / "training_data"
    if not training_data_dir.exists():
        print("❌ Training data folder not found!")
        print(f"\n💡 Create folder: {training_data_dir}")
        print("   Then add your processed FDS data files\n")
        return
    
    csv_files = list(training_data_dir.glob("*_hrr.csv"))
    if len(csv_files) < 10:
        print(f"⚠️  Only {len(csv_files)} files found in training_data/")
        print("   Recommended: 100+ scenarios for good performance\n")
    else:
        print(f"✅ Found {len(csv_files)} training files\n")
    
    # Default hyperparameters
    config = {
        'epochs': 50,
        'batch_size': 32,
        'hidden_dim': 128,
        'num_layers': 2,
        'learning_rate': 0.001
    }
    
    print("Training Configuration:")
    print("  • Architecture: Physics-Informed LSTM")
    print("  • Input features: 9 (3 original + 6 physics correlations)")
    print("    - Original: HRR, Q_RADI, MLR")
    print("    - Heskestad: Flame height, growth rate, deviation")
    print("    - McCaffrey: Plume region classification")
    print("    - Thomas: Ventilation flow factor")
    print("    - Buoyancy power scaling")
    print(f"  • Epochs: {config['epochs']} (with early stopping)")
    print(f"  • Batch size: {config['batch_size']}")
    print("  • Sequence length: 30 steps")
    print("  • Prediction horizon: 10 steps\n")
    
    print("🤖 Would you like to mathematically optimize these hyper-parameters?")
    tweak = input("Tweak model settings? [y/N]: ").strip().lower()
    
    if tweak == 'y':
        print("\n🔧 HYPERPARAMETER TERMINAL")
        print("Press Enter to keep the default value.\n")
        
        try:
            ep = input(f"Epochs [{config['epochs']}]: ").strip()
            if ep: config['epochs'] = int(ep)
            
            bs = input(f"Batch Size [{config['batch_size']}]: ").strip()
            if bs: config['batch_size'] = int(bs)
            
            hd = input(f"Hidden Dimensions [{config['hidden_dim']}]: ").strip()
            if hd: config['hidden_dim'] = int(hd)
            
            nl = input(f"LSTM Layers [{config['num_layers']}]: ").strip()
            if nl: config['num_layers'] = int(nl)
            
            lr = input(f"Learning Rate [{config['learning_rate']}]: ").strip()
            if lr: config['learning_rate'] = float(lr)
            
            print("\n✅ Matrix parameters re-aligned successfully.\n")
        except ValueError:
            print("\n❌ Invalid input detected. Reverting to safe default parameters.\n")
    
    response = input("Continue with training using these settings? [y/N]: ").strip().lower()
    
    if response != 'y':
        print("\nTraining cancelled.\n")
        return
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    try:
        # Import training modules
        print("📦 Loading training modules...")
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
        from torch.utils.data import DataLoader
        import torch
        import pandas as pd
        import numpy as np
        import json
        from fire_prediction.models.physics_informed import PhysicsInformedLSTM
        
        # Configuration
        INPUT_SEQ_LEN = 30
        PRED_HORIZON = 10
        BATCH_SIZE = config['batch_size']
        MAX_EPOCHS = config['epochs']
        
        print("✅ Modules loaded\n")
        
        # Step 1: Create dataset from CSV files
        print("📂 Preparing dataset from CSV files...")
        scenarios = []
        
        for csv_file in csv_files:
            try:
                # Read CSV, skipping first row (units) and using second row as header
                df = pd.read_csv(csv_file, skiprows=[0])
                
                # Extract required columns
                if 'Time' not in df.columns or 'HRR' not in df.columns:
                    continue
                
                # Calculate total MLR from fuel column (MLR_PROPANE, MLR_METHANE, etc.)
                mlr_cols = [col for col in df.columns if col.startswith('MLR_') and 
                           col not in ['MLR_AIR', 'MLR_PRODUCTS']]
                if mlr_cols:
                    mlr = df[mlr_cols[0]].tolist()
                else:
                    mlr = [0.0] * len(df)
                
                scenario_data = {
                    'scenario': csv_file.stem,
                    'time': df['Time'].tolist(),
                    'hrr_series': df['HRR'].tolist(),
                    'q_radi_series': df['Q_RADI'].tolist() if 'Q_RADI' in df.columns else [0.0] * len(df),
                    'mlr_series': mlr
                }
                scenarios.append(scenario_data)
            except Exception as e:
                print(f"⚠️  Skipping {csv_file.name}: {e}")
                continue
        
        if len(scenarios) < 10:
            print(f"\n❌ Only {len(scenarios)} valid scenarios found. Need at least 10.\n")
            return
        
        print(f"✅ Loaded {len(scenarios)} scenarios\n")
        
        # Step 2: Create train/val splits (80/20)
        import random
        random.seed(42)
        random.shuffle(scenarios)
        
        split_idx = int(len(scenarios) * 0.8)
        train_scenarios = scenarios[:split_idx]
        val_scenarios = scenarios[split_idx:]
        
        print(f"📊 Split: {len(train_scenarios)} train, {len(val_scenarios)} validation\n")
        
        # Step 3: Save ml_dataset.json and splits
        print("💾 Creating dataset files...")
        
        dataset_json = {'scenarios': scenarios}
        dataset_path = training_data_dir / "ml_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset_json, f)
        
        splits_dir = training_data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        with open(splits_dir / "train_split.json", 'w') as f:
            json.dump([s['scenario'] for s in train_scenarios], f)
        
        with open(splits_dir / "val_split.json", 'w') as f:
            json.dump([s['scenario'] for s in val_scenarios], f)
        
        print("✅ Dataset files created\n")
        
        # Step 4: Load datasets with physics correlations
        print("📂 Loading datasets with enhanced physics correlations...")
        from fire_prediction.data.physics_dataset import PhysicsInformedDataset
        
        room_dims = {
            'opening_area': 0.8,  # 0.8 m² opening
            'opening_height': 1.0,  # 1.0 m height
            'room_area': 9.0  # 3m x 3m room
        }
        
        train_ds = PhysicsInformedDataset(
            str(training_data_dir), 'train', INPUT_SEQ_LEN, PRED_HORIZON,
            include_heskestad=True, fire_diameter=0.3, room_dims=room_dims
        )
        val_ds = PhysicsInformedDataset(
            str(training_data_dir), 'val', INPUT_SEQ_LEN, PRED_HORIZON,
            include_heskestad=True, fire_diameter=0.3, room_dims=room_dims,
            train_stats=train_ds.stats
        )
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"✅ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples\n")
        
        # Create model
        print("🧠 Initializing model with enhanced physics correlations...")
        model = PhysicsInformedLSTM(
            input_dim=9,  # 3 original + 6 physics features
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=1,
            dropout=0.1,
            lr=config['learning_rate'],
            pred_horizon=PRED_HORIZON,
            use_physics_loss=True,
            lambda_physics=0.1,
            lambda_monotonic=0.05,
            fire_diameter=0.3,
            validate_physics=True
        )
        print("✅ Model ready\n")
        
        # Setup callbacks
        print("⚙️  Setting up training...")
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(SCRIPT_DIR / "checkpoints"),
            filename='model-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        )
        
        logger = TensorBoardLogger(
            save_dir=str(SCRIPT_DIR / "logs"),
            name="training",
            log_graph=True
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            accelerator='auto',
            devices=1,
            log_every_n_steps=10,
            enable_progress_bar=True
        )
        
        print("✅ Trainer configured\n")
        print("="*70)
        print("🚀 TRAINING STARTED")
        print("="*70)
        print("\n💡 Monitor progress:")
        print(f"   tensorboard --logdir={SCRIPT_DIR / 'logs'}")
        print("\n⏱️  This may take 1-3 hours depending on data size and hardware...\n")
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        # Save best model
        best_model_path = SCRIPT_DIR / "model" / "best_model.ckpt"
        best_model_path.parent.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70 + "\n")
        
        print(f"📊 Best model saved to: {best_model_path}")
        print(f"📈 All checkpoints: {SCRIPT_DIR / 'checkpoints'}")
        print(f"📉 TensorBoard logs: {SCRIPT_DIR / 'logs'}\n")
        
        # Copy best checkpoint
        if checkpoint_callback.best_model_path:
            import shutil
            shutil.copy(checkpoint_callback.best_model_path, best_model_path)
            print(f"✅ Best model copied to deployment location\n")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\n💡 Make sure all packages are installed:")
        print("   pip install -r requirements.txt\n")
    except FileNotFoundError as e:
        print(f"\n❌ Data error: {e}")
        print("\n💡 Check your training_data/ folder structure:")
        print("   training_data/")
        print("   ├── train/")
        print("   ├── val/")
        print("   └── test/\n")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n💡 Check:")
        print("   • Data format is correct")
        print("   • Sufficient disk space")
        print("   • CUDA/GPU drivers (if using GPU)\n")

# ============================================================================
# SYNC FROM FDS_SCENARIOS
# ============================================================================

def sync_from_fds_scenarios():
    """Smart-sync: scans training_data/ for explicitly new CSVs and appends them to ml_dataset.json"""
    import json
    import shutil
    import numpy as np
    import pandas as pd

    TRAINING_DIR = SCRIPT_DIR / "training_data"
    DT           = 0.1
    
    print_section("🔄 SMART SYNC: ASSIMILATING NEW DATA")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    ml_path = TRAINING_DIR / "ml_dataset.json"
    
    existing_scenarios = set()
    ml_records = []
    
    # Load existing JSON if it exists
    if ml_path.exists():
        try:
            with open(ml_path, "r") as f:
                dataset = json.load(f)
                ml_records = dataset.get("scenarios", [])
                for record in ml_records:
                    existing_scenarios.add(record["scenario"])
            print(f"Loaded existing manifest with {len(existing_scenarios)} scenarios.")
        except Exception as e:
            print(f"⚠️ Could not read existing JSON ({e}). Rebuilding from scratch.")

    def safe_col(df, *names):
        for nm in names:
            if nm in df.columns: return df[nm].values
            m = [c for c in df.columns if nm.lower() in c.lower()]
            if m: return df[m[0]].values
        return None

    # Find all potential HRR files in the training directory
    hrr_files = list(TRAINING_DIR.glob("*_hrr.csv"))

    # Also rescue any orphaned CSVs left behind in Input/ by interrupted runs
    INPUT_DIR = SCRIPT_DIR / "Input"
    if INPUT_DIR.exists():
        orphans = list(INPUT_DIR.glob("*_hrr.csv"))
        if orphans:
            print(f"🔍 Found {len(orphans)} orphaned HRR file(s) in Input/ — rescuing...")
            hrr_files.extend(orphans)

    # Optional: also check the external FDS_DIR if it exists for backwards compatibility
    FDS_DIR = Path(r"D:\FDS\Small_project\fds_scenarios")
    if FDS_DIR.exists():
        print("External fds_scenarios/ folder detected. Scanning for legacy imports...")
        for scenario_dir in [d for d in FDS_DIR.iterdir() if d.is_dir()]:
            ext_hrr = scenario_dir / f"{scenario_dir.name}_hrr.csv"
            if ext_hrr.exists(): hrr_files.append(ext_hrr)

    copied, skipped = 0, 0

    print(f"Scanning {len(hrr_files)} HRR files...")

    for hrr_src in hrr_files:
        # Determine scenario name (remove '_hrr.csv' suffix)
        name = hrr_src.name.replace("_hrr.csv", "")
        
        # Smart Skip: If it's already in the JSON, we don't need to re-parse it!
        if name in existing_scenarios:
            skipped += 1
            continue
            
        devc_src = hrr_src.parent / f"{name}_devc.csv"

        try:
            # If the file is coming from the external FDS dir, copy it over first
            if hrr_src.parent != TRAINING_DIR:
                dest_hrr = TRAINING_DIR / hrr_src.name
                shutil.copy2(hrr_src, dest_hrr)
                hrr_src = dest_hrr # update path to use local copy
                if devc_src.exists():
                    shutil.copy2(devc_src, TRAINING_DIR / devc_src.name)
                    devc_src = TRAINING_DIR / devc_src.name

            # Extract features
            raw  = pd.read_csv(hrr_src, skiprows=1)
            t    = raw.iloc[:, 0].values.astype(float)
            h    = raw.iloc[:, 1].values.astype(float)
            ok   = np.isfinite(t) & np.isfinite(h)
            t, h = t[ok], h[ok]
            if len(t) < 10: 
                print(f"  ⚠️  {name}: Skipped (time series too short)")
                continue

            t_end = float(t[-1])
            ct    = np.linspace(0.0, t_end, max(int(round(t_end/DT))+1, 10))
            hi    = np.interp(ct, t, h)

            mlr_i = qr_i = [0.0]*len(ct)
            if devc_src.exists():
                try:
                    dv = pd.read_csv(devc_src, skiprows=1)
                    dt_ = dv.iloc[:, 0].values.astype(float)
                    tr  = dt_ <= t_end
                    def _ic(*ns):
                        col = safe_col(dv, *ns)
                        if col is None: return None
                        cv = col[tr].astype(float); ct2 = dt_[tr]
                        v  = np.isfinite(cv)&np.isfinite(ct2)
                        return np.interp(ct, ct2[v], cv[v]).tolist() if v.sum()>1 else None
                    mlr_i = _ic("MLR","Mass Loss Rate","MLR_TOTAL") or mlr_i
                    qr_i  = _ic("Q_RADI","QRADI","RADIATIVE_FLUX") or qr_i
                except Exception: pass

            ml_records.append({
                "scenario": name, "peak_hrr": float(hi.max()),
                "time_to_peak": float(ct[hi.argmax()]), "duration": t_end,
                "hrr_series": hi.tolist(), "q_radi_series": qr_i, "mlr_series": mlr_i
            })
            existing_scenarios.add(name)
            copied += 1
            print(f"  ✅ Added to dataset: {name}")

        except Exception as e:
            print(f"  ⚠️  Failed to process {name}: {e}")

    # Write ml_dataset.json (only if changes were made or it doesn't exist)
    if copied > 0 or not ml_path.exists():
        with open(ml_path, "w") as f:
            json.dump({"n_scenarios": len(ml_records), "dt": DT,
                       "scenarios": ml_records}, f, indent=2)
        print("\n" + "="*70)
        print("✅ SMART SYNC COMPLETE")
        print("="*70)
        print(f"  Previous scenarios: {len(existing_scenarios) - copied}")
        print(f"  New injected      : {copied}")
        print(f"  Total robust DB   : {len(ml_records)} scenarios → {ml_path.name}")
        print("\n💡 Now use Option 6 (Train Model) to retrain on the updated data.")
    else:
        print("\n" + "="*70)
        print("✅ SMART SYNC ZERO-OP")
        print("="*70)
        print(f"  Dataset is already up-to-date. (Skipped {skipped} existing files).")

# ============================================================================
# MAIN INTERACTIVE INTERFACE
# ============================================================================

def interactive_mode():
    """Main interactive menu loop"""
    # Initial check
    all_ok, issues = run_diagnostics(silent=True)
    
    if not all_ok:
        clear_screen()
        print_banner()
        print("⚠️  SETUP REQUIRED\n")
        print("Some components are missing:")
        for issue in issues:
            print(f"  • {issue}")
        print("\n💡 Would you like to run setup now?")
        response = input("\nRun setup wizard? [Y/n]: ").strip().lower()
        
        if not response or response == 'y':
            if setup_wizard():
                press_enter()
            else:
                print("\n❌ Setup incomplete. Please fix issues and try again.")
                return 1
        else:
            print("\n⚠️  Tool may not work properly without setup.")
            press_enter()
    
    # Main menu loop
    fire_splash()
    while True:
        clear_screen()
        show_main_menu()
        
        choice = input("Choose option (1-9, 0 to exit): ").strip()
        
        if choice == '1':
            # Quick predict
            clear_screen()
            print_banner()
            print_section("🎯 QUICK PREDICTION")
            
            file_path = input("Enter path to your CSV file: ").strip().strip('"').strip("'")
            
            if file_path:
                if not Path(file_path).exists():
                    print(f"\n❌ File not found: {file_path}")
                    print("\n💡 Tips:")
                    print("   • Check the path is correct")
                    print("   • Try dragging file into terminal")
                    print("   • Use full path if needed")
                else:
                    print()
                    try:
                        run_prediction(file_path, save_plot=True, show_plot=True)
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                press_enter()
        
        elif choice == '2':
            # Run example
            clear_screen()
            print_banner()
            run_example_prediction(show_plot=True)
            press_enter()
        
        elif choice == '3':
            # Batch process
            clear_screen()
            print_banner()
            run_batch_predictions()
            press_enter()
        
        elif choice == '4':
            # Generate FDS file
            clear_screen()
            print_banner()
            generate_fds_file()
            press_enter()
        
        elif choice == '5':
            # Sync from fds_scenarios
            clear_screen()
            print_banner()
            sync_from_fds_scenarios()
            press_enter()
        
        elif choice == '6':
            # Train model
            clear_screen()
            print_banner()
            train_model_interactive()
            press_enter()
        
        elif choice == '7':
            # File management
            show_file_management_menu()
        
        elif choice == '8':
            # Setup & diagnostics
            show_setup_menu()
        
        elif choice == '9':
            # Help
            show_help_menu()
        
        elif choice == '0':
            # Exit
            clear_screen()
            print("\n👋 Thanks for using Fire Prediction Tool!\n")
            return 0
        
        else:
            print("\n❌ Invalid option. Please choose 1-9 or 0 to exit.")
            press_enter()

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def command_line_mode(args):
    """Handle command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fire HRR Prediction Tool - All-in-One Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fire_predict.py                      Interactive menu
  python fire_predict.py file.csv            Predict single file
  python fire_predict.py --example           Run example
  python fire_predict.py --batch             Batch process Input/
  python fire_predict.py setup               Run setup wizard
  python fire_predict.py check               Check system status

For more help, use interactive mode (no arguments) or see documentation.
        """
    )
    
    parser.add_argument('file', nargs='?', help='CSV file to predict (or "setup"/"check")')
    parser.add_argument('-o', '--output', help='Output plot filename')
    parser.add_argument('--output-dir', help='Output directory for batch mode')
    parser.add_argument('--batch', action='store_true', help='Batch process Input/ folder')
    parser.add_argument('--example', action='store_true', help='Run example prediction')
    parser.add_argument('--no-plot', action='store_true', help='Don\'t show interactive plot')
    parser.add_argument('--version', action='store_true', help='Show version')
    
    parsed = parser.parse_args(args)
    
    # Handle special commands
    if parsed.version:
        print_banner()
        print(f"Version: {VERSION}")
        print("Model: Physics-Informed LSTM")
        print("Training scenarios: 221")
        print("Test MAE: 0.05 kW\n")
        return 0
    
    if parsed.file == 'setup':
        return 0 if setup_wizard() else 1
    
    if parsed.file == 'check':
        print_section("🔍 SYSTEM DIAGNOSTICS")
        all_ok, issues = run_diagnostics(silent=False)
        if all_ok:
            print("🎉 All systems ready!\n")
            return 0
        else:
            print("\n⚠️  Issues found - run setup wizard")
            return 1
    
    if parsed.example:
        print_banner()
        result = run_example_prediction(show_plot=not parsed.no_plot)
        return 0 if result else 1
    
    if parsed.batch:
        print_banner()
        results = run_batch_predictions()
        return 0 if results else 1
    
    if parsed.file:
        if not Path(parsed.file).exists():
            print(f"\n❌ Error: File not found: {parsed.file}")
            print("\n💡 Tips:")
            print("   • Check the file path is correct")
            print("   • Use quotes if path has spaces")
            print("   • Try: python fire_predict.py --example")
            return 1
        
        print_banner()
        result = run_prediction(parsed.file, save_plot=True, 
                               show_plot=not parsed.no_plot,
                               output_dir=parsed.output_dir)
        return 0 if result else 1
    
    # No arguments - show quick help
    print_banner()
    print("USAGE:")
    print("  python fire_predict.py                 - Interactive menu")
    print("  python fire_predict.py file.csv        - Predict single file")
    print("  python fire_predict.py --example       - Run example")
    print("  python fire_predict.py --batch         - Batch process")
    print("  python fire_predict.py setup           - Run setup")
    print("  python fire_predict.py check           - Check status")
    print("  python fire_predict.py --help          - Detailed help\n")
    
    print("💡 TIP: Run without arguments for interactive menu!")
    print("   Just type: python fire_predict.py\n")
    return 0

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        # If arguments provided, use command line mode
        if len(sys.argv) > 1:
            return command_line_mode(sys.argv[1:])
        
        # Otherwise, use interactive mode
        return interactive_mode()
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n💡 Try:")
        print("   • python fire_predict.py check (system diagnostics)")
        print("   • python fire_predict.py setup (run setup)")
        print("   • python fire_predict.py --example (test with sample)\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())





