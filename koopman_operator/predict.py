"""
Koopman Fire Prediction Tool
=============================
Interactive menu for the Koopman-Enhanced LSTM fire dynamics model.

Usage:
    python predict.py              # Interactive menu
    python predict.py --batch      # Batch predict all files in Input/
    python predict.py file.csv     # Predict a single CSV file
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

VERSION    = "1.0.0"
DATA_DIR   = Path(r"D:\FDS\Small_project\ml_data")
FDS_DIR    = Path(r"D:\FDS\Small_project\fds_scenarios")
CKPT_DIR   = SCRIPT_DIR / "checkpoints"
PRED_DIR   = SCRIPT_DIR / "predictions"
INPUT_DIR  = SCRIPT_DIR / "Input"
OUTPUT_DIR = SCRIPT_DIR / "Output"
# ───────────────────────────────────────────────────────────────────────────


# ============================================================================
# TERMINAL UTILITIES
# ============================================================================

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def press_enter():
    input("\nPress Enter to continue...")

# ── ANSI colours ──────────────────────────────────────────────────────────────
R  = "\033[31m";  Y  = "\033[33m";  W  = "\033[97m"
RB = "\033[91m";  YB = "\033[93m";  RST = "\033[0m"

FRAMES = [
    [
        f"     {RB}  ) {Y} ({RB}  ){RST}   ",
        f"    {Y} ( {RB}){Y}  (  {RB}( {RST}  ",
        f"   {RB}){Y}(   {RB}) {Y}( {RB})  {RST} ",
        f"  {Y}(  {RB})   {Y}(   {RB})  {RST} ",
        f" {R}  \\{RB}|{Y}///{RB}|{Y}///{R}|/{RST}",
        f" {R}   \\{RB}|{R}/////|{RB}\\{R}/{RST} ",
        f"  {R}   \\{RB}|||{R}///{RST}    ",
        f"   {R}   \\{RB}|{R}/ {RST}      ",
        f"    {R}───┴───{RST}    ",
    ],
    [
        f"     {Y}(  {RB}) {Y}  ({RB}){RST}  ",
        f"    {RB}){Y}  ({RB})  {Y}(  {RST} ",
        f"   {Y}( {RB}) {Y}(  {RB})  {Y}({RST} ",
        f"  {RB})  {Y}(   {RB})  {Y}(  {RST}",
        f" {R}  \\{Y}|{RB}\\\\\\{Y}|{RB}\\\\\\{R}|/{RST}",
        f" {R}   \\{Y}|{R}/////|{Y}\\{R}/{RST} ",
        f"  {R}   \\{Y}|||{R}///{RST}    ",
        f"   {R}   \\{Y}|{R}/ {RST}      ",
        f"    {R}───┴───{RST}    ",
    ],
    [
        f"    {RB} ( {Y}){RB}  ( {Y}) {RST}  ",
        f"   {Y} ){RB}(  {Y}) {RB} ( {Y}) {RST}",
        f"   {RB}( {Y})  {RB}(  {Y}){RB}( {RST} ",
        f"  {Y})  {RB})   {Y}(   {RB})  {RST}",
        f" {R}  \\{RB}|{R}\\\\\\{RB}|{R}\\\\\\{RB}|/{RST}",
        f" {R}   \\{RB}|{R}/////|{Y}\\{R}/{RST} ",
        f"  {R}   \\{RB}|||{R}///{RST}    ",
        f"   {R}   \\{RB}|{R}/ {RST}      ",
        f"    {R}───┴───{RST}    ",
    ],
]

def fire_splash():
    """Play a brief animated ASCII fire, then clear."""
    import time
    try:
        # Hide cursor
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        clear_screen()
        n_lines = len(FRAMES[0])
        first   = True
        for _ in range(6):           # ~1.5 s total
            for frame in FRAMES:
                if not first:
                    sys.stdout.write(f"\033[{n_lines}A")  # move up
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


def print_banner():
    _A = "\033[91m"; _B = "\033[93m"; _RST = "\033[0m"
    art = [
        f"{_A}██╗  ██╗ ██████╗  ██████╗ ██████╗ ███╗   ███╗ █████╗ ███╗   ██╗{_RST}",
        f"{_A}██║ ██╔╝██╔═══██╗██╔═══██╗██╔══██╗████╗ ████║██╔══██╗████╗  ██║{_RST}",
        f"{_B}█████╔╝ ██║   ██║██║   ██║██████╔╝██╔████╔██║███████║██╔██╗ ██║{_RST}",
        f"{_B}██╔═██╗ ██║   ██║██║   ██║██╔═══╝ ██║╚██╔╝██║██╔══██║██║╚██╗██║{_RST}",
        f"{_A}██║  ██╗╚██████╔╝╚██████╔╝██║     ██║ ╚═╝ ██║██║  ██║██║ ╚████║{_RST}",
        f"{_A}╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝{_RST}",
    ]
    print()
    for line in art:
        print("  " + line)
    print(f"\n  \033[97mKoopman-Enhanced LSTM  ·  Fire Dynamics Forecasting  ·  v{VERSION}\033[0m")
    print()

def print_section(title):
    w = 65
    bar = "─" * (w - 2)
    print()
    print("┌" + bar + "┐")
    print("│  " + title.ljust(w - 4) + "│")
    print("└" + bar + "┘")
    print()


# ============================================================================
# MODEL LOADING
# ============================================================================

def find_checkpoint():
    """Return path to the best or latest checkpoint."""
    if not CKPT_DIR.exists():
        return None
    # Prefer best checkpoint (contains 'best' in name)
    best = [p for p in CKPT_DIR.glob("*.ckpt") if "best" in p.name.lower()]
    if best:
        return str(max(best, key=lambda p: p.stat().st_mtime))
    ckpts = list(CKPT_DIR.glob("*.ckpt"))
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda p: p.stat().st_mtime))


def load_model():
    """Load KoopmanLSTM from latest checkpoint. Returns (model, ckpt_path) or (None, None)."""
    try:
        from koopman_fire.models.koopman_lstm import KoopmanLSTM
        ckpt = find_checkpoint()
        if not ckpt:
            print("❌ No checkpoint found in checkpoints/")
            print("   Run Option 5 (Train Model) first.")
            return None, None
        print(f"  📂 Loading: {Path(ckpt).name}")
        model = KoopmanLSTM.load_from_checkpoint(ckpt)
        model.eval()
        model.cpu()
        return model, ckpt
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install torch pytorch_lightning numpy matplotlib pandas")
        return None, None
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return None, None


def load_dataset():
    """Load train/test datasets. Returns (train_ds, test_ds) or (None, None)."""
    try:
        from koopman_fire.data.dataset import FireTimeSeriesDataset
        train_ds = FireTimeSeriesDataset(str(DATA_DIR), split="train")
        test_ds  = FireTimeSeriesDataset(str(DATA_DIR), split="test",
                                         train_stats=train_ds.stats)
        return train_ds, test_ds
    except FileNotFoundError:
        print(f"❌ Dataset not found at {DATA_DIR}")
        print("   Run Option 4 (Sync from fds_scenarios) first.")
        return None, None
    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        return None, None


# ============================================================================
# CORE PREDICTION LOGIC
# ============================================================================

DT = 0.1  # seconds per timestep

def denormalise_hrr(values, stats):
    return values * stats["std"][0] + stats["mean"][0]


def predict_scenario(model, dataset, scenario_idx):
    """Stitch sliding-window predictions for one scenario. Returns (times, actual_kw, pred_kw, name)."""
    import numpy as np
    import torch

    scenario = dataset.processed[scenario_idx]
    data     = scenario["data"]
    name     = scenario["name"]
    T        = data.shape[0]
    seq_len  = dataset.input_seq_len
    horizon  = dataset.pred_horizon
    stats    = dataset.stats

    all_preds = np.full(T, np.nan)
    all_actual = data[:, 0].numpy()
    count = np.zeros(T)

    model.eval()
    with torch.no_grad():
        for t in range(max(T - seq_len - horizon, 0)):
            x     = data[t: t + seq_len, :].unsqueeze(0)
            y_hat = model(x).squeeze().cpu().numpy()
            for h in range(horizon):
                idx = t + seq_len + h
                if idx < T:
                    if np.isnan(all_preds[idx]):
                        all_preds[idx] = 0.0
                    all_preds[idx] += y_hat[h]
                    count[idx] += 1

    valid = count > 0
    all_preds[valid] /= count[valid]

    actual_kw = denormalise_hrr(all_actual, stats)
    pred_kw   = denormalise_hrr(all_preds,  stats)
    times     = np.arange(T) * DT
    return times, actual_kw, pred_kw, name


def plot_scenario(times, actual, predicted, name, save_path=None, show=False):
    """Plot actual vs predicted HRR with error panel."""
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    valid = ~np.isnan(predicted)
    ax = axes[0]
    ax.plot(times, actual,            color="#7fdbff", lw=1.5,
            label="Actual (FDS)",     alpha=0.9)
    ax.plot(times[valid], predicted[valid], color="#ff6b6b", lw=1.2,
            label="Koopman Prediction", alpha=0.9)
    ax.set_ylabel("HRR (kW)", color="white")
    ax.set_title(f"🔥 {name}", color="white", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.2, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.set_xlim(times[0], times[-1])

    ax2 = axes[1]
    error = np.where(valid, predicted - actual, np.nan)
    mae   = float(np.nanmean(np.abs(error)))
    ax2.fill_between(times, 0, error, alpha=0.4, color="#e74c3c")
    ax2.axhline(0, color="#aaa", lw=0.5)
    ax2.text(0.02, 0.82, f"MAE = {mae:.2f} kW", transform=ax2.transAxes,
             fontsize=9, color="white",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", alpha=0.8))
    ax2.set_xlabel("Time (s)", color="white")
    ax2.set_ylabel("Error (kW)", color="white")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444")
    ax2.set_facecolor("#1a1a2e")
    ax2.grid(True, alpha=0.2, color="white")
    ax2.set_xlim(times[0], times[-1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return mae


# ============================================================================
# MENU ACTIONS
# ============================================================================

def _predict_csv(csv_path: Path, model, stats, show=False):
    """Core: predict a single *_hrr.csv. Returns (mae, save_path)."""
    import numpy as np, torch, pandas as pd

    df        = pd.read_csv(csv_path, skiprows=1)
    times_raw = df.iloc[:, 0].values.astype(float)
    hrr_raw   = df.iloc[:, 1].values.astype(float)
    t_end     = float(times_raw[-1])
    ct        = np.linspace(0, t_end, max(int(round(t_end / DT)) + 1, 40))
    hi        = np.interp(ct, times_raw, hrr_raw)

    hrr_norm = (hi - stats["mean"][0]) / (stats["std"][0] + 1e-8)
    seq      = np.stack([hrr_norm] + [np.zeros_like(hrr_norm)] * 5, axis=1)
    data_t   = torch.tensor(seq, dtype=torch.float32)

    T, seq_len, horizon = data_t.shape[0], 30, 10
    all_preds = np.full(T, np.nan)
    count     = np.zeros(T)

    model.eval()
    with torch.no_grad():
        for t in range(max(T - seq_len - horizon, 0)):
            x     = data_t[t: t + seq_len, :].unsqueeze(0)
            y_hat = model(x).squeeze().cpu().numpy()
            for h in range(horizon):
                idx = t + seq_len + h
                if idx < T:
                    if np.isnan(all_preds[idx]):
                        all_preds[idx] = 0.0
                    all_preds[idx] += y_hat[h]
                    count[idx] += 1

    valid = count > 0
    all_preds[valid] /= count[valid]
    pred_kw = denormalise_hrr(all_preds, stats)
    mae     = float(np.nanmean(np.abs(pred_kw[~np.isnan(pred_kw)] - hi[~np.isnan(pred_kw)])))

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / f"{csv_path.stem}_prediction.png"
    plot_scenario(ct, hi, pred_kw, csv_path.stem,
                  save_path=str(save_path), show=show)
    return mae, save_path


def action_quick_predict():
    """Predict a single *_hrr.csv picked from Input/ or by path."""
    print_section("🎯 QUICK PREDICTION")

    INPUT_DIR.mkdir(exist_ok=True)
    csv_files = sorted(INPUT_DIR.glob("*_hrr.csv"))

    if csv_files:
        print("Files in Input/:")
        for i, f in enumerate(csv_files, 1):
            size_kb = f.stat().st_size / 1024
            print(f"  {i}. {f.name}  ({size_kb:.0f} KB)")
        print()
    else:
        print("💡 Drop *_hrr.csv files into Input/ first, or enter a full path.\n")

    raw = input("File number or full path: ").strip().strip('"').strip("'")
    if not raw:
        return

    if raw.isdigit() and csv_files:
        idx = int(raw) - 1
        if 0 <= idx < len(csv_files):
            csv_path = csv_files[idx]
        else:
            print("❌ Invalid number."); return
    else:
        csv_path = Path(raw)
        if not csv_path.exists():
            print(f"❌ File not found: {csv_path}"); return

    print("\n📂 Loading model...")
    model, _ = load_model()
    if model is None: return
    train_ds, _ = load_dataset()
    if train_ds is None: return

    try:
        print("⚙️  Running prediction...\n")
        mae, save_path = _predict_csv(csv_path, model, train_ds.stats, show=True)
        print(f"\n✅ MAE: {mae:.2f} kW")
        print(f"💾 Saved: {save_path}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")


def action_batch_predict():
    """Predict every *_hrr.csv in Input/ → Output/ (mirrors fire_predict.py)."""
    import numpy as np

    print_section("📦 BATCH PROCESS — Input/ folder")

    INPUT_DIR.mkdir(exist_ok=True)
    csv_files = sorted(INPUT_DIR.glob("*_hrr.csv"))

    if not csv_files:
        print("❌ No *_hrr.csv files found in Input/")
        print(f"   Drop files into: {INPUT_DIR}")
        return

    print(f"Found {len(csv_files)} file(s) in Input/\n")
    print("📂 Loading model...")
    model, _ = load_model()
    if model is None: return
    train_ds, _ = load_dataset()
    if train_ds is None: return

    OUTPUT_DIR.mkdir(exist_ok=True)
    all_mae = []
    errors  = []

    for csv_path in csv_files:
        try:
            mae, save_path = _predict_csv(csv_path, model, train_ds.stats)
            all_mae.append((csv_path.stem, mae))
            print(f"  ✅ {csv_path.name:<48}  MAE: {mae:.2f} kW")
        except Exception as e:
            errors.append(csv_path.name)
            print(f"  ❌ {csv_path.name}: {e}")

    print("\n" + "=" * 65)
    print("  BATCH SUMMARY")
    print("=" * 65)
    if all_mae:
        maes = [m for _, m in all_mae]
        best = min(all_mae, key=lambda x: x[1])
        print(f"  Processed : {len(all_mae)} files")
        print(f"  Mean MAE  : {np.mean(maes):.2f} kW")
        print(f"  Best      : {best[1]:.2f} kW  ({best[0]})")
        print(f"  Baseline  : ~4.85 kW (standard LSTM)")
    if errors:
        print(f"  Errors    : {len(errors)} files failed")
    print(f"  Plots saved → {OUTPUT_DIR}")
    print("=" * 65)


def action_multi_step_demo():
    """Show Koopman multi-step ahead prediction advantage."""
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    print_section("⚡ MULTI-STEP PREDICTION DEMO")
    print("Demonstrates Koopman's K^n power: predict 10/20/50/100 steps")
    print("from a single latent encoding.\n")

    model, _ = load_model()
    if model is None:
        return
    train_ds, test_ds = load_dataset()
    if test_ds is None:
        return

    stats   = train_ds.stats
    scenario = test_ds.processed[0]
    data    = scenario["data"]
    name    = scenario["name"]
    seq_len = test_ds.input_seq_len

    t0 = len(data) // 3
    x  = data[t0: t0 + seq_len, :].unsqueeze(0)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    t_input   = np.arange(seq_len) * DT
    input_hrr = denormalise_hrr(data[t0: t0 + seq_len, 0].numpy(), stats)
    ax.plot(t_input, input_hrr, color="#7fdbff", lw=2, label="Input window")

    horizons = [10, 20, 50, 100]
    colors   = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]

    model.eval()
    with torch.no_grad():
        for h, color in zip(horizons, colors):
            max_h = min(h, len(data) - t0 - seq_len)
            if max_h <= 0:
                continue
            y_hat = model.predict_sequence(x, steps=max_h).squeeze().cpu().numpy()
            t_p   = seq_len * DT + np.arange(max_h) * DT
            ax.plot(t_p, denormalise_hrr(y_hat, stats), "-",
                    color=color, lw=1.5, label=f"K^n {max_h}-step", alpha=0.85)

    future_len = min(100, len(data) - t0 - seq_len)
    if future_len > 0:
        t_a = seq_len * DT + np.arange(future_len) * DT
        ax.plot(t_a, denormalise_hrr(
            data[t0 + seq_len: t0 + seq_len + future_len, 0].numpy(), stats),
            "w--", lw=1, alpha=0.5, label="Actual future")

    ax.axvline(seq_len * DT, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("HRR (kW)", color="white")
    ax.set_title(f"⚡ Koopman Multi-Step: {name}", color="white",
                 fontsize=12, fontweight="bold")
    ax.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.2, color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    plt.tight_layout()
    PRED_DIR.mkdir(exist_ok=True)
    save_path = OUTPUT_DIR / "multi_step_demo.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n💾 Saved: {save_path}")


def action_sync():
    """Sync ml_data/ from fds_scenarios/ and rebuild splits."""
    import numpy as np
    import pandas as pd

    print_section("🔄 SYNC FROM FDS_SCENARIOS")

    if not FDS_DIR.exists():
        print(f"❌ fds_scenarios/ not found at {FDS_DIR}")
        return

    scenario_dirs = sorted([d for d in FDS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(scenario_dirs)} scenario folders\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records  = []
    synced   = 0
    skipped  = 0

    for d in scenario_dirs:
        hrr_src = d / f"{d.name}_hrr.csv"
        if not hrr_src.exists():
            skipped += 1
            continue
        try:
            raw = pd.read_csv(hrr_src, skiprows=1)
            t   = raw.iloc[:, 0].values.astype(float)
            h   = raw.iloc[:, 1].values.astype(float)
            ok  = np.isfinite(t) & np.isfinite(h)
            t, h = t[ok], h[ok]
            if len(t) < 10:
                skipped += 1
                continue

            t_end = float(t[-1])
            ct    = np.linspace(0, t_end, max(int(round(t_end / 0.1)) + 1, 10))
            hi    = np.interp(ct, t, h)

            records.append({
                "scenario":     d.name,
                "peak_hrr":     float(hi.max()),
                "time_to_peak": float(ct[hi.argmax()]),
                "duration":     t_end,
                "hrr_series":   hi.tolist(),
                "q_radi_series": [0.0] * len(ct),
                "mlr_series":    [0.0] * len(ct),
            })
            synced += 1
            print(f"  ✅ {d.name}")
        except Exception as e:
            print(f"  ⚠️  {d.name}: {e}")
            skipped += 1

    # Write ml_dataset.json
    ml_path = DATA_DIR / "ml_dataset.json"
    with open(ml_path, "w") as f:
        json.dump({"n_scenarios": len(records), "dt": 0.1,
                   "scenarios": records}, f, indent=2)

    # Write splits (70/15/15)
    import random
    random.seed(42)
    names = [r["scenario"] for r in records]
    random.shuffle(names)
    n   = len(names)
    t1  = int(n * 0.70)
    t2  = int(n * 0.85)
    splits_path = DATA_DIR / "splits.json"
    with open(splits_path, "w") as f:
        json.dump({"train": names[:t1], "val": names[t1:t2],
                   "test": names[t2:]}, f, indent=2)

    print("\n" + "=" * 65)
    print("  ✅ SYNC COMPLETE")
    print("=" * 65)
    print(f"  Scenarios synced : {synced}")
    print(f"  Skipped          : {skipped}")
    print(f"  ml_dataset.json  : {len(records)} scenarios → {ml_path}")
    print(f"  splits.json      : train={t1}, val={t2-t1}, test={n-t2}")
    print("\n💡 Now use Option 5 (Train Model) to retrain.")


def action_train():
    """Launch training in a subprocess."""
    print_section("🧠 TRAIN KOOPMAN MODEL")
    print("This will retrain the Koopman LSTM on the dataset in ml_data/.")
    print("Training checkpoints → checkpoints/\n")

    epochs = input("Max epochs [50]: ").strip() or "50"
    confirm = input(f"Start training for {epochs} epochs? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Training cancelled.")
        return

    train_script = SCRIPT_DIR / "scripts" / "train.py"
    cmd = [sys.executable, str(train_script), "--epoch", epochs]
    print(f"\n🚀 Running: python scripts/train.py --epoch {epochs}\n")
    try:
        subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"❌ Training failed: {e}")


def action_manage_files():
    """File management submenu."""
    while True:
        clear_screen()
        print_banner()
        print("MANAGE FILES\n")
        print("  1. 📥 List Input/ files")
        print("  2. 📤 List Output/ files")
        print("  3. 🗂️  Open Input/ folder")
        print("  4. 🗂️  Open Output/ folder")
        print("  5. 🗂️  Open checkpoints/ folder")
        print("  6. 🧹 Clear Output/ folder")
        print("  7. ← Back\n")

        c = input("Choose (1-7): ").strip()
        if c == "1":
            files = sorted(INPUT_DIR.glob("*_hrr.csv")) if INPUT_DIR.exists() else []
            print_section("📥 INPUT FILES")
            if files:
                for f in files:
                    print(f"  • {f.name}  ({f.stat().st_size/1024:.0f} KB)")
            else:
                print("  (empty — drop *_hrr.csv files here)")
            press_enter()
        elif c == "2":
            files = sorted(OUTPUT_DIR.glob("*.png")) if OUTPUT_DIR.exists() else []
            print_section("📤 OUTPUT FILES")
            if files:
                for f in files: print(f"  • {f.name}")
            else:
                print("  (empty — run Quick/Batch Predict first)")
            press_enter()
        elif c == "3":
            INPUT_DIR.mkdir(exist_ok=True)
            if sys.platform == "win32": os.startfile(str(INPUT_DIR))
            press_enter()
        elif c == "4":
            OUTPUT_DIR.mkdir(exist_ok=True)
            if sys.platform == "win32": os.startfile(str(OUTPUT_DIR))
            press_enter()
        elif c == "5":
            CKPT_DIR.mkdir(exist_ok=True)
            if sys.platform == "win32": os.startfile(str(CKPT_DIR))
            press_enter()
        elif c == "6":
            files = list(OUTPUT_DIR.glob("*.png")) if OUTPUT_DIR.exists() else []
            if not files:
                print("\n  (Already empty)")
            else:
                confirm = input(f"  Delete {len(files)} plots? [y/N]: ").strip().lower()
                if confirm == "y":
                    for f in files: f.unlink()
                    print(f"  ✅ Deleted {len(files)} files.")
            press_enter()
        elif c == "7":
            break


def action_diagnostics():
    """Quick system / model health check."""
    print_section("🔧 DIAGNOSTICS")

    ok = True

    # Python packages
    for pkg in ["torch", "pytorch_lightning", "numpy", "matplotlib", "pandas"]:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}  ← pip install {pkg}")
            ok = False

    # Dataset
    if (DATA_DIR / "ml_dataset.json").exists():
        d = json.loads((DATA_DIR / "ml_dataset.json").read_text())
        print(f"\n  ✅ ml_dataset.json  — {d['n_scenarios']} scenarios")
    else:
        print(f"\n  ❌ ml_dataset.json missing — run Sync (Option 4)")
        ok = False

    # Checkpoint
    ckpt = find_checkpoint()
    if ckpt:
        size_mb = Path(ckpt).stat().st_size / 1_048_576
        print(f"  ✅ Checkpoint: {Path(ckpt).name}  ({size_mb:.1f} MB)")
    else:
        print("  ❌ No checkpoint found — run Train (Option 5)")
        ok = False

    print()
    if ok:
        print("  🎉 All systems ready!")
    else:
        print("  ⚠️  Some issues found. See above.")


def show_main_menu():
    print_banner()
    w = 65
    bar = "─" * (w - 2)
    print("┌" + bar + "┐")
    print("│" + "  MAIN MENU".ljust(w - 2) + "│")
    print("├" + bar + "┤")
    items = [
        ("1", "🎯", "Quick Predict",    "pick a file from Input/"),
        ("2", "📦", "Batch Process",    "all *_hrr.csv in Input/ → Output/"),
        ("3", "⚡", "Multi-Step Demo",  "Koopman K^n advantage demo"),
        ("4", "🔄", "Sync Scenarios",   "update dataset from fds_scenarios/"),
        ("5", "🧠", "Train Model",      "retrain on updated dataset"),
        ("6", "📁", "Manage Files",     "Input, Output, checkpoints"),
        ("7", "🔧", "Diagnostics",      "check packages & model"),
    ]
    for num, icon, name, desc in items:
        line = f"  {num}.  {icon}  {name:<18} {desc}"
        print("│" + line.ljust(w - 2) + "│")
    print("├" + bar + "┤")
    print("│" + "  0.  🚪  Exit".ljust(w - 2) + "│")
    print("└" + bar + "┘")
    print()


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    ACTIONS = {
        "1": action_quick_predict,
        "2": action_batch_predict,
        "3": action_multi_step_demo,
        "4": action_sync,
        "5": action_train,
        "6": action_manage_files,
        "7": action_diagnostics,
    }

    fire_splash()

    while True:
        clear_screen()
        show_main_menu()
        choice = input("Choose (1-7, 0 to exit): ").strip()

        if choice == "0":
            clear_screen()
            print("\n👋 Goodbye!\n")
            return 0

        if choice in ACTIONS:
            clear_screen()
            print_banner()
            try:
                ACTIONS[choice]()
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted.")
            press_enter()
        else:
            print("❌ Invalid option.")
            press_enter()


# ============================================================================
# COMMAND-LINE MODE
# ============================================================================

def command_line_mode(args):
    import argparse
    parser = argparse.ArgumentParser(
        description="Koopman Fire Prediction Tool",
        epilog="Run without arguments for interactive menu."
    )
    parser.add_argument("file", nargs="?",
                        help="CSV file to predict, or 'check'/'sync'/'train'")
    parser.add_argument("--batch",   action="store_true", help="Batch predict test set")
    parser.add_argument("--demo",    action="store_true", help="Multi-step demo")
    parser.add_argument("--version", action="store_true", help="Show version")
    parsed = parser.parse_args(args)

    if parsed.version:
        print_banner()
        return 0
    if parsed.batch:
        print_banner();  action_batch_predict();  return 0
    if parsed.demo:
        print_banner();  action_multi_step_demo();  return 0
    if parsed.file == "check":
        print_banner();  action_diagnostics();  return 0
    if parsed.file == "sync":
        print_banner();  action_sync();  return 0
    if parsed.file == "train":
        print_banner();  action_train();  return 0
    if parsed.file:
        # Treat as CSV path — quick predict
        print_banner()
        csv_path = Path(parsed.file)
        if not csv_path.exists():
            print(f"❌ File not found: {csv_path}")
            return 1
        # Put file into Input/ and run quick predict
        INPUT_DIR.mkdir(exist_ok=True)
        dst = INPUT_DIR / csv_path.name
        shutil.copy2(csv_path, dst)
        print(f"📥 Copied to Input/{csv_path.name}")
        action_quick_predict()
        return 0
    # No args → show help
    print_banner()
    print("USAGE:")
    print("  python predict.py              — interactive menu")
    print("  python predict.py file.csv     — predict single file")
    print("  python predict.py --batch      — batch predict test set")
    print("  python predict.py --demo       — multi-step demo")
    print("  python predict.py check        — diagnostics")
    print("  python predict.py sync         — sync from fds_scenarios")
    print("  python predict.py train        — train model\n")
    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    try:
        if len(sys.argv) > 1:
            return command_line_mode(sys.argv[1:])
        return interactive_mode()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   Run: python predict.py check")
        return 1


if __name__ == "__main__":
    sys.exit(main())
