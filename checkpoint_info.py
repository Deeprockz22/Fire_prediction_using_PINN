"""
Quick script to display checkpoint information
"""
import torch
from pathlib import Path

checkpoint_path = "model/best_model.ckpt"

print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n" + "="*70)
print("CHECKPOINT INFORMATION")
print("="*70)

print(f"\nEpoch: {ckpt.get('epoch', 'N/A')}")
print(f"Training Loss: {ckpt.get('train_loss', 'N/A')}")
print(f"Validation Loss: {ckpt.get('val_loss', 'N/A')}")
print(f"Timestamp: {ckpt.get('timestamp', 'N/A')}")

print(f"\nConfiguration:")
config = ckpt.get('config', {})
for key, value in config.items():
    print(f"  {key}: {value}")

print(f"\nModel State:")
state_dict = ckpt['model_state_dict']
print(f"  Total parameters: {sum(p.numel() for p in state_dict.values()):,}")
print(f"  Model keys: {list(state_dict.keys())[:5]}...")

print(f"\nCheckpoint size: {Path(checkpoint_path).stat().st_size / 1024:.2f} KB")
print("="*70)
