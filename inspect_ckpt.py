
import torch
import sys

try:
    path = "model/best_model.ckpt"
    print(f"Loading {path}...")
    ckpt = torch.load(path, map_location='cpu')
    print(f"Type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Keys: {list(ckpt.keys())}")
    else:
        print("Checkpoint is not a dict (likely a state_dict direct save)")
except Exception as e:
    print(f"Error: {e}")
