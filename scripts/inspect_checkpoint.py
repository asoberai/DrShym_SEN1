#!/usr/bin/env python3
"""
Inspect checkpoint structure to understand layer naming
"""
import torch
from pathlib import Path

def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint structure"""
    print(f"Inspecting checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("\n=== Checkpoint Keys ===")
    for key in checkpoint.keys():
        print(f"  {key}")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

        print(f"\n=== Model State Dict ({len(state_dict)} parameters) ===")

        # Group by layer type
        layers = {}
        for param_name in state_dict.keys():
            layer_type = param_name.split('.')[0]
            if layer_type not in layers:
                layers[layer_type] = []
            layers[layer_type].append(param_name)

        for layer_type in sorted(layers.keys()):
            print(f"\n{layer_type}:")
            for param in layers[layer_type]:
                shape = state_dict[param].shape
                print(f"  {param}: {shape}")

    if 'config' in checkpoint:
        print(f"\n=== Config ===")
        print(checkpoint['config'])

if __name__ == "__main__":
    checkpoint_path = "artifacts/checkpoints/best_optimized.pt"
    if Path(checkpoint_path).exists():
        inspect_checkpoint(checkpoint_path)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")