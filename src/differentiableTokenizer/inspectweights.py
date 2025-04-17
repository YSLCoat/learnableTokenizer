#!/usr/bin/env python3
"""
Print the weights of `edge_detection_conv` from a PyTorch checkpoint.

Usage
-----
python print_edge_weights.py /path/to/learnableGradMapTokenizer_boundary_path_finder_196_.pt
"""

import sys
import re
import torch

def load_state_dict(path):
    """Return a pure state‑dict regardless of how the checkpoint was saved."""
    obj = torch.load(path, map_location="cpu")

    # Case 1: a full nn.Module was saved
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()

    # Case 2: obj is already a state‑dict
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    # Case 3: obj is a dict wrapper (typical training checkpoint)
    for key in ("state_dict", "model_state_dict", "net", "model"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    raise RuntimeError(
        "Could not find a state_dict in the checkpoint. "
        "Inspect the keys manually: " + ", ".join(obj.keys())
    )

def main(path):
    sd = load_state_dict(path)

    # Look for any parameter whose name contains "edge_detection_conv"
    pattern = re.compile(r"edge[_\.]?detection[_\.]?conv.*weight", re.I)

    found = False
    for name, tensor in sd.items():
        if pattern.search(name):
            found = True
            print(f"\n{name}  –  shape: {tuple(tensor.shape)}")
            print(tensor)                    # raw values
            print("-" * 80)

    if not found:
        print("No parameter matching 'edge_detection_conv.*weight' was found.\n"
              "Try printing all keys to see the exact naming:\n"
              "for k in sd: print(k)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python print_edge_weights.py <checkpoint.pt>")
    main(sys.argv[1])
