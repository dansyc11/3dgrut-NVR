#!/usr/bin/env python3
import torch
import os

print("Working directory:", os.getcwd())

pth_path = "data/load_from_ply/chkpnt20000.pth"
print("Loading checkpoint:", pth_path)
ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)

# We already know ckpt is a tuple of length 2
print("Top‑level:", type(ckpt), "length =", len(ckpt))
print(" Index 0 type:", type(ckpt[0]), "length =", len(ckpt[0]))
print(" Index 1 type:", type(ckpt[1]), "value =", ckpt[1])

# Now inspect the contents of ckpt[0], which is another tuple
nested = ckpt[0]
for i, element in enumerate(nested):
    print(f"  → nested index {i}: type {type(element)}")
    if isinstance(element, dict):
        print(f"     ⤷ This dict has keys: {list(element.keys())}")
    else:
        # If it’s not a dict, but might be a tensor, just print its type/shape
        if torch.is_tensor(element):
            print(f"     ⤷ Tensor with shape {tuple(element.shape)}")
        else:
            print(f"     ⤷ {repr(element)} (not a dict or tensor)")
