import torch
import argparse

checkpoint_path = "models/mbart50-one-to-many/model.pt"
output_path = "models/mbart50-one-to-many/model_fixed.pt"

# Load checkpoint and allow unpickling Namespace
from torch.serialization import add_safe_globals
add_safe_globals([argparse.Namespace])

# Load checkpoint
print(f"Loading checkpoint from {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Fix architecture name if needed
if "args" in ckpt and hasattr(ckpt["args"], "arch"):
    if ckpt["args"].arch == "denoising_large":
        print("Renaming architecture 'denoising_large' -> 'mbart_large'")
        ckpt["args"].arch = "mbart_large"

# Save clean version
print(f"Saving fixed checkpoint to {output_path}")
torch.save(ckpt, output_path)
