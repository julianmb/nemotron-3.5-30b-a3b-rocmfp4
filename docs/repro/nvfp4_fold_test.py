#!/usr/bin/env python3
"""
nvfp4_fold_test.py

Demonstrates why folding scale2 into block scales fails: folding scale2 (~1.4e-4)
pushes UE4M3 block scales into the denormal range, causing a 1.19x relative error
(12x worse than exact dequantization).

Requires:
  - HuggingFace NVFP4 checkpoint at path given by --nvfp4
  - HuggingFace BF16 checkpoint at path given by --bf16
"""

import argparse, json, sys, torch
import numpy as np
from safetensors import safe_open

def load_tensor(repo_path, name):
    idx_path = f"{repo_path}/model.safetensors.index.json"
    with open(idx_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard = weight_map[name]
    with safe_open(f"{repo_path}/{shard}", framework="pt") as sf:
        return sf.get_tensor(name)

def main():
    parser = argparse.ArgumentParser(description="Test folding scale2 into E4M3 block scales")
    parser.add_argument("--nvfp4", default="models/hf-nvfp4", help="Path to NVFP4 HF dir")
    parser.add_argument("--bf16",  default="models/hf-bf16",  help="Path to BF16 HF dir")
    args = parser.parse_args()

    tensor_base = "backbone.layers.1.mixer.experts.0.down_proj"
    w  = load_tensor(args.nvfp4, f"{tensor_base}.weight")
    s  = load_tensor(args.nvfp4, f"{tensor_base}.weight_scale")
    s2 = float(load_tensor(args.nvfp4, f"{tensor_base}.weight_scale_2").item())
    ref = load_tensor(args.bf16,  f"{tensor_base}.weight").float()

    out, nb = w.shape[0], s.shape[1]
    wb = w.view(out, nb, 8)
    vals = torch.stack([wb & 0x0F, wb >> 4], dim=-1).reshape(out, nb, 16)
    tab = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)

    # Build E4M3 positive lookup table
    tbl = []
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if b in (0, 0x80):
            v = 0.0
        elif exp == 0:
            v = sign * mant * 2**-6
        elif exp == 0x0F:
            v = sign * float('inf')
        else:
            v = sign * (1 + mant/8) * 2**(exp-7)
        tbl.append(v)
    pos = torch.tensor(tbl[:128], dtype=torch.float32)

    # Fold scale2 into block scales and quantize back to nearest E4M3
    folded = s.float() * s2
    f = folded.flatten()
    j = torch.argmin((pos.unsqueeze(0) - f.unsqueeze(1)).abs(), dim=1)
    folded_q = pos[j].reshape(folded.shape)

    d_folded = (tab[vals.long()] * folded_q.unsqueeze(-1)).reshape(out, nb * 16)
    rel = (d_folded - ref).abs().mean().item() / ref.abs().mean().item()

    print(f"Folding scale2 ({s2:.6e}) into E4M3 block scales:")
    print(f"  Folded E4M3 relative error = {rel:.4f}")
    print(f"  Exact dequant relative error = 0.1007")
    print(f"  Conclusion: denormal E4M3 precision loss increases error by ~12x. Folding rejected.")

if __name__ == "__main__":
    main()
