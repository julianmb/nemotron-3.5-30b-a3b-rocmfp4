#!/usr/bin/env python3
"""
nvfp4_scale2_check.py

Demonstrates the 7110x relative error caused when dequantizing NVFP4 weights
without applying ModelOpt's companion scale2 factor.

Requires:
  - HuggingFace NVFP4 checkpoint at path given by --nvfp4
  - HuggingFace BF16 checkpoint at path given by --bf16
"""

import argparse, json, sys, torch
from safetensors import safe_open

def load_tensor(repo_path, name):
    idx_path = f"{repo_path}/model.safetensors.index.json"
    with open(idx_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard = weight_map[name]
    with safe_open(f"{repo_path}/{shard}", framework="pt") as sf:
        return sf.get_tensor(name)

def main():
    parser = argparse.ArgumentParser(description="Check effect of companion scale2 on NVFP4 expert weights")
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

    d_block = (tab[vals.long()] * s.float().unsqueeze(-1)).reshape(out, nb * 16)
    rel_without = (d_block - ref).abs().mean().item() / ref.abs().mean().item()

    d_full = d_block * s2
    rel_with = (d_full - ref).abs().mean().item() / ref.abs().mean().item()

    print(f"Target tensor: {tensor_base}")
    print(f"  scale2 factor = {s2:.6e}")
    print(f"  WITHOUT scale2 (block scales only): rel_err = {rel_without:.2f}x")
    print(f"  WITH scale2    (full dequant):      rel_err = {rel_with:.4f} (~fp4 noise)")

if __name__ == "__main__":
    main()
