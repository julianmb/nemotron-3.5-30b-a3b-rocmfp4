#!/usr/bin/env python3
"""
nvfp4_dequant_match.py

Validates that the signed E2M1 fp4 dequantization scheme matches the BF16 ground
truth for output.weight (lm_head.weight) to within fp4 quantization noise.

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
    parser = argparse.ArgumentParser(description="Validate NVFP4 dequant scheme vs BF16 ground truth")
    parser.add_argument("--nvfp4", default="models/hf-nvfp4", help="Path to NVFP4 HF dir")
    parser.add_argument("--bf16",  default="models/hf-bf16",  help="Path to BF16 HF dir")
    args = parser.parse_args()

    print(f"Loading lm_head from {args.nvfp4} and {args.bf16}...")
    w  = load_tensor(args.nvfp4, "lm_head.weight")        # uint8 packed
    s  = load_tensor(args.nvfp4, "lm_head.weight_scale")  # e4m3
    s2 = float(load_tensor(args.nvfp4, "lm_head.weight_scale_2").item())
    ref = load_tensor(args.bf16, "lm_head.weight")       # bf16 ground truth

    out_features, n_blocks = w.shape[0], s.shape[1]
    wb = w.view(out_features, n_blocks, 8)
    lo, hi = (wb & 0x0F).float(), (wb >> 4).float()
    vals = torch.stack([lo, hi], dim=-1).reshape(out_features, n_blocks, 16)

    # Signed E2M1 (fp4) lookup table
    tab = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)

    dq = tab[vals.long()].reshape(out_features, n_blocks * 16) * s.float().repeat_interleave(16, dim=-1) * s2
    err = (dq - ref.float()).abs().max().item()
    rms = ((dq - ref.float()).pow(2).mean().sqrt()).item()

    print(f"Dequant vs Ground Truth: max_err = {err:.5f}, rms_err = {rms:.5f}")
    if rms < 0.01:
        print("PASS: Signed E2M1 fp4 dequantization verified.")
    else:
        print("FAIL: Mismatch in dequantization logic.")
        sys.exit(1)

if __name__ == "__main__":
    main()
