# Nemotron 3.5 Lightning 30B-A3B → ROCmFP4

Converter fixes, measurements, and build/run notes for quantizing
[NVIDIA Nemotron 3.5 Lightning 30B-A3B Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16)
to the experimental **ROCmFP4** GGUF family on AMD (Strix Halo / `gfx1151`).

Pre-built GGUF files: **[julianmb/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-GGUF](https://huggingface.co/julianmb/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-GGUF)** on Hugging Face.

---

## Model Variants & Performance Summary

All benchmarks run on **Framework AMD Strix Halo** (Ryzen AI Max, `gfx1151`, 128 GB unified memory, ROCm 7.2.3) using Vulkan0:

| Preset | BPW | Size | Prompt Eval (pp512) | Decode Speed (tg128) | Recommended Use Case |
|:-------|----:|-----:|--------------------:|---------------------:|:---------------------|
| **`STRIX_LEAN`** | ~4.38 | 15.73 GiB | 1299.7 t/s | 85.6 t/s | **General / Balanced** (Strix Halo K/V recipe + Q5_K token embeddings) |
| **`FAST`** | ~4.25 | 15.66 GiB | 199.8 t/s | 83.4 t/s | **Maximum Speed** (Single-scale speed layout) |
| **`COHERENT`** | ~4.70 | 16.74 GiB | 188.8 t/s | 81.0 t/s | **Agentic / Coding** (Protected token & output embeddings) |

*Note: Token embeddings for 2688 dimensions fall back from `Q5_K` / `Q6_K` to `Q5_1` / `Q8_0` because 2688 is not divisible by 256.*

---

## What this repo contains

- `patches/` — two `git format-patch` fixes against commit `00d54526e24e3aba4c76474e3147cbf9c7cc034c` for `convert_hf_to_gguf.py`:
  - `0001-converter-detect-w4a16-nvfp4.patch`: Fixes detection for `W4A16_NVFP4` ModelOpt tags.
  - `0002-converter-dequant-output-to-f16.patch`: Dequantizes `output.weight` to F16 (prevents unloadable `output.scale`).
- `docs/benchmarks.md` — detailed speed benchmarks (Vulkan vs ROCm) and perplexity results.
- `docs/nvfp4-findings.md` — analysis of native NVFP4 vs ROCmFP4 remap.
- `docs/repro/` — self-contained Python scripts (`nvfp4_dequant_match.py`, `nvfp4_scale2_check.py`, `nvfp4_fold_test.py`) verifying the math.
- `docs/how_to_run.md` — step-by-step build, conversion, and quantization commands.
- `AI_CHANGES.md` — change log of AI-assisted work per project policy.

---

## Key Conclusions

1. **Clean BF16 → ROCmFP4 Path**: Excellent quality and speed (**PPL 5.9936 ± 0.0358** on wikitext-2).
2. **NVFP4 Path Findings**:
   - **Native NVFP4** loads and runs without crashing after applying our patches, but gets PPL **109.79** because runtime kernels/graph do not integrate ModelOpt's companion `scale2` factor (`~1.4e-4`).
   - **NVFP4 → ROCmFP4 Remap** fails because the quantizer's `dequantize_row_nvfp4` ignores `scale2`, causing a 7110× error on expert weights.
3. **Vulkan > ROCm on Strix Halo**: Vulkan0 outperforms ROCm0 by ~21% on prompt evaluation (1299.7 vs 1075.4 t/s) and ~8% on decode (85.6 vs 79.4 t/s).
4. **Context Length**: Fixed `context_length = 262144` (256K) via `--override-kv nemotron_h_moe.context_length=int:262144`.

---

## Provenance & AI Disclosure

- **AI-Assisted Work**: The converter fixes and documentation were developed with AI assistance in accordance with [AGENTS.md](AI_CHANGES.md).
- **Upstream Policy**: Per project guidelines, these patches are provided as standalone patches for manual application (`git am`) rather than as automated pull requests.

## Base Commit & Pinning

- **Upstream Fork**: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)
- **Base Commit**: `00d54526e24e3aba4c76474e3147cbf9c7cc034c` (Branch `main`, 2026-08-09)

## License & Credits

- Model Weights: NVIDIA ([OpenMDW-1.1 License](https://openmdw.ai/license/1-1/))
- Code & Patches: MIT License (llama.cpp lineage)
- Quantization & Benchmarks: **julianmb**
