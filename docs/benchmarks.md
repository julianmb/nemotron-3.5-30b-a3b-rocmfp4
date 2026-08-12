# Benchmarks & Quality

All numbers were measured locally on a **Framework AMD Strix Halo** (Ryzen AI Max 395+, Radeon 8050S/8060S, `gfx1151`, 128 GB Unified RAM, ROCm 7.2.3), build `b209-00d5452` of the ROCmFPX fork.

---

## Hardware & Environment Notes

- **Unified Memory Architecture (UMA)**: The GPU is an APU iGPU with a 512 MB fixed VRAM carve-out. Setting `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` enables ROCm's HIP backend to allocate directly from system RAM (ROCm reports **122,880 MiB VRAM**), allowing 16+ GB models to fully offload (`-ngl 999`).
- **Target Version Override**: `HSA_OVERRIDE_GFX_VERSION=11.5.1` is required for `gfx1151` on ROCm releases that do not list the target.

---

## Weight-Quantization Benchmarks (`llama-bench`)

Parameters: `llama-bench -p 512 -n 128`, `q8_0` KV cache (`-ctk q8_0 -ctv q8_0`), FlashAttention enabled (`-fa 1`), full GPU offload (`-ngl 999`), 16 threads (`-t 16`).

| Preset | GGUF Size | Vulkan0 Prompt (pp512) | Vulkan0 Decode (tg128) | ROCm0 Prompt (pp512) | ROCm0 Decode (tg128) |
|:-------|:---------:|:----------------------:|:---------------------:|:-------------------:|:-------------------:|
| **`FAST`** (`Q4_0_ROCMFP4_FAST`) | 15.65 GiB | **1310.46 ± 7.64 t/s** | **85.99 ± 0.11 t/s** | 1079.34 ± 7.51 t/s | 80.32 ± 0.08 t/s |
| **`STRIX_LEAN`** (`Q4_0_ROCMFP4_STRIX_LEAN`) | 15.72 GiB | **1299.73 ± 6.93 t/s** | **85.62 ± 0.28 t/s** | 1075.44 ± 6.93 t/s | 79.38 ± 0.28 t/s |
| **`COHERENT`** (`Q4_0_ROCMFP4_COHERENT`) | 16.74 GiB | **1290.38 ± 13.74 t/s** | **81.57 ± 0.27 t/s** | 1302.21 ± 6.94 t/s | 77.75 ± 0.24 t/s |

> **Backend Comparison:** Vulkan0 outperforms ROCm0 on Strix Halo by **~21%** in prompt processing and **~8%** in decode speed.

---

## Interactive-Generation Speeds (`llama-completion`)

Parameters: `llama-completion`, 7-token prompt, greedy sampling (`--temp 0`), Vulkan0, full GPU offload:

| Artifact | Prompt Speed | Decode Speed |
|:---------|:------------:|:------------:|
| **`FAST`** (BF16-derived) | 199.8 t/s | 83.4 t/s |
| **`STRIX_LEAN`** (BF16-derived) | 184–200 t/s | 76–83 t/s |
| **`COHERENT`** (BF16-derived) | 188.8 t/s | 81.0 t/s |
| **CPU Reference** (`-ngl 0`, same model) | 47.4 t/s | 37.2 t/s |

---

## Quality & Perplexity (`llama-perplexity`)

Dataset: `wikitext-2-raw/wiki.test.raw`, `-c 512 -b 512`, `q8_0` KV cache, FlashAttention enabled (`-fa on`), Vulkan0:

| Model / Source Path | Perplexity | Status / Findings |
|:-------------------|:----------:|:------------------|
| **BF16 → ROCmFP4 STRIX_LEAN** | **5.9936 ± 0.0358** | **Promoted & Delivered** (Clean quality baseline) |
| **Native NVFP4** | 109.7910 ± 0.9683 | Incoherent (Missing `scale2` factor in kernels) |
| **NVFP4 → ROCmFP4 Remap** | Incoherent | Incoherent (Missing `scale2` factor in quantizer) |

---

## How to Reproduce These Numbers

```bash
# 1. Run llama-bench
build-strix-rocmfp4/bin/llama-bench \
  -m models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-ROCmFP4-STRIX_LEAN.gguf \
  -p 512 -n 128 -t 16 -dev Vulkan0 -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0

# 2. Run perplexity evaluation
scripts/get-wikitext-2.sh
build-strix-rocmfp4/bin/llama-perplexity \
  -m models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-ROCmFP4-STRIX_LEAN.gguf \
  -f wikitext-2-raw/wiki.test.raw -c 512 -b 512 -t 16 -dev Vulkan0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0
```
