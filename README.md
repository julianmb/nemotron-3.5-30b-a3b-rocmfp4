# ⚡ NVIDIA Nemotron 3.5 Lightning 30B-A3B → ROCmFP4

<p align="center">
  <a href="https://huggingface.co/julianmb/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-GGUF"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GGUF%20Models-yellow?style=for-the-badge" alt="Hugging Face Models"></a>
  <a href="https://github.com/charlie12345/ROCmFPX"><img src="https://img.shields.io/badge/Fork-ROCmFPX-blue?style=for-the-badge" alt="ROCmFPX Fork"></a>
  <img src="https://img.shields.io/badge/Hardware-AMD%20Strix%20Halo-red?style=for-the-badge" alt="AMD Strix Halo">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License MIT">
</p>

Converter fixes, empirical findings, and Strix Halo benchmarks for quantizing [NVIDIA Nemotron 3.5 Lightning 30B-A3B Base (BF16)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16) to the experimental **ROCmFP4** GGUF family on AMD hardware (`gfx1151`).

> [!NOTE]
> **Pre-built GGUF Models Available:** Download ready-to-run models directly from Hugging Face:
> **[julianmb/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-GGUF](https://huggingface.co/julianmb/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-GGUF)**

---

## 📌 Table of Contents

- [Model Variants & Benchmarks](#model-variants--benchmarks)
- [Repository Structure](#what-this-repo-contains)
- [Key Engineering Findings](#key-engineering-findings)
- [Sample Output](#sample-generation)
- [Quick Start](#quick-start)
- [Provenance & License](#provenance--license)

---

## 📊 Model Variants & Benchmarks

All performance metrics were measured on a **Framework AMD Strix Halo** (Ryzen AI Max 395+, `gfx1151`, 128 GB Unified RAM, ROCm 7.2.3) using full GPU offload (`-ngl 999`), FlashAttention enabled (`-fa on`), and `q8_0` KV cache:

| Preset | BPW | Size | Vulkan0 Prompt (pp512) | Vulkan0 Decode (tg128) | Recommended Use Case |
|:-------|:---:|:----:|:------------------:|:-------------------:|:---------------------|
| 🏆 **`STRIX_LEAN`** | ~4.38 | 15.73 GiB | **1,299.7 t/s** | **85.6 t/s** | **General / Balanced** (Strix Halo K/V recipe + Q5_K token embeddings) |
| ⚡ **`FAST`** | ~4.25 | 15.66 GiB | **1,310.5 t/s** | **86.0 t/s** | **Maximum Speed** (Single-scale speed layout) |
| 🛠️ **`COHERENT`** | ~4.70 | 16.74 GiB | **1,290.4 t/s** | **81.6 t/s** | **Agentic / Coding** (Protected token & output embeddings) |

> [!TIP]
> **Vulkan vs ROCm Performance:** On Strix Halo (`gfx1151`), **Vulkan0** outperforms **ROCm0** by **~21%** in prompt processing on `STRIX_LEAN` & `FAST` (1299.7 vs 1075.4 t/s) and **~8%** in decode speed. On `COHERENT`, prompt eval is tied (~1290 vs ~1302 t/s) while Vulkan retains the decode edge (81.6 vs 77.8 t/s).

*Note: Token embeddings for 2688 hidden dimensions fall back from `Q5_K`/`Q6_K` to `Q5_1`/`Q8_0` because 2688 is not divisible by 256.*

---

## 📁 What This Repo Contains

```text
.
├── patches/
│   ├── 0001-converter-detect-w4a16-nvfp4.patch   # Fixes ModelOpt W4A16_NVFP4 conversion crash
│   └── 0002-converter-dequant-output-to-f16.patch # Fixes unloadable GGUFs due to output.scale
├── docs/
│   ├── benchmarks.md                            # Complete speed & perplexity tables
│   ├── nvfp4-findings.md                        # Deep-dive on ModelOpt NVFP4 scale2 issues
│   ├── how_to_run.md                            # Step-by-step build, quantize, & run commands
│   └── repro/                                   # Standalone Python scripts verifying the math
│       ├── nvfp4_dequant_match.py               # Verifies signed E2M1 fp4 dequant (RMS 0.0023)
│       ├── nvfp4_scale2_check.py                # Demonstrates the 7110x error without scale2
│       └── nvfp4_fold_test.py                   # Proves why E4M3 scale folding fails (1.19x rel err)
├── AI_CHANGES.md                                # Authoritative AI session change log
├── test_patches.sh                              # Validation script for git am / git apply
├── push.sh                                      # GitHub repository management script
└── LICENSE                                      # MIT License (ggml-org / llama.cpp lineage)
```

---

## 🔍 Key Engineering Findings

1. **Clean BF16 → ROCmFP4 Path**: Produces exceptional quality and speed (**Perplexity 5.9936 ± 0.0358** on `wikitext-2`).
2. **NVFP4 Path Root Causes**:
   - **Fix 1 (Detection)**: ModelOpt tags per-layer quantization as `W4A16_NVFP4`. Fixed in `0001-converter-detect-w4a16-nvfp4.patch`.
   - **Fix 2 (Unloadable GGUFs)**: `output.weight` in ModelOpt NVFP4 emits an unmapped `output.scale` tensor. Fixed in `0002-converter-dequant-output-to-f16.patch` by dequantizing `output.weight` to F16.
   - **Requantization Defect**: Requantizing NVFP4 → ROCmFP4 produces garbage because `dequantize_row_nvfp4` in C++ ignores ModelOpt's companion `scale2` factor (`~1.4e-4`), causing a **7,110× weight scaling error**.
3. **Context Length Override**: GGUF `context_length` is corrected to **262,144** (256K tokens) via `--override-kv nemotron_h_moe.context_length=int:262144`.

Read the full analysis in [`docs/nvfp4-findings.md`](docs/nvfp4-findings.md).

---

## 💬 Sample Generation

Running `llama-completion` on `STRIX_LEAN.gguf` via Vulkan0:

```text
$ llama-completion -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
    -p "What is 2+2?" -n 8 --temp 0 -dev Vulkan0 -ngl 999

user: What is 2+2?
assistant: Answer: 4

[ Prompt: 184.0 t/s | Generation: 68.1 t/s (cold-start, 7 tokens) ]
```

*Standardized `llama-bench` tg128 decode: 85.6 t/s on Vulkan0.*

---

## 🚀 Quick Start

> [!IMPORTANT]
> These custom `Q4_0_ROCMFP4_*` GGUFs require the [ROCmFPX llama.cpp fork](https://github.com/charlie12345/ROCmFPX).

### 1. Build the Fork
```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
```

### 2. Run Inferences
```bash
# Vulkan0 (Recommended)
build-strix-rocmfp4/bin/llama-completion \
  -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
  -p "Explain quantum computing in one sentence:" -n 64 \
  -dev Vulkan0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 8192

# HIP/ROCm (Unified Memory)
HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
build-strix-rocmfp4/bin/llama-completion \
  -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
  -p "Explain quantum computing in one sentence:" -n 64 \
  -dev ROCm0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 8192
```

For complete instructions and quantization guides, see [`docs/how_to_run.md`](docs/how_to_run.md).

---

## 📄 Provenance & License

- **Base Model Weights**: NVIDIA ([OpenMDW-1.1 License](https://openmdw.ai/license/1-1/))
- **Source Code & Patches**: MIT License (llama.cpp / ggml-org lineage)
- **Base Commit**: `00d54526e24e3aba4c76474e3147cbf9c7cc034c` (Branch `main`, 2026-08-09) on [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)
- **AI Disclosure**: Converter patches and documentation were developed with AI assistance in accordance with project standards. Patches are formatted for `git am` application.

---

<p align="center">
  Maintained by <b><a href="https://github.com/julianmb">julianmb</a></b>
</p>
