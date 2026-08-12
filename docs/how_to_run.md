# How to Build, Convert, Quantize, and Run

This document provides complete instructions for building the ROCmFPX fork, converting Nemotron 3.5 Lightning 30B-A3B from Hugging Face, quantizing to ROCmFP4, and running inferences on AMD hardware.

---

## 1. Environment & Build (Strix Halo / `gfx1151`)

### System Requirements
- Linux (Ubuntu 24.04+ / Arch / Fedora)
- C++ toolchain: `clang` / `gcc`, `cmake`, `ninja`
- ROCm / HIP SDK (tested on ROCm 7.2.3)
- Vulkan tools & headers (`glslc`, `vulkan-tools`, `libvulkan-dev`)
- Python 3.10+ with `torch` and `transformers`

### Build the ROCmFPX Fork

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX

# Apply the two converter format-patches from this repo (git am or git apply)
git am /path/to/patches/0001-converter-detect-w4a16-nvfp4.patch
git am /path/to/patches/0002-converter-dequant-output-to-f16.patch

# Build Strix Halo binaries (gfx1151)
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
```

Binaries land in `build-strix-rocmfp4/bin/`:
- `llama-completion` / `llama-cli`
- `llama-quantize`
- `llama-bench`
- `llama-perplexity`

---

## 2. Conversion & Quantization (BF16 Source)

To achieve maximum quality (**PPL 5.9936**), quantize from the clean **BF16 Base checkpoint**:

```bash
# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers gguf

# 1. Download HF BF16 Base Model
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16 --local-dir models/hf-bf16

# 2. Convert HF Checkpoint -> BF16 GGUF
python convert_hf_to_gguf.py models/hf-bf16 \
  --outtype bf16 \
  --outfile models/gguf/Nemotron-3.5-Lightning-30B-A3B-Base-BF16.gguf

# 3. Quantize to ROCmFP4 (with context length fix)
# Note: --override-kv MUST precede positional arguments, and type token is int:

# Recommended (Strix Halo K/V recipe):
build-strix-rocmfp4/bin/llama-quantize \
  --override-kv nemotron_h_moe.context_length=int:262144 \
  models/gguf/Nemotron-3.5-Lightning-30B-A3B-Base-BF16.gguf \
  models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
  Q4_0_ROCMFP4_STRIX_LEAN 32

# Speed-first (Single-scale layout):
build-strix-rocmfp4/bin/llama-quantize \
  --override-kv nemotron_h_moe.context_length=int:262144 \
  models/gguf/Nemotron-3.5-Lightning-30B-A3B-Base-BF16.gguf \
  models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-FAST.gguf \
  Q4_0_ROCMFP4_FAST 32

# Agent / Tool Calling (Protected embeddings):
build-strix-rocmfp4/bin/llama-quantize \
  --override-kv nemotron_h_moe.context_length=int:262144 \
  models/gguf/Nemotron-3.5-Lightning-30B-A3B-Base-BF16.gguf \
  models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-COHERENT.gguf \
  Q4_0_ROCMFP4_COHERENT 32
```

---

## 3. Running Inferences

### Vulkan0 (Recommended)

```bash
build-strix-rocmfp4/bin/llama-completion \
  -m models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
  -p "What is 2+2?" -n 64 \
  -dev Vulkan0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 8192
```

### HIP/ROCm0 (Unified Memory for iGPUs/APUs)

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
build-strix-rocmfp4/bin/llama-completion \
  -m models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
  -p "What is 2+2?" -n 64 \
  -dev ROCm0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 8192
```

### Interactive Chat Mode

```bash
build-strix-rocmfp4/bin/llama-cli \
  -m models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
  --jinja -if -dev Vulkan0 -ngl 999 -fa on -c 32768
```

---

## 4. Measuring Benchmarks & Quality

### Speed Benchmark (`llama-bench`)
```bash
build-strix-rocmfp4/bin/llama-bench \
  -m models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
  -p 512 -n 128 -t 16 -dev Vulkan0 -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0
```

### Perplexity Evaluation (`llama-perplexity`)
```bash
scripts/get-wikitext-2.sh
build-strix-rocmfp4/bin/llama-perplexity \
  -m models/gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
  -f wikitext-2-raw/wiki.test.raw -c 512 -b 512 -t 16 -dev Vulkan0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0
```

---

## 5. Verification Scripts (`docs/repro/`)

If you have downloaded both HF checkpoints (`models/hf-nvfp4` and `models/hf-bf16`), you can run the reproduction scripts to verify the NVFP4 math and scaling findings:

```bash
# Verify signed E2M1 fp4 dequantization against BF16 ground truth
python3 docs/repro/nvfp4_dequant_match.py

# Check the 7110x relative error when omitting scale2
python3 docs/repro/nvfp4_scale2_check.py

# Demonstrate why folding scale2 into E4M3 block scales causes 1.19x relative error
python3 docs/repro/nvfp4_fold_test.py
```
