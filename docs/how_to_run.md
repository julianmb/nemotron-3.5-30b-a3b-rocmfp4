# How to run

The GGUFs use the ROCmFPX custom quant types and require the **ROCmFPX fork** of
llama.cpp. This documents the exact flow used.

## Build (Strix Halo / gfx1151)

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
# other GPUs: scripts/build-rdna2.sh | build-rdna3.sh | build-rdna4.sh
```

Binaries: `build-strix-rocmfp4/bin/{llama-cli,llama-completion,llama-quantize,llama-bench,llama-perplexity}`.

## Run

### Vulkan (recommended on Strix Halo)

```bash
build-strix-rocmfp4/bin/llama-completion \
  -m model-ROCmFP4-STRIX_LEAN.gguf \
  -p "What is 2+2?" -n 64 -dev Vulkan0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 8192
```

### HIP/ROCm (unified memory — enables APU/iGPU with small VRAM carve-out)

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
build-strix-rocmfp4/bin/llama-completion \
  -m model-ROCmFP4-STRIX_LEAN.gguf \
  -p "What is 2+2?" -n 64 -dev ROCm0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 8192
```

Interactive chat:

```bash
build-strix-rocmfp4/bin/llama-cli -m model.gguf --jinja -if \
  -dev Vulkan0 -ngl 999 -fa on -c 32768
```

## Quantize from BF16 source

```bash
# 1. HF -> BF16 GGUF (requires the fork's convert script + torch env)
python convert_hf_to_gguf.py /path/to/hf-bf16 --outtype bf16 --outfile model-BF16.gguf

# 2. BF16 GGUF -> ROCmFP4 (STRIX_LEAN / FAST / COHERENT)
build-strix-rocmfp4/bin/llama-quantize \
  model-BF16.gguf out-STRIX_LEAN.gguf Q4_0_ROCMFP4_STRIX_LEAN

# apply the context-length fix at the same time:
build-strix-rocmfp4/bin/llama-quantize \
  --override-kv nemotron_h_moe.context_length=int:262144 \
  model-BF16.gguf out-STRIX_LEAN.gguf Q4_0_ROCMFP4_STRIX_LEAN 32
```

> `--override-kv` must come **before** the positional args, and the type token
> is `int:` (not `u32:`), or the override is silently ignored.

## Measure

```bash
build-strix-rocmfp4/bin/llama-bench -m model.gguf -p 512 -n 128 -dev Vulkan0 -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0
build-strix-rocmfp4/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw -c 512 -b 512 -dev Vulkan0 -ngl 999 -fa on -ctk q8_0 -ctv q8_0
```

## Apply the converter patches

Both fixes apply to the ROCmFPX fork's `convert_hf_to_gguf.py`:

```bash
git apply patches/0001-converter-detect-w4a16-nvfp4.patch
git apply patches/0002-converter-dequant-output-to-f16.patch
```

Patch 1 alone fixes the NVFP4 checkpoint conversion crash; patch 2 makes the
resulting NVFP4 GGUF loadable. (The NVFP4→ROCmFP4 remap still produces garbage
for models with non-trivial `scale2` — see `nvfp4-findings.md`.)
