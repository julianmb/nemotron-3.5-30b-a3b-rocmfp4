# Nemotron 3.5 Lightning 30B-A3B → ROCmFP4

Converter fixes, measurements, and build/run notes for quantizing
[NVIDIA Nemotron 3.5 Lightning 30B-A3B Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16)
to the experimental **ROCmFP4** GGUF family on AMD (Strix Halo / `gfx1151`).

Pre-built GGUF files: **[julianmb/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-GGUF](https://huggingface.co/julianmb/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-GGUF)** on the Hugging Face Hub.

## What this repo contains

- `patches/` — two minimal fixes to the ROCmFPX fork's `convert_hf_to_gguf.py`
  that were needed to convert this model's NVFP4 checkpoint and to produce a
  loadable GGUF.
- `docs/benchmarks.md` — measured decode/prompt speeds (Vulkan vs ROCm) and
  perplexity.
- `docs/nvfp4-findings.md` — why the NVFP4 → ROCmFP4 remap path was **dropped**
  for this model (two defects found; one fixed, one needs a core-quantizer change).
- `docs/how_to_run.md` — build + run instructions.
- `AI_CHANGES.md` — change log of the AI-assisted work (per the fork's policy).

## tl;dr conclusions

- The **clean BF16 → ROCmFP4** path works and is the deliverable. The resulting
  `Q4_0_ROCMFP4_STRIX_LEAN` GGUF scores **PPL 5.9936 ± 0.0358** on wikitext-2
  and runs fully on GPU via unified memory.
- The **NVFP4 → ROCmFP4 remap** path silently produces **garbage** for this
  model because the quantizer's NVFP4 dequant ignores ModelOpt's companion
  `scale2`/`input_scale` factors (this model: `scale2 ≈ 1.4e-4` → weights
  ~7110× off). The README's 9B example only worked because its factors were ≈ 1.0.
- On this Strix Halo box, **Vulkan beats ROCm**: pp512 1299.7 vs 1075.4 t/s,
  tg128 85.6 vs 79.4 t/s.
- The model's **MTP/NextN head is not included** (converter skips `mtp.*` for
  MoE), so no self-speculative decode here.
- GGUF advertises **262,144** context (the real limit), overriding the converter's
  hardcoded 1M for hybrid mamba/attention archs.

## Reproducing

See `docs/how_to_run.md`. Everything was built with the
[ROCmFPX fork](https://github.com/charlie12345/ROCmFPX) (`b209-00d5452`) on a
Framework AMD Strix Halo (Ryzen AI Max, `gfx1151`, 128 GB unified RAM,
ROCm 7.2.3).

## License

The converter patches are MIT (llama.cpp lineage). The model weights are
NVIDIA's under [OpenMDW-1.1](https://openmdw.ai/license/1-1/). See `LICENSE`.

## Credits

- NVIDIA — model weights and architecture
- charlie12345 — ROCmFPX / ROCmFP4 quantization + kernels
- julianmb — this conversion work and documentation
