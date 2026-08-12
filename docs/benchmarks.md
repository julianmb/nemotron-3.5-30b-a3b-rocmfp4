# Benchmarks & quality

All numbers measured locally on a **Framework AMD Strix Halo** (Ryzen AI Max,
Radeon 8050S/8060S, `gfx1151`, 128 GB unified RAM, ROCm 7.2.3), build
`b209-00d5452` of the ROCmFPX fork.

## Hardware notes

- The GPU is an APU iGPU with a small fixed VRAM carve-out. With
  `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` the HIP backend sees the full unified
  memory pool (ROCm reports 122,880 MiB), so 16 GB models fully offload.
- `HSA_OVERRIDE_GFX_VERSION=11.5.1` is required for `gfx1151` on ROCm releases
  that do not list the target.

## Weight-quant benchmarks

`llama-bench -p 512 -n 128`, q8_0/q8_0 KV cache, FlashAttention on, full offload
(`-ngl 999`), model = `...ROCmFP4-STRIX_LEAN.gguf` (BF16-derived).

| Backend | Prompt (pp512) | Decode (tg128) |
|--------:|---------------:|---------------:|
| **Vulkan0** | **1299.7 t/s** | **85.6 t/s** |
| ROCm0   | 1075.4 t/s | 79.4 t/s |

Vulkan was ~21% faster at prompt processing and ~8% faster at decode on this
system — consistent with the fork's README.

## Interactive-generation numbers

`llama-completion`, 7-token prompt, greedy (`--temp 0`), Vulkan0, full offload:

| Artifact | Prompt | Generation |
|----------|-------:|-----------:|
| STRIX_LEAN (BF16-derived) | 184–200 t/s | 76–83 t/s |
| FAST (BF16-derived)       | 199.8 t/s | 83.4 t/s |
| COHERENT (BF16-derived)   | 188.8 t/s | 81.0 t/s |

CPU-only reference (`-ngl 0`, same model): prompt ~47 t/s, generation ~37 t/s.

## Quality

| Model | Task | Result |
|-------|------|--------|
| `...ROCmFP4-STRIX_LEAN.gguf` (BF16-derived) | wikitext-2 perplexity | **5.9936 ± 0.0358** |

Reference for the NVFP4-derived remap: **not measurable** — the remap produced
garbage weights (see `nvfp4-findings.md`).

## Context length

The GGUF advertises `context_length = 262144` (`n_ctx_train = 262144` at load).
This is the model's real `max_position_embeddings`; the converter hardcodes
2^20 for hybrid mamba2/attention archs, which we overrode during re-quantization
(`--override-kv nemotron_h_moe.context_length=int:262144` — note the `int:`
token and that the flag must precede positional args).
