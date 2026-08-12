# NVFP4 → ROCmFP4: findings and why we dropped it

We originally intended to also remap NVIDIA's pre-quantized
[NVFP4 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
onto the ROCmFP4 kernel path. Two independent defects surfaced; the resulting
artifact was dropped in favor of the clean BF16 → ROCmFP4 path.

## Defect 1 — unloadable NVFP4 GGUF when `output.weight` is NVFP4 (FIXED)

**Symptom:** converted NVFP4 GGUF (and its ROCmFP4 remap) failed to load:
`done_getting_tensors: wrong number of tensors; expected 494, got 493`.

**Root cause:** every tensor in the GGUF must have a counterpart in the runtime
tensor mapping. The converter's NVFP4 path writes a per-tensor `output.scale`
(a "scale2" factor) for the packed `output.weight`. llama.cpp has **no
output-scale tensor** (`output_s`), so one file tensor goes unmapped and the
strict count check aborts. The 92 per-expert `ffn_*_exps/shexp.scale` tensors
are fine — the runtime auto-creates those (`llama-model.cpp`).

**Fix** (`patches/0002-converter-dequant-output-to-f16.patch`): dequantize the
GGUF output tensor to F16 instead of packing it as NVFP4, so no `output.scale`
is emitted. The dequant was verified against the BF16 checkpoint's
`lm_head.weight`: signed-E2M1 fp4, low-nibble-first, one E4M3 scale per 16-group,
× scale2 × input_scale, matching to fp4 noise (rms 0.0023).

Why F16 and not BF16: the dequantized values only carry fp4 precision, and the
environment's numpy lacks bfloat16 support.

## Defect 2 — quantizer ignores NVFP4 companion scales (NOT FIXED)

**Symptom:** after fixing defect 1, the NVFP4 → ROCmFP4 artifact loaded but
produced **garbage** ("No Yes No Yes" vs the correct "Answer: 4").

**Root cause:** `dequantize_row_nvfp4` (`ggml-quants.c`) applies **only the block
scales**; `llama_tensor_dequantize_impl` (`llama-quant.cpp`) has no access to
the NVFP4 companion `.scale` / `.input_scale` tensors. For ModelOpt checkpoints
with non-trivial per-tensor/per-expert `scale2`/`input_scale`, the requantized
weights are wrong by that factor.

Measured on one expert: **without** `scale2` → relative error **7110×** vs the
BF16 source; **with** `scale2` → 0.10 (~fp4 noise). This model's `scale2 ≈
1.4e-4`.

The fork's README claims NVFP4 → ROCmFP4 is the "closest-matching conversion";
the 9B example it cites worked because that model's factors are ≈ 1.0 (and its
`output.weight` stays high-precision, which is why defect 1 never triggered).

**Why it wasn't fixed here:** a correct fix requires the quantizer to look up and
apply each NVFP4 tensor's companion scales (per-expert broadcast over the merged
expert dim) during dequant and then drop them from the output — a change to core
C++ quantization behavior. Per this project's AI-use policy, that is a
human-owned change. We also tested folding the factors into the block scales in
the converter; it was rejected because `scale2 = 1.4e-4` pushes UE4M3 block
scales into denormal range (relative error 1.19 vs 0.10 exact).

## Decision

Keep the **BF16 → ROCmFP4** artifacts only. They are correct, validated, and
cleanly reproducible. The NVFP4 remap is documented here so the next person does
not repeat the investigation.
