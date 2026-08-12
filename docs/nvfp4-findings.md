# NVFP4 Findings & Analysis

We evaluated both **Native NVFP4** inference and the **NVFP4 → ROCmFP4 Remap** path for NVIDIA's pre-quantized [Nemotron 3.5 Lightning 30B-A3B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) checkpoint.

---

## Defect 1 — Unloadable GGUF due to `output.scale` (FIXED)

**Symptom:** Converted NVFP4 GGUF files failed to load in llama.cpp:
```text
error loading model: done_getting_tensors: wrong number of tensors; expected 494, got 493
```

**Root Cause:** Every tensor in a GGUF file must map to a tensor created in `llama-model.cpp`. ModelOpt's NVFP4 export packs `output.weight` and writes a separate `output.scale` ("scale2") factor tensor. Because llama.cpp has no `output_s` scale tensor mapping for `output.weight`, one tensor goes unmapped, failing the `done_getting_tensors` count check.

**Fix (`patches/0002-converter-dequant-output-to-f16.patch`):**
In `convert_hf_to_gguf.py`, we detect when `output.weight` is NVFP4-quantized and dequantize it in-place to F16 (signed E2M1 lookup table, low nibble first, per-16 group scale, times `scale2`). This eliminates `output.scale` while preserving exact weights. Verified against the BF16 ground truth in `docs/repro/nvfp4_dequant_match.py` (RMS error = **0.00230**).

*Note: F16 is used instead of BF16 because numpy 1.26 lacks native bfloat16 array support.*

---

## Defect 2 — Ignored Companion `scale2` Factor in Runtime & Quantizer

**Symptom:**
- **NVFP4 → ROCmFP4 Remap:** The remapped GGUF loads, but generates gibberish ("No Yes No Yes").
- **Native NVFP4:** The GGUF loads, but scores **PPL 109.7910 ± 0.96827** on wikitext-2 (vs **5.9936** for BF16-derived ROCmFP4).

**Root Cause:**
`dequantize_row_nvfp4` (`ggml-quants.c`) applies **only the per-subblock E4M3 scales**. It does not apply ModelOpt's companion `.scale` (`scale2`) or `.input_scale` tensors.
For expert weights in this model, `scale2 ≈ 0.0001386`.

Empirical verification (`docs/repro/nvfp4_scale2_check.py`):
- **Without `scale2`**: Relative error vs BF16 ground truth = **7110.49×** (garbage)
- **With `scale2`**: Relative error vs BF16 ground truth = **0.1007** (~10%, expected FP4 quantization noise)

**Why folding `scale2` into block scales in the converter fails:**
We tested folding `scale2` into the E4M3 block scales during conversion (`docs/repro/nvfp4_fold_test.py`). Because `scale2 = 1.38e-4`, folding it pushes the E4M3 block scales into the denormal range, resulting in a **1.19× relative error** (~120% error, 12× worse than exact dequantization) due to 3-bit denormal mantissa precision loss.

---

## Summary Table

| Path | Load Status | PPL (wikitext-2) | Status |
|:-----|:------------|:-----------------|:-------|
| **BF16 → ROCmFP4 STRIX_LEAN** | Loads OK | **5.9936 ± 0.0358** | **Promoted & Delivered** |
| **Native NVFP4 (with Patch 1 & 2)** | Loads OK | 109.7910 ± 0.9683 | Incoherent (Missing `scale2` in kernels) |
| **NVFP4 → ROCmFP4 Remap** | Loads OK | Incoherent | Incoherent (Missing `scale2` in quantizer) |

---

## Conclusion

The **clean BF16 → ROCmFP4 path** is the only viable conversion path for ModelOpt NVFP4 checkpoints with non-trivial `scale2` factors.
