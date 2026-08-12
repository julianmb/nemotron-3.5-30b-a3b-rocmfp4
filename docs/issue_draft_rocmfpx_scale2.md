# [Bug Report / Feature Request] `llama-quantize` NVFP4 -> ROCmFP4 remap ignores ModelOpt companion `scale2` / `input_scale` factors

### Title
> **`llama-quantize`: NVFP4 → ROCmFP4 remap produces garbage weights due to unhandled companion `scale2` factors**

---

### Issue Body

#### Summary
Requantizing ModelOpt `W4A16_NVFP4` checkpoints to `Q4_0_ROCMFP4_*` using `llama-quantize --allow-requantize` produces garbage output ("No Yes No Yes", perplexity >100) on models whose ModelOpt quantization uses non-trivial per-tensor or per-expert `scale2` (`.weight_scale_2`) or `input_scale` (`.input_scale`) factors.

#### Root Cause Analysis
During requantization in `llama_tensor_quantize_impl` (`llama-quant.cpp`), quantized source tensors are dequantized to `f32` via `dequantize_row_nvfp4` (`ggml-quants.c`). 

`dequantize_row_nvfp4` applies **only the per-subblock UE4M3 scales** embedded inside the packed `block_nvfp4` structures. It has no access to ModelOpt's separate companion `.scale` (`weight_scale_2`) or `.input_scale` tensors stored in the GGUF.

While native NVFP4 inference in `llama-model.cpp` applies companion scale tensors at graph execution time via `ggml_mul` (e.g. `layer.ffn_down_exps_s`), tensor-level requantization in `llama-quantize` operates on raw tensor data alone, bypassing the graph.

#### Empirical Reproduction
On [NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4):
- Expert down-projection `backbone.layers.1.mixer.experts.0.down_proj`:
  - Block scales: `~36.0 – 64.0`
  - Per-expert `scale2`: `0.00013860066` (`~1.38e-4`)
  - Ground truth BF16 weight max-abs: `0.2148`

When `dequantize_row_nvfp4` dequantizes without multiplying `scale2`:
- **Without `scale2`**: Mean relative error vs BF16 ground truth = **7,110.49×** (garbage)
- **With `scale2`**: Mean relative error vs BF16 ground truth = **0.1007** (~10%, expected FP4 quantization noise)

#### Why the 9B Example Worked
The README cites a 9B NVFP4 example where `Q4_0_ROCMFP4` perplexity matched the source. That model's ModelOpt export had `scale2 ≈ 1.0`. For models trained/export-quantized with small `scale2` multipliers (like Nemotron 3.5), the remap silently fails.

#### Why Converter-Side Scale Folding Fails
We tested folding `scale2` into the UE4M3 block scales in `convert_hf_to_gguf.py` during GGUF creation. Because `scale2 ≈ 1.38e-4`, folding it pushes the UE4M3 block scales into the denormal range, causing a **1.19× relative error** (~120% error, 12× worse than exact dequantization) due to 3-bit denormal mantissa precision loss.

#### Suggested Fix
In `llama_model_quantize_internal` (`llama-quant.cpp`), when source tensor type is `GGML_TYPE_NVFP4`:
1. Look up companion `.scale` (`weight_scale_2`) and `.input_scale` tensors for the source tensor from the GGUF metadata.
2. After `llama_tensor_dequantize_impl`, multiply `f32_data` by the companion scale factor(s) (broadcasting across experts for merged MoE tensors) before invoking `ggml_quantize_chunk`.
3. Drop or set companion scale tensors to `1.0` in the output GGUF so `llama-model.cpp` does not double-apply the scale at inference time.
