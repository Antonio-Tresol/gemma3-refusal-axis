# Milestone 0: Project Bootstrap, Findings

**Date:** 2026-03-13
**Status:** COMPLETE (with one plan correction)

## Validation Gate Results

| Check | Result | Status |
|-------|--------|--------|
| Model VRAM | 22.70 GB allocated | PASS |
| SAE VRAM (1M) | 15.01 GB allocated | PASS |
| Combined VRAM | 37.71 GB > 32 GB limit | PASS (confirms they cannot coexist) |
| Hook captures shape (seq_len, 3840) at layer 41 | (16, 3840) for 16-token prompt | PASS |
| Can extract last-prompt-token activation | Shape (3840,) | PASS |
| Can extract mean-response-token activation | Shape (3840,) | PASS |
| SAE L0 in range [30, 120] | **L0 = 57** | PASS |
| Matryoshka nesting verified | See correction below | CORRECTED |
| Full load-unload cycle without VRAM leaks | 0.01 GB after cleanup | PASS |

## Measured Numbers

### Model (Gemma 3 12B-IT, bf16)
- VRAM allocated: **22.70 GB**
- Layer access path: `model.model.language_model.layers[41]` (NOT `model.model.layers`; Gemma 3 is multimodal)
- 48 transformer layers total
- Layer 41 output: residual stream, shape (seq_len, 3840)
- Loading: `dtype=torch.bfloat16` (not `torch_dtype`, which is deprecated in transformers 5.x)
- Requires `accelerate` for `device_map="auto"`
- `apply_chat_template` returns `BatchEncoding` dict with `return_dict=True`; must index `["input_ids"]`

### SAE (1M Matryoshka, layer 41, resid_post, l0_medium)
- VRAM allocated: **15.01 GB**
- HuggingFace path: `resid_post/layer_41_width_1m_l0_medium/params.safetensors`
- Parameter keys: `w_enc` (3840, 1048576), `w_dec` (1048576, 3840), `threshold` (1048576,), `b_enc` (1048576,), `b_dec` (3840,), all lowercase, all float32 on disk
- L0 on real activation: **57** (within expected ~60 for "medium")
- SAE params are stored as float32 but we cast to bf16 on GPU

### Matryoshka Prefix Slicing (with real activation)

| Width | Active features | FVU | Cosine similarity |
|-------|----------------|------|-------------------|
| 16k | 15/57 | 0.0331 | 0.9841 |
| 65k | 42/57 | 0.0192 | 0.9904 |
| 262k | 53/57 | 0.0178 | 0.9912 |
| 1M | 57/57 | 0.0172 | 0.9915 |

Reconstruction degrades gracefully. Even at 16k prefix (26% of active features), FVU is 3.3% and cosine similarity is 0.984.

## Plan Correction: Step 0i

**Original plan assumption (WRONG):** "Compare the first 16384 entries of the 1M encoding with the full 16k encoding. They should be identical (or near-identical within floating point tolerance). This confirms that prefix slicing is valid."

**Why it was wrong:** The standalone 16k SAE and the 1M SAE are independently trained models with completely different encoder weights. Matryoshka training adds a loss term that orders features by importance within a single SAE, so that the first N features capture the most variance. It does NOT make the 1M encoder's first 16k columns match a standalone 16k encoder.

**Correct verification:** Encode with the 1M SAE, prefix-slice to different widths, and check that reconstruction quality degrades gracefully (which it does; see table above).

**Implication for the pipeline:** We use only the 1M SAE throughout. Encode once, then prefix-slice the 1048576-dimensional feature vector to get width-specific sub-dictionaries (first 16k, 65k, 262k, or all 1M features). No need to load separate SAEs per width.

## Technical Notes for Future Milestones

1. **GPU Phase A (model):** Load with `device_map="auto"`, `dtype=torch.bfloat16`, `attn_implementation="eager"`. Access layers via `model.model.language_model.layers`. Unload with `del model; gc.collect(); torch.cuda.empty_cache()`.
2. **GPU Phase B (SAE):** Load params with `load_file`, instantiate `JumpReLUSAE(d_in=3840, d_sae=1048576)`, cast to bf16 on GPU. Unload similarly.
3. **Hook pattern:** `register_forward_hook` on layer module. Output is a tuple; `output[0]` is the residual stream tensor.
4. **Generation:** Full sequence shape from `model.generate` includes prompt + response. Hook during a separate `model(full_ids)` forward pass captures all positions.
