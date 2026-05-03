---
name: thesis-experiment
description: Background knowledge for running mechanistic interpretability experiments in this thesis project. Covers GPU memory pattern, tensor shape conventions, reproducibility requirements, and evidence pinning. Use when writing experiment code, analysis scripts, or any GPU-touching work.
user-invocable: false
---

# Thesis Experiment Conventions

## GPU Memory Pattern (RTX 5090, 32 GB VRAM)

Model (~22.7 GB) + 1M SAE (~15 GB) = ~37.7 GB. They cannot coexist.

**Two-phase pattern** (first-class concern, not an afterthought):

1. **Phase A — Model.** Load model → hook layer → forward pass → save activations to disk → `del model; torch.cuda.empty_cache(); gc.collect()` → verify VRAM ≈ 0.
2. **Phase B — SAE.** Load SAE → encode activations from disk → save features → `del sae; torch.cuda.empty_cache(); gc.collect()` → verify VRAM ≈ 0.
3. **Phase C — Analysis.** CPU only. Load features from disk. Plot, cluster, compare.

**Verified loading patterns:**
- Model: `dtype=torch.bfloat16` (not `torch_dtype=`, deprecated)
- Layer path: `model.model.language_model.layers[N]` (Gemma 3 is multimodal wrapper)
- SAE params: lowercase keys (`w_enc`, `w_dec`, `threshold`, `b_enc`, `b_dec`), float32 on disk
- Tokenizer: `apply_chat_template(..., return_dict=True)` → index `["input_ids"]`
- Matryoshka: feature ordering within ONE SAE. Prefix-slice the 1M encoding for sub-dictionaries.

## GPU Script Requirements

Every GPU script MUST have:
- Per-item progress logging (not just start/end)
- Elapsed time and ETA
- Resume capability (skip items already saved to disk)
- Unbuffered output (`PYTHONUNBUFFERED=1` or `flush=True`)
- VRAM verification after cleanup (`torch.cuda.memory_allocated()`)

## Code Conventions

- **Typed shapes.** Prefer explicit tensor shapes in comments: `# (seq_len, d_model)`. A reader should never guess what a tensor holds.
- **Reproducibility.** All random seeds explicit and recorded. Any result must be re-derivable from code + seed.
- **Minimal dependencies.** If we can build it in a few lines, don't add a library.
- **Findings trace to code.** Every `findings/` entry references the script and ideally the git commit that produced it.
- **No drift.** If code changes, docs change.

## Evidence Pinning

Every methodological decision must cite its source. Format: `[Decision] — [Source, Section/Page]`.

Applies to:
- Code comments explaining WHY, not just what
- Findings documents linking each choice to literature
- Prompts and rubrics referencing the methodology they implement
- If no prior work exists, state it explicitly as novel and justify from first principles
- Keep references in a .bib in `references/`, organized by topic

## Matryoshka SAE — Common Misconception

There is ONE 1M SAE. Matryoshka widths (16k, 65k, 262k, 1M) are prefix slices of the same encoding, NOT separate SAE models. The standalone width-specific SAEs on HuggingFace are separate models with different training. Our analysis uses prefix-slicing of the 1M SAE.
