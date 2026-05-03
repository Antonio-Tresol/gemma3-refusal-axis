# Plan: Feature Hierarchy Analysis Across Matryoshka Widths

## Research question

When a broad refusal feature at 16k width disappears at 65k, does its signal **decompose** into multiple finer features, or does it simply vanish?

## Hypothesis to test

**H1 (Hierarchical decomposition):** Broad 16k features split into domain-specific children at wider widths. A "general refusal" feature at 16k becomes "safety refusal" + "ethical refusal" at 65k.

**H0 (No hierarchy / reorganisation):** Disappeared 16k features are genuinely lost: the wider SAE learns unrelated features that happen to cover the same behavioural territory but are not geometrically related to the parent.

## Falsification criteria (checked BEFORE investing compute)

These must be true for the analysis to be meaningful:

1. **Decoder vectors must be geometrically meaningful.** If `w_dec[i]` for feature i is a unit-norm vector in R^3840, then cosine similarity between decoder vectors is a meaningful measure of feature relatedness.
   - **Evidence:** Gemma Scope 2 paper (line 174): "we restrict the columns of W_dec to have unit norm by renormalizing after every update." Confirmed.
   - **Check:** Verify empirically that decoder vectors are unit norm after loading.

2. **Matryoshka prefix slicing must actually share decoder weights.** If 16k and 65k use different decoders, cosine similarity across widths is meaningless.
   - **Evidence:** milestone_0_bootstrap.md confirms single SAE prefix-sliced. `w_dec` shape is (1048576, 3840), one decoder matrix. Feature i at "16k width" uses `w_dec[i]`, and the same `w_dec[i]` at "65k width."
   - **Check:** Verify that loading the 1M SAE and taking `w_dec[:16384]` gives the same features as a "16k" prefix slice.

3. **There must be enough disappeared features to analyse.** If all 16k features survive to 65k, there's nothing to decompose.
   - **Evidence:** LPT 16k has 36 relevant features. LPT 65k has 34. Only 11 overlap. So **25 features disappeared** from 16k to 65k, and **23 new features appeared** at 65k. Sufficient sample.
   - **Data source:** `data/milestone_7_results/width_metrics.json`, feature_ids fields.

4. **Feature absorption (Chanin et al. (2024, "A is for Absorption," arXiv:2409.14507)) must not confound results.** If parent features stop firing due to sparsity optimization (not because they were replaced), we'd see false "disappearance."
   - **Mitigation:** After finding candidate parent-child relationships via decoder cosine, validate with co-activation analysis on actual prompts. If a parent's decoder vector is similar to children but the parent still fires on the same prompts (just with lower activation), that's absorption, not splitting.

## Methods (ordered by cost)

### Method 1: Decoder cosine similarity (CPU only, ~15 min)

**What:** For each of the 25 disappeared LPT features (at 16k, not at 65k), compute cosine similarity of their decoder vector with all 23 new features at 65k.

**How:**
```
For each disappeared feature d (feature_id in 16k but not in 65k):
    d_vec = w_dec[d]  # shape (3840,)
    For each new feature n (feature_id in 65k but not in 16k):
        cos = cosine(d_vec, w_dec[n])
    Report top-3 matches for each d
```

**Evidence pin:** Decoder cosine similarity is the standard method for feature matching in SAEs.
- Bricken et al. (2023, "Towards Monosemanticity," Transformer Circuits, Anthropic) use decoder cosine to identify feature splitting.
- Bussmann et al. (2025, "Learning Multi-Level Features with Matryoshka SAEs," arXiv:2503.17547) validate hierarchies via decoder geometry.
- Luo et al. (2026, "From Atoms to Trees: HSAE," arXiv:2602.11881) use structural alignment of decoder vectors as their primary hierarchy metric.

**Falsification:** If NO disappeared feature has cosine > 0.5 with ANY new feature, H1 is falsified: the dictionary genuinely reorganizes without hierarchical structure.

**Success criterion:** At least 5 disappeared features have cosine > 0.5 with 2+ new features (suggesting splitting, not just replacement).

**GPU requirement:** Load 1M SAE (~15 GB) once, extract w_dec, then CPU only. Two-phase pattern: load SAE → extract decoder → del SAE → CPU analysis.

### Method 2: Co-activation validation (uses saved activations, CPU only)

**What:** For each parent-child pair found in Method 1, check if they activate on the same prompts.

**How:** We already have SAE feature activations from the M4 pipeline. Load the encoded features for each width and check:
- Does parent feature d activate on prompt set P at 16k width?
- Do candidate children n1, n2, ... activate on subsets of P at 65k width?
- Does union(n1_prompts, n2_prompts, ...) ≈ P? (coverage)

**Evidence pin:** Co-activation is used by Chanin et al. (2024) to distinguish absorption from splitting. If children cover the parent's activation pattern, it's splitting. If they don't, the geometric similarity is coincidental.

**Falsification:** If geometrically similar features (cos > 0.5) do NOT co-activate on the same prompts, the decoder similarity is an artefact of the high-dimensional space, not meaningful hierarchy.

**Data requirement:** Check if `data/milestone_4_results/` contains per-feature activation data, or if we need to re-encode.

### Method 3: Hierarchy quality metrics (CPU only)

**What:** Apply the structural alignment metric from Luo et al. (2026) (HSAE): a parent's decoder vector should be approximately a weighted combination of its children's decoder vectors.

**How:**
```
For each parent d with children {n1, n2, ...}:
    Fit: d_vec ≈ α1 * n1_vec + α2 * n2_vec + ...
    Measure R² of this fit
    High R² = parent decomposes into children
    Low R² = coincidental similarity
```

**Evidence pin:** Luo et al. (2026) Sec 3.2: structural constraint loss encourages parent-child decoder alignment. We apply their evaluation metric, not their training method.

**Falsification:** If R² < 0.3 for most parent-child groupings, the decoder vectors are not hierarchically structured despite pairwise similarities.

## What we DON'T do (and why)

- **Train new SAEs.** We use the pre-trained Gemma Scope 2 SAE. Novel SAE training is out of scope for this work.
- **Gradient-based attribution.** Requires model forward passes. We can do this later if Methods 1-3 show promise.
- **Cross-layer analysis.** The hierarchy question is within-layer (layer 41). Cross-layer features are a different research question.

## Expected outcomes

| Outcome | What it means | Next step |
|---------|--------------|-----------|
| H1 confirmed (clear hierarchies) | Broad refusal features split into domain-specific children. Matryoshka ordering encodes a meaningful feature taxonomy. | Visualize as tree/Sankey. Report as project finding. |
| H0 confirmed (no hierarchy) | The SAE genuinely reorganizes at larger widths. Decoder similarity across widths is low. | Report reorganisation finding. Still valuable: contradicts naive "features just get finer" assumption. |
| Mixed (some hierarchy, some reorganisation) | Some features split hierarchically, others are replaced. | Report the mix. Characterize which feature types split vs. reorganize. |

## Deliverables

1. Parent-child cosine similarity matrix (Method 1)
2. Co-activation validation table (Method 2)
3. Hierarchy quality R² scores (Method 3)
4. Interactive marimo notebook showing the feature tree (if hierarchies found)
5. Section in findings report documenting the analysis

## Resource requirements

- **GPU:** One SAE load (~15 GB VRAM, ~2 min) to extract w_dec. Then del SAE.
- **CPU:** All three methods run on CPU after decoder extraction.
- **Time estimate:** Method 1: ~30 min implementation + ~15 min compute. Method 2: ~1 hour (depends on data availability). Method 3: ~30 min.
- **Total:** ~3 hours including validation and notebook creation.
