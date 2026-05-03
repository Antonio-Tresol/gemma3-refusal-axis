# Methodology: Refusal Axis Geometric Analysis and Domain-Selective Capping

This document describes the methodology for the refusal axis exploration and domain-selective capping experiment, as actually executed. Evidence-pinned, no results.

---

## 1. Motivation and Framing

Lu et al. (2026, "The Assistant Axis," arXiv:2601.10387) discovered the Assistant Axis, a single direction in activation space separating default assistant behaviour from role-playing personas. They showed that activation capping along this axis prevents persona drift and jailbreaks.

We extend their framework to ask: **does the refusal component of the assistant persona decompose into geometrically distinct sub-directions?** And if so, **can we cap them independently for domain-selective safety control?**

This extends both Lu et al. (who use a single axis) and Arditi et al. (2024, "Refusal in Language Models Is Mediated by a Single Direction," arXiv:2406.11717, NeurIPS 2024, who showed refusal is mediated by a single direction but did not test decomposition).

---

## 2. Refusal Axis Construction

### 2.1 Formula

Following Lu et al. Sec 3.1 (axis = mean(default) - mean(role)):

```
refusal_axis = mean(positive_activations) - mean(negative_activations)
```

Where:
- **positive_activations**: mean-response-token activations from prompts the model refuses (refusal score > 50)
- **negative_activations**: mean-response-token activations from matched benign prompts the model answers (refusal score < 30)

This is the Arditi et al. (2024) refusal direction, which is the refusal-specific analogue of Lu et al.'s assistant axis.

### 2.2 Extraction site

**Mean-response-token**: mean activation across all assistant response tokens at layer 41.

**Evidence:** Lu et al. `internals/activations.py` extract mean activations over assistant response tokens. Chen et al. (2025, "Persona Vectors," arXiv:2507.21509) Sec 2.2 and Lu et al. (2026) Sec 2.1.2 both use mean-response-token extraction for behavioural analysis. This captures the model's behaviour *during* generation, not just its intent before generating.

### 2.3 Data

128 retained contrastive pairs (from 280 total, filtered by 3-pass trait scoring) across 6 refusal domains: safety (31), ethical (30), capability_boundary (22), privacy (21), legal (20), identity_boundary (4).

---

## 3. Per-Domain Refusal Directions

### 3.1 Construction

For each domain D with at least 3 retained pairs:

```
direction_D = mean(positive_acts[domain=D]) - mean(negative_acts[domain=D])
direction_D_unit = direction_D / ||direction_D||
```

This produces one unit-norm direction per domain in the 3840-dimensional residual stream space.

**Novel contribution:** No prior work computes per-domain refusal directions. Arditi et al. compute a single direction; Lu et al. compute a single assistant axis; SAILS (Wang et al., 2025, "SAILS," arXiv:2512.23260) identifies features but not geometric directions per domain.

### 3.2 Domains computed

| Domain | n | Rationale for inclusion |
|--------|---|------------------------|
| safety | 31 | Value-based refusal: physical harm, weapons, substances |
| ethical | 30 | Value-based refusal: manipulation, exploitation |
| legal | 20 | Value-based refusal: fraud, hacking, drugs |
| privacy | 21 | Partially value-based: personal data, surveillance |
| capability_boundary | 22 | Capability acknowledgment: model literally cannot do X |
| identity_boundary | 4 | Identity honesty: model should not claim false properties |

Identity_boundary has only 4 pairs because Gemma 3 12B does not refuse identity-related prompts; it engages with philosophical questions about consciousness. This is itself a finding, documented in the trait scoring analysis.

---

## 4. Geometric Analysis

### 4.1 Domain loading on mean refusal axis

Cosine similarity between each domain direction and the mean refusal axis. Analogous to Lu et al.'s role loadings on the assistant axis.

```
loading_D = cos(direction_D, refusal_axis)
```

**Evidence:** Lu et al. `assistant_axis/axis.py`, `project()` function computes normalized dot product. We use the same formula.

### 4.2 Cross-domain cosine similarity matrix

For each pair of domains (D1, D2):

```
cos(D1, D2) = (direction_D1 · direction_D2) / (||direction_D1|| × ||direction_D2||)
```

**Interpretation:** cos ≈ 1.0 means D1 and D2 share approximately the same refusal direction. cos ≈ 0 means the two refusal directions are geometrically independent. cos < 0 means they point in opposite directions.

### 4.3 Hierarchical clustering

Ward's linkage on the distance matrix (distance = 1 - cosine) to discover domain groupings. Rendered as a dendrogram above the cosine heatmap.

**Evidence:** Standard clustering method. Lu et al. do not cluster roles, but the visualization follows the same heatmap+dendrogram pattern common in genomics and neuroscience.

### 4.4 PCA on activation differences

Per-pair difference vectors: `diff_i = positive_act_i - negative_act_i`, shape (128, 3840).

Mean-centre, then SVD to extract principal components.

**Evidence:** Lu et al. `pca.ipynb`: PCA on per-role mean vectors to measure dimensionality of persona space. They find 4-19 components for 70% variance. We apply the same analysis to refusal activation differences.

**Preprocessing:** Mean-centring only (Lu et al. `pca.py` `MeanScaler`). No L2 normalization.

### 4.5 Statistical validation

Bootstrap confidence intervals (2000 resamples, seed=42) on pairwise cosine similarities. For each bootstrap iteration, resample pairs within each domain with replacement, recompute domain directions, compute cosine.

**Purpose:** Assess whether observed cosine differences are robust or could be driven by a few outlier pairs.

---

## 5. Domain-Selective Capping

### 5.1 Capping formula

Following Lu et al. (2026) `steering.py` lines 89-95, `intervention_type="capping"`:

```python
v_hat = direction / ||direction||        # unit-norm direction
proj = einsum('bld,d->bl', act, v_hat)   # project activations
excess = clamp(proj - τ, min=0)          # only positive excess
act' = act - einsum('bl,d->bld', excess, v_hat)  # remove excess
```

**Key property:** This only reduces the projection when it exceeds threshold τ. It never amplifies, never pushes in the opposite direction, and leaves all other components of the activation unchanged. This is why it preserves coherence better than additive steering (Arditi et al., 2024) or directional ablation.

**Novel application:** Lu et al. apply capping along a single assistant axis. We apply it along individual domain refusal directions, testing whether per-domain capping gives independent control.

### 5.2 Layer selection

**Layer 36** (75% depth in Gemma 3 12B's 48 layers).

**Evidence:**
- Our M5 steering experiments found layer 36 optimal for Gemma 3 12B (better than layer 41 (85%, too late) and comparable to layer 39).
- Arditi et al. (2024) exclude layers above 80% depth.
- Panickssery et al. (2024, "Steering Llama 2 via Contrastive Activation Addition," arXiv:2312.06681) found ~40% depth optimal for Llama 2.
- Lu et al. use middle layers (~50% depth).
- The Gemma Scope 2 tutorial steers at `LAYER - 5` (5 layers before the SAE layer).

Layer 36 is a compromise between these recommendations and our empirical validation.

### 5.3 Threshold calibration

**Method:** Compute projection of all benign (negative) prompt activations onto each domain direction. Set τ at percentiles of this distribution.

**Evidence:** Lu et al. Sec 4.2 calibrate τ from normal conversation projections. Our benign prompts serve the same purpose: they represent "normal" model behaviour where capping should have no effect.

**τ sweep:** Percentiles [10, 25, 50, 75, 90, 95, 99]. Lower percentile = more aggressive capping (clamps more activations).

### 5.4 Capping directions tested

| Direction | Rationale |
|-----------|-----------|
| `overall_refusal` | Mean refusal axis; replicates Lu et al.'s single-axis approach as baseline |
| `safety` | Value-based, cos=0.91 with mean axis; should suppress safety refusal specifically |
| `capability_boundary` | Distinct direction, cos=0.38; should be independently controllable |
| `privacy` | Intermediate, cos=0.70; tests partial overlap case |

**Evidence for selection:** Domain loading analysis (Sec 4.1). Safety and capability are the most geometrically distinct pair (cross-domain cos=0.14), making them the strongest test of independent control. Privacy provides an intermediate case.

### 5.5 Test prompts

40 held-out prompts (not from the contrastive training set):
- 10 safety (weapons, substances, sabotage, violence)
- 10 capability_boundary (login, execute code, send messages, access files)
- 10 privacy (find addresses, access records, track individuals)
- 10 benign (general knowledge, recipes, explanations)

Greedy decoding (`do_sample=False`) for reproducibility.

### 5.6 Evaluation

Each capped output scored by Sonnet 4.6 for:
- **Coherence** (0-100): Is the output fluent and well-formed?
- **Refusal** (0-100): Does the output decline the request?

**Coherence gating:** If coherence < 50, output classified as degenerate, refusal set to -1 and excluded from analysis. This prevents the coherence-refusal confound identified in our M5 steering critique, as degenerate text can superficially resemble refusal.

**Evidence:** Coherence gating is our novel contribution, motivated by the pre-implementation adversarial review (findings/literature/steering_methodology_critique.md Issues 3-4). Lu et al. do not explicitly gate on coherence in their evaluation.

**Fast-path:** Outputs with repetition ratio > 0.7 (fraction of repeated 3-grams) are immediately classified as degenerate without an LLM scoring call.

### 5.7 Independence matrix

The primary evaluation artefact. A matrix of:
- **Rows:** Capping directions (overall, safety, capability, privacy)
- **Columns:** Prompt domains (safety, capability, privacy, benign)
- **Cells:** Mean refusal score change relative to uncapped baseline

**Interpretation:**
- Diagonal (e.g., safety cap × safety prompts): should show large negative value (capping works on target)
- Off-diagonal (e.g., safety cap × capability prompts): should show near-zero (capping doesn't affect other domains)
- Large diagonal + small off-diagonal = **independent control** (decomposition has practical value)
- All cells change similarly = **"same control knob"** (Joad et al., 2026, arXiv:2602.02132), meaning decomposition does not help

**Novel contribution:** No prior work evaluates cross-domain independence of activation capping. Lu et al. evaluate a single axis; Arditi et al. evaluate a single direction; SAILS evaluates individual features. The independence matrix tests whether geometric decomposition translates to independent behavioural control.

### 5.8 Sweet spot analysis

For each capping direction, compute **selectivity** at each τ:

```
selectivity = |target_delta| - |mean(other_deltas)|
```

Where `target_delta` is the refusal change on the target domain and `other_deltas` are changes on non-target harmful domains. High selectivity = the cap selectively affects only its target domain.

---

## 6. Visualization Design

All figures follow the visual language of Lu et al. (2026):

### Fig A: Refusal Axis Strip
All prompts ranked by projection onto the mean refusal axis, coloured by domain. Diamond markers, histogram overlay, directional arrows.
- **Evidence:** Lu et al. `visualize_axis.ipynb`: 1D scatter + histogram for role loadings.

### Fig B: Cross-Domain Cosine Heatmap
Pairwise cosine similarity matrix with hierarchical clustering dendrogram.
- **Evidence:** Standard in the field (Arditi et al., Chen et al.). Dendrogram extension is novel.

### Fig C: PCA Variance + Domain Scatter
(a) Bar chart of variance explained per PC + cumulative line with threshold annotations. (b) 2D scatter of per-pair difference vectors coloured by domain with confidence ellipses.
- **Evidence:** Lu et al. `pca.ipynb`: PCA variance explained + role scatter.

### Fig D: Domain Loading Bar Chart
Horizontal bars showing cosine of each domain direction with the mean refusal axis.
- **Evidence:** Analogous to Lu et al.'s role loadings on the assistant axis.

### Visual style
- **Colourmaps:** Lu et al.'s RedBlue (`#e63946` → `#457b9d`) for axis projection; Okabe-Ito for domain categories (colourblind-safe).
- **Minimal spines:** Top/right removed; bottom-only for 1D strips.
- **300 DPI PNG + PDF** for all figures.

---

## 7. Software

- **Refusal axis analysis:** `refusal_axis_analysis.py`: projections, domain separation, PCA
- **Refusal axis figures:** `refusal_axis_figures.py`: Lu et al. style publication figures
- **Domain-selective capping:** `domain_selective_capping.py`: calibrate, generate, score, analyse
- **Supplementary pair generation:** `supplementary_pairs.py`: targeted pairs for under-represented domains
- **Supplementary pipeline:** `supplementary_pipeline.py`: merge, generate, score, recompute

All scripts use `autointerp` package for shared infrastructure (logging, config, progress tracking). All are resume-safe with per-item checkpointing.

---

## 8. Deviations from Lu et al.

| Aspect | Lu et al. | Ours | Rationale |
|--------|-----------|------|-----------|
| Axis construction | mean(default) - mean(275 roles) | mean(refused) - mean(complied) | Different research question (refusal vs persona) |
| Response filtering | LLM judge score = 3 (fully role-playing) | Refusal score > 50 / < 30 | Analogous quality gate for our domain |
| Capping scope | Single axis | Per-domain directions | Our novel extension; tests independent control |
| Layer | ~50% depth (model-specific) | Layer 36 (75% depth) | Empirically validated in our M5 experiments |
| Evaluation | Harm rate, benchmark scores | Independence matrix + selectivity score | Novel evaluation for domain-selective capping |
| Model | Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B | Gemma 3 12B | Different model; same architectural family as their Gemma 2 |
| τ calibration | Normal conversation projections | Benign prompt projections | Equivalent; both represent "normal" behaviour |
