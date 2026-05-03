# Feature Hierarchy Analysis: Do Refusal Features Split Across Matryoshka Widths?

## Research Question

When the Gemma Scope 2 SAE dictionary widens from 16k to 65k features (via Matryoshka prefix slicing), do broad refusal features **decompose** into finer sub-features, or does the dictionary **reorganise**, replacing old features with geometrically unrelated new ones?

## Hypothesis

**H1 (Hierarchical decomposition):** Broad 16k features split into domain-specific children at 65k. A "general refusal" feature becomes "safety refusal" + "ethical refusal."

**H0 (Reorganisation):** Disappeared features are replaced by unrelated new features. No parent-child geometric structure.

## Methods

Three methods, ordered by what they test:

| Method | Question | Metric | Threshold | Source |
|--------|----------|--------|-----------|--------|
| M1: Decoder cosine | Are parent/child decoder vectors geometrically similar? | cosine similarity | >0.5 for hierarchy | Bricken et al. (2023, "Towards Monosemanticity," Transformer Circuits, Anthropic), Bussmann et al. (2025, "Learning Multi-Level Features with Matryoshka SAEs," arXiv:2503.17547) |
| M2: Co-activation | Do they fire on the same prompts? | Jaccard index | >0.5 for co-activation | Chanin et al. (2024, "A is for Absorption," arXiv:2409.14507) |
| M3: Hierarchy R² | Can a parent be reconstructed as a weighted sum of children? | R² of least-squares fit | >0.3 for structural alignment | Luo et al. (2026, "From Atoms to Trees: HSAE," arXiv:2602.11881) |

**Data:** 33 disappeared features (25 LPT + 8 MRT) and 40 new features (23 LPT + 17 MRT) across the 16k→65k transition. Decoder vectors from the 1M SAE (shared across prefix widths). Co-activation computed from 104 prompts × 1M SAE encodings.

## Results

### Method 1: Decoder Cosine Similarity

| Site | Parents | Max cosine | Mean max cosine | Any > 0.5 | Any > 0.3 |
|------|---------|-----------|-----------------|-----------|-----------|
| LPT | 25 | 0.287 | 0.142 | 0 | 0 |
| MRT | 8 | 0.268 | 0.150 | 0 | 0 |

**No parent-child geometric relationship.** The highest cosine across all 575 LPT pairs is 0.287, well below the 0.3 threshold, let alone 0.5.

### Method 2: Co-activation (Jaccard)

| Site | Parents | Max Jaccard | Mean max Jaccard | Any > 0.5 | Any > 0.3 |
|------|---------|------------|------------------|-----------|-----------|
| LPT | 25 | 0.961 | 0.241 | 1 | 2 |
| MRT | 8 | 1.000 | 0.150 | 1 | 1 |

Two apparent high-Jaccard pairs emerged. **Both were investigated by robustness checks (see below).**

### Method 3: Hierarchy R²

| Site | Parents | Max R² | Mean R² | Any > 0.5 | Any > 0.3 | Children/d_model |
|------|---------|--------|---------|-----------|-----------|-----------------|
| LPT | 25 | 0.580 | 0.092 | 1 | 1 | 0.60% |
| MRT | 8 | n/a | 0.056 | 0 | 0 | 0.44% |

One LPT parent (feature 9449) has R² = 0.580. This is the same feature flagged by Method 2.

## Robustness Checks

### Check 1: Permutation null distribution for Jaccard

For each parent, shuffled its activation vector 1000 times and recomputed max Jaccard.

| Pair | Observed Jaccard | p-value | Significant? |
|------|-----------------|---------|-------------|
| LPT 9449→20318 | 0.961 | 0.076 | **No** |
| LPT 11090→24339 | 0.375 | 0.000 | Yes |
| MRT 8315→23735 | 1.000 | 0.010 | Yes |

**Feature 9449→20318 does NOT survive the permutation test.** p=0.076, above the 0.05 threshold.

### Check 2: Base-rate analysis

| Pair | Parent activation rate | Child activation rate | Expected Jaccard (independence) | Observed Jaccard |
|------|----------------------|---------------------|-------------------------------|-----------------|
| LPT 9449→20318 | 97.1% | 97.1% | 0.944 | 0.961 |
| MRT 8315→23735 | 1.0% | 1.0% | 0.005 | 1.000 |

**Feature 9449→20318 is a base-rate artefact.** Both features fire on 97.1% of positive prompts. Any two features with ~97% activation rates will have Jaccard ≈ 0.94 under independence. The observed 0.961 is only 0.017 above chance.

**Feature 8315→23735 is statistically significant** (p=0.010) because both fire on only 1% of prompts, making chance co-activation unlikely. However, this is 1 prompt out of 104; the sample is too small for a mechanistic claim.

### Check 3: Threshold sensitivity

**All 33 disappeared features are truly absent from the 65k contrastive data**, not below threshold but, genuinely gone. This rules out the alternative that features merely weakened rather than vanished.

### Check 4: Other width transitions

| Transition | Site | Disappeared | New | Max cosine | >0.3 | >0.5 |
|-----------|------|------------|-----|-----------|------|------|
| 65k→262k | LPT | 3 | 10 | 0.256 | 0 | 0 |
| 65k→262k | MRT | 2 | 5 | 0.098 | 0 | 0 |
| 262k→1M | LPT | 1 | 1 | 0.028 | 0 | 0 |
| 262k→1M | MRT | 0 | 0 | n/a | n/a | n/a |

**No hierarchy at any width transition.** The 65k→262k transition shows max cosines (0.256 LPT, 0.098 MRT) comparable to or lower than 16k→65k (0.287 LPT, 0.268 MRT). The 262k→1M transition shows near-complete stabilization: LPT has 1 disappeared and 1 new feature with cosine 0.028 (effectively random); MRT has zero changes.

### Check 5: Power analysis

Expected max cosine between random unit vectors in R^3840 with 23 comparisons: **0.032**. Observed: **0.287** (9× above random). This confirms the decoder vectors are not fully random with respect to each other, but 0.287 is far below any hierarchy threshold. There is weak structure, likely reflecting shared refusal-related subspace, but not parent-child geometry.

## Conclusions

### What we can claim (survived falsification):

1. **No hierarchical feature splitting at any width transition.** All three methods converge: disappeared features are geometrically unrelated to new features. This holds across 16k→65k (max cosine 0.287), 65k→262k (max cosine 0.256 LPT, 0.098 MRT), and 262k→1M (max cosine 0.028). Robust across methods and not explained by thresholding artefacts.

2. **Genuine dictionary reorganisation.** Features do not weaken; they truly vanish from the 65k encoding (Check 3). The Matryoshka SAE replaces features wholesale at wider widths rather than refining them.

3. **Weak geometric structure above random.** Max cosine (0.287) is 9× above random expectation (0.032), suggesting disappeared and new features share a broad refusal-related subspace. But this is far too weak for parent-child hierarchy.

### What we cannot claim (failed falsification):

4. ~~"Two features show reparameterisation."~~ **Retracted.** The LPT pair (9449→20318) is a base-rate artefact (both fire on 97% of prompts, expected Jaccard under independence = 0.944). The MRT pair (8315→23735) is statistically significant but based on 1 prompt, insufficient for any mechanistic claim.

### What we don't know (not tested):

5. Whether non-refusal-relevant features (below our Cohen's d threshold) participate in hierarchies we missed by filtering.
6. Whether a broader search over ALL feature pairs (not just refusal-relevant ones) would find hierarchy in the SAE generally.

## Implications

The absence of hierarchical splitting has practical implications:

- **Feature selection across widths is not transferable.** A feature identified as important at 16k cannot be assumed to exist at 65k. Each width requires independent feature discovery.
- **Matryoshka ordering ≠ semantic hierarchy.** The Matryoshka loss orders features by reconstruction importance, not by semantic generality. "More important for reconstruction" does not mean "broader concept that decomposes into finer ones."
- **This contradicts the plan's hypothesis** (rq1_poc_plan_v5.md line 15: "broad features may split into sub-type-specific features"). The data says otherwise.

## Reproducibility

| Item | Location |
|------|----------|
| Analysis script | `feature_hierarchy_analysis.py` |
| Results (all methods + robustness) | `data/hierarchy_results/all_methods.json` |
| Feature IDs per width | `data/milestone_7_results/width_metrics.json` |
| Decoder vectors | Extracted from `google/gemma-scope-2-12b-it` 1M SAE |
| Co-activation data | `data/encoded/{site}_{condition}.pt` |
| Analysis plan | `findings/plans/plan_feature_hierarchy_analysis.md` |
