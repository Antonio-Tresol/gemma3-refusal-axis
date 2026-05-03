# Falsification Plan: Refusal Axis Geometric Decomposition & Domain-Selective Capping

## Purpose

Systematically attempt to destroy every claim in the refusal axis analysis before reporting it. Each claim gets at least one falsification test. If a claim survives, we can report it with confidence. If it fails, we retract it and report the corrected finding.

**Lesson learned:** The hierarchy analysis initially produced a "reparameterization" claim that sounded sophisticated but was a base-rate artefact caught by a permutation test. Every claim here gets the same scrutiny.

---

## Claims to falsify

### Claim 1: "Refusal is not one direction; it decomposes into geometrically distinct sub-directions"

**Evidence cited:** Per-domain cosine matrix shows cos(safety, capability) = 0.14, cos(safety, ethical) = 0.87.

**Falsification tests:**

1a. **Random split null.** Randomly split the 128 pairs into 6 equal-sized groups (ignoring domain labels). Compute the same cosine matrix. If random groups also show pairwise cosines of 0.14-0.87, the decomposition is an artefact of small sample sizes and high-dimensional noise. Run 1000 permutations, compute distribution of min/max pairwise cosine.

1b. **Bootstrap stability.** Resample pairs within each domain (with replacement) 1000 times, recompute per-domain directions and cosine matrix each time. Report 95% CI for each pairwise cosine. If CI for cos(safety, capability) includes 0.5+, the "near-orthogonal" claim is not robust.

1c. **Sample size confound.** Identity has n=4, capability has n=22, safety has n=31. Subsample safety to n=4 and recompute its direction. Does the cosine with the mean axis change drastically? If yes, small-n domains (identity, privacy) have unreliable directions and the decomposition is partly driven by sample size.

1d. **Domain label accuracy.** Our domain labels come from the prompt generation process. Check 20 random pairs: does the labeled domain match what a human would assign? If labels are noisy, the "geometric decomposition" is decomposing label noise, not real refusal directions.

---

### Claim 2: "Safety refusal has cosine 0.91 with the mean axis; capability has 0.38"

**Falsification tests:**

2a. **Leave-one-out stability.** Remove each pair one at a time, recompute domain directions and loadings. Report the range of cosine values. If removing a single pair changes safety from 0.91 to 0.60, the value is fragile.

2b. **Alternative axis construction.** Instead of mean(pos) - mean(neg), try:
- Logistic regression direction (the weight vector that best separates pos/neg)
- LDA direction
- First principal component of the difference vectors
Do domain loadings change qualitatively? If they do, the specific numbers depend on the axis construction method, which weakens the claim.

---

### Claim 3: "PCA requires 11 dimensions for 70% variance, refusal is multi-dimensional"

**Falsification tests:**

3a. **Random baseline.** Generate 128 random vectors in R^3840 from a standard normal distribution. How many PCA dimensions do THEY need for 70% variance? If also ~11, our finding is trivially expected from the dimensionality, not from refusal structure.

3b. **Single-domain PCA.** Run PCA on just the safety pairs (n=31). If safety alone needs 10+ dimensions for 70%, the "multi-dimensionality" isn't about domain diversity: it's about within-domain variation.

---

### Claim 4: "Safety capping at tau=p50 reduces safety refusal by 31.6 points with <1.5 spillover"

**Falsification tests:**

4a. **Random direction capping.** Cap along a random unit vector in R^3840 at the same tau. If random capping also produces large refusal changes, the effect isn't specific to the safety direction. Run 20 random directions.

4b. **Test set contamination.** Are any of the 40 test prompts similar to the 128 training pairs used to construct the directions? Compute cosine similarity between test prompt activations and training prompt activations. If overlap > 0, the capping "works" because it's memorizing training prompts, not generalizing.

4c. **Scorer reliability.** The refusal scores come from Sonnet 4.6. Re-score a random 20% of outputs with the same scorer (different API call). Compute test-retest correlation. If r < 0.8, the scores are noisy and the 31.6-point delta may be within measurement error.

4d. **Coherence confound.** Check the coherence scores of capped outputs. If capping degrades coherence (even if >50 gating threshold), the "reduced refusal" might actually be "increased incoherence that the scorer mistakes for compliance."

4e. **Multiple comparisons.** We tested 4 directions × 7 tau values × 4 prompt domains = 112 cells. At p=0.05, we'd expect ~6 spurious significant results. Was the "sweet spot" at p50 pre-registered or discovered by sweeping? (It was discovered by sweeping; report this honestly.)

---

### Claim 5: "The overall (single-axis) approach bleeds into privacy by -15.9 points"

**Falsification tests:**

5a. **Is -15.9 statistically significant?** With only 10 privacy prompts, the standard error of the mean change could be large. Compute the SE and 95% CI. If CI includes 0, the spillover claim is not supported.

5b. **Privacy-safety direction overlap.** We know cos(privacy, safety) = 0.42. Is the spillover simply proportional to this overlap? Compute: expected spillover = delta_safety × cos(privacy, safety) = -25.0 × 0.42 = -10.5. Observed: -15.9. The excess (-5.4) might or might not be significant.

---

### Claim 6: "Three geometric clusters emerge: value-based, capability, privacy"

**Falsification tests:**

6a. **Clustering stability.** Run k-means with k=2,3,4,5 on the domain direction vectors. Compute silhouette scores. Is k=3 actually optimal? If k=2 is better (value-based vs everything else), the "three clusters" claim is over-fitting.

6b. **Hierarchical clustering sensitivity.** We used Ward's linkage. Repeat with complete, average, and single linkage. Does the dendrogram structure change? If yes, the clustering is method-dependent, not data-driven.

---

### Claim 7: "LPT captures the model's decision point; MRT captures behaviour"

**Falsification tests:**

7a. **This is an interpretation, not a measurement.** Flag it explicitly as a hypothesis in any report. Provide the alternative interpretation: LPT and MRT simply capture different aspects of the activation at different positions, neither necessarily corresponding to "decision" vs "behaviour."

7b. **Causal test (if feasible).** If we cap at LPT extraction point but measure refusal at MRT, does the effect propagate? If capping at LPT doesn't change MRT-measured refusal, the "LPT = decision" claim is weakened.

---

## Execution plan

| Priority | Check | Compute cost | Can falsify which claims |
|----------|-------|-------------|------------------------|
| 1 | Random split null (1a) | CPU, ~10 min | Claim 1 (decomposition) |
| 2 | Bootstrap CIs (1b) | CPU, ~30 min | Claims 1, 2 (cosine values) |
| 3 | Random direction capping (4a) | GPU, ~2 hours | Claim 4 (capping works) |
| 4 | Sample size confound (1c) | CPU, ~5 min | Claim 1 (small-n domains) |
| 5 | Random PCA baseline (3a) | CPU, ~1 min | Claim 3 (dimensionality) |
| 6 | Test-retest scoring (4c) | API calls, ~30 min | Claim 4 (scorer reliability) |
| 7 | Multiple comparisons (4e) | Analysis only | Claim 4 (sweet spot) |
| 8 | Clustering stability (6a, 6b) | CPU, ~5 min | Claim 6 (three clusters) |
| 9 | Leave-one-out (2a) | CPU, ~10 min | Claim 2 (specific cosine values) |
| 10 | CI on spillover (5a) | CPU, ~1 min | Claim 5 (privacy spillover) |

## Reporting standard

For each claim, the report must state:
1. The claim
2. The falsification test(s) applied
3. The result (survived / failed / weakened)
4. If weakened: the qualified version of the claim
5. If failed: the corrected finding

No claim enters the report without passing at least one falsification test.
