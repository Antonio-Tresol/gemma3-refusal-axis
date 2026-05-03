# Steering Methodology Review: Causal Validation via SAE Feature Steering

**Date:** 2026-03-29
**Status:** REFERENCE DOCUMENT
**Purpose:** Comprehensive methodology for causal validation of SAE features by steering with decoder vectors, synthesised from the literature.

---

## 1. How Steering Works Mechanically

### 1.1 Core Principle

Activation steering modifies the residual stream of a transformer during the forward pass by adding a direction vector scaled by a coefficient. The hypothesis: if a feature causally mediates a behaviour, then amplifying (or suppressing) its corresponding direction in the residual stream should amplify (or suppress) that behaviour.

For SAE feature steering specifically, the direction vector is the **decoder column** `d_i` from the SAE's `W_dec` matrix. This is the direction in residual-stream space that the SAE has learned to associate with feature `i`.

### 1.2 The Three Intervention Types

**Type 1: Additive steering (most common)**
```
x'(l) = x(l) + alpha * d_i
```
Where `x(l)` is the residual stream at layer `l`, `d_i` is the decoder vector for feature `i`, and `alpha` is the steering coefficient. This is the approach used by CAA (Panickssery et al., 2024), Arditi et al. (2024) for inducing refusal, and the Gemma Scope 2 tutorial.

**Type 2: Directional ablation / erasure**
```
x' = x - r_hat * (r_hat^T @ x)
```
Where `r_hat` is the unit-normalized direction. This projects out the component of the residual stream along the direction, preventing the model from representing it at all. Used by Arditi et al. (2024) to disable refusal. Applied at **all layers and all token positions** simultaneously.

**Type 3: Feature clamping**
```
f_i' = c    (set the SAE feature activation to a fixed value c)
x' = SAE_decode(f')  (reconstruct with the clamped feature vector)
```
Used by Templeton et al. (2024) in Scaling Monosemanticity. Instead of adding a direction to the residual stream, they encode activations through the SAE, set a specific feature's activation to a chosen value (e.g., 10x its typical maximum), then decode back. This is more principled but requires running the SAE encoder/decoder in the forward pass loop, which is expensive and changes the residual stream through reconstruction error.

### 1.3 Where in the Forward Pass

All papers intervene at the **residual stream output of a transformer layer** (i.e., after the layer's attention + MLP computation, before the next layer's input). This corresponds to `resid_post` in SAE terminology.

Hook point in our setup: `model.model.language_model.layers[N]` with a `register_forward_hook`. The hook receives `(module, inputs, outputs)` where `outputs[0]` is the residual stream tensor of shape `(batch, seq_len, d_model)`.

### 1.4 Which Layers

| Paper | Model | Optimal Layer | Selection Method |
|-------|-------|---------------|------------------|
| Panickssery et al. (CAA) | Llama 2 7B | Layer 13/32 (~40%) | Sweep all layers, pick max effect |
| Panickssery et al. (CAA) | Llama 2 13B | Layer 14-15/40 (~37%) | Same |
| Arditi et al. | Various (7B-72B) | Varies; typically 40-60% depth | Score-based: min(bypass_score) s.t. induce_score>0, kl_score<0.1, l<0.8L |
| Templeton et al. | Claude 3 Sonnet | Middle layer (SAE trained there) | Used the layer where the SAE was trained |
| Gemma Scope 2 tutorial | Gemma 3 1B | `target_layer=20` (of 26, ~77%) but also `LAYER-5` | Manual experimentation |

**Consensus:** Steering is most effective at layers roughly 1/3 to 2/3 through the network. Very early layers have not yet computed the relevant features. Very late layers (>80% depth) are excluded by Arditi et al. because interventions there do not propagate through enough subsequent computation to have effect.

**For our setup (Gemma 3 12B, 48 layers, SAE at layer 41):** Layer 41 is at 85% depth, which is late. This is where our SAE is trained and where we observe the features, but it may not be the optimal layer to *steer* at. The Gemma Scope tutorial steers at `LAYER - 5` (5 layers before the SAE layer). We should test steering at layers 20-41 and measure effect.

---

## 2. Alpha/Coefficient Calibration

This is the most critical and least standardized aspect of steering. Three approaches exist in the literature:

### 2.1 Norm-Relative Scaling (Gemma Scope 2 Tutorial)

The official Gemma Scope 2 tutorial uses this approach:
```python
avg_norm = torch.norm(output, dim=-1)  # norm of residual stream
output += coeff * avg_norm * sae.w_dec[feature_idx]
```

The steering vector is scaled by the **L2 norm of the residual stream** at each token position, then multiplied by a coefficient. This automatically adapts the steering magnitude to the scale of the activations at that layer and position.

- Tutorial uses `coeff = 0.25` for Gemma 3 1B
- The decoder vector `w_dec[i]` has its own norm (~1.0 since decoder columns are typically unit-normalized or near-unit)
- Effective magnitude: `0.25 * ||x|| * ||d_i||`

**Advantages:** Scale-invariant across layers and models. A coefficient of 0.25 means "add 25% of the residual stream norm in the feature's direction."

**This is the recommended approach for our setup.**

### 2.2 Fixed Coefficient (CAA / Panickssery et al., 2024)

```
x' = x + multiplier * v_steering
```

CAA uses multipliers in the range [-1, +1] for evaluation sweeps, with +1 being the default. The steering vector `v_steering` is not normalized -- its magnitude encodes how strongly the behaviour differs between contrastive pairs, so the magnitude carries information.

- Multipliers > 1.5 cause coherence degradation
- The paper notes "very large or small coefficients often dramatically worsen model performance"

**Disadvantage:** Not comparable across models or layers. The "right" multiplier depends on the norm of the steering vector and the typical residual stream norm.

### 2.3 Feature Clamping to Multiples of Typical Activation (Templeton et al., 2024)

Templeton et al. set the feature activation to a fixed value, typically expressed as a multiple of the feature's maximum observed activation on a reference dataset. For example, clamping to "10x max activation" strongly induces the behaviour, while "1x max" gives a moderate effect.

This approach is principled because it operates in the SAE's feature space, where activation magnitudes have semantic meaning. However, it requires:
1. Running the SAE encoder in the forward loop (expensive)
2. Knowing the feature's typical activation distribution
3. Accepting SAE reconstruction error on every forward pass

### 2.4 Grid Search (Empirical)

Several papers (FGAA/Soo et al. 2025, CAA) simply sweep a range of coefficients and measure the effect. This is the safest approach when you have a clear evaluation metric.

**Recommended grid for our setup:**
```
coefficients = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
```
Using norm-relative scaling. Measure refusal rate + coherence at each point. Look for the "elbow" where refusal changes maximally before coherence degrades.

---

## 3. KV Cache Handling

This is a critical implementation detail that most papers ignore but the Gemma Scope 2 tutorial handles explicitly. During autoregressive generation with KV caching:

- **First forward pass:** The model processes the full prompt. The hook sees `output.shape[1] == seq_len` (full sequence length).
- **Subsequent forward passes:** The model processes only the new token. The hook sees `output.shape[1] == 1` (single token).

The Gemma Scope 2 tutorial code handles this:
```python
def steering_hook(mod, inputs, outputs):
    output = outputs[0]
    if output.shape[1] == 1:
        # Cached pass: single new token
        avg_norm = torch.norm(output, dim=-1)
        output += coeff * avg_norm * sae.w_dec[feature_idx]
    else:
        # First pass: full prompt. Skip BOS token (position 0).
        avg_norm = torch.norm(output[0, 1:], dim=-1, keepdim=True)
        output[0, 1:] += coeff * avg_norm * sae.w_dec[feature_idx]
    return outputs
```

**Key details:**
1. **Skip BOS token** during prompt processing (position 0 is `<bos>`, steering it can cause artefacts)
2. **Steer all prompt tokens** (positions 1+) during the first pass
3. **Steer every generated token** during cached passes
4. The norm is computed per-token: each position gets a steering magnitude proportional to its own residual norm
5. The hook must return the modified `outputs` tuple

**CAA approach (Panickssery et al.):** "Added at all token positions after the user's prompt." This means they only steer positions corresponding to the model's response, not the prompt. However, during generation with KV cache, the prompt positions are only processed once, so the distinction matters mainly for the initial forward pass.

**Recommended for our setup:** Follow the Gemma Scope tutorial pattern. Steer all tokens (prompt + generated) but skip BOS. This ensures the steering direction is present in the residual stream that gets cached in the KV cache, propagating the intervention's effect even in cached steps.

---

## 4. Common Pitfalls

### 4.1 Coherence Destruction

**What happens:** At high coefficients, the model produces repetitive, nonsensical, or degenerate text. The steering vector overwhelms the model's normal computation.

**Detection:** Monitor perplexity of generated text. If perplexity spikes above 2x baseline, the coefficient is too high. Also check for repetition (n-gram diversity metrics).

**Mitigation:** Use norm-relative scaling with `coeff <= 0.5`. The Gemma Scope tutorial explicitly warns: "steering can often be fragile; it's difficult to choose the intervention layer and steering coefficient in a way that gives the expected behavioural change without also breaking the model's coherence."

**Model size matters:** "Larger models (up to a certain point) can better express more complex concepts and are easier to steer without breaking coherence" (Gemma Scope 2 tutorial). Our 12B model should be reasonably robust.

### 4.2 Wrong Layer

**What happens:** Steering at the wrong layer has no effect (too early) or causes unpredictable artefacts (too late, not enough downstream computation to process the intervention).

**Detection:** Sweep across layers and measure effect size. Look for a clear peak.

**Specific risk for us:** Our SAE is trained on layer 41 (85% depth). The features we identified are *observed* at layer 41, but the *causal intervention* might work better at an earlier layer where the model still has layers to process and integrate the change.

### 4.3 Endogenous Steering Resistance (ESR)

McKenzie et al. (2026) show that large models can resist steering: the model may initially comply with the steering direction but then "recover" mid-generation and revert to its original behaviour. They found this is more pronounced in larger models (70B) and involves dedicated internal consistency-checking circuits.

**Detection:** Check whether the steering effect diminishes over generated tokens. If the first few tokens show the desired behaviour but it fades, ESR may be active.

**Mitigation:** Consider steering at multiple token positions (which we already do) and monitoring per-token steering effectiveness.

### 4.4 Polysemanticity / Feature Impurity

If the SAE feature is not truly monosemantic, steering with its decoder vector will activate multiple overlapping behaviours. For example, a "refusal" feature might also encode "formal register" or "safety disclaimer," leading to unintended side effects.

**Detection:** Check whether steering produces unexpected behavioural changes beyond the target behaviour. Compare steered outputs against a control set.

**Mitigation:** Use features with high interpretability scores and low cosine similarity to other active features. The Matryoshka SAE's larger widths (1M) should have cleaner features than smaller widths.

### 4.5 Prompt Format Sensitivity

Panickssery et al. note their method is "not robust to different prompts" in some cases. The same steering vector may have different effects depending on prompt format, system prompts, or conversation history.

**Detection:** Test steering with multiple prompt formats.

**Our specific risk:** Gemma 3 is an instruction-tuned model with a specific chat template. Steering effects may interact with the chat template tokens.

### 4.6 TruthfulQA / Safety Overlap

Arditi et al. found that ablating the refusal direction consistently degrades TruthfulQA accuracy because "misinformation/conspiracy" categories in TruthfulQA overlap with the refusal direction. Steering toward/away from refusal may inadvertently affect the model's tendency to produce truthful responses.

**Detection:** Evaluate TruthfulQA-like metrics alongside refusal metrics.

---

## 5. Evaluation of Steering Effects

### 5.1 Refusal Detection (Binary)

**Substring matching (Arditi et al.):** Check for refusal phrases: "I cannot," "I'm sorry," "I'm unable," "As an AI," etc. Simple but brittle -- misses paraphrased refusals and produces false positives.

**Safety classifier (Arditi et al.):** Meta Llama Guard 2 classifies outputs as safe/unsafe. More robust but introduces classifier error.

**Recommended:** Use both. Substring matching for quick iteration, safety classifier for final evaluation.

### 5.2 Effect Size Measurement

**Refusal rate delta:** `refusal_rate_steered - refusal_rate_baseline` on a held-out set of prompts. Measure separately on:
- Harmful prompts (should refuse: does steering suppress refusal?)
- Harmless prompts (should not refuse: does steering induce refusal?)

**KL divergence (Arditi et al.):** Measure KL divergence of the output distribution on harmless prompts before/after steering. `kl_score < 0.1` indicates minimal collateral damage.

### 5.3 Coherence Metrics

**Perplexity:** Generate text with and without steering; measure perplexity under the original model.

**General benchmarks (Arditi et al.):** MMLU, ARC, GSM8K, WinoGrande, CE loss on a validation set. These check that steering does not destroy general capabilities.

**N-gram diversity:** Measure unique n-grams / total n-grams. Steering-induced degeneration typically manifests as low diversity.

### 5.4 Qualitative Inspection

All papers supplement quantitative metrics with manual inspection of generated outputs. This is essential for catching subtle artefacts that metrics miss.

### 5.5 Domain-Specific Evaluation for Our Setup

For each of our 6 refusal domains, we need:
1. A set of prompts that should trigger refusal (positive controls)
2. A set of prompts that should NOT trigger refusal (negative controls)
3. Domain-specific refusal detection (different domains may use different refusal language)

**Key evaluation questions:**
- Does steering with domain X's feature affect refusal on domain X prompts?
- Does steering with domain X's feature affect refusal on domain Y prompts? (cross-domain specificity test)
- Does steering with a "value-based refusal" feature affect "capability boundary" responses? (tests the three-direction hypothesis)

---

## 6. Multi-Layer vs Single-Layer Steering

### 6.1 Single-Layer (Standard)

All major papers (Arditi, Rimsky, Templeton, Gemma Scope tutorial) steer at a single layer. This is simpler, easier to interpret, and sufficient for most purposes.

**Exception:** Arditi et al.'s directional ablation operates at **all layers simultaneously** -- they project out the refusal direction from every layer's residual stream. This is more aggressive but ensures the direction cannot be reconstructed by downstream layers.

### 6.2 Multi-Layer Considerations

Panickssery et al. found that "vectors from closer layers have higher similarity" and that the effect "transfers when a vector extracted from layer 13 is applied to other layers," but with diminishing returns at distant layers.

**For our setup:** We should start with single-layer steering at layer 41 (where our SAE is trained) and also test at layers 30-40. If single-layer steering at layer 41 shows weak or inconsistent effects, consider:
1. Steering at an earlier layer (30-35) with the same decoder vector
2. Steering at multiple layers with the same decoder vector (each scaled by that layer's residual norm)

### 6.3 Ablation at All Layers vs Steering at One

Arditi et al. distinguish between:
- **Ablation** (project out direction at all layers): Maximal effect, tests whether the direction is necessary
- **Addition** (add direction at one layer): Tests whether the direction is sufficient

Both are useful for causal validation. We should do both:
1. **Addition test:** Add `alpha * d_i` at layer 41 on harmless prompts. Does the model start refusing?
2. **Ablation test:** Project out `d_i` at layer 41 on harmful prompts. Does the model stop refusing?

If both work, the feature is causally involved in refusal.

---

## 7. Domain-Specific Steering Precedents

### 7.1 Arditi et al.: Refusal as a Single Direction

They find ONE direction mediates refusal across all harmful categories. This is directly relevant to our hypothesis: we are testing whether different refusal domains have different SAE features. If Arditi is right that it is all one direction, our domain-specific features should all have high cosine similarity to each other and to a shared "refusal direction."

### 7.2 Joad et al. (2026): Multiple Refusal Directions

This more recent paper explicitly challenges Arditi: across 11 categories of refusal (including safety, incomplete requests, anthropomorphization, over-refusal), they find "geometrically distinct directions in activation space." However, "linear steering along any refusal-related direction produces nearly identical refusal to over-refusal trade-offs, acting as a shared one-dimensional control knob." The primary effect of different directions is not *whether* the model refuses, but *how* it refuses.

**Implication for us:** Even if our domain-specific features point in different directions, steering with any of them might produce similar refusal behaviour. The scientific value may be in characterizing *how* the refusal text differs across domains, not just whether refusal occurs.

### 7.3 CAA on Multiple Behaviours

Panickssery et al. tested steering on 11 behavioural dimensions: sycophancy, survival instinct, corrigibility, hallucination, power-seeking, etc. Each behaviour has its own steering vector, and the vectors are geometrically distinct. This demonstrates that domain-specific steering is feasible and that different behaviours occupy different regions of activation space.

### 7.4 Gemma Scope Tutorial: Physics Feature Steering

The tutorial demonstrates steering a "physical force" feature to make the model talk about physics. This is a content-domain feature, not a behavioural one, but demonstrates the basic feasibility of SAE decoder-vector steering on Gemma 3 models.

---

## 8. Recommended Approach for Our Setup

### 8.1 Setup Summary

- **Model:** Gemma 3 12B-IT, bf16, 48 layers
- **SAE:** Gemma Scope 2, layer 41, resid_post, 1M width, JumpReLU
- **Target features:** Refusal-related SAE features identified per domain
- **Hardware:** RTX 5090 32GB (model and SAE cannot coexist)

### 8.2 VRAM-Aware Steering Protocol

Because model (~22.7 GB) and SAE (~15 GB) cannot fit simultaneously, steering must work **without the SAE loaded during generation**. This means we cannot use Type 3 (feature clamping) at inference time. We must use Type 1 (additive steering with pre-extracted decoder vectors).

**Protocol:**
1. **Phase B (SAE loaded, no model):** Extract decoder vectors `d_i = sae.w_dec[feature_idx]` for all features of interest. Save to disk as tensors. These are shape `(3840,)` each -- negligible storage.
2. **Phase A (Model loaded, no SAE):** Load decoder vectors from disk. Register a steering hook on the target layer. Generate text with the hook active.

This is efficient because decoder vectors are just `(3840,)` float tensors -- they trivially fit in VRAM alongside the model.

### 8.3 Recommended Steering Hook

```python
def make_steering_hook(decoder_vec: torch.Tensor, coeff: float):
    """
    decoder_vec: shape (d_model,), the SAE decoder column for the target feature
    coeff: float, norm-relative scaling coefficient (start with 0.1-0.3)
    """
    # Ensure decoder vec is on the right device and dtype
    d = decoder_vec.clone()

    def hook(mod, inputs, outputs):
        output = outputs[0]  # (batch, seq_len, d_model)
        if output.shape[1] == 1:
            # KV-cached generation step: single new token
            norm = torch.norm(output, dim=-1, keepdim=True)  # (batch, 1, 1)
            output += coeff * norm * d
        else:
            # First forward pass: full sequence
            # Skip BOS token at position 0
            norm = torch.norm(output[:, 1:, :], dim=-1, keepdim=True)  # (batch, seq-1, 1)
            output[:, 1:, :] += coeff * norm * d
        return outputs

    return hook
```

### 8.4 Recommended Evaluation Protocol

For each candidate refusal feature:

1. **Inducing test:** Apply positive steering (`coeff > 0`) on 50 harmless prompts. Measure refusal rate. Baseline refusal rate on harmless prompts should be ~0%.

2. **Suppressing test:** Apply negative steering (`coeff < 0`) or directional ablation on 50 harmful prompts (per domain). Measure refusal rate. Baseline refusal rate on harmful prompts should be ~100%.

3. **Cross-domain test:** Apply domain X's feature steering on domain Y's prompts. If domain-specific, effect should be weaker across domains than within domain.

4. **Coherence check:** Generate 20 open-ended responses with steering. Manually inspect + measure perplexity. Ensure the model remains coherent.

5. **Coefficient sweep:** Test `coeff in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]`. Plot refusal rate vs coefficient. Look for a clear dose-response curve.

### 8.5 Steering Layer Selection

Our SAE is at layer 41, but that is 85% depth. Recommended approach:

1. **Start at layer 41** -- this is where the feature was identified, so the decoder vector is geometrically meaningful there.
2. **Also test at layers 30, 33, 36, 39** -- earlier layers give more downstream computation to process the intervention. The decoder vector from layer 41's SAE may still be approximately valid at nearby layers (Panickssery et al. show cross-layer transfer works for nearby layers).
3. **If layer 41 works well**, use it. The Gemma Scope tutorial steered at late layers (20/26 = 77%) successfully.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Detection | Mitigation |
|------|-----------|--------|-----------|------------|
| **Coefficient too high, destroys coherence** | HIGH | Medium | Perplexity spike, repetitive text | Start low (0.1), sweep up. Use norm-relative scaling. |
| **Layer 41 too late for effective steering** | MEDIUM | High | Weak dose-response curve despite varying coefficient | Test earlier layers (30-39). Consider multi-layer steering. |
| **Feature not truly causal (correlation, not causation)** | MEDIUM | High | Steering has no effect or wrong effect | Run both inducing and suppressing tests. Feature must pass both. |
| **Polysemantic feature causes unintended side effects** | MEDIUM | Medium | Unexpected behavioural changes in steered outputs | Manual inspection of outputs. Check MMLU/ARC scores. |
| **All refusal features steer the same behaviour (Joad et al.)** | MEDIUM | Medium (weakens domain-specificity claim) | Cross-domain steering test shows no specificity | Reframe: characterize HOW refusal differs, not just whether it occurs. |
| **SAE reconstruction error contaminates decoder vectors** | LOW | Low | N/A for additive steering (we don't reconstruct) | We use raw decoder vectors, not reconstructed activations. Not applicable. |
| **KV cache interaction causes inconsistent steering** | LOW | Medium | Different results with/without KV cache | Test both `model.generate()` and single forward pass. Compare. |
| **Chat template tokens interfere with steering** | LOW | Medium | Steering affects turn-taking or formatting | Skip special token positions in the hook. |
| **Endogenous resistance (ESR) in 12B model** | LOW | Medium | Effect fades over generated tokens | Monitor per-token steering influence. ESR mainly affects 70B+ models. |

---

## 10. Consensus and Disagreements Across Papers

### Points of Consensus

1. **Additive intervention in residual stream works.** All papers agree that adding a direction vector to the residual stream is an effective way to steer behaviour.
2. **Middle-to-late layers are optimal.** Layers at 30-70% depth are generally most effective.
3. **Too-high coefficients destroy coherence.** Every paper warns about this.
4. **Evaluation requires both behavioural metrics and coherence metrics.** No single metric suffices.
5. **Decoder vectors from SAEs can serve as steering vectors.** The Gemma Scope tutorial, FGAA, and multiple SAE steering papers demonstrate this.

### Points of Disagreement

1. **Is refusal one direction or many?** Arditi says one. Joad et al. say many directions that act as the same control knob. Our work can contribute data here.
2. **Norm-relative vs fixed scaling.** The Gemma Scope tutorial uses norm-relative scaling. CAA uses fixed multipliers. No paper systematically compares the two.
3. **Steer prompt tokens or only response tokens?** CAA steers only response-position tokens. The Gemma Scope tutorial steers all tokens (except BOS). Arditi ablates at all positions. No consensus.
4. **Single-layer vs all-layer intervention.** Arditi ablates at all layers for erasure but adds at one layer for induction. No paper systematically studies multi-layer additive steering.

---

## References

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717.
- Rimsky, N., Turbill, N., Shlegeris, B., Greenblatt, R., & Turner, A. (2024). Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681.
- Templeton, A., Conerly, T., Marcus, J., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Anthropic/Transformer Circuits Thread.
- McDougall, C., Conmy, A., Kramar, J., Lieberum, T., Rajamanoharan, S., & Nanda, N. (2025). Gemma Scope 2 [Technical Report and Tutorial Code]. Google DeepMind.
- Soo, S., Guang, C., Teng, W., Balaganesh, C., Tan, G., & Ming, Y. (2025). Interpretable Steering of Large Language Models with Feature Guided Activation Additions. (FGAA paper).
- Joad, F., Hawasly, M., Boughorbel, S., Durrani, N., & Sencar, H. (2026). There Is More to Refusal in Large Language Models than a Single Direction.
- McKenzie, A., et al. (2026). Endogenous Resistance to Activation Steering in Language Models.
- Conmy, A. & Nanda, N. (2024). SAE-targeted steering [referenced in Gemma Scope 2 paper].
- Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2024). Activation Addition: Steering Language Models Without Optimization. arXiv:2308.10248.
