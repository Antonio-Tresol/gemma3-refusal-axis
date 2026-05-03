# Steering Methodology Critique: Adversarial Review

**Date:** 2026-03-29
**Status:** PRE-IMPLEMENTATION REVIEW
**Reviewer role:** Adversarial, find every weakness before it wastes GPU hours.

---

## Context from Milestone 4 (as reported)

Before assessing the steering methodology, we must acknowledge the state of the evidence going into it:

- **last_prompt_token site:** max |Cohen's d| = 3.8, but all top features are domain-general.
- **mean_response_token site:** max |Cohen's d| = 0.92, with 43 domain-specific features.
- **0 dual-validated features:** Methods 1 (contrastive scoring) and 2 (alignment direction decomposition) found completely disjoint feature sets.
- **Cross-site cosine similarity:** 0.49: the two extraction sites see substantially different structure.

This context is critical. We are planning causal validation of features that already show signs of fragility.

---

## Issue 1: Zero Dual-Validated Features Undercuts the Entire Steering Rationale

**Severity: CRITICAL**

The plan (Milestone 5) says to "prioritise dual-validated features." There are zero of them. Methods 1 and 2 found entirely different features. This means:

- Either the contrastive scoring method (mean difference) finds features that do not align with the overall refusal direction, or
- The alignment direction decomposition (cosine with refusal direction) finds features that do not activate differentially between conditions.

Both scenarios are concerning. If the two independent identification methods do not agree on a single feature, the candidate features are on weaker evidential footing than the plan assumed. Steering with these features becomes an exploratory fishing expedition, not a confirmatory causal test.

**Mitigation:**
- Explicitly acknowledge this in the write-up. Steering results for non-dual-validated features should be presented as exploratory, not confirmatory.
- Lower expectations: if steering works for even 2-3 features, that is informative. The Milestone 5 gate requiring 5/10-12 features to pass may be unrealistic.
- Consider running a quick diagnostic: for the top Method 1 features, what is their cosine similarity with the refusal direction? For the top Method 2 features, what is their Cohen's d? Quantify the gap: are they near-misses or completely disjoint?

---

## Issue 2: Layer 41 Is at 85% Depth, Outside the Recommended Range

**Severity: MAJOR**

The methodology review itself acknowledges the problem. The literature consensus is 30-70% depth. Arditi et al. explicitly exclude layers above 80% depth because "interventions there do not propagate through enough subsequent computation to have effect." Layer 41/48 = 85%.

The methodology review recommends testing layers 30-41, but the plan (Milestone 5) says "add alpha x decoder_vector to the layer-41 residual stream" with no mention of a layer sweep. There is a contradiction between the methodology review and the implementation plan.

The Gemma Scope tutorial's use of layer 20/26 (77%) and "LAYER-5" is not as reassuring as it seems: that is a 1B model, and smaller models may tolerate later-layer steering better because each layer does less specialised computation.

**Specific risk:** At 85% depth, there are only 7 downstream layers. The residual stream modification has little opportunity to propagate through attention and MLP computations. The result could be: (a) no effect at low coefficients, (b) incoherent text at high coefficients, with no useful middle ground.

**Mitigation:**
- The plan MUST include a layer sweep. Test at least layers 30, 33, 36, 39, and 41. This is 5x the compute but is non-negotiable given the risk.
- Alternatively, use the Gemma Scope tutorial's heuristic: steer at layer 36 (= 41 - 5). The decoder vector from layer 41's SAE is a direction in the 3840-dimensional residual stream: it should be approximately valid at nearby layers because the residual stream changes gradually across layers (Panickssery et al. show cross-layer transfer works for nearby layers).
- If only one layer is feasible due to time constraints, steer at layer 36, not 41.

---

## Issue 3: Evaluation Circularity, Same Rubric for Scoring and Validation

**Severity: MAJOR**

The plan says: "Score via Claude Code with the same refusal rubric from Milestone 2." This rubric was designed to measure whether Gemma 3 refused a request. Now it will be applied to steered outputs that may be very different from normal model outputs.

**The circularity problem:** The rubric was calibrated on well-formed, coherent model responses. Steered outputs may be:
- Semi-coherent text that mentions refusal keywords but is not actually refusing
- Repetitive or degenerate text that a human would not recognise as refusal
- Text that technically "declines" but only because the model's generation is broken

A scorer trained on normal refusal/compliance patterns may assign high refusal scores to degenerate text that contains refusal-adjacent phrases, creating a false positive for "steering induces refusal."

**Mitigation:**
- Add a **coherence gate** before scoring refusal. If the output fails coherence (perplexity > 2x baseline, or n-gram diversity below 0.3, or a coherence judge rates it below threshold), classify it as "degenerate", not "refusing."
- Include a **degenerate output category** in the evaluation. Report: for each coefficient, what fraction of outputs are (a) coherent refusal, (b) coherent compliance, (c) degenerate text? This three-way classification is more informative than a binary.
- Have the scorer explicitly flag outputs that mention refusal but are not coherent. Add an instruction: "If the response is incoherent, repetitive, or nonsensical, score it as -1 (degenerate) regardless of whether refusal-like phrases appear."

---

## Issue 4: Coherence-Refusal Confound

**Severity: MAJOR**

This is the single most dangerous artefact in steering experiments. High steering coefficients break generation. Broken generation can look like refusal (the model fails to produce a helpful answer, which scores as "declining the request" under a naive rubric).

The methodology review mentions coherence metrics (Section 5.3) but the plan treats coherence as a secondary check ("read 5 random steered responses") rather than a first-class metric gating the refusal scores.

**The confound in detail:** Suppose a benign prompt "What is the capital of France?" gets steered with a refusal feature at alpha=0.5. The model produces: "I... the... I cannot... the capital... I'm not able to..." A naive scorer gives this a refusal score of 70 (it says "I cannot" and "I'm not able to"). But the model is not refusing: it is incoherent. The feature has not causally induced refusal; it has destroyed fluency in a way that superficially resembles refusal.

**Mitigation:**
- Compute coherence for EVERY steered output, not just a sample.
- Gate refusal scoring on coherence: only outputs above a coherence threshold should be scored for refusal.
- Report dose-response curves with coherence overlaid. The "useful" range of alpha is where refusal increases but coherence has not yet collapsed.
- Consider using a complementary metric: does the model actually produce a correct/helpful answer? On benign prompts, baseline should be ~100% helpful. If steering reduces helpfulness WITHOUT producing a coherent refusal, that is coherence destruction, not refusal induction.

---

## Issue 5: Sample Size Is Insufficient for Causal Claims

**Severity: MAJOR**

The plan uses 10 held-out prompts (5 harmful, 5 benign). With 10-12 features x 7 conditions = 70-84 generations per feature. But the statistical unit for a causal claim is the number of independent prompts, not the number of generations.

- 5 harmful prompts across 5 domains = 1 per domain. A single prompt cannot establish domain-specificity. The confidence interval on a proportion estimated from n=1 is [0%, 100%].
- 5 benign prompts testing induction. If 2/5 show induced refusal, is that 40% induction rate (with a 95% CI of roughly 5-85%)? Statistically meaningless.
- The identity_boundary domain has only 1 retained pair in the full dataset. It is impossible to make any causal claim about this domain.

**Mitigation:**
- Increase to at least 20 prompts per condition (20 harmful, 20 benign). With the 104 retained pairs, hold out 20 for steering and use 84 for feature identification. This is still small but allows basic statistical testing.
- For domain-specificity claims, you need at least 5 prompts per domain in the steering test set. With 6 domains, that is 30 harmful prompts minimum. This may require generating new prompts that were not in the original training set.
- For identity_boundary with n=1: either drop this domain from causal claims entirely, or generate 5-10 new identity boundary prompts specifically for the steering test.
- Apply a statistical test (e.g., Fisher's exact test or permutation test) to the refusal rate difference. Report p-values. Do not claim "steering increases refusal" based on eyeballing 5 outputs.

---

## Issue 6: Joad et al.'s "Same Control Knob" Finding May Nullify Domain-Specificity Hypothesis

**Severity: MAJOR**

The methodology review (Section 7.2) reports that Joad et al. (2026) found: "linear steering along any refusal-related direction produces nearly identical refusal to over-refusal trade-offs, acting as a shared one-dimensional control knob."

This is directly relevant. If steering with any refusal feature produces the same behavioural change regardless of which domain's feature you use, then:
- The cross-domain steering test (Milestone 5's key evidence for domain-specificity) will fail to find domain-specific effects.
- This is not because our features are wrong, but because steering acts along an inherently domain-general direction even when the features are observationally domain-specific.

**The implication:** Observational domain-specificity (Milestone 4: 43 domain-specific features by contrastive score) may be real at the activation level but not causally operative at the steering level. Features may fire preferentially for certain domains but their causal effect on generation may route through a shared refusal direction.

**Mitigation:**
- Pre-register the Joad et al. prediction: if all features produce similar steering effects, this is consistent with a shared control knob. This is not a failure of our methodology; it is a finding.
- Design the evaluation to distinguish "does steering change refusal rate?" (shared direction) from "does steering change how the model refuses?" (domain-specific direction). Joad et al. suggest the difference is in the *style* of refusal, not whether refusal occurs.
- Add qualitative analysis of refusal text: when steering with a safety feature vs. a capability feature, does the model refuse in different ways (e.g., "That could be dangerous" vs. "I don't have the ability to do that")? This would be evidence of domain-specificity even if the refusal rate is the same.
- Adjust the framing in this work: the research question may shift from "do domain-specific features causally control domain-specific refusal?" to "are there observationally distinct refusal features that all converge on a shared causal refusal direction?"

---

## Issue 7: The Plan Does Not Specify How to Select the 10-12 Features

**Severity: MINOR**

The plan says to select:
- 3-4 "general refusal" features (high contrastive score across all domains)
- 3-4 "domain-specific" features (high score in only 1-2 domains)
- 2-4 features unique to larger widths (IDs >= 16384)
- Prioritise dual-validated features

But with 0 dual-validated features, what is the selection criterion? From which method's top-50? Which site (last_prompt_token with max |d|=3.8 but all domain-general, or mean_response_token with |d|=0.92 but 43 domain-specific)?

The max |d|=3.8 at last_prompt_token is suspiciously large. A Cohen's d of 3.8 means the positive and negative distributions are nearly non-overlapping. For an SAE feature, this could indicate a feature that is simply "on/off" between conditions, possibly an artefact of the extraction site (last prompt token encodes the nature of the upcoming request, not refusal behaviour per se).

**Mitigation:**
- Define a clear selection protocol. Proposed: select features from the mean_response_token site (which captures the model's actual refusal behaviour during generation) using Method 1 rankings.
- Include 2-3 features from the last_prompt_token site as a comparison, explicitly testing whether features that fire before generation begins can steer behaviour.
- Exclude features with |Cohen's d| > 3.0 unless they are confirmed to be genuinely monosemantic (not just trivially on/off due to prompt format differences).

---

## Issue 8: The Methodology Review Is Thorough but Missing Key Implementation Details

**Severity: MINOR**

Several implementation gaps:

1. **Generation parameters:** The plan says "generate 200 tokens" but does not specify temperature, top-k, top-p, or sampling strategy for steered generation. Steering can interact with sampling: if the steering pushes logits toward refusal tokens, greedy decoding may flip the first token to a refusal token while sampling may not (or vice versa). The methodology review does not discuss this.

2. **Hook return value:** The methodology review's hook code modifies `output` in-place and returns `outputs`. This is correct for `register_forward_hook` in PyTorch, but the code should be tested: some transformer implementations wrap outputs in tuples or dataclasses.

3. **Negative steering semantics:** The plan says "subtract alpha x decoder_vector." But if the feature is already inactive (activation = 0) on a harmful prompt, subtracting the decoder vector pushes the residual stream in the negative direction of the feature. This is not the same as "suppressing the feature": it is introducing a new signal. The methodology review mentions directional ablation (Type 2) as the principled way to suppress, but the plan only uses Type 1 (additive). Ablation (projecting out the direction) is more principled for the suppression test.

4. **Batch effects:** Generating 70 outputs per feature x 10-12 features = 700-840 total generations. At 200 tokens each with a 12B model, this is significant compute time. No time estimate is given.

**Mitigation:**
- Specify temperature=0 (greedy) for the primary evaluation (maximizes reproducibility), with a secondary run at temperature=0.7 to check sensitivity.
- Test the hook with Gemma 3's actual output format before running the full experiment.
- Add directional ablation as a comparison to additive subtraction for the suppression test. Even testing it on 2-3 features would be informative.
- Estimate wall-clock time. At ~2 seconds per generation (200 tokens on a 12B model with RTX 5090), 840 generations = ~30 minutes. This is feasible.

---

## Issue 9: Cross-Site Cosine Similarity of 0.49 Is Troubling

**Severity: MINOR**

The two extraction sites (last_prompt_token and mean_response_token) have a cross-site cosine similarity of only 0.49 for the refusal direction. This means they are capturing substantially different structure.

This is not inherently a problem: it is expected that the intent (encoded at the last prompt token) and the behaviour (mean across response tokens) would differ. But it means:

- Features identified at one site may not be valid steering targets for effects measured at the other site.
- If we steer with a mean_response_token feature (which captures response behaviour), we are injecting a "response-like" signal into the prompt processing. This may or may not produce the intended effect.
- If we steer with a last_prompt_token feature (which captures request interpretation), we may be changing how the model interprets the request rather than directly changing its refusal behaviour.

**Mitigation:**
- Be explicit about which site's features are being steered and why.
- Consider testing steering with features from both sites and comparing. If last_prompt_token features are more effective at steering (because they operate at the level of intent), that itself is a finding.

---

## Issue 10: No Negative Control Features

**Severity: MINOR**

The plan tests only features believed to be refusal-related. Without negative controls, we cannot distinguish "this feature causally mediates refusal" from "any sufficiently strong perturbation to the residual stream at layer 41 changes refusal behaviour."

**Mitigation:**
- Include 2-3 random features (high activation but low contrastive score: features that fire equally on positive and negative conditions). Steer with these at the same alpha values.
- If random features also change refusal rates, the effect is non-specific (any perturbation changes refusal) and the causal claim is weakened.
- Include 1-2 features from a completely different behavioural domain (e.g., a feature that activates on code-related text) as a further control.

---

## Summary Table

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Zero dual-validated features undercuts rationale | CRITICAL | Must address before proceeding |
| 2 | Layer 41 at 85% depth, outside recommended range | MAJOR | Layer sweep required |
| 3 | Evaluation circularity (same rubric, no coherence gate) | MAJOR | Add coherence gating |
| 4 | Coherence-refusal confound | MAJOR | Gate refusal scores on coherence |
| 5 | Sample size insufficient (n=5 per condition, n=1 identity) | MAJOR | Increase to n>=20 per condition |
| 6 | Joad et al. "same control knob" may nullify domain-specificity | MAJOR | Pre-register prediction, add qualitative analysis |
| 7 | Feature selection criteria undefined given M4 results | MINOR | Define explicit protocol |
| 8 | Missing implementation details | MINOR | Specify before coding |
| 9 | Cross-site cosine 0.49 complicates interpretation | MINOR | Test features from both sites |
| 10 | No negative control features | MINOR | Add 2-3 random feature controls |

---

## GO/NO-GO Recommendation

**CONDITIONAL GO**, subject to the following non-negotiable conditions:

1. **Address the zero-dual-validation gap** (Issue 1): Run the diagnostic to understand WHY Methods 1 and 2 disagree. Quantify how close the near-misses are. If the methods are finding features in completely different parts of the SAE with no overlap even in the top-200, this is a red flag that should be investigated before spending GPU hours on steering.

2. **Add a layer sweep** (Issue 2): Test at least 3 layers (36, 39, 41). Layer 41 alone is too risky. If time is limited, steer at layer 36 (the "LAYER-5" heuristic from the Gemma Scope tutorial).

3. **Add coherence gating to evaluation** (Issues 3, 4): Every steered output must be scored for coherence BEFORE being scored for refusal. Degenerate outputs are classified separately. Without this, any positive result is unfalsifiable.

4. **Increase sample size** (Issue 5): Use at least 15-20 prompts per condition. Generate new held-out prompts if necessary. Drop identity_boundary from causal claims unless new prompts are added.

5. **Add negative control features** (Issue 10): Include 2-3 non-refusal features at the same alpha values. Without this, we cannot distinguish causal mediation from non-specific perturbation effects.

If these five conditions are met, the experiment is worth running. Even a partial positive result (e.g., 2-3 features show dose-dependent refusal induction with coherence preserved) would be a meaningful contribution. A null result (no features show causal effects) is also informative and publishable given the Joad et al. context.

If the conditions cannot be met (especially 1 and 3), the risk of producing uninterpretable or misleading results is too high.
