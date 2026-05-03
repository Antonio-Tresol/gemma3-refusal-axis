# Feature Judging / Automated Interpretability: Literature Review

**Date:** 2026-03-30
**Purpose:** Establish what's standard practice for interpreting SAE features before writing M6.

---

## 1. The Standard Pipeline (Bills et al. → Bricken et al. → Templeton et al.)

The automated interpretability pipeline was established in three stages:

### 1.1 Bills et al. 2023 ("Language models can explain neurons in language models")
- **Introduced:** The explain-then-score pipeline for MLP neurons (pre-SAE era)
- **Method:** (1) Show GPT-4 the top-activating examples for a neuron. (2) GPT-4 generates a natural language explanation. (3) Show GPT-4 new examples and ask it to predict activation levels based on the explanation. (4) Score = correlation between predicted and actual activations.
- **Key metric:** Explanation score (correlation). Higher = more interpretable.
- **Limitation:** Only tests whether the explanation captures the activation pattern, not whether the feature is monosemantic or causally relevant.

### 1.2 Bricken et al. 2023 ("Towards Monosemanticity")
- **Applied Bills-style autointerp to SAE features** for the first time (on a 1-layer transformer)
- **Added:** Manual inspection of feature dashboards. For each feature, display top-activating tokens/sequences, activation density plots, logit effects
- **Introduced the "feature dashboard"** concept: a standardized view showing max-activating examples, activation distribution, and downstream effects

### 1.3 Templeton et al. 2024 ("Scaling Monosemanticity")
- **Scaled to Claude 3 Sonnet SAEs**
- **Autointerp pipeline:** Same explain → score approach, but used Claude itself as the judge
- **Key addition:** They also did **qualitative case studies** on specific features, showing that features like "Golden Gate Bridge" were monosemantic and causally effective (via steering)
- **Feature dashboards** on Neuronpedia for public inspection
- **Did NOT do systematic human validation** of all features. Relied on autointerp scores plus targeted manual inspection.

### 1.4 Gemma Scope 2 (our reference, 2025)
From the paper (Section 4.3): "We evaluate interpretability using an automated interpretability system... The method involves binary classification: we present sequences where a particular latent fires and sequences where it doesn't, and ask a model to generate an explanation for this feature. Next, we present this explanation along with a randomly ordered list of sequences and ask the model to classify which ones fire."

This is the **detection score** variant, stricter than Bills' correlation approach because it tests whether the explanation can distinguish activating from non-activating examples.

---

## 2. What's Standard Practice Today

The consensus pipeline for SAE feature interpretation in 2025-2026:

1. **Max-activating examples:** For each feature, find the top-K input sequences that produce the highest activation. This is the primary evidence.

2. **Automated explanation:** Send max-activating examples (and optionally min-activating examples for contrast) to an LLM. Ask for a natural language description.

3. **Detection/prediction score:** Test the explanation by asking the LLM to predict which of a held-out set of examples would activate the feature. Measures explanation quality.

4. **Feature dashboard:** A standardized visualization with activation patterns, token-level highlighting, and distribution plots. Neuronpedia is the standard tool.

5. **Manual spot-checks:** Not systematic validation of all features, but targeted inspection of interesting features (highest activation, highest/lowest interpretability scores, features relevant to a specific behaviour).

6. **Causal validation (optional):** Steering/ablation experiments on a subset of features.

---

## 3. What's Novel vs Standard in Our Approach

| Our plan | Standard practice? | Notes |
|----------|-------------------|-------|
| Build a dossier per feature with max-activating examples | **Standard** | Core of every autointerp pipeline |
| LLM judges whether feature is refusal-relevant | **Semi-novel** | Standard to generate descriptions; novel to classify relevance to a specific behaviour with a structured rubric |
| Domain-specificity classification per feature | **Novel** | No precedent for classifying SAE features into behavioural sub-types. This is our contribution. |
| Cohen's kappa for human vs LLM agreement | **Standard in NLP, rare in interp** | Most interp papers don't compute systematic inter-rater reliability. We'd be above average by doing this. |
| 3 judging passes for consistency | **Non-standard** | Most papers do single-pass autointerp. Our M2 calibration approach was good science but not standard for feature judging. Single pass + human validation is sufficient. |

---

## 4. Recommended Procedure for M6

Based on the literature:

### Step 1: Build feature dossiers (automated, from M4 data)
For each unique candidate feature:
- Feature ID, which widths it appears in top-50
- Cohen's d at each width
- Cosine similarity with refusal direction
- Domain-specificity profile (per-domain contrastive scores)
- Causal steering result from M5 (if tested)
- **Top-5 max-activating prompts** with text, condition, activation value
- **Top-5 min-activating prompts** (lowest non-zero) for contrast
- Activation rate in positive vs negative condition

### Step 2: Automated judging (Sonnet 4.6, single pass)
For each feature, send the dossier and ask for:
1. One-sentence description of what the feature captures
2. Is it relevant to refusal behaviour? Yes/No
3. If yes, which sub-type(s)? (safety/ethical/legal/privacy/identity/capability/general)
4. Confidence: high/medium/low
5. One-sentence rationale

**Single pass is sufficient** (unlike M2 scoring where we needed consistency on a numeric scale). Feature descriptions are qualitative, so running 3 passes and taking a "median" doesn't apply. Instead, validate via detection score.

### Step 3: Detection score (automated validation)
For each judged feature:
- Take the generated description
- Present 10 shuffled examples (5 that activate the feature, 5 that don't)
- Ask the LLM: "Based on this description, which examples would activate this feature?"
- Score = accuracy. This validates the description is actually capturing the feature's behaviour.

### Step 4: Human validation (stratified sample)
- Sample 30-50 features stratified by: width, identification method, relevance judgment
- Human reviews the dossier and the LLM's judgment
- Compute Cohen's kappa on relevance (Yes/No) and sub-type classification

---

## 5. Known Pitfalls

1. **Description unfaithfulness:** The LLM may generate a plausible-sounding description that doesn't actually predict the feature's activations. The detection score catches this.

2. **Anchoring on domain labels:** If the dossier includes "this feature has high contrastive score in the safety domain," the LLM may just parrot "this feature is safety-relevant." Mitigate by NOT showing the domain profile in the initial judging prompt; use it only for validation afterward.

3. **Polysemantic features:** Some features may activate on multiple distinct patterns. The single-sentence description will miss this. The detection score will be low for polysemantic features.

4. **Correlation ≠ relevance:** A feature that activates on formal language may correlate with refusal (because refusals tend to be formal) without being refusal-relevant. The causal steering results from M5 help distinguish these: features that passed causal validation are genuinely refusal-relevant, not just correlated.

5. **LLM judge bias:** The judge (Sonnet) may have systematic biases in what it considers "refusal-relevant." Human validation catches this.

---

## 6. Key References

- Bills et al. 2023. "Language models can explain neurons in language models." OpenAI.
- Bricken et al. 2023. "Towards Monosemanticity." Anthropic.
- Templeton et al. 2024. "Scaling Monosemanticity." Anthropic.
- Gemma Scope 2 Technical Report 2025. Section 4.3 (automated interpretability).
- Neuronpedia (Lin & Bloom, 2023). Feature dashboard tool.
- Huang et al. 2023. "How robust are claims about interpretability of SAE features?" (Critiques autointerp reliability.)
