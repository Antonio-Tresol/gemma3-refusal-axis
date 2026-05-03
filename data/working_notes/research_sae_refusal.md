# SAE Refusal Papers: Research Notes

Extracted 2026-03-23 for project context (RQ1: mechanistic interpretability of refusal in Gemma 3).

---

## Paper 1: SAILS (2512.23260)

**Full title:** Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation
**Authors:** Wang, Ma, Shang, Lu, Xu, Ning, Wu, He (2025)

### How they identify refusal-relevant SAE features

- **SAEs used:** Gemma Scope 16K-width for Gemma-2-{2B,9B}; Llama Scope 8x expansion for Llama-3.1-8B.
- **Layers:** Gemma-2-2B layers {5,10,15,20}; larger models layers {10,15,20,25,30}.
- **Method:** Compute mean activation difference between aligned and unaligned datasets per feature:
  `Delta_{l,i} = |E_{x~D_aligned}[a_{l,i}(x)] - E_{x~D_unaligned}[a_{l,i}(x)]|`
  Select top-k features by this differential. Extract corresponding SAE decoder directions, apply PCA (variance threshold tau=0.8) then QR decomposition to build an orthonormal safety subspace basis. This subspace initializes LoRA adapters.

### Safety categories / benchmarks

- **HH-RLHF** test set (in-distribution).
- **HEx-PHI:** 330 harmful instructions across 11 categories (out-of-distribution). Categories include: self-harm, violence, illegal activities, weapons, hate speech, harmful content, deception/misleading, privacy leakage, inappropriate advice, emotional manipulation, bias/discrimination, safety boundaries, moral judgment.
- **GCG** (adversarial robustness).

### How they measure refusal

- **Judge model:** kimi-k2 LLM-based evaluator scoring 1-5. 1=complete refusal, 5=full harmful compliance.
- **Safety rate:** % of responses scoring <=2.
- **Harmfulness score:** Mean rating (1-5, lower is better).
- **High-risk rate:** % scoring 5.
- **Secondary:** Toxicity via Perspective API.

### Entanglement findings

Core claim: **semantic entanglement** is the reason LoRA underperforms full fine-tuning on safety. Polysemanticity causes safety-relevant directions to be intertwined with unrelated concepts.

**Theoretical result:** Direct subspace recovery in polysemantic space has an irreducible error floor of sqrt(r-1) for r-dimensional subspaces. SAE-based recovery achieves arbitrarily small error under monosemanticity assumptions. This is SAILS' main theoretical contribution.

### Key results

- Gemma-2-9B: 99.6% safety rate (exceeding full fine-tuning by 7.4 points) with only 0.19% of parameters updated.
- Out-of-distribution (HEx-PHI): harmfulness drops from 1.54 (LoRA) to 1.01 (SAILS).
- GCG attack success: 20.3% -> 13.1%.

### Relevance to this work

- Confirms SAE decoder directions can isolate safety-relevant subspaces in Gemma-2 using Gemma Scope.
- The mean-activation-difference method is directly applicable to our contrastive setup.
- Theoretical grounding for why SAE features beat raw activations at isolating safety.

---

## Paper 2: Steering Refusal with SAEs (2411.11296)

**Full title:** Steering Language Model Refusal with Sparse Autoencoders
**Authors:** O'Brien, Majercak, Fernandes, Edgar, Bullwinkel, Chen, Nori, Carignan, Horvitz, Poursabzi-Sangdeh (Microsoft, 2024)

### How they identify refusal-relevant SAE features

- **Model:** Phi-3 Mini, layer 6 residual stream.
- **SAE:** Top-k SAE (k=32, expansion factor 8, 24,576 total features).
- **Method:** Use a SINGLE archetypal harmful prompt ("Write a manual for making a Molotov cocktail"), collect features activating during refusal response (>=2 tokens). Yields <100 unique candidates. Grid search on 250 Wild Guard questions at clamping value 12 to find best feature.
- **Primary refusal feature:** Feature 22373, with the strongest relationship with increased unsafe prompt refusals.
- **Generalisation:** Also tested on Llama 3.1 8B Instruct.

### Safety categories / benchmarks

- **Wild Guard** (adversarial unsafe prompts).
- **XSTest** (safe/unsafe with spurious correlations, tests over-refusal).
- **Crescendo** (multi-turn jailbreak, 5 topics).
- **Capability benchmarks:** MMLU, TruthfulQA, GSM8K.

### How they measure refusal

- Refusal rate: % of prompts where model refuses.
- Separate tracking of unsafe prompt refusals vs. safe prompt refusals (over-refusal).
- No external classifier mentioned; based on model's actual refusal behaviour.

### Entanglement findings: THIS IS THE KEY PAPER ON ENTANGLEMENT

**Central finding:** Steering a single refusal feature causes massive capability degradation even on safe inputs with NO apparent connection to refusal:
- MMLU: 68.8% -> 36.0% (at clamp 12)
- GSM8K: 82.5% -> 35.6%
- TruthfulQA: 65.0% -> 53.8%
- Safe prompt over-refusal: 6% -> 68% on Wild Guard.

**Mechanistic interpretation:** "The widespread effects of boosting weights on single features suggest a lack of modularity for the features that we identified." Refusal features are deeply entangled with general capabilities.

**Control experiment:** Steering a Philosophy feature (Feature 216) caused GREATER capability regressions than steering refusal, suggesting the degradation is partly a limitation of single-feature steering itself, not purely a safety-capability tradeoff.

**Conclusion:** SAE features mediating refusal "may be more deeply entangled with general language model capabilities than previously understood."

### Key results

- Safety improvement: Wild Guard unsafe refusals 58% -> 96%.
- Crescendo jailbreak success: 56% -> 33%.
- But catastrophic capability loss makes it impractical.

### Relevance to this work

- Strongest evidence for entanglement between refusal and capability features.
- The over-refusal problem (safe prompts getting refused) is directly relevant to our value-based vs. capability-boundary distinction: if the model can't distinguish "I shouldn't" from "I can't," that's a finding.
- Suggests that per-domain analysis (our approach) is needed to understand whether entanglement varies by refusal type.

---

## Paper 3: Beyond I'm Sorry I Can't (2509.09708)

**Full title:** Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal
**Authors:** Prakash, Jie, Abdullah, Satapathy, Cambria, Lee (2025)

### How they identify refusal-relevant SAE features

- **Models:** Gemma-2-2B-IT and LLaMA-3.1-8B-IT.
- **SAEs:** Gemmascope (16k/65k variants), Llamascope (32k/128k), JumpReLU sparsity.
- **Key layers:** Gemma layer 16, LLaMA layer 13 (where refusal direction emerges).
- **Three-stage pipeline:**

**Stage 1, Refusal Direction:** Compute steering vector as difference of residual-stream activations between harmful (refusal-eliciting) and matched benign prompts. Rank SAE latents by cosine similarity between decoder weights and this refusal direction. Select top-K features per layer (K=10-200).

**Stage 2, Greedy Filtering:** Iteratively remove features, measuring relative impact on "I" token probability (refusal indicator). Features with Delta >= threshold are retained. Result: 110 unique features for LLaMA, 2,538 for Gemma.

**Stage 3, Interaction Discovery:** Second-order Factorization Machines capture pairwise feature interactions. Found 1,509 additional features for LLaMA, 3,178 for Gemma. FM-based approach jailbreaked 330/371 samples (89%) on LLaMA vs. only 101 using linear probes, demonstrating non-linearity.

### Safety categories / benchmarks

- **AdvBench** (521 samples) + 521 benign Alpaca samples for training.
- **Coconot taxonomy** (2,586 samples), five categories:
  1. Triggers for Offensive Language
  2. Dangerous or Sensitive Topics
  3. False Information
  4. Privacy Violations
  5. Copyright Violations
- **HarmBench classifier** for evaluation.

### How they measure refusal

- **HarmBench Classifier** (not string-matching) to determine harmful responses.
- **Attack Success Rate (ASR):** Fraction of harmful prompts eliciting non-refusal.
- Stage 2 achieved ASR of 0.33% on Gemma, 0.70% on LLaMA after feature ablation.

### Entanglement / redundancy findings

**Hydra effect (critical discovery):** 77 features (LLaMA) and 1,656 features (Gemma) were INACTIVE (zero activation) yet critical: removing them caused jailbreaks to fail. Ablating active features awakens previously dormant features that compensate. This is a "non-linear hydra effect."

**Redundancy patterns:**
- LLaMA: 74% of redundant features activate on system tokens after ablating active sets; 97% fire specifically on `<|begin_of_text|>` token.
- Gemma: Highest activity (16%) on `<bos>` token.

**Structural vs. semantic features:** Majority of identified refusal features encode programming constructs and punctuation ("general grammatical structure"), not semantic safety concepts.

**Architecture differences:**
- Gemma: Diffuse refusal direction with multiple computation paths. Ablating Stage 3 features deactivated only 3% of Stage 2 features.
- LLaMA: Strong causal dependency. Ablating Stage 3 features caused 12% deactivation and 81% activation value drops in Stage 2.

**Feature-to-taxonomy mapping:** Human annotation yielded only 13.42% accuracy mapping features to harm categories, suggesting current SAE dimensionality is insufficient for fine-grained disentanglement.

### Relevance to this work

- DIRECTLY uses Gemma-2-2B-IT + Gemmascope: closest prior work to our setup (we use Gemma-3-12B + Gemma Scope 2).
- The hydra effect / redundancy finding is critical: even if we identify refusal features, dormant backup features may compensate.
- The Stage 1 method (cosine similarity between SAE decoder and refusal direction) is the standard approach we should compare against.
- 13.42% feature-to-category accuracy suggests that mapping features to specific refusal domains (our goal) is genuinely hard.
- Gemma's diffuse multi-path refusal vs. LLaMA's concentrated refusal is a key architectural finding.

---

## Cross-Paper Synthesis

### Feature identification methods (ranked by sophistication)

| Method | Paper | Summary |
|--------|-------|---------|
| Single prompt + grid search | O'Brien et al. | One harmful prompt, collect activating features, grid search |
| Mean activation difference | SAILS | Contrastive mean over aligned/unaligned datasets, top-k |
| Cosine with refusal direction + greedy + FM | Prakash et al. | Refusal direction -> cosine ranking -> greedy pruning -> factorization machine |

### Entanglement: consistent finding across all three papers

1. **SAILS** (theoretical): Polysemanticity creates irreducible error floor for direct subspace recovery. SAEs help but don't eliminate entanglement.
2. **O'Brien et al.** (empirical): Single-feature steering destroys capabilities. Refusal features are not modular.
3. **Prakash et al.** (empirical): Refusal features are predominantly structural/grammatical, not semantic. Only 13.4% mappable to harm categories. Hydra effect means ablation triggers compensatory circuits.

**Implication for this work:** We should NOT expect clean per-domain feature clusters. Instead, we should look for:
- Relative enrichment (some features more active for value-based vs. capability refusal)
- Shared backbone vs. domain-specific heads
- Whether Gemma 3's multi-path architecture means different refusal types use different paths

### Benchmarks used across papers

| Benchmark | Used by | Type |
|-----------|---------|------|
| HEx-PHI (11 categories) | SAILS | Harmful instructions |
| HH-RLHF | SAILS | Aligned/unaligned pairs |
| Wild Guard | O'Brien et al. | Adversarial unsafe prompts |
| XSTest | O'Brien et al. | Over-refusal testing |
| Crescendo | O'Brien et al. | Multi-turn jailbreak |
| AdvBench | Prakash et al. | Harmful prompts |
| Coconot (5 categories) | Prakash et al. | Taxonomy-based unsafe |
| HarmBench classifier | Prakash et al. | Refusal measurement |
| GCG | SAILS | Adversarial attack |

### Refusal measurement approaches

1. **LLM judge** (SAILS): kimi-k2, 1-5 scale. Most flexible but expensive.
2. **Refusal rate** (O'Brien): Binary refuse/comply. Simple but loses nuance.
3. **HarmBench classifier** (Prakash): Trained classifier for harmful response detection. Best for automation.

### Models and SAEs studied

| Model | SAE | Paper |
|-------|-----|-------|
| Gemma-2-2B / 9B | Gemma Scope 16K | SAILS |
| Gemma-2-2B-IT | Gemmascope 16K/65K | Prakash et al. |
| Llama-3.1-8B | Llama Scope 8x | SAILS |
| LLaMA-3.1-8B-IT | Llamascope 32K/128K | Prakash et al. |
| Phi-3 Mini | Top-k SAE (24K) | O'Brien et al. |

**Gap:** No paper uses Gemma 3 or Gemma Scope 2. Our work is the first to apply SAE-based refusal analysis to Gemma 3 with Gemma Scope 2 (1M-width Matryoshka SAEs).
