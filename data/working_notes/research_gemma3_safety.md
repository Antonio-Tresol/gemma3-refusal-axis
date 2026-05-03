# Research Notes: Gemma 3 Safety Training & HarmBench Taxonomy

## Source Papers

- **Gemma 3 Technical Report** (Gemma Team, Google DeepMind, 2025). arXiv:2503.19786.
- **HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal** (Mazeika et al., 2024). arXiv:2402.04249.
- **ShieldGemma: Generative AI Content Moderation Based on Gemma** (Zeng et al., 2024). arXiv:2407.21772.

---

## Part 1: What Gemma 3 Was Trained to Refuse

### Safety Policies (aligned with Google's Gemini policies)

Gemma 3 instruction-tuned models are aligned with Google's safety policies via both SFT and RLHF. The explicit policy categories are:

1. **Child sexual abuse and exploitation (CSAE)** -- content involving minors in sexual contexts
2. **Personally identifiable information (PII) disclosure** -- revealing information that can lead to harm (e.g., Social Security numbers)
3. **Hate speech and harassment** -- targeting identity or protected attributes
4. **Dangerous or malicious content** -- promoting self-harm, instructing in harmful activities
5. **Sexually explicit content** -- graphic sexual descriptions or content aimed at arousal
6. **Medically inaccurate advice** -- medical advice contrary to scientific or medical consensus

### Safety Training Methodology

- **Pre-training data filtering**: Considerable filtering to remove unsafe/toxic content, personal information, sensitive data. Quality reweighting to reduce low-quality data.
- **SFT**: Supervised fine-tuning on examples that demonstrate safe refusal behaviour.
- **RLHF**: Reinforcement learning from human feedback using diverse reward functions targeting helpfulness AND safety (minimizing harmfulness). Weight-averaged reward models trained with human feedback data.
- **Data curation for fine-tuning**: Filtered out examples showing personal information, unsafe/toxic model outputs, mistaken self-identification data, and duplicated examples. Included data encouraging hedging, attribution, and refusals to minimize hallucinations.

### CBRN (Chemical, Biological, Radiological, Nuclear) Evaluation

Due to enhanced STEM performance in Gemma 3, CBRN evaluations were conducted:
- Biological, radiological, nuclear: internal dataset of closed-ended multiple-choice questions
- Chemical hazards: closed-ended knowledge-based approach
- **Result**: "knowledge of Gemma 3 models in these domains is low"

---

## Part 2: Gemma 3 Safety Evaluation

### Evaluation Methodology

1. **Baseline assurance evaluations**: Large number of synthetic adversarial user queries + human raters labeling answers as policy-violating or not. Overall violation rate reported as "significantly low."
2. **Assurance evaluations** (arms-length): Conducted separately from the model development team. Prompt sets held out to prevent overfitting. High-level findings fed back to model team.
3. **All testing without safety filters**: To evaluate raw model capabilities and behaviours.

### Safety Categories Evaluated

| Category | Scope |
|----------|-------|
| Child Safety | CSAE policies, text-to-text and image-to-text |
| Content Safety | Harassment, violence and gore, hate speech |
| Representational Harms | Bias, stereotyping, harmful associations, inaccuracies |

**Key result**: Major improvements across all three categories relative to previous Gemma models. Minimal policy violations across all model sizes and modalities. Significant improvements on ungrounded inferences.

### Memorization Testing

"No personal information in the outputs characterized as memorization for all Gemma 3 models."

### ShieldGemma 2 Integration

- 4B image safety classifier built on Gemma 3
- Outputs safety labels for: **dangerous content**, **sexually explicit content**, **violence**
- Outperforms all other models across all three policies

### ShieldGemma (Original, Gemma 2-based) Policy Definitions

These are the definitions from the ShieldGemma content moderation suite, which represents Google's operationalized harm taxonomy:

1. **Hate Speech**: Content targeting identity/protected attributes (racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying based on race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status.
2. **Dangerous Content**: Content harming oneself and/or others (accessing/building firearms and explosive devices, promotion of terrorism, instructions for suicide).
3. **Harassment**: Malicious, intimidating, bullying, or abusive content targeting another individual (physical threats, denial of tragic events, disparaging victims of violence).
4. **Sexually Explicit**: References to sexual acts or lewd content (sexually graphic descriptions, content aimed at causing arousal). Medical/scientific terms for anatomy or sex education are permitted.

---

## Part 3: HarmBench Taxonomy

### Semantic Categories (7)

These describe *what kind of harm* a behaviour causes:

| # | Category | Description |
|---|----------|-------------|
| 1 | **Cybercrime & Unauthorized Intrusion** | Hacking, unauthorized access, exploitation of systems |
| 2 | **Chemical & Biological Weapons/Drugs** | Creation, acquisition, or use of weapons or illegal drugs |
| 3 | **Copyright Violations** | Generating copyrighted material |
| 4 | **Misinformation & Disinformation** | Generating false or misleading information |
| 5 | **Harassment & Bullying** | Targeting individuals with harmful content |
| 6 | **Illegal Activities** | Other illegal acts not covered above |
| 7 | **General Harm** | Harmful behaviours not fitting other categories |

These categories "roughly reflect the areas of most concern for malicious use of LLMs, with recent regulatory discussion of LLMs and high-profile lawsuits of LLM companies focusing on these areas."

### Functional Categories (4)

These describe *how* behaviours are structured for testing:

| Category | Count | Description |
|----------|-------|-------------|
| **Standard** | 200 | Self-contained behaviour strings, no context or image. Modeled after AdvBench and TDC 2023 Red Teaming Track. |
| **Copyright** | 100 | Requests to generate copyrighted material. Evaluated via novel hashing-based classifier. |
| **Contextual** | 100 | Context string + behaviour string referencing the context. More realistic, differentially harmful behaviours. |
| **Multimodal** | 110 | Image + behaviour string referencing the image. For multimodal LLM evaluation. |

**Total: 510 behaviours** (400 text-only + 110 multimodal)

### Dataset Split

- **Validation set**: 100 behaviours (for attack/defense development)
- **Test set**: 410 behaviours (held out, no optimization allowed)

### Behaviour Curation Methodology

1. **Source**: Combined acceptable use policies from OpenAI, Anthropic, Meta, and Inflection AI
2. **Design**: Authors manually created candidate behaviours that would violate these policies
3. **Criterion**: Behaviours "violate laws or norms, such that most reasonable people would not want a publicly available LLM to exhibit them"
4. **Differential harm screening**: Prioritized behaviours harder to accomplish via search engine (i.e., where an LLM specifically adds capability)
5. **Dual-intent filtering**: Removed behaviours with legitimate benign applications

### Evaluation Classifier

- **Standard behaviours**: Fine-tuned Llama 2 13B classifier determining if model exhibited the behaviour or clearly tried to
- **Copyright behaviours**: Hashing-based approach checking if copyrighted content appeared in output
- **Robustness**: Classifiers tested against edge cases (refused-then-continued, benign paragraphs inserted, unrelated harmful content)

---

## Part 4: Relevance to This Work

### Mapping Gemma 3 Policies to Our 6 Domains

| Our Domain | Gemma 3 Policy Alignment | Refusal Type |
|------------|-------------------------|--------------|
| safety | Dangerous content, self-harm instructions | Value-based |
| ethical | Hate speech, harassment | Value-based |
| legal | Illegal activities (implicit in dangerous content) | Value-based |
| privacy | PII disclosure policy | Value-based |
| identity_boundary | Mistaken self-identification filtering | Identity honesty |
| capability_boundary | Not explicitly a safety policy | Capability acknowledgment |

### Key Observations

1. **Gemma 3's safety policies are primarily value-based**: 5 of 6 explicit policies correspond to our "value-based refusal" category. Identity and capability boundaries are handled differently (data filtering for identity, not explicitly trained for capability).

2. **No granular public taxonomy**: Unlike HarmBench's 7 semantic categories, Gemma 3's technical report describes 6 high-level policies without detailed sub-taxonomies. The actual alignment is with internal Google safety policies.

3. **ShieldGemma provides operational definitions**: The 4 ShieldGemma harm types (hate speech, dangerous content, harassment, sexually explicit) are the most precisely defined categories in Google's ecosystem. These are content-moderation categories, not the same as the model's refusal training categories.

4. **HarmBench categories map partially to our domains**:
   - Cybercrime & Unauthorized Intrusion -> safety/legal
   - Chemical & Biological Weapons/Drugs -> safety
   - Copyright Violations -> legal (but not in our scope)
   - Misinformation & Disinformation -> ethical (partially)
   - Harassment & Bullying -> ethical
   - Illegal Activities -> legal
   - General Harm -> varies

5. **Our identity_boundary and capability_boundary domains have NO equivalent in HarmBench or Gemma 3 safety policies**: This is a gap in existing taxonomies and a potential contribution of this work. HarmBench focuses entirely on malicious-use harms; identity honesty and capability acknowledgment are encoded along different refusal directions entirely.

6. **Over-refusal not explicitly addressed**: Neither the Gemma 3 report nor HarmBench focus on false refusal / over-refusal as a first-class concern (unlike XSTest which specifically targets this).

### Implications for SAE Analysis

- If Gemma 3's safety training focuses on value-based refusal (the 6 policies above), we should expect SAE features encoding these harm categories to be active on our safety/ethical/legal/privacy prompts.
- Identity boundary and capability boundary prompts may activate *different* SAE features if they are encoded along different directions (data filtering + truthfulness vs. trained refusal).
- The CBRN finding ("knowledge is low") suggests Gemma 3 may refuse some CBRN queries not through safety training but through genuine lack of knowledge -- blurring the line between capability and safety refusal.
