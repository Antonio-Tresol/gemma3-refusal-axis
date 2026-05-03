// Glossary entries for the explainer. Each entry is keyed by a slug used in
// `<Term name="slug">` and as the in-page anchor on `/glossary`. Definitions are
// plain-English first; sources cite the most relevant published work for a reader
// who wants to drill in. Sources are stored as compact citation labels; the
// Glossary page renders them as arxiv/HF links where applicable.

export const glossaryEntries = [
  {
    slug: "refusal",
    term: "Refusal",
    short:
      "When a language model says \"I can't help with that\"; a class of trained behaviours that decline to comply with a prompt.",
    long: "Refusal is a learned behaviour shaped by instruction tuning, RLHF, and safety training. Arditi et al. (2024) showed that, in 13 open-source chat models, refusal is mediated by a single direction in the residual stream and can be suppressed by ablating that direction. This work and several recent papers argue that refusal is not unitary: it spans value-based declines (\"I won't help you make a weapon\"), capability acknowledgements (\"I can't run code\"), and identity statements (\"I'm not conscious\"). Wollschläger et al. (2025, ICML) extend Arditi to a multi-dimensional concept cone, and Zhao et al. (2025, NeurIPS) further separate a harmfulness-detection direction from a refusal-emission direction.",
    sources: ["arditi2024refusal", "wollschlager2025cones", "zhao2025harmfulness", "joad2026refusal"],
  },
  {
    slug: "residual-stream",
    term: "Residual stream",
    short:
      "The vector that flows from the bottom to the top of a transformer, accumulating contributions from every attention and MLP layer.",
    long: "The residual stream is the per-token vector that every transformer layer reads from and writes to in place; the additive structure comes from the residual connection of Vaswani et al. (2017). Elhage et al. (2021, \"A Mathematical Framework for Transformer Circuits\") sharpened the framing: the residual stream is a high-bandwidth communication channel that successive attention heads and MLPs use to pass information forward, layer after layer. All of our geometric work happens on this vector, taken at layer 41 of Gemma 3 12B (48 layers total).",
    sources: ["elhage2021framework", "vaswani2017attention", "gemma3team2025"],
  },
  {
    slug: "layer",
    term: "Layer (transformer block)",
    short:
      "One unit of computation in a transformer. Each layer has an attention sub-layer and an MLP sub-layer.",
    long: "A transformer layer (sometimes called a transformer block) is the unit of computation introduced by Vaswani et al. (2017): an attention sub-layer followed by a feed-forward (MLP) sub-layer, each wrapped in a residual connection and layer norm. Gemma 3 12B stacks 48 such layers, so \"layer 41\" refers to depth index 41 out of 48, near the top of the stack. Empirically, earlier layers tend to track surface features and later layers carry decision-relevant abstractions like \"is this a refusal context.\" We extract activations at layer 41 and apply steering interventions slightly below, at layer 36.",
    sources: ["vaswani2017attention", "gemma3team2025"],
  },
  {
    slug: "activation",
    term: "Activation",
    short:
      "The numerical vector that represents a token at a given point inside the network.",
    long: "In this work, an activation is the residual-stream vector at one token position at one layer; for Gemma 3 12B that vector is 3,840-dimensional. The term is overloaded: in classical neural-network usage it means the output of a non-linearity, but in mechanistic interpretability (following Elhage et al. 2021) it almost always refers to the per-token, per-layer residual-stream vector. Treating those vectors as data (averaging, projecting, comparing, and editing them mid-forward-pass) is the basic move of representation engineering (Zou et al. 2023) and contrastive activation addition (Panickssery et al. 2024).",
    sources: ["elhage2021framework", "zou2023repe", "panickssery2024caa"],
  },
  {
    slug: "direction",
    term: "Direction (activation axis)",
    short:
      "A unit vector in activation space that, by hypothesis, points along some abstract concept the model uses.",
    long: "A direction is a unit vector in activation space along which a behaviour or attribute is hypothesised to be encoded; projecting an activation onto it yields a scalar score for how strongly that attribute is present. The view that human-interpretable concepts are encoded along specific directions is the linear representation hypothesis, formalised by Park et al. (2024). Most directions in this work are computed by subtracting the mean activation of one condition from the mean of another, the contrastive recipe used in CAA (Panickssery et al. 2024) and adapted to refusal by Arditi et al. (2024).",
    sources: ["park2024geometry", "arditi2024refusal", "panickssery2024caa"],
  },
  {
    slug: "refusal-axis",
    term: "Refusal axis",
    short:
      "A direction in activation space whose magnitude tracks how much the model is about to refuse.",
    long: "A direction in activation space whose magnitude tracks how strongly the model is moving towards a refusal. We compute it as mean(refusing-activations) − mean(complying-activations) over matched prompt pairs, the standard contrastive recipe from Arditi et al. (2024, NeurIPS), who showed a single such direction mediates refusal across 13 chat models. Wollschläger et al. (2025, ICML) extend the picture by identifying multiple representationally-independent refusal directions (up to 5 ablated jointly in Gemma 2 2B, Llama 3 8B, and Qwen 2.5 3B/7B) that each suppress refusal on top of one another, and Joad et al. (2026) find geometrically distinct per-category directions that nonetheless act as a shared one-dimensional control knob. This work tests whether refusal really is one direction or several across six refusal domains.",
    sources: ["arditi2024refusal", "wollschlager2025cones", "joad2026refusal"],
  },
  {
    slug: "concept-cone",
    term: "Concept cone",
    short:
      "A multi-dimensional region of activation space where any vector inside the region triggers a behaviour like refusal.",
    long: "A region of activation space, geometrically a polyhedral cone, in which any direction within the region triggers a target behaviour. Wollschläger et al. (2025, ICML) introduce the construction for refusal: using gradient-based representation engineering (their Refusal Direction Optimisation), they find that refusal in instruction-tuned LLMs is mediated by multi-dimensional cones rather than a single direction. Their reported operating cone for Gemma 2 2B is four-dimensional, performance degrades at larger dims (Figure 17), and the cone construction generalises to the Qwen 2.5 family (Figure 18). They separately define representational independence (Definition 6) as a stricter criterion than orthogonality for two directions to remain mutually unaffected under intervention, and identify up to five jointly representationally-independent refusal directions in their tested models. The 11-dimensional subspace found here for Gemma 3 12B is consistent with the multi-dimensional picture but does not test the cone-shape property directly.",
    sources: ["wollschlager2025cones", "arditi2024refusal"],
  },
  {
    slug: "harmfulness",
    term: "Harmfulness vs refusal",
    short:
      "Two related but distinguishable directions: detecting that a request is harmful, and producing a refusal in the response.",
    long: "Zhao et al. (2025, NeurIPS) decompose what looked like one refusal direction into two: a harmfulness direction encoded at the last token of the user instruction (their t_inst, where the model decides whether the request is harmful), and a refusal direction encoded at the last token of the post-instruction tokens (their t_post-inst, after the chat-template close, where the model commits to the refusal). The two are correlated but causally distinct: steering along harmfulness can flip the model's internal judgment of whether a request is harmful, whereas steering along refusal directly elicits refusal text without reversing that judgment. They also show that certain jailbreaks suppress refusal without changing the model's internal belief about harmfulness. The site uses 'refusal' as the umbrella term and points to Zhao et al. for the finer decomposition.",
    sources: ["zhao2025harmfulness", "arditi2024refusal"],
  },
  {
    slug: "projection",
    term: "Projection",
    short:
      "The scalar you get when you take the dot product of an activation with a unit direction.",
    long: "If v̂ is a unit-norm direction and a is an activation vector, then projⱽ(a) = v̂ · a is a scalar saying how much of a lies along v̂. In representation engineering (Zou et al. 2023) the projection is the readout: it measures how strongly the model encodes a property along that direction. Lu et al. (2026) use projection onto the Assistant Axis to track persona drift across turns, and capping operates directly on this scalar (their Section 5).",
    sources: ["zou2023repe", "lu2026assistant"],
  },
  {
    slug: "cosine-similarity",
    term: "Cosine similarity",
    short:
      "The cosine of the angle between two vectors. 1 = identical direction, 0 = orthogonal, −1 = opposite.",
    long: "Cosine similarity is the dot product of two unit vectors, equal to the cosine of the angle between them, with values in [−1, 1]. We use it to compare refusal-related directions: cosine 0.87 between two domain axes says they point almost the same way; cosine near 0 says they are geometrically orthogonal. Wollschläger et al. (2025) caution that geometric orthogonality alone does not establish causal independence, so cosines reported here are read alongside intervention tests, not as standalone proof of distinct directions.",
    sources: ["wollschlager2025cones"],
  },
  {
    slug: "pca",
    term: "PCA (Principal Component Analysis)",
    short:
      "A way to find the directions of largest variance in a set of vectors.",
    long: "Given a set of activation vectors, principal component analysis returns an ordered list of orthogonal directions where the first captures the most variance, the second the most of what is left, and so on. The count of components needed to reach a given variance threshold (typically 70%) measures how high-dimensional the represented concept is. We follow the procedure Lu et al. (2026) used to map persona space (their Section 2.1.3, where 4–19 components reach 70% across three frontier models) and apply it to refusal-related activations on Gemma 3 12B; 11 components reach 70%, against a median of 80 for random vectors in the same space.",
    sources: ["lu2026assistant"],
  },
  {
    slug: "bootstrap-ci",
    term: "Bootstrap confidence interval",
    short:
      "A confidence interval computed by repeatedly resampling the data with replacement and recomputing the statistic.",
    long: "The bootstrap (Efron 1979, \"Bootstrap Methods: Another Look at the Jackknife\") is a non-parametric procedure for estimating the sampling distribution of a statistic by repeatedly resampling the observed data with replacement. To gauge how stable a per-domain cosine of 0.87 is, we resample the matched prompt pairs within that domain 2,000 times with replacement, recompute the cosine each time, and report the 2.5th and 97.5th percentiles of the resulting distribution as a 95% percentile-bootstrap confidence interval. A wide interval such as [−0.32, 0.64] flags that the point estimate is fragile; the procedure makes no normality assumption.",
    sources: ["efron1979bootstrap"],
  },
  {
    slug: "permutation-test",
    term: "Permutation test",
    short:
      "A null-hypothesis test that shuffles labels many times to see how often you'd see a result this extreme by chance.",
    long: "A permutation test (Fisher 1935, The Design of Experiments) computes a p-value by repeatedly shuffling the labels or group assignments under the null hypothesis of exchangeability and counting how often the shuffled statistic is at least as extreme as the observed one. This work uses one-sided permutation tests in two places: the cosine-range null (relabel prompts into fake \"domains\" 1,000 times and recompute the spread of per-domain cosines) and the dimensionality null (compare the 70%-variance dimension of refusal differences against random vectors in the same ℝ³⁸⁴⁰ space). p = 0.054 is borderline; p < 0.001 indicates structure beyond chance.",
    sources: ["fisher1935design"],
  },
  {
    slug: "sae",
    term: "Sparse autoencoder (SAE)",
    short:
      "A separately-trained network that re-expresses one model activation as a small number of human-interpretable \"features.\"",
    long: "A model activation is a dense 3,840-dim vector that mixes many concepts at once (superposition). A sparse autoencoder learns a wider over-complete dictionary by training an encoder that maps the activation to a much higher-dimensional, mostly-zero feature vector, and a decoder that reconstructs the original activation from those features, with a reconstruction loss plus a sparsity penalty (Cunningham et al. 2023; Bricken et al. 2023). Each non-zero feature ideally corresponds to one human-readable concept (\"DNA sequence\", \"refusal context\", \"Python code\"). This work uses the Gemma Scope 2 JumpReLU SAEs released by Google DeepMind (Lieberum et al. 2024 for Gemma Scope; McDougall et al. 2025 for Gemma Scope 2).",
    sources: ["cunningham2023sparse", "bricken2023monosemanticity", "lieberum2024gemmascope", "mcdougall2025gemmascope2"],
  },
  {
    slug: "jumprelu",
    term: "JumpReLU",
    short:
      "A sparse activation function used inside Gemma Scope SAEs that combines a learned threshold with a step gate.",
    long: "JumpReLU is the discontinuous gating function used inside Gemma Scope SAEs (Rajamanoharan et al. 2024, \"Jumping Ahead\"). Each feature has a learned threshold; the feature output is the pre-activation when it exceeds the threshold and exactly zero otherwise. The discontinuity allows the L0 sparsity penalty to be optimised directly via straight-through estimators rather than through an L1 proxy, which Rajamanoharan et al. report as a better sparsity-fidelity trade-off than ReLU or TopK SAEs at matched L0. Gemma Scope and Gemma Scope 2 both use JumpReLU.",
    sources: ["rajamanoharan2024jumprelu", "lieberum2024gemmascope", "mcdougall2025gemmascope2"],
  },
  {
    slug: "matryoshka",
    term: "Matryoshka SAE",
    short:
      "A training trick that lets one wide SAE be sliced into several narrower SAEs nested inside it, like Russian dolls.",
    long: "Bussmann et al. (2025, \"Learning Multi-Level Features with Matryoshka Sparse Autoencoders\") train a single wide SAE under a sum of reconstruction losses, one per nested prefix: the first 16k features alone must reconstruct well, the first 65k must also reconstruct well, the first 262k as well, and so on up to the full 1M. The result is a single 1M-feature SAE that can be sliced down to any of those prefix widths without retraining. This work uses the Gemma Scope 2 1M Matryoshka SAE at layer 41 (McDougall et al. 2025) to test whether prefix-slicing recovers a parent-child hierarchy in which wider-width features refine coarse ones.",
    sources: ["bussmann2025matryoshka", "mcdougall2025gemmascope2", "luo2026hsae"],
  },
  {
    slug: "decoder",
    term: "Decoder vector",
    short:
      "The column of the SAE's decoder matrix corresponding to one feature; the direction that feature \"writes\" into the residual stream when it fires.",
    long: "An SAE has a decoder matrix W_dec whose columns are unit-norm vectors in residual-stream space, one per dictionary feature; the reconstructed activation is the sum of those columns weighted by the (mostly zero) feature activations (Bricken et al. 2023, \"Towards Monosemanticity\"). The decoder vector is therefore the geometric identity of a feature: cosine similarity between two decoder vectors tells you whether the features point the same way in activation space. We use decoder cosine to test whether 65k features inherit from 16k parents under Matryoshka prefix-slicing (Bussmann et al. 2025).",
    sources: ["bricken2023monosemanticity", "bussmann2025matryoshka", "chanin2024absorption"],
  },
  {
    slug: "feature",
    term: "Feature (SAE feature)",
    short:
      "One slot in the SAE's output; by hypothesis, a unit corresponding to one human-interpretable concept.",
    long: "\"Feature\" is overloaded. In the broad mechanistic-interpretability sense, a feature is a direction in activation space that the network uses to encode a property of its input (Elhage et al. 2022, \"Toy Models of Superposition\"). In the SAE sense, used throughout this work, a feature is one slot of the learned dictionary: a column of the decoder, paired with an encoder row that detects when to fire it (Bricken et al. 2023). Some SAE features really do track clean concepts (Bible verses, base64, refusal contexts); others fire on broad statistical co-occurrence patterns, and SAEBench (Karvonen et al. 2025) is one benchmark that measures how monosemantic they actually are.",
    sources: ["elhage2022toy", "bricken2023monosemanticity", "karvonen2025saebench", "cunningham2023sparse"],
  },
  {
    slug: "jaccard",
    term: "Jaccard similarity",
    short:
      "Size of the intersection of two sets divided by size of the union. 1 = identical sets, 0 = disjoint.",
    long: "Jaccard similarity (Jaccard 1901) is the size of the intersection of two sets divided by the size of their union, |A ∩ B| / |A ∪ B|; 1 means identical sets, 0 means disjoint. We apply it to the sets of prompts on which two SAE features fire: if a 16k parent feature and a 65k child feature fire on exactly the same 47 prompts, their co-activation Jaccard is 1.0 and we suspect they encode the same concept. Co-activation Jaccard is one of three independent hierarchy tests in this work; the related literature on feature splitting and absorption (Chanin et al. 2024) uses different metrics, so the choice of Jaccard for our M2 is the project's own.",
    sources: [],
  },
  {
    slug: "r-squared",
    term: "R² (coefficient of determination)",
    short:
      "How much of the variance in one signal is explained by another. 1 = perfect, 0 = none.",
    long: "R², the coefficient of determination, is the standard regression statistic 1 − SS_res / SS_tot: it reports the fraction of variance in a target signal explained by a linear fit, with 1 meaning a perfect fit and 0 meaning no better than the mean. We use it as the third hierarchy test (Method 3 in this work): for each candidate 16k parent feature we regress its activation vector across prompts onto the activations of its candidate 65k children, asking whether the parent decodes as a weighted sum of children. The R² metric is the project's own design choice; related work on SAE hierarchy (Luo et al. 2026, \"From Atoms to Trees\") works on the same question with different metrics (parent-child co-activation probability and a structural-constraint reconstruction loss). One feature reaches R² = 0.580 here, but parent and child both fire on 97% of prompts, so the regression is fitting the shared firing rate, not a hierarchy.",
    sources: ["luo2026hsae"],
  },
  {
    slug: "capping",
    term: "Activation capping",
    short:
      "A steering intervention: subtract any projection of the activation onto a target direction that exceeds a threshold τ, then leave the rest alone.",
    long: "Activation capping is an intervention introduced by Lu et al. (2026, Section 5, Eq. 1): h ← h − v · min(⟨h, v⟩ − τ, 0), which clamps the projection of an activation onto a chosen direction v to be at least τ. Their footnote 3 notes the symmetric maximum-cap variant, h ← h − max(⟨h, v⟩ − τ, 0) · v, which clamps projection to be at most τ; that is the form used here, since we want to suppress refusal projections that sit above a benign-prompt percentile rather than enforce a floor. Capping is gentler than directional ablation, which zeroes the projection at every layer. We apply it at layer 36 to each candidate refusal direction and measure which ones produce selective behaviour change.",
    sources: ["lu2026assistant", "arditi2024refusal"],
  },
  {
    slug: "tau-threshold",
    term: "τ-threshold",
    short:
      "The percentile of benign-prompt projections used as the cap point for activation capping.",
    long: "τ is the projection threshold used by activation capping (Lu et al. 2026): the cap point above (or below) which the intervention starts to bite. Lu et al. set τ by computing percentiles of projections on a calibration set of model responses (their Section 5.1.1, where the 25th percentile gave the best safety-capability trade-off). We adopt their calibration procedure and sweep τ across {p10, p25, p50, p75, p90, p95, p99} of benign-prompt projections. The p₅₀ setting produced the cleanest safety-selective effect on Gemma 3 12B, but the sweep is exploratory, not pre-registered.",
    sources: ["lu2026assistant"],
  },
  {
    slug: "falsification",
    term: "Falsification",
    short:
      "A pre-registered test that, if it failed, would force you to retract or weaken a specific claim.",
    long: "Following Popper (1959, The Logic of Scientific Discovery), a claim is scientific only insofar as it forbids some observable outcome and so could in principle be shown false; mere consistency with data is not evidence. Before running the analyses on the held-out evaluation set, we wrote down ten tests in advance, each paired with the headline claim it would refute (a bootstrap CI that crossed sign, a permutation null with p > 0.05, a leave-one-out run that destabilised a cosine, and so on). The Falsification page reports which claims survived, which were weakened in writing, and which rest on evidence too small to confirm without replication.",
    sources: ["popper1959falsification"],
  },
  {
    slug: "subspace",
    term: "Subspace",
    short:
      "A linear region of activation space, characterised by its dimension; e.g. a single direction is a 1-dimensional subspace, and the 11-dim subspace this work finds is a subspace of dimension 11.",
    long: "A subspace of activation space is the span of one or more direction vectors: a flat region through the origin, characterised by its dimension k. The project's working hypothesis (after Park et al. 2024) is that interpretable concepts are encoded in low-dimensional subspaces. Arditi et al. (2024) found refusal in a 1-dimensional subspace (a single \"refusal direction\"). Wollschläger et al. (2025, ICML) extended that to multi-dimensional concept cones up to dim 5. This work finds an 11-dimensional subspace captures 70% of refusal-related activation-difference variance for Gemma 3 12B at layer 41 (versus a median of 80 for random vectors in the same ℝ³⁸⁴⁰ space). \"Subspace dimension\" and \"rank\" of the corresponding difference matrix refer to the same number.",
    sources: ["park2024geometry", "arditi2024refusal", "wollschlager2025cones"],
  },
  {
    slug: "geometry",
    term: "Geometry (of activation space)",
    short:
      "The spatial structure of activation vectors in residual-stream space: angles between directions, distances, dimensionality of the subspace they occupy.",
    long: "Activations in a transformer are vectors in d_model-dimensional space (3,840 for Gemma 3 12B). \"Geometry\" is the standard term in interpretability for the spatial structure of those vectors, captured by cosine similarity (angles), dot products (projections), PCA (variance directions), and subspace dimension. The framing comes from Park et al. (2024), who define a \"concept geometry\" under the linear representation hypothesis: human-interpretable concepts correspond to specific directions, so the residual stream's geometry carries semantic content. The reason such low-dimensional structure can exist inside a 3,840-dim space at all is superposition (Elhage et al. 2022). \"Decoder geometry\" specifically refers to the angular relationships between SAE decoder vectors, used here to test the parent-child hierarchy hypothesis. Caveat: not every analysis on this site is geometric. Co-activation Jaccard is set-theoretic; R² decomposition is statistical. \"Geometry\" applies to the cosine, projection, capping, and PCA results; it does not extend to those other tests.",
    sources: ["park2024geometry", "elhage2022toy", "lu2026assistant"],
  },
];

// Map of bibtex key -> {label, url} used to render source pills inline.
export const sources = {
  lu2026assistant: {
    label: "Lu et al. (2026)",
    url: "https://arxiv.org/abs/2601.10387",
  },
  arditi2024refusal: {
    label: "Arditi et al. (2024)",
    url: "https://arxiv.org/abs/2406.11717",
  },
  chen2025persona: {
    label: "Chen et al. (2025)",
    url: "https://arxiv.org/abs/2507.21509",
  },
  joad2026refusal: {
    label: "Joad et al. (2026)",
    url: "https://arxiv.org/abs/2602.02132",
  },
  bricken2023monosemanticity: {
    label: "Bricken et al. (2023)",
    url: "https://transformer-circuits.pub/2023/monosemantic-features/index.html",
  },
  bussmann2025matryoshka: {
    label: "Bussmann et al. (2025)",
    url: "https://arxiv.org/abs/2503.17547",
  },
  chanin2024absorption: {
    label: "Chanin et al. (2024)",
    url: "https://arxiv.org/abs/2409.14507",
  },
  luo2026hsae: {
    label: "Luo et al. (2026)",
    url: "https://arxiv.org/abs/2602.11881",
  },
  panickssery2024caa: {
    label: "Panickssery et al. (2024)",
    url: "https://arxiv.org/abs/2312.06681",
  },
  templeton2024scaling: {
    label: "Templeton et al. (2024)",
    url: "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html",
  },
  gemma3team2025: {
    label: "Gemma 3 Tech Report",
    url: "https://arxiv.org/abs/2503.19786",
  },
  mcdougall2025gemmascope2: {
    label: "Gemma Scope 2",
    url: "https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf",
  },
  alagharu2026categorical: {
    label: "Alagharu et al. (2026)",
    url: "https://arxiv.org/abs/2603.13359",
  },
  obrien2024steering: {
    label: "O'Brien et al. (2024)",
    url: "https://arxiv.org/abs/2411.11296",
  },
  rajamanoharan2024jumprelu: {
    label: "Rajamanoharan et al. (2024)",
    url: "https://arxiv.org/abs/2407.14435",
  },
  lieberum2024gemmascope: {
    label: "Lieberum et al. (2024)",
    url: "https://arxiv.org/abs/2408.05147",
  },
  karvonen2025saebench: {
    label: "Karvonen et al. (2025)",
    url: "https://arxiv.org/abs/2503.09532",
  },
  cunningham2023sparse: {
    label: "Cunningham et al. (2023)",
    url: "https://arxiv.org/abs/2309.08600",
  },
  zou2023repe: {
    label: "Zou et al. (2023)",
    url: "https://arxiv.org/abs/2310.01405",
  },
  park2024geometry: {
    label: "Park et al. (2024)",
    url: "https://arxiv.org/abs/2406.01506",
  },
  wollschlager2025cones: {
    label: "Wollschläger et al. (2025, ICML)",
    url: "https://arxiv.org/abs/2502.17420",
  },
  zhao2025harmfulness: {
    label: "Zhao et al. (2025, NeurIPS)",
    url: "https://arxiv.org/abs/2507.11878",
  },
  elhage2021framework: {
    label: "Elhage et al. (2021)",
    url: "https://transformer-circuits.pub/2021/framework/index.html",
  },
  vaswani2017attention: {
    label: "Vaswani et al. (2017)",
    url: "https://arxiv.org/abs/1706.03762",
  },
  elhage2022toy: {
    label: "Elhage et al. (2022)",
    url: "https://transformer-circuits.pub/2022/toy_model/index.html",
  },
  efron1979bootstrap: {
    label: "Efron (1979)",
    url: "https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full",
  },
  fisher1935design: {
    label: "Fisher (1935)",
    url: "https://en.wikipedia.org/wiki/The_Design_of_Experiments",
  },
  popper1959falsification: {
    label: "Popper (1959)",
    url: "https://en.wikipedia.org/wiki/The_Logic_of_Scientific_Discovery",
  },
};

export const glossaryBySlug = Object.fromEntries(
  glossaryEntries.map((e) => [e.slug, e])
);
