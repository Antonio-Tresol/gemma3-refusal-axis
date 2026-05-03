// Methodology entries for the explainer. Each entry maps a procedural term to
// the section of the Methodology page that describes exactly what we did.
// These complement the Glossary (which explains what a word means); here the
// focus is "this is exactly what we did and where to read the detailed write-up."
//
// Shape: { slug: string, term: string, short: string, anchor: string }
// - slug      used in <MethodTerm name="slug"> and as the tooltip key
// - term      display title in the popover header
// - short     one sentence summarising the procedural choice (British English)
// - anchor    the id of the target H2/H3 in Methodology.jsx (no leading #)

export const methodologyEntries = [
  {
    slug: "mean-response-token",
    term: "Mean-response-token site",
    short:
      "We extract activations by averaging across all assistant response tokens at layer 41, following Chen et al. (2025) Persona Vectors.",
    anchor: "regime-raw",
  },
  {
    slug: "last-prompt-token",
    term: "Last-prompt-token site",
    short:
      "The activation of the final prompt token at layer 41; captures model intent before any response token is generated, following Arditi et al. (2024).",
    anchor: "regime-raw",
  },
  {
    slug: "trait-scoring",
    term: "Trait scoring",
    short:
      "Each Gemma 3 response is scored 0-100 for refusal expression by Claude Sonnet 4.6 across three independent passes; the final score is the median (inter-pass Pearson r = 0.990-0.991).",
    anchor: "dataset",
  },
  {
    slug: "pair-construction",
    term: "Contrastive pair construction",
    short:
      "Each pair is one positive prompt (should trigger refusal) matched with one negative prompt (same domain, benign); pairs are retained only when positive score > 50 and negative score < 30.",
    anchor: "dataset",
  },
  {
    slug: "layer-36",
    term: "Layer 36 (steering layer)",
    short:
      "Activation capping is applied at layer 36 (75% depth), distinct from the layer 41 extraction site; M5 steering experiments found this optimal for Gemma 3 12B.",
    anchor: "regime-raw",
  },
  {
    slug: "bootstrap",
    term: "Bootstrap confidence intervals",
    short:
      "2000 within-domain resamples with replacement; domain directions and pairwise cosines are recomputed each time and the 2.5/97.5 percentiles reported.",
    anchor: "statistics",
  },
  {
    slug: "permutation-null",
    term: "Permutation null",
    short:
      "1000 random shuffles of domain labels (or feature activation vectors for Jaccard) generate a null distribution; the p-value is the fraction of shuffles matching or exceeding the observed statistic.",
    anchor: "statistics",
  },
  {
    slug: "leave-one-out",
    term: "Leave-one-out stability",
    short:
      "Each prompt pair is dropped in turn and the domain direction recomputed; the cosine of the LOO direction with the full-data direction gives the LOO range.",
    anchor: "statistics",
  },
  {
    slug: "decoder-cosine-method",
    term: "Decoder cosine (M1)",
    short:
      "We measure cosine similarity between the decoder vectors of a candidate parent feature (16k width) and each child feature (65k width) to test geometric inheritance.",
    anchor: "regime-sae",
  },
  {
    slug: "coactivation-jaccard",
    term: "Co-activation Jaccard (M2)",
    short:
      "We compute Jaccard index between the binary firing sets of candidate parent and child SAE features across 104 prompts, then validate against a permutation null and base-rate analysis.",
    anchor: "regime-sae",
  },
  {
    slug: "r2-decomposition",
    term: "Hierarchy R² (M3)",
    short:
      "We fit a least-squares regression of each parent feature’s activations onto those of its candidate children; R² > 0.3 would indicate a parent reconstructible as a weighted sum of children.",
    anchor: "regime-sae",
  },
];

export const methodologyBySlug = Object.fromEntries(
  methodologyEntries.map((e) => [e.slug, e])
);
