import {
  Section,
  H1,
  H2,
  P,
  Card,
  Label,
  Mono,
  Arrow,
  InlineLink,
} from "../components/Primitives.jsx";
import { palette, fonts } from "../tokens.js";

const REFS = [
  // ---------------------------------------------------------------------------
  // Group 1: Refusal directions and capping (no-SAE regime)
  // ---------------------------------------------------------------------------
  {
    group: "Refusal directions and activation capping",
    groupNote:
      "Foundational papers for the no-SAE regime. Refusal as a single direction; per-category extensions; activation capping as a steering primitive.",
    items: [
      {
        key: "arditi2024",
        authors: "Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N.",
        year: "2024",
        title: "Refusal in Language Models Is Mediated by a Single Direction",
        venue: "NeurIPS 2024",
        arxiv: "2406.11717",
        relevance:
          "Foundational. Refusal mediated by a one-dimensional subspace across 13 chat models (≤72B). We extend their contrastive direction extraction to per-domain axes and add capping.",
      },
      {
        key: "lu2026assistant",
        authors: "Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J.",
        year: "2026",
        title: "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models",
        arxiv: "2601.10387",
        relevance:
          "Direct methodological precedent. We follow their axis formula (Sec 3.1), capping intervention (Sec 4.2), and visual language. Our contribution is per-domain decomposition.",
      },
      {
        key: "joad2026",
        authors: "Joad, F., Hawasly, M., Boughorbel, S., Durrani, N., & Sencar, H. T.",
        year: "2026",
        title: "There Is More to Refusal in Large Language Models than a Single Direction",
        arxiv: "2602.02132",
        relevance:
          "Closest published challenge. Across 11 categories, refusal corresponds to geometrically distinct directions, but steering along any of them produces a 'shared one-dimensional control knob.' Our domain-selective capping result speaks directly to this.",
      },
      {
        key: "alagharu2026",
        authors: "Alagharu, et al.",
        year: "2026",
        title:
          "From Refusal Tokens to Refusal Control: Discovering and Steering Category-Specific Refusal Directions",
        arxiv: "2603.13359",
        relevance:
          "Closest published analogue. Extracts category-aligned directions in a Llama-3-8B fine-tuned with categorical refusal tokens, then steers per category via low-rank whitened combination. We achieve domain decomposition unsupervised, on Gemma 3 12B, with capping (not addition).",
      },
      {
        key: "zhao2025",
        authors: "Zhao, et al.",
        year: "2025",
        title: "LLMs Encode Harmfulness and Refusal Separately",
        arxiv: "2507.11878",
        venue: "NeurIPS 2025",
        relevance:
          "Identifies a harmfulness direction (encoded at the last user-token) distinct from the refusal direction (encoded at the last sequence-token), with causal evidence. Supports the premise that refusal is not monolithic; multiple decomposable subspaces exist.",
      },
      {
        key: "wollschlager2025",
        authors: "Wollschläger, T., Elstner, J., Geisler, S., Cohen-Addad, V., Günnemann, S., & Gasteiger, J.",
        year: "2025",
        title: "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence",
        arxiv: "2502.17420",
        venue: "ICML 2025",
        relevance:
          "Directly extends the rank-1 picture: refusal is mediated by multi-dimensional concept cones (up to dim 5 in tested models), with a property they call representational independence. Cited in Part I §1 background and §2 reading; supports our 11-dimensional subspace finding.",
      },
      {
        key: "coalson2026",
        authors: "Coalson, et al.",
        year: "2026",
        title: "Fail-Closed Alignment for Large Language Models",
        arxiv: "2602.16977",
        relevance:
          "Shows progressively ablating refusal directions induces multiple causally independent refusal subspaces. Both motivates and constrains domain-selective capping.",
      },
      {
        key: "maskey2026",
        authors: "Maskey, et al.",
        year: "2026",
        title:
          "Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs",
        arxiv: "2603.27518",
        relevance:
          "Distinguishes task-agnostic harmful-refusal directions from task-dependent over-refusal subspaces. Relevant to whether our 6 domains are mutually orthogonal or share a global axis.",
      },
      {
        key: "panickssery2024",
        authors: "Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M.",
        year: "2024",
        title: "Steering Llama 2 via Contrastive Activation Addition",
        arxiv: "2312.06681",
        relevance:
          "CAA. First author also known as Nina Rimsky. ~40% depth optimal for Llama 2; we cite this when justifying our layer 36 steering choice.",
      },
      {
        key: "chen2025persona",
        authors: "Chen, R., et al.",
        year: "2025",
        title: "Persona Vectors: Monitoring and Controlling Character Traits in Language Models",
        arxiv: "2507.21509",
        relevance:
          "Sec 2.2: trait scoring methodology and mean-response-token extraction we follow. Appendix M: decoder-cosine decomposition of 'evil' persona into SAE features; directly informs Method 2.",
      },
    ],
  },

  // ---------------------------------------------------------------------------
  // Group 2: SAE methodology (SAE regime)
  // ---------------------------------------------------------------------------
  {
    group: "Sparse autoencoders for LM interpretability",
    groupNote:
      "Foundational methodology for the SAE regime: JumpReLU SAEs, Gemma Scope, Matryoshka, feature absorption, hierarchy.",
    items: [
      {
        key: "rajamanoharan2024",
        authors: "Rajamanoharan, S., et al.",
        year: "2024",
        title: "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders",
        arxiv: "2407.14435",
        relevance:
          "JumpReLU SAE architecture used by Gemma Scope and Gemma Scope 2. Required when describing our SAE.",
      },
      {
        key: "lieberum2024",
        authors: "Lieberum, T., et al.",
        year: "2024",
        title: "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2",
        arxiv: "2408.05147",
        relevance:
          "Original Gemma Scope release. Lineage and design rationale that Gemma Scope 2 builds on.",
      },
      {
        key: "mcdougall2025gemmascope2",
        authors:
          "McDougall, C., Conmy, A., Kramár, J., Lieberum, T., Rajamanoharan, S., & Nanda, N.",
        year: "2025",
        title: "Gemma Scope 2",
        venue: "Google DeepMind Technical Report",
        relevance:
          "The SAE we use. Trains expanded-width SAEs at four layers for 12B (Table 1: {12, 24, 31, 41} at 25/50/65/85% depth). Layer 41 was our choice (the deepest available, not a paper recommendation).",
      },
      {
        key: "bussmann2025matryoshka",
        authors: "Bussmann, B., et al.",
        year: "2025",
        title: "Learning Multi-Level Features with Matryoshka Sparse Autoencoders",
        arxiv: "2503.17547",
        relevance:
          "Validates hierarchies via decoder geometry with explicit cosine thresholds. Matryoshka prefix-slicing methodology.",
      },
      {
        key: "chanin2024",
        authors: "Chanin, D., et al.",
        year: "2024",
        title: "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders",
        arxiv: "2409.14507",
        relevance:
          "Feature absorption framework. We use co-activation to distinguish absorption from splitting (Method 2).",
      },
      {
        key: "luo2026hsae",
        authors: "Luo, Y., Zhan, Y., Jiang, J., Liu, T., Wu, M., Zhou, Z., & Dong, B.",
        year: "2026",
        title: "From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders",
        arxiv: "2602.11881",
        relevance:
          "HSAE evaluation metric (Method 3). Parent decoder as weighted sum of children's decoders.",
      },
      {
        key: "wang2025sails",
        authors: "Wang, D., et al.",
        year: "2025",
        title:
          "Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation (SAILS)",
        arxiv: "2512.23260",
        relevance:
          "SAE-based contrastive identification avoids the irreducible error floor of direct subspace recovery (Stage 1 proof). Identifies refusal features at 16k width but does not examine larger widths.",
      },
      {
        key: "obrien2024",
        authors:
          "O'Brien, K., Majercak, D., Fernandes, X., Edgar, R., Bullwinkel, B., Chen, J., Nori, H., Carignan, D., Horvitz, E., & Poursabzi-Sangdeh, F.",
        year: "2024",
        title: "Steering Language Model Refusal with Sparse Autoencoders",
        arxiv: "2411.11296",
        relevance:
          "Empirical foil. SAE-based refusal steering causes systematic capability degradation on benign tasks. Any SAE-steering follow-up must address this.",
      },
      {
        key: "prakash2025",
        authors: "Prakash, et al.",
        year: "2025",
        title: "Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal",
        arxiv: "2509.09708",
        relevance:
          "SAE-feature search + factorisation-machine interaction analysis on Gemma 2 2B-IT and Llama 3.1 8B-IT. Direct methodological precedent for SAE-feature-based refusal dissection in the Gemma family.",
      },
      {
        key: "karvonen2025saebench",
        authors: "Karvonen, et al.",
        year: "2025",
        title:
          "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability",
        arxiv: "2503.09532",
        relevance:
          "Eight-metric SAE evaluation. Matryoshka SAEs have a feature-disentanglement advantage that grows with width; motivates our 65k→262k width transition.",
      },
      {
        key: "bricken2023monosemanticity",
        authors: "Bricken, T., et al.",
        year: "2023",
        title: "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning",
        venue: "Transformer Circuits, Anthropic",
        relevance:
          "Decoder cosine similarity as a hierarchy metric (Method 1). Note: not on arxiv; published at transformer-circuits.pub.",
      },
      {
        key: "templeton2024scaling",
        authors: "Templeton, A., et al.",
        year: "2024",
        title: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet",
        venue: "Transformer Circuits, Anthropic",
        relevance:
          "Scales SAE features to Claude 3 Sonnet. SAE feature clamping for steering. We follow their autointerp evaluation methodology.",
      },
      {
        key: "cunningham2023",
        authors: "Cunningham, H., et al.",
        year: "2023",
        title: "Sparse Autoencoders Find Highly Interpretable Features in Language Models",
        arxiv: "2309.08600",
        relevance:
          "First demonstration that SAEs recover interpretable, causally-relevant features from LM activations. Pairs with Bricken 2023 as canonical 'SAEs work on LMs' citation.",
      },
    ],
  },

  // ---------------------------------------------------------------------------
  // Group 3: Geometric/representational background
  // ---------------------------------------------------------------------------
  {
    group: "Background: concept geometry and representations",
    groupNote:
      "Theoretical underpinnings: superposition, linear representation hypothesis, concept hierarchy as polytopes.",
    items: [
      {
        key: "elhage2022",
        authors: "Elhage, N., et al.",
        year: "2022",
        title: "Toy Models of Superposition",
        arxiv: "2209.10652",
        relevance:
          "Origin of the superposition / polysemanticity story that motivates SAEs.",
      },
      {
        key: "park2024",
        authors: "Park, et al.",
        year: "2024",
        title: "The Geometry of Categorical and Hierarchical Concepts in Large Language Models",
        arxiv: "2406.01506",
        relevance:
          "Formalises categorical concepts as polytopes; relates concept hierarchy to representation geometry. Validated on Gemma and LLaMA-3. Theoretical underpinning for hierarchy analysis.",
      },
      {
        key: "zou2023repe",
        authors: "Zou, A., et al.",
        year: "2023",
        title: "Representation Engineering: A Top-Down Approach to AI Transparency",
        arxiv: "2310.01405",
        relevance:
          "Foundational RepE paper. Population-level activation interventions for honesty, harmlessness, power-seeking. Establishes the broader paradigm in which our work sits.",
      },
      {
        key: "bills2023neurons",
        authors:
          "Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J., & Saunders, W.",
        year: "2023",
        title: "Language models can explain neurons in language models",
        venue: "OpenAI",
        relevance:
          "Bills → Bricken → Templeton autointerp pipeline. Used for feature judging in our M6.",
      },
    ],
  },

  // ---------------------------------------------------------------------------
  // Group 4: Models
  // ---------------------------------------------------------------------------
  {
    group: "Models",
    groupNote: "The model under analysis.",
    items: [
      {
        key: "gemma3team2025",
        authors: "Gemma Team, Kamath, A., et al.",
        year: "2025",
        title: "Gemma 3 Technical Report",
        arxiv: "2503.19786",
        relevance:
          "Gemma 3 12B-IT: 48 layers, d_model = 3840, multimodal architecture (nested layer path).",
      },
    ],
  },
];

function RefRow({ item }) {
  const url = item.arxiv ? `https://arxiv.org/abs/${item.arxiv}` : null;
  const Tag = url ? "a" : "div";
  return (
    <Tag
      {...(url
        ? {
            className: "link-a hover-card",
            href: url,
            target: "_blank",
            rel: "noopener noreferrer",
            style: {
              textDecoration: "none",
              color: "inherit",
              display: "block",
              padding: "20px 24px",
              borderRadius: "10px",
              border: `1px solid ${palette.border}`,
              background: palette.surface,
            },
          }
        : {
            style: {
              padding: "20px 24px",
              borderRadius: "10px",
              border: `1px solid ${palette.border}`,
              background: palette.surface,
            },
          })}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "16px", marginBottom: "8px" }}>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: "11px",
            color: palette.muted,
            letterSpacing: "0.04em",
          }}
        >
          {item.authors}{" "}
          <span style={{ color: palette.text, fontWeight: 500 }}>· {item.year}</span>
        </div>
        {item.arxiv && (
          <div style={{ fontFamily: fonts.mono, fontSize: "10.5px", color: palette.muted, whiteSpace: "nowrap" }}>
            arXiv:{item.arxiv} <Arrow size={9} color={palette.orange} />
          </div>
        )}
      </div>
      <div
        style={{
          fontFamily: fonts.display,
          fontSize: "16px",
          fontWeight: 600,
          color: palette.text,
          letterSpacing: "-0.01em",
          marginBottom: "6px",
          lineHeight: 1.4,
        }}
      >
        <em style={{ fontStyle: "normal" }}>{item.title}</em>
      </div>
      {item.venue && (
        <Mono style={{ fontSize: "11px", color: palette.muted }}>{item.venue}</Mono>
      )}
      <P style={{ fontSize: "14.5px", marginTop: "10px", marginBottom: 0, color: palette.body }}>
        {item.relevance}
      </P>
    </Tag>
  );
}

export function ReferencesPage() {
  return (
    <>
      <Section style={{ paddingTop: "64px" }}>
        <Label>References</Label>
        <H1>Bibliography</H1>
        <P lead>
          Grouped by what each paper supports in the project. Each entry includes the arXiv ID
          where available and a one-paragraph note on relevance to this work.
        </P>
        <Mono>
          Full BibTeX:{" "}
          <InlineLink
            href="https://github.com/Antonio-Tresol/gemma3-refusal-axis/blob/dev/references/from-safety-prompts-project/relevant_references.bib"
            style={{ color: palette.text }}
          >
            references/from-safety-prompts-project/relevant_references.bib
          </InlineLink>
        </Mono>
      </Section>

      {REFS.map((group) => (
        <Section key={group.group} style={{ marginTop: "56px" }}>
          <H2 style={{ marginTop: 0 }}>{group.group}</H2>
          <P style={{ color: palette.muted, fontStyle: "italic" }}>{group.groupNote}</P>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginTop: "24px" }}>
            {group.items.map((item) => (
              <RefRow key={item.key} item={item} />
            ))}
          </div>
        </Section>
      ))}

      <Section style={{ marginTop: "64px" }}>
        <Mono>
          <InlineLink href="#/methodology" arrow="left" style={{ color: palette.text }}>
            Back to Methodology
          </InlineLink>
          {"   |   "}
          <InlineLink href="#/" arrow="left" style={{ color: palette.text }}>
            Back to overview
          </InlineLink>
        </Mono>
      </Section>
    </>
  );
}
