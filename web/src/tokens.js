// Design tokens for the refusal-axis explainer.
// Palette borrows the Far AI / Anthropic-inspired neutral system from the
// agentic-se-course site (gray surfaces, near-black text, single clay-orange
// accent), with two domain colours added for the refusal-axis red->blue scale.

export const palette = {
  bg: "#F4F4F4", // page background
  surface: "#FFFFFF", // card surfaces
  surfaceAlt: "#FAFAFA", // hover background, code blocks
  border: "#E5E5E5", // card borders, dividers
  text: "#0A0A0A", // headings + emphasised body
  body: "#262626", // paragraph text
  muted: "#737373", // labels, meta, mono captions
  stepNumber: "#525252", // step indices in lists
  orange: "#CC785C", // Anthropic clay-orange, used for arrows + accents
  // Refusal-axis-specific (Lu et al. RedBlue diverging scale):
  refuse: "#1d3557", // navy; strong refusal end of the axis
  comply: "#e63946", // red; compliance / "no refusal" end
  highlight: "#457b9d", // muted teal-blue; secondary accent
  // Domain colours (Okabe-Ito, colourblind-safe):
  domainSafety: "#D55E00",
  domainEthical: "#CC79A7",
  domainLegal: "#E69F00",
  domainPrivacy: "#0072B2",
  domainIdentity: "#009E73",
  domainCapability: "#56B4E9",
};

export const fonts = {
  display: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  body: "'Source Serif 4', Georgia, 'Times New Roman', serif",
  mono: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
};

export const layout = {
  readingMaxWidth: "820px",
  wideMaxWidth: "1080px",
  pagePadding: "0 28px",
};

// CSS custom properties applied at the root so they're available in <style> blocks.
export const cssVars = {
  "--bg": palette.bg,
  "--surface": palette.surface,
  "--surface-alt": palette.surfaceAlt,
  "--border": palette.border,
  "--text": palette.text,
  "--body": palette.body,
  "--muted": palette.muted,
  "--orange": palette.orange,
  "--refuse": palette.refuse,
  "--comply": palette.comply,
  "--highlight": palette.highlight,
  "--display": fonts.display,
  "--body-font": fonts.body,
  "--mono": fonts.mono,
};

// Domains in canonical display order (highest cosine with mean axis -> lowest).
export const domains = [
  { id: "safety", label: "Safety", colour: palette.domainSafety, n: 31, loading: 0.91 },
  { id: "ethical", label: "Ethical", colour: palette.domainEthical, n: 30, loading: 0.91 },
  { id: "legal", label: "Legal", colour: palette.domainLegal, n: 20, loading: 0.89 },
  { id: "privacy", label: "Privacy", colour: palette.domainPrivacy, n: 21, loading: 0.70 },
  { id: "identity_boundary", label: "Identity", colour: palette.domainIdentity, n: 4, loading: 0.58 },
  { id: "capability_boundary", label: "Capability", colour: palette.domainCapability, n: 22, loading: 0.38 },
];

// Methodological regimes the project operates in. The distinction matters:
// raw activations are the residual-stream geometry; SAE features are the
// Gemma Scope 2 dictionary. Different tools, different evidence.
export const regimes = {
  raw: {
    id: "raw",
    label: "Raw activation space",
    short: "no SAEs",
    description:
      "We compute refusal directions and per-domain axes directly in the 3840-dim residual stream of Gemma 3 12B at layer 41. Following Arditi et al. (2024) and Lu et al. (2026): mean(positive) − mean(negative). Capping operates as soft-clamping along these directions. PCA, cosine matrices, falsification tests all live here.",
    badgeFg: "#0A0A0A",
    badgeBg: "#F4F4F4",
  },
  sae: {
    id: "sae",
    label: "SAE feature space",
    short: "Gemma Scope 2",
    description:
      "We encode the same activations through Gemma Scope 2 SAEs (1M-width JumpReLU SAE at layer 41) and compare features across Matryoshka prefix widths {16k, 65k, 262k, 1M}. The hierarchy question lives here: does a coarse 16k feature decompose into multiple 65k children?",
    badgeFg: "#FFFFFF",
    badgeBg: "#1d3557",
  },
};
