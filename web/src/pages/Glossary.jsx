import { useEffect } from "react";
import {
  Section,
  H1,
  H2,
  P,
  Card,
  Label,
  InlineLink,
  SourcePill,
  Mono,
} from "../components/Primitives.jsx";
import { palette, fonts } from "../tokens.js";
import { glossaryEntries } from "../glossary.js";

// Group glossary entries by reading-friendly category so the page is skimmable.
const groups = [
  {
    id: "what-the-model-is-doing",
    title: "What the model is doing",
    blurb: "The objects of study: the words for the things inside Gemma 3 that we are looking at.",
    slugs: ["refusal", "residual-stream", "layer", "activation"],
  },
  {
    id: "geometric-tools",
    title: "Geometric tools (raw activations, no SAEs)",
    blurb:
      "How we treat activations as data. The Part I machinery: directions, projections, distances, dimensionality.",
    slugs: ["geometry", "subspace", "direction", "refusal-axis", "concept-cone", "harmfulness", "projection", "cosine-similarity", "pca"],
  },
  {
    id: "interventions",
    title: "Interventions",
    blurb: "Editing activations mid-forward-pass to steer the model.",
    slugs: ["capping", "tau-threshold"],
  },
  {
    id: "saes",
    title: "Sparse autoencoders (Part II)",
    blurb:
      "Re-expressing activations as a small number of human-interpretable features. Gemma Scope 2 vocabulary.",
    slugs: ["sae", "feature", "decoder", "jumprelu", "matryoshka"],
  },
  {
    id: "hierarchy-tests",
    title: "Hierarchy tests",
    blurb: "How we ask whether SAE features at different widths form a parent-child hierarchy.",
    slugs: ["jaccard", "r-squared"],
  },
  {
    id: "statistics",
    title: "Statistics & epistemics",
    blurb: "How we decide a result is real, weak, or wrong.",
    slugs: ["bootstrap-ci", "permutation-test", "falsification"],
  },
];

function GlossaryEntry({ entry }) {
  return (
    <Card
      as="article"
      hover={false}
      id={entry.slug}
      style={{ scrollMarginTop: "100px", padding: "22px 24px" }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: "12px",
          flexWrap: "wrap",
          marginBottom: "10px",
        }}
      >
        <h3
          style={{
            fontFamily: fonts.display,
            fontSize: "20px",
            fontWeight: 600,
            color: palette.text,
            letterSpacing: "-0.01em",
          }}
        >
          {entry.term}
        </h3>
        <Mono style={{ fontSize: "10.5px", color: palette.muted }}>#{entry.slug}</Mono>
      </div>
      <P style={{ fontSize: "16px", color: palette.text, marginBottom: "10px" }}>
        {entry.short}
      </P>
      <P style={{ fontSize: "15px", marginBottom: entry.sources.length ? "14px" : 0 }}>
        {entry.long}
      </P>
      {entry.sources.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center" }}>
          <Mono style={{ fontSize: "10.5px", marginRight: "2px" }}>Sources</Mono>
          {entry.sources.map((s) => (
            <SourcePill key={s} bibKey={s} />
          ))}
        </div>
      )}
    </Card>
  );
}

export function GlossaryPage() {
  // Allow nested-hash anchors like `#/glossary#refusal-axis` to scroll the
  // matching card into view on initial load.
  useEffect(() => {
    const hash = window.location.hash;
    const idx = hash.indexOf("#", 1);
    if (idx < 0) return;
    const slug = hash.slice(idx + 1);
    if (!slug) return;
    requestAnimationFrame(() => {
      const el = document.getElementById(slug);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }, []);

  const entriesBySlug = Object.fromEntries(glossaryEntries.map((e) => [e.slug, e]));

  return (
    <>
      <Section style={{ paddingTop: "64px" }}>
        <Label>For readers new to mechanistic interpretability</Label>
        <H1>Glossary</H1>
        <P lead>
          Plain-English definitions for the vocabulary used across the rest of this site.
          Hover any dotted-underlined word in the prose to get the short version; click to land on
          the full entry below.
        </P>
        <P>
          Sources point to the published work most relevant to each term, for the reader who wants
          to drill in. Definitions are written for someone who knows what a neural network is but
          has not read interpretability papers.
        </P>
      </Section>

      <Section style={{ marginTop: "16px" }}>
        <Label>Quick index</Label>
        <Card hover={false} style={{ padding: "18px 22px" }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
              gap: "10px 18px",
              fontFamily: fonts.body,
              fontSize: "14.5px",
            }}
          >
            {glossaryEntries.map((e) => (
              <a
                key={e.slug}
                href={`#${e.slug}`}
                className="link-plain"
                style={{ color: palette.body, textDecoration: "none" }}
              >
                {e.term}
              </a>
            ))}
          </div>
        </Card>
      </Section>

      {groups.map((g) => (
        <Section key={g.id} style={{ marginTop: "56px" }}>
          <Label>{g.title}</Label>
          <H2 id={g.id} style={{ marginTop: "8px" }}>
            {g.title}
          </H2>
          <P>{g.blurb}</P>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "16px",
              marginTop: "20px",
            }}
          >
            {g.slugs.map((slug) =>
              entriesBySlug[slug] ? (
                <GlossaryEntry key={slug} entry={entriesBySlug[slug]} />
              ) : null
            )}
          </div>
        </Section>
      ))}

      <Section style={{ marginTop: "72px" }}>
        <Mono>
          <InlineLink href="#/" arrow="left">
            Back to overview
          </InlineLink>
          {"   |   "}
          <InlineLink href="#/refusal-axis" arrow="right">
            Continue to Part I
          </InlineLink>
        </Mono>
      </Section>
    </>
  );
}
