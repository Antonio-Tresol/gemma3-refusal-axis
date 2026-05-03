import {
  Section,
  H1,
  H2,
  H3,
  P,
  Code,
  Card,
  Label,
  RegimeBadge,
  InlineLink,
  Mono,
  Callout,
  DataTable,
} from "../components/Primitives.jsx";
import { palette, fonts, regimes } from "../tokens.js";

const FALSIFICATION_TESTS = [
  {
    id: "1a",
    name: "Random split null",
    regime: regimes.raw,
    target: "Domain decomposition",
    method:
      "Shuffle domain labels 1000 times. Compute the cosine range across permuted-label domain directions. p = fraction of permutations with cosine range >= observed.",
    result: "p = 0.054 (one-sided)",
    verdict: "Borderline / weakened",
    detail:
      "The overall cosine range across domains is borderline distinguishable from random splits. Headline language was weakened from 'three geometric clusters' to 'capability separates, but the cluster structure depends on linkage method.'",
  },
  {
    id: "1b",
    name: "Bootstrap CIs",
    regime: regimes.raw,
    target: "Specific cosine values",
    method:
      "Stratified bootstrap with 2000 within-domain resamples (replacement). Recompute domain directions and pairwise cosines per draw; take 2.5/97.5 percentiles.",
    result: "9 of 15 pairwise CIs exclude zero",
    verdict: "Mixed / weakened",
    detail:
      "Safety-capability cosine point estimate is 0.14, but the 95% CI is [-0.316, 0.639]; the CI crosses both 0 and 0.5. 'Near-orthogonal' was weakened to 'low cosine with wide CI.'",
  },
  {
    id: "1c",
    name: "Sample size confound",
    regime: regimes.raw,
    target: "Small-n domain reliability",
    method:
      "Subsample large domains down to n = 4 (the size of identity_boundary). Repeat 100 times; report cosine range.",
    result: "n = 4 safety: range [0.47, 0.88]; n = 4 capability: [-0.20, 0.64]",
    verdict: "Concern confirmed",
    detail:
      "Domains with fewer than ~20 retained pairs are unreliable. Identity (n=4) and capability (n=22) flagged as suggestive only.",
  },
  {
    id: "2a",
    name: "Leave-one-out stability",
    regime: regimes.raw,
    target: "Cosine stability",
    method:
      "Drop each pair, recompute domain direction, cosine with full-data direction. Report LOO range.",
    result: "Safety: [0.900, 0.921]; Ethical: [0.905, 0.916]",
    verdict: "Survives",
    detail:
      "Value-based domain directions are stable under leave-one-out for the larger domains.",
  },
  {
    id: "3a",
    name: "Random PCA baseline",
    regime: regimes.raw,
    target: "11-dim subspace claim",
    method:
      "Generate 1000 sets of random vectors in R^3840 with the same n. Run PCA, count dims for 70% variance.",
    result: "Real = 11 dims; random median = 80; 0/1000 <= real",
    verdict: "Survives (p < 0.001)",
    detail:
      "The low-dimensional structure is real, not a dimensionality artefact. p < 0.001 (one-sided).",
  },
  {
    id: "3b",
    name: "Single-domain PCA",
    regime: regimes.raw,
    target: "Source of multi-dimensionality",
    method:
      "Run PCA within each domain separately. If multi-dimensionality comes only from cross-domain variation, single-domain PCA should be ~1D.",
    result: "Within-domain dims for 70%: 1 to 9",
    verdict: "Informative",
    detail:
      "Some domains are more diffuse than others; but most of the cross-domain decomposition is real, not single-domain noise.",
  },
  {
    id: "4a",
    name: "Random direction capping",
    regime: regimes.raw,
    target: "Safety direction specificity",
    method:
      "Compare safety projection variance on refusal prompts to that of 100 random directions in R^3840.",
    result: "Safety / random ratio = 60.9x",
    verdict: "Survives",
    detail:
      "The capping effect is specific to the safety direction, not a generic activation perturbation.",
  },
  {
    id: "4e",
    name: "Multiple comparisons",
    regime: regimes.raw,
    target: "tau = p50 selection",
    method:
      "Account for the multiplicity in selecting tau = p50 across 7 thresholds x 4 directions x 4 prompt domains = 112 cells.",
    result: "~6 cells significant by chance at p = 0.05",
    verdict: "Honest report / exploratory",
    detail:
      "tau = p50 was sweep-selected, not pre-registered. Reported as exploratory, hypothesis-generating; needs held-out replication.",
  },
  {
    id: "5a",
    name: "Spillover CI",
    regime: regimes.raw,
    target: "Privacy spillover significance",
    method:
      "Attempt bootstrap CI on the -15.9 overall->privacy spillover.",
    result: "Cannot assess; only n = 9 coherent outputs and aggregate scores",
    verdict: "Cannot assess",
    detail:
      "Privacy spillover from overall capping flagged as suggestive only; CI not computable from available aggregate data.",
  },
  {
    id: "6",
    name: "Clustering stability",
    regime: regimes.raw,
    target: "Three-cluster structure",
    method:
      "Repeat hierarchical clustering with Ward's, average, complete, single linkage. Compare orderings.",
    result: "Ward / average separate capability; complete / single produce different orderings",
    verdict: "Method-dependent / weakened",
    detail:
      "The 'three clusters' framing is method-dependent. Capability separation is directionally consistent but not robust to clustering choice.",
  },
];

function VerdictBadge({ text }) {
  const lower = text.toLowerCase();
  let bg = "#F4F4F4";
  let fg = palette.text;
  if (lower.includes("survives")) {
    bg = "#e8f4ec";
    fg = "#1f6f3f";
  } else if (lower.includes("weakened") || lower.includes("borderline") || lower.includes("mixed")) {
    bg = "#fff4e1";
    fg = "#a55a00";
  } else if (lower.includes("cannot")) {
    bg = "#f0f0f0";
    fg = palette.muted;
  } else if (lower.includes("confirmed") || lower.includes("falsified")) {
    bg = "#fde8eb";
    fg = "#a23244";
  } else if (lower.includes("honest") || lower.includes("exploratory") || lower.includes("informative")) {
    bg = "#fff4e1";
    fg = "#a55a00";
  }
  return (
    <span
      style={{
        display: "inline-block",
        background: bg,
        color: fg,
        fontFamily: fonts.mono,
        fontSize: "10.5px",
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        padding: "3px 9px",
        borderRadius: "999px",
        fontWeight: 500,
      }}
    >
      {text}
    </span>
  );
}

function FalsificationCard({ test }) {
  return (
    <Card style={{ padding: "22px 24px" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
          marginBottom: "12px",
          flexWrap: "wrap",
        }}
      >
        <Label style={{ marginBottom: 0 }}>Test {test.id}</Label>
        <RegimeBadge regime={test.regime} />
        <VerdictBadge text={test.verdict} />
      </div>
      <h4
        style={{
          fontFamily: fonts.display,
          fontSize: "17px",
          fontWeight: 600,
          color: palette.text,
          marginBottom: "10px",
          letterSpacing: "-0.01em",
        }}
      >
        {test.name}
      </h4>
      <P style={{ fontSize: "14.5px", marginBottom: "8px" }}>
        <strong style={{ color: palette.text }}>Target:</strong> {test.target}
      </P>
      <P style={{ fontSize: "14.5px", marginBottom: "8px" }}>
        <strong style={{ color: palette.text }}>Method:</strong> {test.method}
      </P>
      <P style={{ fontSize: "14.5px", marginBottom: "8px" }}>
        <strong style={{ color: palette.text }}>Result:</strong> <Mono>{test.result}</Mono>
      </P>
      <P style={{ fontSize: "14.5px", marginBottom: 0, color: palette.body }}>{test.detail}</P>
    </Card>
  );
}

export function MethodologyPage() {
  return (
    <>
      <Section style={{ paddingTop: "64px" }}>
        <Label>Methodology</Label>
        <H1>How we measure refusal</H1>
        <P lead>
          The project operates in two regimes that produce different evidence and warrant different
          claims. This page tells you exactly what we did, in which regime, and what each tool
          can and cannot show.
        </P>
      </Section>

      <Section>
        <H2 id="model-and-sae">1 · Model and SAE</H2>
        <P>
          <strong style={{ color: palette.text }}>Model:</strong> Gemma 3 12B-IT in bf16. 48
          transformer blocks, <Code>d_model = 3840</Code>. Multimodal architecture, so layer
          access path is <Code>model.model.language_model.layers[N]</Code>. For full
          architecture details, see the{" "}
          <InlineLink href="https://arxiv.org/abs/2503.19786">
            Gemma 3 Technical Report
          </InlineLink>
          ; for a plain-English explainer of what changed from Gemma 2, see{" "}
          <InlineLink href="https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/">
            Gemma Explained: what's new in Gemma 3
          </InlineLink>
          .
        </P>
        <P>
          <strong style={{ color: palette.text }}>SAE:</strong> Gemma Scope 2 (
          <InlineLink href="https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf">
            McDougall et al. 2025
          </InlineLink>
          ) JumpReLU SAE (
          <InlineLink href="https://arxiv.org/abs/2407.14435">
            Rajamanoharan et al. 2024
          </InlineLink>
          ) at layer 41, 1M-width with Matryoshka prefix-slicing to {"{"}16k, 65k, 262k, 1M{"}"}.
          Layer 41 is the deepest of the four expanded-width layers Gemma Scope 2 trains for 12B
          (Table 1: layers {"{"}12, 24, 31, 41{"}"} at 25 / 50 / 65 / 85% depth). It is{" "}
          <em>our project choice</em>, not a paper recommendation.
        </P>
        <P>
          <strong style={{ color: palette.text }}>Hardware:</strong> RTX 5090 (32 GB VRAM). Model
          and SAE cannot coexist on a single GPU (~22.7 + ~15.0 GB); pipeline swaps phases.
        </P>
      </Section>

      <Section>
        <H2 id="dataset">2 · Dataset</H2>
        <P>
          <strong>240 contrastive prompt pairs</strong> across 6 domains (40 per domain). Each pair
          has a positive prompt (should trigger refusal) and a negative prompt (same topic, benign,
          should be answered). Construction follows{" "}
          <InlineLink href="https://arxiv.org/abs/2406.11717">Arditi et al. (2024)</InlineLink>.
        </P>
        <P>
          <strong>Pair generation:</strong> All 240 pairs were generated by{" "}
          <InlineLink href="https://www.anthropic.com/claude/opus">Claude Opus 4.6</InlineLink>{" "}
          via Claude Code, from a specification prompt that supplied the six domain definitions,
          the matching rules, and two human-written example pairs per domain. The output JSON was
          then put through a vocabulary-confound check (no content word in &gt;40% of one condition
          and &lt;10% of the other) and a 60-pair stratified human review (10 per domain,{" "}
          <em>seed=42</em>; 60/60 passed). 40 supplementary pairs targeted at under-covered domains
          (ethical, privacy, identity_boundary) were added later via the Claude Agent SDK using the
          same model and rules; total candidate pool is 280.
        </P>
        <P>
          <strong>Trait scoring:</strong> Each Gemma 3 response scored 0–100 for refusal expression
          by{" "}
          <InlineLink href="https://www.anthropic.com/claude/sonnet">Claude Sonnet 4.6</InlineLink>{" "}
          via the Claude Agent SDK. Three independent scoring passes, inter-pass Pearson
          correlation 0.990–0.991, final score is the median. This follows{" "}
          <InlineLink href="https://arxiv.org/abs/2507.21509">
            Chen et al. (2025) Sec 2.2
          </InlineLink>{" "}
          (Persona Vectors): &quot;We filter the responses based on their trait expression scores,
          retaining only those that align with the intended system prompt.&quot;
        </P>
        <P>
          <strong>Retention:</strong> 128/280 pairs retained where positive score &gt; 50 AND
          negative score &lt; 30 (104 from the initial 240 + 24 from the supplementary 40).
        </P>
      </Section>

      <Section>
        <H2 id="regime-raw">3 · Regime A: Raw activation space</H2>
        <RegimeBadge regime={regimes.raw} />

        <H3>3.1 · Refusal axis construction</H3>
        <P>
          Following Lu et al. Sec 3.1: <Code>axis = mean(positive) − mean(negative)</Code>. Per
          domain D: <Code>direction_D = mean(positive[D]) − mean(negative[D])</Code>, unit-normalised.
        </P>
        <H3>3.2 · Extraction site</H3>
        <P>
          <strong>Mean-response-token at layer 41.</strong> Mean activation across all assistant
          response tokens. We also extract <strong>last-prompt-token</strong> as a comparison site.
        </P>
        <DataTable
          headers={["Site", "Captures", "Source"]}
          rows={[
            [
              "Last-prompt-token",
              'Model "intent" before generating',
              <InlineLink key="a" href="https://arxiv.org/abs/2406.11717">
                Arditi et al. 2024
              </InlineLink>,
            ],
            [
              "Mean-response-token",
              "Behaviour during generation",
              <InlineLink key="c" href="https://arxiv.org/abs/2507.21509">
                Chen et al. 2025
              </InlineLink>,
            ],
          ]}
        />
        <H3>3.3 · Geometric analyses</H3>
        <ul style={{ marginLeft: "20px", lineHeight: 1.7, fontFamily: fonts.body, fontSize: "16px", color: palette.body }}>
          <li>
            <strong style={{ color: palette.text }}>Domain loadings on mean axis</strong>; cosine
            similarity, analogous to Lu et al.'s role loadings.
          </li>
          <li>
            <strong style={{ color: palette.text }}>Cross-domain cosine matrix</strong>; pairwise
            cosines between unit-normalised domain directions.
          </li>
          <li>
            <strong style={{ color: palette.text }}>Hierarchical clustering</strong>; Ward's
            linkage on 1 − cosine. Method-dependent (test 6).
          </li>
          <li>
            <strong style={{ color: palette.text }}>PCA</strong>; on per-pair difference vectors,
            mean-centred (no L2 normalisation), following Lu et al. <Code>pca.ipynb</Code>.
          </li>
        </ul>

        <H3>3.4 · Domain-selective capping</H3>
        <P>
          Following Lu et al. (2026) <Code>steering.py</Code>{" "}
          <Code>intervention_type=&quot;capping&quot;</Code>. Soft-clamp:{" "}
          <Code>act −= max(proj − τ, 0) · v_hat</Code>. Applied at <strong>layer 36</strong> (75%
          depth), distinct from layer 41 where activations are extracted. M5 steering experiments
          found 36 optimal for Gemma 3 12B; consistent with{" "}
          <InlineLink href="https://arxiv.org/abs/2312.06681">
            Panickssery et al. (2024) (CAA)
          </InlineLink>{" "}
          finding ~40% depth optimal for Llama 2.
        </P>
        <P>
          τ values are percentiles of the benign-prompt projection distribution, sweeping{" "}
          <Code>{`[10, 25, 50, 75, 90, 95, 99]`}</Code>.
        </P>
      </Section>

      <Section>
        <H2 id="regime-sae">4 · Regime B: SAE feature space</H2>
        <RegimeBadge regime={regimes.sae} />
        <P>
          Same activations as Regime A, encoded through the 1M-width Gemma Scope 2 JumpReLU SAE. We
          prefix-slice the encoding to {"{"}16k, 65k, 262k, 1M{"}"} via Matryoshka and compare
          features across widths. Three independent methods test the hierarchy hypothesis:
        </P>
        <DataTable
          headers={["Method", "Tool", "Source"]}
          rows={[
            [
              "Decoder cosine",
              "Cosine similarity between parent / child decoder vectors",
              <InlineLink key="b" href="https://transformer-circuits.pub/2023/monosemantic-features/">
                Bricken et al. 2023
              </InlineLink>,
            ],
            [
              "Co-activation Jaccard",
              "Binary intersection / union over prompts",
              <InlineLink key="c" href="https://arxiv.org/abs/2409.14507">
                Chanin et al. 2024
              </InlineLink>,
            ],
            [
              "Hierarchy R²",
              "Least-squares fit of parent decoder as weighted sum of children",
              <InlineLink key="l" href="https://arxiv.org/abs/2602.11881">
                Luo et al. 2026 (HSAE)
              </InlineLink>,
            ],
          ]}
        />
        <P>
          Robustness checks: permutation null distributions for Jaccard (1000 shuffles), base-rate
          analysis under feature-rate independence assumption, threshold sensitivity, and tests at
          all width transitions (16k→65k, 65k→262k, 262k→1M).
        </P>
      </Section>

      <Section>
        <H2 id="statistics">5 · Statistical methods</H2>
        <P>
          <strong style={{ color: palette.text }}>Bootstrap CIs.</strong> 2000 within-domain
          resamples with replacement. Recompute domain directions, recompute pairwise cosines, take
          2.5/97.5 percentiles. Stratified per domain.
        </P>
        <P>
          <strong style={{ color: palette.text }}>Permutation tests.</strong> One-sided. For test
          1a (random split null) and 3a (random PCA baseline): 1000 permutations comparing the
          observed statistic to the null distribution. For Jaccard: shuffle the parent's binary
          activation vector while keeping child marginals fixed.
        </P>
        <P>
          <strong style={{ color: palette.text }}>Leave-one-out stability.</strong> Drop each pair
          in turn, recompute domain direction, measure cosine with the full-data direction. Reports
          the LOO range.
        </P>
      </Section>

      <Section>
        <H2 id="falsification-step">6 · Pre-registered falsification</H2>
        <P>
          Falsification is a step in the methodology, not a separate exhibit. Before any headline
          claim was written up, a pre-registered suite of tests tried to break it. The numbers
          that survived are the ones reported; the numbers that did not survive are recorded
          below, alongside the language change they forced.
        </P>
        <Callout>
          Working with an LLM-based agent partner makes it easy to drift into believing
          nicely-formatted numbers, neat tables, and a clean narrative. Falsification is the
          discipline that keeps both the human and the agent honest: never trust a number until
          something has tried to break it, and never let a failed test quietly disappear.
        </Callout>
        <P>
          The step has three rules.{" "}
          <strong style={{ color: palette.text }}>Pre-registration:</strong> the test plan is
          committed before the tests run, so a disappointing result cannot be reframed after the
          fact.{" "}
          <strong style={{ color: palette.text }}>Honest reporting:</strong> exploratory results
          (such as the τ = p<sub>50</sub> sweet-spot) are flagged as exploratory rather than
          laundered into confirmatory ones.{" "}
          <strong style={{ color: palette.text }}>Update or weaken, never delete:</strong> a test
          that conflicts with a claim weakens or qualifies the claim; the test itself is never
          dropped from the record.
        </P>

        <H3 id="falsification-ledger">6.1 · The ledger</H3>
        <P>
          Ten pre-registered tests, each paired with a specific claim it could break. For each:
          what it tried to break, the method, the observed result, and how the prose was updated
          afterwards.
        </P>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "16px",
            marginTop: "16px",
          }}
        >
          {FALSIFICATION_TESTS.map((t) => (
            <FalsificationCard key={t.id} test={t} />
          ))}
        </div>
        <P style={{ marginTop: "20px" }}>
          The implementation lives in{" "}
          <InlineLink href="https://github.com/Antonio-Tresol/gemma3-refusal-axis/blob/dev/src/refusal_decomposition/analysis/falsification.py">
            <Code>src/refusal_decomposition/analysis/falsification.py</Code>
          </InlineLink>
          ; the test plan was committed before the tests ran (see{" "}
          <Code>plans/plan_refusal_axis_falsification.md</Code> in the repo).
        </P>
      </Section>

      <Section>
        <H2 id="reproducibility">7 · Reproducibility</H2>
        <DataTable
          headers={["Item", "Where"]}
          rows={[
            [
              "Refusal axis analysis",
              <InlineLink
                key="ra"
                href="https://github.com/Antonio-Tresol/gemma3-refusal-axis/blob/dev/src/refusal_decomposition/analysis/refusal_axis.py"
              >
                <Code>src/refusal_decomposition/analysis/refusal_axis.py</Code>
              </InlineLink>,
            ],
            [
              "Domain-selective capping",
              <InlineLink
                key="dc"
                href="https://github.com/Antonio-Tresol/gemma3-refusal-axis/blob/dev/src/refusal_decomposition/analysis/domain_capping.py"
              >
                <Code>src/refusal_decomposition/analysis/domain_capping.py</Code>
              </InlineLink>,
            ],
            [
              "Feature hierarchy (SAE)",
              <InlineLink
                key="fh"
                href="https://github.com/Antonio-Tresol/gemma3-refusal-axis/blob/dev/src/refusal_decomposition/analysis/feature_hierarchy.py"
              >
                <Code>src/refusal_decomposition/analysis/feature_hierarchy.py</Code>
              </InlineLink>,
            ],
            [
              "Falsification suite",
              <InlineLink
                key="fs"
                href="https://github.com/Antonio-Tresol/gemma3-refusal-axis/blob/dev/src/refusal_decomposition/analysis/falsification.py"
              >
                <Code>src/refusal_decomposition/analysis/falsification.py</Code>
              </InlineLink>,
            ],
            [
              "Figures (no-SAE)",
              <InlineLink
                key="fr"
                href="https://github.com/Antonio-Tresol/gemma3-refusal-axis/blob/dev/src/refusal_decomposition/viz/refusal_axis.py"
              >
                <Code>src/refusal_decomposition/viz/refusal_axis.py</Code>
              </InlineLink>,
            ],
            [
              "Public dataset",
              <InlineLink
                key="hf"
                href="https://huggingface.co/datasets/abotresol/gemma3-refusal-axis-data"
              >
                abotresol/gemma3-refusal-axis-data
              </InlineLink>,
            ],
          ]}
        />
      </Section>

      <Section style={{ marginTop: "64px" }}>
        <Mono>
          <InlineLink href="#/feature-hierarchy" arrow="left" style={{ color: palette.text }}>
            Back to Part II: Feature hierarchy
          </InlineLink>
          {"   |   "}
          <InlineLink href="#/references" arrow="right" style={{ color: palette.text }}>
            Continue to References
          </InlineLink>
        </Mono>
      </Section>
    </>
  );
}
