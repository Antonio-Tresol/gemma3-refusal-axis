import {
  Section,
  H1,
  H2,
  P,
  Card,
  Label,
  RegimeBadge,
  InlineLink,
  Mono,
  Callout,
  DataTable,
  Term,
  MethodTerm,
  Figure,
} from "../components/Primitives.jsx";
import { palette, fonts, regimes } from "../tokens.js";

const FIG_PART1 = import.meta.env.BASE_URL + "figures/refusal_axis/";
const FIG_PART2 = import.meta.env.BASE_URL + "figures/sae_width_scaling/";

function FindingCard({ regime, title, body, falsification, payoff }) {
  return (
    <Card style={{ padding: "26px 28px" }}>
      <div style={{ marginBottom: "14px" }}>
        <RegimeBadge regime={regime} />
      </div>
      <h3
        style={{
          fontFamily: fonts.display,
          fontSize: "20px",
          fontWeight: 600,
          color: palette.text,
          marginBottom: "10px",
          letterSpacing: "-0.01em",
        }}
      >
        {title}
      </h3>
      <div
        style={{
          fontFamily: fonts.body,
          fontSize: "15.5px",
          lineHeight: 1.65,
          color: palette.body,
        }}
      >
        {body}
      </div>
      {payoff && (
        <div
          style={{
            marginTop: "14px",
            paddingTop: "14px",
            borderTop: `1px solid ${palette.border}`,
            fontFamily: fonts.body,
            fontStyle: "italic",
            fontSize: "14.5px",
            lineHeight: 1.55,
            color: palette.text,
          }}
        >
          {payoff}
        </div>
      )}
      {falsification && (
        <Mono
          style={{
            display: "block",
            marginTop: payoff ? "10px" : "16px",
            fontSize: "10.5px",
            color: palette.muted,
          }}
        >
          <InlineLink
            href="#/methodology#falsification-ledger"
            style={{ color: palette.muted }}
          >
            {falsification}
          </InlineLink>
        </Mono>
      )}
    </Card>
  );
}

export function LandingPage() {
  return (
    <>
      <Section style={{ paddingTop: "72px" }}>
        <H1>The Refusal Axis</H1>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "baseline",
            gap: "10px 18px",
            marginTop: "8px",
            marginBottom: "24px",
            fontFamily: fonts.body,
            fontSize: "16px",
            color: palette.body,
          }}
        >
          <span style={{ color: palette.text, fontWeight: 600 }}>Antonio Badilla-Olivas</span>
          <span style={{ fontFamily: fonts.mono, fontSize: "12px", color: palette.muted }}>
            independent · 2026
          </span>
          <InlineLink
            href="https://github.com/Antonio-Tresol/gemma3-refusal-axis"
            style={{ fontFamily: fonts.mono, fontSize: "12px" }}
          >
            github.com/Antonio-Tresol/gemma3-refusal-axis
          </InlineLink>
        </div>

        <P lead>
          A <strong style={{ color: palette.text, fontWeight: 600 }}>refusal axis</strong> is a
          direction in a language model's <Term name="residual-stream">residual stream</Term>{" "}
          that points "toward refusing": the average{" "}
          <Term name="activation">activation</Term> difference between matched prompts the
          model declines and prompts it complies with. The question we ask of{" "}
          <strong style={{ color: palette.text, fontWeight: 600 }}>Gemma 3 12B</strong> is
          whether one such axis is enough, or whether refusal decomposes into
          distinguishable per-domain directions (safety, ethical, legal, privacy, identity,
          capability).
        </P>
      </Section>

      <Section style={{ marginTop: "40px" }}>
        <Label>One model, two views</Label>
        <P style={{ marginTop: "8px" }}>
          Both regimes start from the same layer-41 captures of the same prompt-response
          pairs. Part I works on the raw 3,840-dim vectors; Part II passes them through
          Gemma Scope 2's <Term name="sae">SAE</Term>. Click any figure to enlarge.
        </P>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
            gap: "28px",
            marginTop: "20px",
          }}
        >
          <div>
            <Figure
              src={FIG_PART1 + "fig_hero_3d_refusal_axis.png"}
              alt="3D PCA scatter of 128 contrastive activation differences. Per-domain centroids float in PC1/PC2/PC3 space; the dashed line marks the mean refusal axis through them."
              regime={regimes.raw}
              caption={
                <>
                  128 contrastive activation differences in the top three PCs. Per-domain
                  centroids spread out in this subspace, and the mean refusal axis (dashed)
                  threads through them. Visual evidence that refusal is not noise: it
                  occupies a structured low-dimensional region.
                </>
              }
              style={{ margin: "0" }}
            />
            <Mono style={{ display: "block", marginTop: "12px" }}>
              <InlineLink href="#/refusal-axis" arrow="right" style={{ color: palette.text }}>
                Read Part I
              </InlineLink>
            </Mono>
          </div>

          <div>
            <Figure
              src={FIG_PART2 + "fig2_feature_genealogy.png"}
              alt="Feature genealogy bar charts. Each row is a Matryoshka SAE width; bars show how few of the 16k baseline features survive into 65k, 262k, 1M. Most features at the larger widths emerged later, not split from 16k."
              regime={regimes.sae}
              caption={
                <>
                  At finer <Term name="matryoshka">Matryoshka</Term> widths, only a minority
                  of 16k features survive (28% LPT, 41% MRT at 1M). Features appear{" "}
                  <em>replaced</em> by unrelated new ones rather than splitting into
                  domain-specific children. No apparent parent-child hierarchy.
                </>
              }
              style={{ margin: "0" }}
            />
            <Mono style={{ display: "block", marginTop: "12px" }}>
              <InlineLink
                href="#/feature-hierarchy"
                arrow="right"
                style={{ color: palette.text }}
              >
                Read Part II
              </InlineLink>
            </Mono>
          </div>
        </div>
      </Section>

      <Section style={{ marginTop: "56px" }}>
        <Card
          style={{
            padding: "24px 28px",
            background: palette.surfaceAlt,
            borderLeft: `3px solid ${palette.text}`,
          }}
        >
          <Label style={{ color: palette.muted, marginBottom: "10px" }}>TL;DR</Label>
          <P style={{ marginBottom: "10px", fontSize: "15.5px" }}>
            <strong style={{ color: palette.text }}>Two regimes, two views.</strong>{" "}
            <strong>Part I</strong> works in raw activation space (no{" "}
            <Term name="sae">SAE</Term>): refusal seems to occupy a structured
            11-dimensional subspace (<em>p</em> &lt; 0.001 vs random); per-domain directions
            look distinguishable but bootstrap CIs are wide; and{" "}
            <Term name="capping">capping</Term> along the safety direction reduces
            safety refusal by 31.6 points on the 0–100{" "}
            <MethodTerm name="trait-scoring">trait-score</MethodTerm> scale, while
            capability, privacy, and benign responses move by &lt; 1.5 on the same scale.
          </P>
          <P style={{ marginBottom: "10px", fontSize: "15.5px" }}>
            <strong>Part II</strong> works in Gemma Scope 2 <Term name="sae">SAE</Term>{" "}
            feature space across <Term name="matryoshka">Matryoshka</Term> widths{" "}
            <Mono>{"{"}16k → 65k → 262k → 1M{"}"}</Mono>: three independent methods (decoder
            cosine, co-activation Jaccard, R² decomposition) all fail to find any{" "}
            <em>apparent parent-child hierarchy</em>. The hypothesis that broad refusal
            features split into domain-specific children at finer widths does not survive
            pre-registered testing. The two regimes thus give different views of the same
            activations: the per-domain structure that supports selective capping in raw
            space does not appear as a parent-child hierarchy in SAE space.
          </P>
          <P style={{ marginBottom: 0, fontSize: "15.5px" }}>
            <strong>Significance.</strong> Our 11-dimensional{" "}
            <Term name="subspace">subspace</Term> finding sits within a recent
            literature triangulation: Lu et al. argue for a single &quot;assistant
            axis&quot;, Wollschläger et al. (ICML 2025) find{" "}
            <Term name="concept-cone">concept cones</Term> up to dim 5, and Joad et al.
            find 11 geometrically distinct category directions that share a
            one-dimensional control knob. Our per-domain decomposition is weaker than Joad's
            headline and stronger than Lu's single axis, broadly consistent with
            Wollschläger's multi-dimensional picture. Safety-selective capping
            is exploratory (τ chosen post-hoc, <em>n</em> = 10 / domain) and would need
            pre-registered replication before deployment.
          </P>
        </Card>

        <Mono style={{ display: "block", marginTop: "20px" }}>
          New to the vocabulary?{" "}
          <InlineLink href="#/glossary" arrow="right" style={{ color: palette.text }}>
            Start at the glossary
          </InlineLink>
          {"  ·  "}Already comfortable with SAEs?{" "}
          <InlineLink href="#/feature-hierarchy" arrow="right" style={{ color: palette.text }}>
            Jump to Part II
          </InlineLink>
        </Mono>
      </Section>

      <Section style={{ marginTop: "56px" }}>
        <Label>What we asked</Label>
        <H2 style={{ marginTop: "8px" }}>The questions</H2>
        <P>
          One overarching ask (<em>is refusal one direction or several?</em>), decomposed
          into four operational questions: three in raw activation space, one inside the SAE
          dictionary. Short answers at the bottom of this section; full evidence further down.
        </P>

        <div style={{ marginTop: "24px", display: "flex", flexDirection: "column", gap: "14px" }}>
          {[
            {
              num: "Q1",
              regime: regimes.raw,
              text: (
                <>
                  Does refusal occupy a structured low-dimensional subspace, or is it
                  scattered through the residual stream?
                </>
              ),
            },
            {
              num: "Q2",
              regime: regimes.raw,
              text: (
                <>
                  Are the per-domain refusal directions distinct, or do they collapse to a
                  single shared axis?
                </>
              ),
            },
            {
              num: "Q3",
              regime: regimes.raw,
              text: (
                <>
                  Can we cap one refusal sub-type (safety) without disturbing the others
                  (capability, privacy, benign)?
                </>
              ),
            },
            {
              num: "Q4",
              regime: regimes.sae,
              text: (
                <>
                  Across <Term name="matryoshka">Matryoshka</Term> widths{" "}
                  <Mono>{"{"}16k → 65k → 262k → 1M{"}"}</Mono>, do broad refusal features
                  split hierarchically into domain-specific children?
                </>
              ),
            },
          ].map(({ num, regime, text }) => (
            <div
              key={num}
              style={{
                display: "flex",
                gap: "14px",
                alignItems: "baseline",
                paddingBottom: "12px",
                borderBottom: `1px solid ${palette.border}`,
              }}
            >
              <Mono
                style={{
                  color: palette.text,
                  fontSize: "13px",
                  fontWeight: 600,
                  minWidth: "30px",
                }}
              >
                {num}
              </Mono>
              <span style={{ flexShrink: 0 }}>
                <RegimeBadge regime={regime} />
              </span>
              <P style={{ marginBottom: 0, flex: 1, fontSize: "15.5px" }}>{text}</P>
            </div>
          ))}
        </div>

        <P style={{ marginTop: "20px", fontSize: "14.5px", color: palette.muted }}>
          <strong style={{ color: palette.text, fontWeight: 600 }}>Short answers.</strong>{" "}
          Q1 appears so (11 dims, <em>p</em> &lt; 0.001). Q2 seems so, weakly (value-domains
          cluster, safety↔capability has a wide CI). Q3 might, for safety only (exploratory;
          τ chosen post-hoc). Q4 apparently not (three pre-registered tests fail to find a
          hierarchy).
        </P>
      </Section>

      <Section style={{ marginTop: "56px" }}>
        <Label>Two methodological regimes</Label>
        <H2 style={{ marginTop: "8px" }}>What we measure, and where</H2>
        <P>
          The project operates in <em>two distinct regimes</em>, and the distinction matters. The
          tools, the evidence they produce, and what we can conclude from each are different. Read
          every figure with the regime badge in mind.
        </P>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: "20px",
            marginTop: "24px",
          }}
        >
          <Card style={{ padding: "26px 28px" }}>
            <RegimeBadge regime={regimes.raw} />
            <h3
              style={{
                fontFamily: fonts.display,
                fontSize: "22px",
                fontWeight: 600,
                color: palette.text,
                marginTop: "14px",
                marginBottom: "10px",
                letterSpacing: "-0.01em",
              }}
            >
              Part I: Raw activation geometry
            </h3>
            <P style={{ fontSize: "15px", marginBottom: "8px" }}>
              We work directly in Gemma 3 12B's <Term name="residual-stream">residual stream</Term>{" "}
              at <Term name="layer">layer 41</Term>: a 3,840-dim space.{" "}
              <Term name="refusal-axis">Refusal directions</Term> are mean activation differences
              between matched harmful and benign prompts.{" "}
              <strong style={{ color: palette.text }}>No SAE.</strong> Per-domain axes,{" "}
              <Term name="cosine-similarity">cosines</Term>, <Term name="pca">PCA</Term>, and{" "}
              <Term name="capping">capping</Term> all live here. This regime tells us about{" "}
              <em>behaviour we can steer</em>.
            </P>
            <Mono style={{ display: "block", marginTop: "10px" }}>
              <InlineLink href="#/refusal-axis" arrow="right" style={{ color: palette.text }}>
                Read Part I
              </InlineLink>
            </Mono>
          </Card>

          <Card style={{ padding: "26px 28px" }}>
            <RegimeBadge regime={regimes.sae} />
            <h3
              style={{
                fontFamily: fonts.display,
                fontSize: "22px",
                fontWeight: 600,
                color: palette.text,
                marginTop: "14px",
                marginBottom: "10px",
                letterSpacing: "-0.01em",
              }}
            >
              Part II: SAE feature space
            </h3>
            <P style={{ fontSize: "15px", marginBottom: "8px" }}>
              Same activations, encoded through Gemma Scope 2's 1M-width{" "}
              <Term name="jumprelu">JumpReLU</Term> SAE and prefix-sliced to{" "}
              {"{"}16k, 65k, 262k, 1M{"}"} via <Term name="matryoshka">Matryoshka</Term>. We ask
              whether coarse refusal <Term name="feature">features</Term> hierarchically split into
              domain-specific children at finer widths. This regime tells us about{" "}
              <em>the dictionary's internal organisation</em>.
            </P>
            <Mono style={{ display: "block", marginTop: "10px" }}>
              <InlineLink href="#/feature-hierarchy" arrow="right" style={{ color: palette.text }}>
                Read Part II
              </InlineLink>
            </Mono>
          </Card>
        </div>
      </Section>

      <Section style={{ marginTop: "72px" }}>
        <Label>Headline findings</Label>
        <H2 id="headline-findings" style={{ marginTop: "8px" }}>What we found</H2>
        <P>
          Each claim below is phrased as an observation ("appears", "seems", "might suggest"),
          not a verdict. Every one of these statements is the version that survived a
          pre-registered set of tests trying to break it: some came through intact, some were
          rewritten in weaker language, one was retracted. The full ledger of which test
          changed which claim sits inside Methodology (see{" "}
          <InlineLink href="#/methodology#falsification-step">§6 Pre-registered falsification</InlineLink>
          ), where it belongs as a hygiene step rather than a separate exhibit.
        </P>

        <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: "24px" }}>
          <FindingCard
            regime={regimes.raw}
            title="Refusal appears to occupy a structured low-dimensional subspace"
            falsification="Test 3a · survives"
            body={
              <>
                <Term name="pca">PCA</Term> on activation differences needs{" "}
                <strong>11 dimensions</strong> to capture 70% of variance, versus a median of 80
                for random vectors in the same space (one-sided{" "}
                <Term name="permutation-test">permutation</Term>, <em>p</em> &lt; 0.001;
                0 / 1000 shuffled label assignments matched the real structure).
              </>
            }
            payoff={
              <>
                Reading: refusal activations appear to occupy a structured low-dimensional
                subspace in the residual stream that survives null-distribution testing,
                not a scattered noise pattern. There seems to be real structure to decompose.
              </>
            }
          />
          <FindingCard
            regime={regimes.raw}
            title="Per-domain directions seem to differ, but only suggestively"
            falsification="Tests 1a, 1b · weakened"
            body={
              <>
                Value-based domains cluster tightly (safety↔ethical{" "}
                <Term name="cosine-similarity">cosine</Term> 0.87). The safety↔capability cosine
                is lower (point estimate 0.14) but its 95%{" "}
                <Term name="bootstrap-ci">bootstrap CI</Term> spans{" "}
                <Mono>[−0.316, 0.639]</Mono>, so the data alone cannot tell the two directions
                apart from "near-orthogonal" or "moderately aligned". The cosine range across
                all six domains is only borderline distinguishable from a random split of the
                same prompts (one-sided permutation test, <em>p</em> = 0.054).
              </>
            }
            payoff={
              <>
                Reading: domains might differ, but not cleanly enough to bet on. Safety vs
                ethical looks solid; safety vs capability is suggestive at best.
              </>
            }
          />
          <FindingCard
            regime={regimes.raw}
            title="Safety capping might be selective (exploratory)"
            falsification="Test 4a · survives, 4e · exploratory"
            body={
              <>
                At <Term name="tau-threshold">τ = p<sub>50</sub></Term>,{" "}
                <Term name="capping">capping</Term> along the safety refusal direction
                reduces safety refusal by <strong>31.6 points</strong> on the
                0–100 <MethodTerm name="trait-scoring">trait-score</MethodTerm> scale,
                while capability, privacy, and benign responses move by &lt; 1.5 on the
                same scale. Refusal-prompt activations align{" "}
                <strong>60.9× more strongly</strong> with the safety direction than with
                random directions on average.
                But τ = p<sub>50</sub> was selected by sweeping 7 values, not pre-registered,
                and with <em>n</em> = 10 prompts per domain, ~6 of the 112 cells tested
                would appear significant by chance at <em>p</em> = 0.05.
              </>
            }
            payoff={
              <>
                Reading: one of three refusal sub-types we tried looks like it can be turned
                down independently. The effect reproduces, but the operating point was chosen
                post-hoc; treat as hypothesis-generating until a pre-registered replication
                confirms it.
              </>
            }
          />
          <FindingCard
            regime={regimes.sae}
            title="No apparent parent-child hierarchy across Matryoshka widths"
            falsification="Three methods agree · H0 holds"
            body={
              <>
                Across the 16k → 65k <Term name="matryoshka">Matryoshka</Term> transition, no
                parent-child <Term name="decoder">decoder</Term> geometry was found. Max{" "}
                <Term name="cosine-similarity">decoder cosine</Term> is 0.287 (below the 0.3
                threshold), and the one feature that approached the R² threshold (max{" "}
                <Term name="r-squared">R²</Term> 0.580) turns out to be a base-rate artefact
                (both parent and child fire on 97% of prompts; permutation <em>p</em> = 0.076).
                Disappearing features look <strong>replaced</strong> by unrelated new ones,
                not split.
              </>
            }
            payoff={
              <>
                Reading: the SAE dictionary does not appear to encode refusal as a parent-
                child feature hierarchy in this setup. Three independent pre-registered tests
                point the same way, so we accept the null and move on.
              </>
            }
          />
        </div>
      </Section>

      <Section style={{ marginTop: "72px" }}>
        <Label>Where this sits</Label>
        <H2 style={{ marginTop: "8px" }}>How this work compares</H2>
        <P>
          Two recent papers bracket this work.{" "}
          <InlineLink href="https://arxiv.org/abs/2601.10387">Lu et al. (2026)</InlineLink> argued
          that a single &quot;assistant axis&quot; mediates refusal across multiple models;{" "}
          <InlineLink href="https://arxiv.org/abs/2602.02132">Joad et al. (2026)</InlineLink>{" "}
          countered that 11 category directions are geometrically distinct yet act as a shared 1D
          control knob. We sit between them: we find structure, but the per-domain decomposition is
          weaker than Joad's headline and stronger than Lu's single axis.
        </P>
        <DataTable
          headers={["Aspect", "Lu et al. (2026)", "Joad et al. (2026)", "This work"]}
          rows={[
            [
              "Direction",
              "Single assistant axis",
              "11 category directions",
              "6 domain refusal directions",
            ],
            [
              "Identification",
              "Contrastive (assistant vs role)",
              "Refusal token labels",
              "Contrastive (refuse vs comply)",
            ],
            [
              "Control claim",
              "Single-axis capping prevents jailbreaks",
              "Shared 1D control knob",
              "Safety-selective capping (1/3 domains)",
            ],
            [
              "Models",
              "Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B",
              "Llama 3.1 8B, Qwen 2.5 7B",
              "Gemma 3 12B",
            ],
          ]}
        />
        <P style={{ marginTop: "12px" }}>
          For the closest published analogue,{" "}
          <InlineLink href="https://arxiv.org/abs/2603.13359">Alagharu et al. (2026)</InlineLink>{" "}
          extracted category-aligned directions in a Llama-3-8B fine-tuned with refusal tokens; we
          test for the same per-domain control without supervised refusal labels.
        </P>
      </Section>

      <Section style={{ marginTop: "72px" }}>
        <Label>Reproduce locally</Label>
        <H2 style={{ marginTop: "8px" }}>Run the analyses yourself</H2>
        <P>
          The full pipeline is <Mono style={{ fontSize: "13px" }}>uv</Mono>-managed. Activations,
          encodings, and refusal direction vectors (3.3 GB) are public on Hugging Face.
        </P>

        <Card style={{ padding: "24px 28px" }}>
          <div style={{ fontFamily: fonts.mono, fontSize: "13px", lineHeight: 1.7, color: palette.body }}>
            <div style={{ color: palette.muted }}># 1. Clone + install</div>
            <div>git clone https://github.com/Antonio-Tresol/gemma3-refusal-axis</div>
            <div>cd gemma3-refusal-axis && uv sync</div>
            <div style={{ color: palette.muted, marginTop: "12px" }}>
              # 2. Fetch the data (3.3 GB from Hugging Face){" "}
              <InlineLink
                href="https://huggingface.co/datasets/abotresol/gemma3-refusal-axis-data"
                style={{ color: palette.muted, fontSize: "11px" }}
              >
                HF dataset
              </InlineLink>
            </div>
            <div>uv run python scripts/cli/download_data.py</div>
            <div style={{ color: palette.muted, marginTop: "12px" }}>
              # 3. Reproduce the refusal-axis (no-SAE) analyses
            </div>
            <div>uv run python -m refusal_decomposition.analysis.refusal_axis</div>
            <div>uv run python -m refusal_decomposition.analysis.falsification</div>
            <div style={{ color: palette.muted, marginTop: "12px" }}>
              # 4. Reproduce the SAE feature-hierarchy analysis
            </div>
            <div>uv run python -m refusal_decomposition.analysis.feature_hierarchy</div>
          </div>
        </Card>
      </Section>

      <Section style={{ marginTop: "72px" }}>
        <Label>What you'll find here</Label>
        <H2 style={{ marginTop: "8px" }}>Reading guide</H2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "16px",
            marginTop: "24px",
          }}
        >
          {[
            {
              hash: "#/glossary",
              title: "Glossary",
              blurb:
                "Plain-English definitions for refusal, residual stream, SAE, JumpReLU, Matryoshka, capping, bootstrap CI, and the rest of the vocabulary.",
            },
            {
              hash: "#/refusal-axis",
              title: "Part I: Refusal Axis",
              regime: regimes.raw,
              blurb:
                "The full geometric analysis: per-domain directions, PCA, cosine matrix, and the safety-selective capping result.",
            },
            {
              hash: "#/feature-hierarchy",
              title: "Part II: Feature Hierarchy",
              regime: regimes.sae,
              blurb:
                "Three methods (decoder cosine, co-activation Jaccard, R² decomposition) all agree H1 is falsified.",
            },
            {
              hash: "#/methodology",
              title: "Methodology",
              blurb:
                "Detailed methods organised by regime, plus the falsification ledger (10 pre-registered tests, what each changed) inside §6.",
            },
            {
              hash: "#/references",
              title: "References",
              blurb:
                "Full bibliography with arXiv IDs. Includes pointers to the closest published analogues (Alagharu, Zhao, Joad) that bracket this work.",
            },
          ].map((item) => (
            <Card
              as="a"
              href={item.hash}
              key={item.hash}
              style={{
                textDecoration: "none",
                color: palette.text,
                padding: "22px 24px",
                display: "flex",
                flexDirection: "column",
                gap: "12px",
              }}
            >
              {item.regime && <RegimeBadge regime={item.regime} />}
              <h4
                style={{
                  fontFamily: fonts.display,
                  fontSize: "17px",
                  fontWeight: 600,
                  color: palette.text,
                  letterSpacing: "-0.01em",
                }}
              >
                {item.title}
              </h4>
              <P style={{ fontSize: "14.5px", marginBottom: 0, color: palette.body }}>
                {item.blurb}
              </P>
            </Card>
          ))}
        </div>
      </Section>

      <Section style={{ marginTop: "72px" }}>
        <Label>About this site</Label>
        <H2 style={{ marginTop: "8px" }}>Colophon and disclaimer</H2>
        <Callout tone="neutral">
          <strong style={{ color: palette.text }}>This is a first attempt at interpretability
          work.</strong>{" "}
          It is also an effort to see how to work with an LLM-based agent, and how to make it do{" "}
          <em>in silico</em> research reliably. I learned a lot, and I might do a write-up of the
          collaboration itself later.{" "}
          <InlineLink
            href="https://www.anthropic.com/claude-code"
            style={{ color: palette.text }}
          >
            Claude Code
          </InlineLink>{" "}
          was used heavily across the project: writing analysis code, monitoring long-running
          experiments, computing statistics, drafting reports, designing figures, and running the
          falsification loop. The full Claude harness (hooks, skills, settings) is committed to the
          GitHub repo for transparency. Things might be wrong. Honest critique, corrections, and
          methodological flags are genuinely welcome:{" "}
          <InlineLink
            href="https://github.com/Antonio-Tresol/gemma3-refusal-axis/issues"
            style={{ color: palette.text }}
          >
            open an issue
          </InlineLink>
          .
        </Callout>
      </Section>
    </>
  );
}
