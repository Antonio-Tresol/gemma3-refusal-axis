import {
  Section,
  H1,
  H2,
  H3,
  P,
  Code,
  Card,
  Figure,
  Label,
  RegimeBadge,
  InlineLink,
  Mono,
  Callout,
  DataTable,
  Term,
  MethodTerm,
  Math,
} from "../components/Primitives.jsx";
import { palette, fonts, regimes } from "../tokens.js";

const FIG_BASE = import.meta.env.BASE_URL + "figures/sae_width_scaling/";

export function FeatureHierarchyPage() {
  return (
    <>
      <Section style={{ paddingTop: "64px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
          <Label style={{ marginBottom: 0 }}>Part II</Label>
          <RegimeBadge regime={regimes.sae} />
        </div>
        <H1>Feature Hierarchy</H1>
        <P lead>
          Do refusal-relevant <Term name="sae">SAE</Term>{" "}
          <Term name="feature">features</Term> split hierarchically across{" "}
          <Term name="matryoshka">Matryoshka</Term> widths, or does the dictionary{" "}
          <em>reorganise</em> as we widen? Three independent methods all point the same way:
          H1 (hierarchical decomposition) does not survive falsification.
        </P>

        <Figure
          src={FIG_BASE + "fig2_feature_genealogy.png"}
          alt="Feature genealogy bar charts. Each row is a Matryoshka SAE width; bars show how few of the 16k baseline features survive into 65k, 262k, 1M. The majority of relevant features at the larger widths emerged later rather than splitting from 16k parents."
          label="Figure 1 · Feature genealogy across Matryoshka widths"
          regime={regimes.sae}
          caption={
            <>
              Each row is a Matryoshka SAE width (16k baseline through 1M). Bars show
              how many of the 16k baseline relevant features survive into the wider
              widths (a 16k feature counts as surviving if a wider-width feature has
              cosine above 0.5 with it). A bar near the top of its row means the wider
              dictionary is mostly inherited from 16k (decomposition / splitting); a
              bar low along the row means most relevant features at the wider width
              are <em>new</em> (replacement).
            </>
          }
        />
        <P>
          At 1M, only 28% of LPT relevant features and 41% of MRT relevant features
          survive from the 16k baseline. The remainder appears to be features present
          only at the larger Matryoshka widths, with no parent-child geometric
          relation to the 16k features.
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: visually this looks more like replacement than splitting. The three
          independent hierarchy tests in §2 say the same thing once base-rate and
          permutation checks are applied.
        </P>

        <Callout>
          <strong style={{ color: palette.text }}>The question.</strong> When the Gemma Scope 2
          dictionary widens from 16k to 65k features (via Matryoshka prefix-slicing), do broad
          refusal features <em>decompose</em> into finer sub-features? Or do disappeared features
          get <em>replaced</em> by geometrically unrelated new ones?
          <ul style={{ marginLeft: "20px", marginTop: "10px", lineHeight: 1.6 }}>
            <li>
              <strong>H1 (decomposition)</strong>: 16k features split into domain-specific 65k
              children with strong parent-child geometry.
            </li>
            <li>
              <strong>H0 (reorganisation)</strong>: disappeared features replaced by unrelated new
              features. No parent-child structure.
            </li>
          </ul>
        </Callout>
      </Section>

      <Section>
        <H2 id="methods">1 · Three methods, three thresholds</H2>
        <DataTable
          headers={["Method", "Question", "Metric", "Threshold for H1", "Source"]}
          rows={[
            [
              "M1 · Decoder cosine",
              "Are parent / child decoder vectors geometrically similar?",
              "cosine similarity",
              "> 0.5",
              <InlineLink key="b" href="https://transformer-circuits.pub/2023/monosemantic-features/">
                Bricken et al. 2023
              </InlineLink>,
            ],
            [
              "M2 · Co-activation",
              "Do they fire on the same prompts?",
              "Jaccard index",
              "> 0.5",
              <InlineLink key="ch" href="https://arxiv.org/abs/2409.14507">
                Chanin et al. 2024
              </InlineLink>,
            ],
            [
              "M3 · Hierarchy R²",
              "Can the parent be reconstructed as a weighted sum of children?",
              "R² of least-squares fit",
              "> 0.3",
              <InlineLink key="l" href="https://arxiv.org/abs/2602.11881">
                Luo et al. 2026 (HSAE)
              </InlineLink>,
            ],
          ]}
        />
        <P>
          <strong style={{ color: palette.text }}>Data.</strong> 33 disappeared features (25 LPT +
          8 MRT) and 40 new features (23 LPT + 17 MRT) across the 16k→65k transition. Decoder
          vectors come from the 1M SAE (shared across prefix widths). Co-activation computed from
          104 prompts × 1M SAE encodings.
        </P>
      </Section>

      <Section>
        <H2 id="results">2 · Results</H2>
        <H3>2.1 · M1 · <MethodTerm name="decoder-cosine-method">Decoder cosine</MethodTerm></H3>
        <DataTable
          headers={["Site", "Parents", "Max cos", "Mean max cos", "Any > 0.5", "Any > 0.3"]}
          rows={[
            ["LPT", "25", "0.287", "0.142", "0", "0"],
            ["MRT", "8", "0.268", "0.150", "0", "0"],
          ]}
        />
        <P>
          <strong style={{ color: palette.text }}>No apparent parent-child geometry.</strong>{" "}
          The highest cosine across all 575 LPT pairs is 0.287, well below the 0.3 threshold
          and far below the 0.5 hierarchy threshold. Expected max cosine between random unit
          vectors in <Math>{String.raw`\mathbb{R}^{3840}`}</Math> with 23 comparisons:{" "}
          <strong>0.032</strong>. The
          observed 0.287 is ~9× above random; there appears to be weak shared-subspace
          structure, just not parent-child.
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: M1 fails for both extraction sites. Disappeared 16k features and
          new 65k features sit at near-random angles relative to one another;
          whatever shared-subspace structure is present is well below the parent-child
          cutoff.
        </P>

        <H3>2.2 · M2 · <MethodTerm name="coactivation-jaccard">Co-activation Jaccard</MethodTerm></H3>
        <DataTable
          headers={["Site", "Parents", "Max Jaccard", "Mean max", "Any > 0.5", "Any > 0.3"]}
          rows={[
            ["LPT", "25", "0.961", "0.241", "1", "2"],
            ["MRT", "8", "1.000", "0.150", "1", "1"],
          ]}
        />
        <P>
          Two apparent high-Jaccard pairs emerge. Both fail robustness checks.
        </P>
        <Card style={{ padding: "20px 24px", margin: "20px 0" }}>
          <Label>Robustness check 1 · <MethodTerm name="permutation-null">permutation null</MethodTerm></Label>
          <DataTable
            headers={["Pair", "Observed Jaccard", "p-value", "Significant?"]}
            rows={[
              ["LPT 9449→20318", "0.961", "0.076", "No"],
              ["LPT 11090→24339", "0.375", "0.000", "Yes"],
              ["MRT 8315→23735", "1.000", "0.010", "Yes"],
            ]}
          />
          <Label style={{ marginTop: "16px" }}>Robustness check 2 · base-rate analysis</Label>
          <DataTable
            headers={[
              "Pair",
              "Parent rate",
              "Child rate",
              "E[Jaccard | independent]",
              "Observed",
            ]}
            rows={[
              ["LPT 9449→20318", "97.1%", "97.1%", "0.944", "0.961"],
              ["MRT 8315→23735", "1.0%", "1.0%", "0.005", "1.000"],
            ]}
          />
        </Card>
        <P>
          The high LPT pair (9449→20318) is a <strong>base-rate artefact</strong>: both features
          fire on 97.1% of positive prompts. Any two ~97% features have expected Jaccard ≈ 0.944
          under independence; the observed 0.961 is only 0.017 above chance, and the permutation
          test gives <em>p</em> = 0.076 (above the 0.05 threshold). The MRT pair (8315→23735) is
          statistically significant (<em>p</em> = 0.010) but based on a single co-firing prompt out
          of 104; too small for a mechanistic claim.
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: the one feature that <em>looked</em> like a parent fires on 97% of prompts;
          it appears to be a near-constant rather than a refusal sub-concept. The apparent
          hierarchy seems to disappear once you control for how often each feature fires.
        </P>

        <H3>2.3 · M3 · <MethodTerm name="r2-decomposition">Hierarchy R²</MethodTerm></H3>
        <DataTable
          headers={["Site", "Parents", "Max R²", "Mean R²", "Any > 0.5", "Any > 0.3"]}
          rows={[
            ["LPT", "25", "0.580", "0.092", "1", "1"],
            ["MRT", "8", "n/a", "0.056", "0", "0"],
          ]}
        />
        <P>
          One LPT parent (feature 9449) shows R² = 0.580; but this is the <em>same</em> feature
          flagged by M2 and shown to be a base-rate artefact. Mean R² across all parents is 0.09
          (LPT) and 0.06 (MRT).
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: M3 fails too. The single high-R² hit collapses to the same base-rate
          artefact already disqualified in §2.2; the rest of the parents sit close to
          zero.
        </P>
      </Section>

      <Section>
        <H2 id="other-transitions">3 · Other Matryoshka transitions</H2>
        <DataTable
          headers={["Transition", "Site", "Disappeared", "New", "Max cos", "> 0.3", "> 0.5"]}
          rows={[
            ["65k → 262k", "LPT", "3", "10", "0.256", "0", "0"],
            ["65k → 262k", "MRT", "2", "5", "0.098", "0", "0"],
            ["262k → 1M", "LPT", "1", "1", "0.028", "0", "0"],
            ["262k → 1M", "MRT", "0", "0", "n/a", "n/a", "n/a"],
          ]}
        />
        <P>
          <strong style={{ color: palette.text }}>No hierarchy at any width transition.</strong>{" "}
          The 65k→262k transition shows max cosines comparable to or lower than 16k→65k. The
          262k→1M transition shows near-complete stabilisation: a single LPT pair with cosine 0.028
          (effectively random) and zero changes for MRT.
        </P>

        <Figure
          src={FIG_BASE + "fig1_width_scaling.png"}
          label="Figure 2 · Width scaling curves"
          regime={regimes.sae}
          caption="Refusal-relevant feature counts and effect sizes across Matryoshka prefix widths {16k, 65k, 262k, 1M}. Each panel plots a different summary against width on the x-axis; a curve that climbs and then flattens means the dictionary keeps adding refusal-relevant features up to some width, then saturates."
        />
        <P>
          Most of the growth happens at the 16k → 65k transition; effect-size and
          count both flatten by 262k, and 262k → 1M is essentially flat (one LPT pair
          changes, MRT does not change at all).
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: the dictionary looks stable from 262k onward for refusal-relevant
          features. Adding more capacity beyond that does not seem to recruit
          additional refusal-relevant features at either extraction site.
        </P>

        <Figure
          src={FIG_BASE + "fig3_domain_specificity.png"}
          label="Figure 3 · Domain specificity at each width"
          regime={regimes.sae}
          caption="Distribution of refusal-relevant features by sub-type at each Matryoshka width. Bars stack so the column height is the total relevant-feature count; the segments show how that total breaks down across sub-types."
        />
        <P>
          At 16k, the relevant features tend to be domain-broad (most categorise as
          general refusal rather than a specific sub-type). At 262k and 1M, the same
          totals include more sub-type-specific features. The new sub-type features
          at the larger widths do not have the geometric or co-activation signature
          of children-of-16k-parents (§2).
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: domain specificity appears to emerge with width, but it seems to
          come from new features rather than from splitting of 16k parents.
          Specificity-via-replacement, not specificity-via-decomposition.
        </P>
      </Section>

      <Section>
        <H2 id="conclusions">4 · Conclusions</H2>
        <Callout>
          <strong style={{ color: palette.text }}>The two regimes give different views.</strong>{" "}
          Part I (raw activation space) shows refusal living in a structured low-dimensional
          subspace with distinguishable per-domain directions, and capping along the safety
          direction is selective on a 0–100 trait scale (<InlineLink href="#/refusal-axis#capping">§4.1 of Part I</InlineLink>).
          Part II (SAE feature space) shows that the same activations do not recover that
          per-domain structure as a parent-child hierarchy across Matryoshka widths. The two
          answers do not have to agree: raw activations appear to encode per-domain distinctions
          that predict refusal behaviour, while the SAE dictionary does not appear to decompose
          refusal along those same directions. That's a finding, not a contradiction; it says
          the lens you put on the activations matters.
        </Callout>
        <P>
          <strong style={{ color: palette.text }}>What we can claim (after falsification).</strong>
        </P>
        <ul style={{ marginLeft: "20px", lineHeight: 1.7, fontFamily: fonts.body, fontSize: "16px", color: palette.body }}>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>H1 (hierarchical decomposition) does not survive falsification</strong>{" "}
            for the 16k→65k Matryoshka transition. All three pre-registered methods agree:
            no apparent parent-child geometric structure, no co-activation pattern, and no R²
            decomposition that survives base-rate and permutation checks.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>The dictionary appears to reorganise across widths</strong>{" "}
            rather than decompose: disappeared features look replaced by geometrically
            unrelated new ones.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Domain specificity seems to emerge</strong>,
            but via replacement at finer widths, not via splitting of coarser parents.
          </li>
        </ul>

        <P style={{ marginTop: "20px" }}>
          <strong style={{ color: palette.text }}>Caveats.</strong>
        </P>
        <ul style={{ marginLeft: "20px", lineHeight: 1.7, fontFamily: fonts.body, fontSize: "16px", color: palette.body }}>
          <li style={{ marginBottom: "8px" }}>
            33 disappeared LPT + 8 disappeared MRT parents is small. With more refusal-relevant
            parents, a rare splitting case might emerge.
          </li>
          <li style={{ marginBottom: "8px" }}>
            Matryoshka prefix-slicing the 1M SAE is not the same as training an independent 64k
            SAE. Parent-child geometry might appear across independently-trained
            widths.
          </li>
          <li style={{ marginBottom: "8px" }}>
            Tested only on Gemma 3 12B at layer 41. Generalisation to other models / layers
            unverified.
          </li>
        </ul>

      </Section>

      <Section>
        <H2 id="future-work">5 · What's next</H2>
        <P>
          The negative result here is sharper than the positive one in Part I, but it lives
          inside a specific test: one model, one layer, one SAE family, the Matryoshka
          prefix-slicing construction. Each of those is a condition under which H1
          (hierarchical decomposition) could still hold while our test fails to detect it.
        </P>
        <ul style={{ marginLeft: "20px", lineHeight: 1.7, fontFamily: fonts.body, fontSize: "16px", color: palette.body }}>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Independently trained SAE widths.</strong>{" "}
            Matryoshka prefix-slicing shares encoder weights across {"{"}16k, 65k, 262k, 1M{"}"}
            ; an independently trained 65k SAE on the same activations might learn a
            different decomposition. If parent-child geometry only emerges across separately
            trained SAEs, our Matryoshka-only test would miss it. Re-running M1–M3 with two
            independently trained SAEs would close this.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Larger refusal-relevant parent pool.</strong>{" "}
            Only 33 LPT + 8 MRT relevant 16k features. Weak hierarchy may exist among rare
            sub-types we don't have enough power to see. A larger contrastive prompt set would
            increase the relevant-parent pool and let rare splitting cases surface above the
            base-rate noise floor.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Other Gemma Scope 2 layers.</strong>{" "}
            Tested at layer 41 (85% depth). Gemma Scope 2 trains expanded-width SAEs at layers{" "}
            {"{"}12, 24, 31, 41{"}"} of the 12B model. Parent-child structure may be
            detectable at mid-depth (layer 24, ~50%), where steering is empirically more
            effective and refusal may be encoded differently in the residual stream.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Other behaviours beyond refusal.</strong>{" "}
            We tested whether <em>refusal</em> features split across widths. The same
            replacement-not-splitting pattern may hold for honesty, sycophancy, deception,
            instruction-following, etc., or may not. Replicating the M1–M3 pipeline on a
            non-refusal behaviour set would tell us whether reorganisation-not-decomposition
            is a refusal property or a Matryoshka SAE property.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Cross-model SAE comparison.</strong> Gemma
            Scope (Gemma 2) vs Gemma Scope 2 (Gemma 3) at comparable layers / widths. If the
            same refusal features are present and structured similarly, the dictionary
            organisation is reproducible across model generations; if not, it's a per-training
            artefact.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>SAE-feature causal interventions.</strong>{" "}
            Steering through the SAE dictionary by toggling individual relevant features at
            different widths. If turning off a 16k "general refusal" feature has the same
            downstream effect as turning off the 1M descendants we'd expect under H1, we'd
            recover hierarchy by behaviour even though the geometry doesn't show it.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Reconcile with Part I.</strong> Part I
            finds per-domain structure in raw activations. Part II finds none in the SAE
            feature space. A focused study of the disconnect (do per-domain raw directions
            project onto specific clusters of SAE features that we miss because they're not
            individually flagged as refusal-relevant?) could explain both observations
            simultaneously.
          </li>
        </ul>
      </Section>

      <Section style={{ marginTop: "64px" }}>
        <Mono>
          <InlineLink href="#/refusal-axis" arrow="left" style={{ color: palette.text }}>
            Back to Part I: Refusal axis
          </InlineLink>
          {"   |   "}
          <InlineLink href="#/methodology" arrow="right" style={{ color: palette.text }}>
            Continue to Methodology
          </InlineLink>
        </Mono>
      </Section>
    </>
  );
}
