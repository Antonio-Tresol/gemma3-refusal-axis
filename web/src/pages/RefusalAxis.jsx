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
  Math,
  CodeTabs,
  MethodTerm,
  Term,
} from "../components/Primitives.jsx";
import { palette, fonts, regimes } from "../tokens.js";
import { CosineHeatmap } from "../components/CosineHeatmap.jsx";
import { CappingMatrix } from "../components/CappingMatrix.jsx";

const FIG_BASE = import.meta.env.BASE_URL + "figures/refusal_axis/";

export function RefusalAxisPage() {
  return (
    <>
      <Section style={{ paddingTop: "64px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
          <Label style={{ marginBottom: 0 }}>Part I</Label>
          <RegimeBadge regime={regimes.raw} />
        </div>
        <H1>The Refusal Axis</H1>
        <P lead>
          A geometric analysis of refusal in the residual stream of Gemma 3 12B at layer 41. No
          SAE: every cosine, projection, and capping intervention happens directly in the 3,840-dim
          activation space.
        </P>

        <Figure
          src={FIG_BASE + "fig_hero_3d_refusal_axis.png"}
          alt="3D PCA of refusal activation differences"
          label="Figure 1 · The refusal axis in domain space"
          regime={regimes.raw}
          caption="128 contrastive activation differences projected into PC1/PC2/PC3. Colour encodes projection onto the mean refusal axis (red = compliance, navy = refusal). Diamonds mark domain centroids; the dashed line is the mean refusal axis. Analogous to Lu et al. (2026, 'The Assistant Axis,' arXiv:2601.10387) Fig 1."
        />
      </Section>

      <Section>
        <H2 id="background">1 · Background</H2>
        <P>
          <InlineLink href="https://arxiv.org/abs/2406.11717">
            Arditi et al. (2024)
          </InlineLink>{" "}
          showed that refusal in 13 open-source chat models (up to 72B parameters) is mediated by a
          one-dimensional subspace; a single direction whose removal disables refusal.{" "}
          <InlineLink href="https://arxiv.org/abs/2601.10387">Lu et al. (2026)</InlineLink>{" "}
          generalised the construction with the &quot;Assistant Axis&quot; and the soft-clamping{" "}
          <em>activation capping</em> intervention used here.
        </P>
        <P>
          More recent work has extended the rank-1 picture in two complementary directions.{" "}
          <InlineLink href="https://arxiv.org/abs/2502.17420">
            Wollschläger et al. (2025, ICML)
          </InlineLink>{" "}
          formalised refusal as a <Term name="concept-cone">concept cone</Term>: a
          multi-dimensional subspace (up to dim 5 in the models tested) where any
          internal vector induces refusal, with a property they call{" "}
          <em>representational independence</em>.{" "}
          <InlineLink href="https://arxiv.org/abs/2507.11878">
            Zhao et al. (2025, NeurIPS)
          </InlineLink>{" "}
          decomposed the picture further, finding that{" "}
          <Term name="harmfulness">harmfulness</Term> (encoded at the last user-token)
          and <em>refusal</em> (encoded at the last sequence-token) are two
          distinguishable directions.
        </P>
        <P>
          But{" "}
          <InlineLink href="https://arxiv.org/abs/2602.02132">Joad et al. (2026)</InlineLink>{" "}
          found that across 11 refusal categories, refusal behaviours correspond to{" "}
          <em>geometrically distinct</em> directions, yet steering along any of them produces
          nearly identical trade-offs; a &quot;shared one-dimensional control knob.&quot;{" "}
          <InlineLink href="https://arxiv.org/abs/2603.13359">Alagharu et al. (2026)</InlineLink>{" "}
          extracted category-aligned directions in a Llama-3-8B fine-tuned with refusal tokens, with
          per-category steering. This work tests, with unsupervised contrastive directions on
          Gemma 3 12B, whether such per-domain control is real.
        </P>

        <Callout>
          <strong style={{ color: palette.text }}>The question.</strong> Does the refusal component
          of the assistant persona decompose into geometrically distinct sub-types? And if so, can
          we cap them <em>independently</em>, relaxing capability refusal while keeping
          safety refusal strict?
        </Callout>
      </Section>

      <Section>
        <H2 id="construction">2 · Refusal axis construction</H2>
        <H3>2.1 · Formula</H3>
        <P>
          Following Lu et al. Sec 3.1, the refusal axis is the mean activation difference between
          the prompts the model refuses and the matched prompts it answers:
        </P>
        <CodeTabs
          label="Refusal axis"
          math={String.raw`v_{\text{refusal}} \;=\; \frac{1}{|P|}\sum_{x \in P} a(x) \;-\; \frac{1}{|N|}\sum_{x \in N} a(x)`}
          python={`# v_refusal: ndarray, shape (d_model,)
positive_acts = activations[positive_prompts]   # shape (|P|, d_model)
negative_acts = activations[negative_prompts]   # shape (|N|, d_model)
v_refusal = positive_acts.mean(axis=0) - negative_acts.mean(axis=0)
v_hat = v_refusal / np.linalg.norm(v_refusal)`}
        />
        <P>
          where <em>P</em> is the set of prompts the model refuses (refusal score &gt; 50) and{" "}
          <em>N</em> is the set of matched benign prompts the model answers (score &lt; 30).
          Activations <Math>a(x)</Math> are extracted at the{" "}
          <strong><MethodTerm name="mean-response-token">mean-response-token</MethodTerm></strong>{" "}
          site, averaged across all assistant response tokens at layer 41.{" "}
          <InlineLink href="https://arxiv.org/abs/2507.21509">
            Chen et al. (2025) Sec 2.2 (Persona Vectors)
          </InlineLink>{" "}
          uses the same site for behavioural analysis.
        </P>

        <H3>2.2 · Data</H3>
        <P>
          <strong>128 retained pairs</strong> across 6 domains, filtered down from 280 candidate
          pairs by three independent passes of{" "}
          <MethodTerm name="trait-scoring">trait scoring</MethodTerm>.
        </P>
        <Callout>
          <strong style={{ color: palette.text }}>What is a pair?</strong> Each pair is one{" "}
          <em>positive prompt</em> (something we expect the model to refuse, e.g. &quot;Walk me
          through synthesising methamphetamine&quot;) and one matched <em>negative prompt</em> (the
          same domain, similar surface form, but benign, e.g. &quot;Walk me through synthesising
          aspirin&quot;) that the model answers. The contrast cancels out everything except the
          refusal-relevant signal, which is what the difference of means then isolates.
        </Callout>
        <P>
          <strong style={{ color: palette.text }}>Where the 280 candidates came from.</strong>{" "}
          All 280 contrastive pairs were generated by{" "}
          <InlineLink href="https://www.anthropic.com/claude/opus">
            Claude Opus 4.6
          </InlineLink>
          . The initial 240 (40 per domain) were produced by Claude Code from a specification
          prompt containing the six domain definitions, the matching rules (same topic, similar
          surface form, only the refusal-relevant dimension differs), and two human-written example
          pairs per domain. Early trait scoring then revealed thin coverage in three domains
          (ethical, privacy, identity_boundary) where Gemma 3 tends to comply-with-disclaimers
          rather than cleanly refuse, so 40 supplementary pairs targeted at those failure modes
          were generated via the Claude Agent SDK with explicit per-domain guidance. Pairs were
          then put through a vocabulary-confound check (no content word in &gt;40% of one condition
          and &lt;10% of the other) and a 60-pair stratified human review (10 per domain,{" "}
          <em>seed=42</em>; 60/60 passed). Trait scoring of Gemma's responses is then Claude
          Sonnet 4.6 via the Claude Agent SDK: three independent passes rating each response 0–100
          for refusal expression, taking the median (inter-pass Pearson 0.990–0.991). The
          rubric follows{" "}
          <InlineLink href="https://arxiv.org/abs/2507.21509">
            Chen et al. (2025) Sec 2.2 (Persona Vectors)
          </InlineLink>
          ; the three-pass calibration is our extension. Chen et al. uses single-pass scoring;
          we added independent passes to surface inter-pass disagreement before retention. A
          pair is <strong>retained</strong> if its positive scored above 50 (the model did
          refuse) and its negative scored below 30 (the model did not refuse). 152 candidates were
          dropped for failing one of those gates. Gemma 3 12B's role in this pipeline is the
          subject under study; it produces the responses, not the pairs and not the scores.
        </P>
        <DataTable
          headers={["Domain", "n", "Examples"]}
          rows={[
            ["Safety", "31", "Physical harm, weapons, drugs"],
            ["Ethical", "30", "Manipulation, exploitation"],
            [
              "Capability boundary",
              "22",
              "Tasks the model cannot perform without external tools (e.g. browsing the live internet without a web-search tool, executing code without a sandbox)",
            ],
            ["Privacy", "21", "Personal data, surveillance"],
            ["Legal", "20", "Fraud, hacking"],
            [
              "Identity boundary",
              "4",
              "Things the model is not (e.g. claims of being conscious, sentient, having feelings, or being human)",
            ],
          ]}
        />

        <H3>2.3 · Separability of the mean axis</H3>
        <DataTable
          headers={["Metric", "Last prompt token", "Mean response token"]}
          rows={[
            ["Cohen's d", "3.25", "1.92"],
            ["Classification accuracy", "93.0%", "84.8%"],
            ["PCA dims for 70%", "23", "11"],
            ["PC1 alignment with refusal dir", "0.415", "0.449"],
          ]}
        />
        <P>
          The refusal axis cleanly separates refusing from compliant responses. But it is{" "}
          <strong>not one-dimensional</strong>: 11 dimensions are needed for 70% variance at the
          mean-response-token site, against a median of 80 for random vectors in{" "}
          <Math>{String.raw`\mathbb{R}^{3840}`}</Math> (one-sided{" "}
          <MethodTerm name="permutation-null">permutation</MethodTerm>, <em>p</em> &lt; 0.001;
          0/1000 permutations matched).
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: refusal is encoded in a structured low-dimensional{" "}
          <Term name="subspace">subspace</Term> of the residual stream (11 dims for
          70% variance vs random median 80, <em>p</em> &lt; 0.001), and that subspace
          looks to have internal structure rather than collapsing to one direction. The
          original single-direction (rank-1) model of{" "}
          <InlineLink href="https://arxiv.org/abs/2406.11717">Arditi et al. (2024)</InlineLink>{" "}
          seems insufficient here, consistent with{" "}
          <InlineLink href="https://arxiv.org/abs/2502.17420">
            Wollschläger et al. (2025)
          </InlineLink>{" "}
          finding refusal-mediating concept cones up to dim 5; an 11-dimensional
          subspace looks like the appropriate target for decomposition.
        </P>
      </Section>

      <Section>
        <H2 id="decomposition">3 · Domain decomposition</H2>
        <P>
          For each domain <em>D</em> with at least 3 retained pairs, the per-domain refusal
          direction is <Code>mean(positive[D]) − mean(negative[D])</Code>. We then ask three
          questions: how each domain direction aligns with the mean refusal axis (§3.1), how
          the domains relate to each other pairwise (§3.2 cosine matrix), and how many
          independent dimensions the whole refusal subspace occupies (§3.3 PCA). Each
          subsection leads with its figure; the explanation follows underneath.
        </P>

        <H3>3.1 · Domain alignment with the mean axis</H3>
        <Figure
          src={FIG_BASE + "fig_d_domain_loading.png"}
          label="Figure 2 · Domain loadings on the mean refusal axis"
          regime={regimes.raw}
          caption="Cosine similarity of each domain's refusal direction with the mean refusal axis (built from all retained pairs together). A loading near 1 means the domain pushes the model in roughly the same direction as the overall refusal signal; near 0 means it is nearly orthogonal to the average. The 0.5 line is a useful rule of thumb. Bars sorted high to low: safety / ethical 0.91, legal 0.89, privacy 0.70, identity 0.58, capability 0.38."
        />
        <P>
          The four value-based domains (safety, ethical, legal, privacy) load between
          0.70 and 0.91 on the mean axis, pointing in roughly the same direction as the
          overall refusal signal. Identity_boundary loads at 0.58 (n = 4, treat as
          unstable). Capability_boundary loads at <strong>0.38</strong>, clearly below
          0.5 and well below the value-based cluster.
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: capability refusal appears to point in a different direction from
          value-based refusal even before we look at any cross-domain comparison. This
          gap (0.38 vs ≥ 0.70) is what motivates testing whether capping the safety
          direction leaves capability refusal intact in §4.
        </P>

        <H3>3.2 · Cross-domain cosine matrix (interactive)</H3>
        <P>
          Each cell is the cosine similarity between two per-domain refusal directions.
          A cosine of 1.0 means the two domains push the model along the same direction
          in activation space; 0.0 means the directions are independent (so a
          31.6-point safety cap should leave the other domain untouched); −1.0 means
          opposite. The diagonal is 1.0 by construction. Hover any cell for the cosine
          value and its <MethodTerm name="bootstrap">95% bootstrap CI</MethodTerm> (a
          wide CI, e.g. one that crosses 0.5, means the data is too small to tell apart
          &quot;near-orthogonal&quot; from &quot;moderately aligned&quot; for that
          pair); click a cell for the falsification ledger entry that targets it.
        </P>
        <CosineHeatmap />
        <P>
          Three of the four value-based domains (safety, ethical, legal) form a tight
          block: cosines 0.73 to 0.87, point estimates well above 0.5. Capability, the
          one non-value-based refusal type tested, has the lowest pairwise cosines with
          the value-based block (0.14 to safety, 0.14 to ethical), but its bootstrap CI
          versus safety is <Code>[−0.316, 0.639]</Code>, wide enough that the cosine
          could be anywhere from opposite-direction to moderately aligned.
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: the per-domain directions for value-based refusal (safety / ethical /
          legal) seem to point along approximately the same axis. Capability looks
          distinguishable from them, but the small-sample CIs mean we cannot say its
          direction is independent of safety in the strict sense.
        </P>
        <Callout>
          <strong style={{ color: palette.text }}>Small-n caveat.</strong> Identity
          (n = 4) and capability (n = 22) produce unstable per-domain directions. When
          safety is artificially subsampled to n = 4, its cosine with the mean axis ranges
          from 0.47 to 0.88 across resamples (falsification test 1c). Treat any pairwise
          claim about a domain with fewer than ~20 retained pairs as suggestive.
        </Callout>

        <H3>3.3 · PCA of activation differences</H3>
        <Figure
          src={FIG_BASE + "fig_c_pca.png"}
          label="Figure 3 · PCA of refusal space"
          regime={regimes.raw}
          caption={
            <>
              (a) Cumulative variance versus PC count. The lower the curve crosses
              70%, the lower-dimensional refusal is; random vectors in{" "}
              <Math>{String.raw`\mathbb{R}^{3840}`}</Math> need a median of 80 PCs
              for 70% (one-sided permutation, <em>p</em> &lt; 0.001), comparable to
              Lu et al.'s 4–19 for persona space. (b) 2D PCA scatter of all 128
              retained pairs with 1-σ confidence ellipses per domain; if domains
              shared one direction the points would overlap into a single cloud,
              if not you would see per-domain clusters. PC1 captures 54.8% of
              variance, PC2 captures 3.8%.
            </>
          }
        />
        <P>
          Panel (a) crosses 70% variance at 11{" "}
          <Term name="pca">PCs</Term>, well below the random-vector median of 80;
          refusal occupies a structured low-dimensional{" "}
          <Term name="subspace">subspace</Term>. Panel (b) shows capability_boundary
          points (light blue) sitting in a separate region of the PC1 / PC2 plane from
          the value-based domains, with two long-tail outliers along PC1. The other
          domains overlap considerably in this 2D slice (most of the per-domain
          separation lives in PCs beyond the top two).
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: the dimensionality result is robust to the permutation null (11 PCs
          for 70% variance vs random median 80, <em>p</em> &lt; 0.001); refusal
          occupies a structured low-dimensional subspace, not a scattered noise pattern.
          The 2D scatter is a partial view of an 11-dimensional object, so per-domain
          clusters being only partly visible is expected; capability still looks
          distinguishable from value-based refusal in this slice.
        </P>
      </Section>

      <Section>
        <H2 id="capping">4 · Domain-selective capping</H2>
        <P>
          We apply Lu et al.'s activation capping per domain direction. For any token activation{" "}
          <Math>a</Math>, measure its projection onto the unit refusal direction <Math>{String.raw`\hat{v}`}</Math>;
          if that projection exceeds a threshold <Math>{String.raw`\tau`}</Math>, subtract the excess
          along <Math>{String.raw`\hat{v}`}</Math>. The benign-side projection is never amplified.
        </P>
        <CodeTabs
          label="Activation capping at layer 36"
          math={String.raw`a' \;=\; a \;-\; \max\bigl(\hat{v} \cdot a - \tau,\; 0\bigr)\,\hat{v}`}
          python={`# Lu et al. 2026 steering.py, intervention_type="capping"
proj   = einsum('bld,d->bl', activations, v_hat)   # scalar projection
excess = (proj - tau).clamp(min=0)                 # only positive overflow
activations -= einsum('bl,d->bld', excess, v_hat)  # subtract along v_hat`}
        />
        <P>
          Applied at{" "}
          <strong><MethodTerm name="layer-36">layer 36</MethodTerm></strong>{" "}
          (75% depth). Layer 36 is the steering layer (distinct from layer 41, where
          activations are extracted). Our M5 steering experiments found this optimal,
          consistent with{" "}
          <InlineLink href="https://arxiv.org/abs/2312.06681">
            Panickssery et al. (2024) (CAA)
          </InlineLink>{" "}
          finding ~40% depth optimal for Llama 2.
        </P>

        <H3>4.1 · Independence matrix (interactive)</H3>
        <P>
          Each cell shows the mean change in{" "}
          <MethodTerm name="trait-scoring">refusal score</MethodTerm> relative to baseline
          (Sonnet 4.6 trait-score, 0–100 scale; negative = less refusal). A good
          domain-selective cap shows a large negative number on the diagonal (target domain)
          and near-zero off-diagonal. Drag the τ slider to sweep percentiles.
        </P>

        <CappingMatrix />

        <P style={{ marginTop: "20px" }}>
          Capping the safety direction at τ = p<sub>50</sub> drops safety refusal by
          31.6 points on the 0–100 trait-score scale; capability, privacy, and benign
          all stay within ±1.3. The overall (mean-axis) cap drops safety by 25 but also
          drops privacy by 15.9, less selective by comparison. Neither capability nor
          privacy capping shows a selective effect at this layer.
        </P>
        <P style={{ fontStyle: "italic", color: palette.text, marginTop: "-4px" }}>
          Reading: at this τ and layer, safety capping looks independent of capability,
          privacy, and benign behaviour, while the mean-axis cap is visibly less
          selective. Capability and privacy capping show no detectable effect at layer
          36; whether that reflects a layer mismatch or an absence of an isolated
          direction is open.
        </P>

        <Callout tone="warn">
          <strong style={{ color: palette.text }}>Exploratory caveat.</strong> τ = p<sub>50</sub>{" "}
          was chosen by sweeping 7 percentiles across 4 directions and 4 prompt domains (112
          cells). At <em>p</em> = 0.05, ~6 cells would appear significant by chance (test 4e). The
          p<sub>50</sub> sweet spot is hypothesis-generating, not confirmatory; pre-registered
          replication needed.
        </Callout>
      </Section>

      <Section>
        <H2 id="interpretation">5 · Interpretation</H2>
        <H3>5.1 · Why safety capping works selectively</H3>
        <P>
          Refusal-prompt activations align{" "}
          <strong>60.9× more strongly</strong> with the safety direction than with random
          directions on average (test 4a, ratio of mean projection magnitudes); the effect
          is direction-specific, not a generic activation perturbation.
          Safety's high loading on the mean axis (0.91, stable under{" "}
          <MethodTerm name="leave-one-out">leave-one-out</MethodTerm>) means capping
          along it primarily affects the value-based refusal component without disturbing
          capability or benign responses.
        </P>

        <H3>5.2 · Why capability and privacy capping don't work</H3>
        <P>Two non-mutually-exclusive explanations:</P>
        <ol style={{ marginLeft: "20px", lineHeight: 1.75, fontFamily: fonts.body, fontSize: "16px", color: palette.body }}>
          <li style={{ marginBottom: "12px" }}>
            <strong style={{ color: palette.text }}>Capability refusal is too weak.</strong>{" "}
            Baseline capability refusal is 34.7/100; the model often complies with capability
            requests anyway (e.g., fabricating code execution output). There isn't much refusal to
            suppress.
          </li>
          <li>
            <strong style={{ color: palette.text }}>
              Privacy may need different layer targeting.
            </strong>{" "}
            Capping operates at layer 36 (the steering layer, distinct from layer 41 where
            activations are extracted). Lu et al. found that optimal steering layers vary by
            behaviour type. We did not sweep layers per domain.
          </li>
        </ol>

        <H3>5.3 · Comparison with prior work</H3>
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
      </Section>

      <Section>
        <H2 id="limitations">6 · Limitations</H2>
        <ol style={{ marginLeft: "20px", lineHeight: 1.7, fontFamily: fonts.body, fontSize: "16px", color: palette.body }}>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>
              The per-domain decomposition is suggestive, not confirmatory.
            </strong>{" "}
            Two independent statistical checks agree the data points the right way but does not
            settle the question. (i) The <em>cosine range</em> across the six domain refusal
            directions (the spread between the most-aligned and most-orthogonal pair) is only
            borderline larger than the spread you get if you randomly relabel which prompts
            belong to which domain (one-sided permutation test, <em>p</em> = 0.054). At <em>p</em> = 0.05
            it would clear the bar; at 0.054 it does not, by a hair. (ii) For the
            safety↔capability pair specifically, the 95%{" "}
            <Term name="bootstrap-ci">bootstrap CI</Term> on the cosine spans{" "}
            <Mono>[−0.316, 0.639]</Mono>: resampling 128 pairs with replacement gives such a
            wide range that we cannot distinguish "the two directions are nearly orthogonal"
            from "the two directions are moderately aligned". Treat the per-domain
            decomposition as an observation that needs a larger pre-registered replication,
            not a settled fact about how the refusal-relevant directions in Gemma 3 12B
            decompose by domain.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Only safety capping shows selectivity.</strong>{" "}
            Capability and privacy capping show no selective effect at layer 36.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Small samples throughout.</strong> 128 retained
            pairs total, n = 4 to 31 per domain. 10 test prompts per domain for capping evaluation.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>τ = p<sub>50</sub> is exploratory.</strong> 112
            cells tested, none pre-registered.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Clustering is method-dependent.</strong> Ward's
            and average linkage separate capability; complete and single produce different
            orderings.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Single capping layer (36).</strong> Different
            domains may benefit from different layers.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Single-turn only.</strong> No multi-turn drift
            evaluation.
          </li>
        </ol>
        <P style={{ marginTop: "12px" }}>
          For the full pre-registered falsification suite (10 tests, what changed in each claim
          after running them), see{" "}
          <InlineLink href="#/methodology#falsification-ledger">
            the falsification ledger in Methodology §6
          </InlineLink>
          .
        </P>
      </Section>

      <Section>
        <H2 id="practical">7 · Practical implications</H2>
        <P>
          <strong style={{ color: palette.text }}>For deployment.</strong> A system that needs to
          relax one type of refusal while maintaining others can use domain-selective capping. An
          AI agent framework can cap the capability direction (if effective at a different layer)
          to stop the model saying &quot;I can't execute code,&quot; while keeping safety refusal
          intact. A medical AI can cap the general refusal direction to be more forthcoming with
          clinical information while keeping the privacy direction uncapped.
        </P>
        <P>
          <strong style={{ color: palette.text }}>For interpretability research.</strong>{" "}
          Value-based and capability refusal load very differently on the mean refusal axis (0.91
          vs 0.38), suggesting their per-domain directions are distinguishable. The pairwise cosine
          has a wide bootstrap CI, so
          the degree of independence is uncertain. If confirmed at scale, different refusal types
          may need to be trained and evaluated separately.{" "}
          <InlineLink href="https://arxiv.org/abs/2411.11296">
            O'Brien et al. (2024)
          </InlineLink>{" "}
          showed SAE-based refusal steering causes systematic capability degradation; a follow-up
          should test whether per-domain capping inherits or escapes this.
        </P>
      </Section>

      <Section>
        <H2 id="future-work">8 · What's next</H2>
        <P>
          Part I results are suggestive but exploratory: a structured low-dimensional
          subspace appears present and safety-direction capping appears selective, but every
          design choice (τ, layer, sample size, single turn, single model) is a free
          parameter whose perturbation could reverse the conclusion. The most useful
          follow-ups address those free parameters, with the cheapest-to-falsify moves
          first.
        </P>
        <ul style={{ marginLeft: "20px", lineHeight: 1.7, fontFamily: fonts.body, fontSize: "16px", color: palette.body }}>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Pre-registered safety-capping replication.</strong>{" "}
            Re-run the safety-capping experiment with the operating point pre-committed
            and a held-out prompt set. This is the cheapest way to upgrade the τ = p
            <sub>50</sub> result from hypothesis-generating to confirmatory.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Larger pair set for tighter CIs.</strong>{" "}
            The 95% bootstrap CI for safety↔capability cosine spans{" "}
            <Mono>[−0.316, 0.639]</Mono> with 128 pairs. Roughly 5× more matched pairs would
            be expected to narrow it enough to distinguish near-orthogonal from moderately
            aligned. Identity boundary at <em>n</em> = 4 is also begging for more pairs.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Per-domain layer sweep for capping.</strong>{" "}
            Capability and privacy capping showed no selective effect at layer 36, the steering
            layer that worked for safety. Lu et al. (2026) found that optimal steering layers
            vary by behaviour, so capability or privacy may simply need a different intervention
            layer. A 6-domain × 5-layer sweep would either confirm or falsify the
            per-domain capping hypothesis.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Multi-turn drift.</strong> Capping was
            evaluated at a single response turn. Does the intervention still hold at turn 5
            of a multi-turn conversation, or does the model drift back toward refusing? This
            is what would matter for any deployed agent.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Cross-model generalisation.</strong> Tested
            on Gemma 3 12B at layer 41 only. The same construction on Gemma 3 4B / 27B and on
            Qwen 3 / Llama 3.3 would show whether per-domain decomposition is a Gemma-3-12B
            quirk or an architectural pattern.
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Capping under adversarial prompts.</strong>{" "}
            Selectivity was measured on benign / harmful prompt pairs from our matched set.
            Does the safety direction still cap selectively when the prompt is a jailbreak
            attempt, where the model's refusal response is under adversarial pressure?
          </li>
          <li style={{ marginBottom: "10px" }}>
            <strong style={{ color: palette.text }}>Capping vs. steering-vector addition.</strong>{" "}
            Capping clamps the projection along a direction; activation addition (Panickssery
            et al. 2024) shifts the entire residual stream. Same direction, different
            intervention; comparing them per-domain would clarify whether the apparent
            selectivity is about the direction or about the operator.
          </li>
        </ul>
      </Section>

      <Section style={{ marginTop: "64px" }}>
        <Mono>
          <InlineLink href="#/" arrow="left" style={{ color: palette.text }}>
            Back to overview
          </InlineLink>
          {"   |   "}
          <InlineLink href="#/feature-hierarchy" arrow="right" style={{ color: palette.text }}>
            Continue to Part II: SAE feature hierarchy
          </InlineLink>
        </Mono>
      </Section>
    </>
  );
}
