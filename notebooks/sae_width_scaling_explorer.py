# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy>=1.26.0",
#     "plotly>=5.18.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SAE Width Scaling: How Many Features Do You Need?

    When a language model refuses a request ("I can't help with that"), *something* in its
    internal activations encodes that decision. But what, exactly? The activations at any
    given layer are a dense 3840-dimensional vector -- thousands of numbers tangled together,
    none obviously corresponding to "refusal."

    **Sparse Autoencoders (SAEs)** let us untangle that vector. An SAE takes the dense
    activation and decomposes it into a much larger but *sparse* set of interpretable features.
    Think of it like a prism splitting white light into individual wavelengths: the SAE splits
    the model's internal state into individual concepts, most of which are "off" (zero) for any
    given input.

    In this notebook, we used Gemma Scope 2 (McDougall et al., 2025) SAEs
    (pre-trained on Gemma 3 12B) to find which features activate when the model refuses.
    We tested **four Matryoshka prefix widths** -- 16k, 65k, 262k, and 1M features -- to answer
    a practical question: *how wide does the dictionary need to be before we stop finding
    new refusal-relevant features?*

    **The short answer:** The biggest improvement comes from 16k to 65k features.
    After 262k, returns diminish sharply. But the dictionary does not simply *grow* --
    it *reorganizes*. Only 28-41% of features found at 16k survive at 1M width,
    meaning wider SAEs rearrange what existing features represent rather than just
    adding new ones on top.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, json, make_subplots, np


@app.cell(hide_code=True)
def _(mo):
    ROOT = mo.notebook_dir().parent
    DATA = ROOT / "data"
    return (DATA,)


@app.cell(hide_code=True)
def _(DATA, json):
    width_metrics = json.load(open(DATA / "milestone_7_results" / "width_metrics.json"))
    feature_catalogue = json.load(
        open(DATA / "milestone_7_results" / "feature_catalogue.json")
    )

    WIDTHS = [16384, 65536, 262144, 1048576]
    WIDTH_LABELS = ["16k", "65k", "262k", "1M"]
    SITES = ["last_prompt_token", "mean_response_token"]
    SITE_SHORT = {"last_prompt_token": "LPT", "mean_response_token": "MRT"}
    SITE_COLORS = {"last_prompt_token": "#0072B2", "mean_response_token": "#D55E00"}

    # Index by (site, width)
    metrics_by = {}
    for _m in width_metrics:
        metrics_by[(_m["site"], _m["width_int"])] = _m
    return (
        SITES,
        SITE_COLORS,
        SITE_SHORT,
        WIDTHS,
        WIDTH_LABELS,
        feature_catalogue,
        metrics_by,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is an SAE, and why do we need one?

    A transformer's residual stream at layer 41 is a vector of 3840 floating-point numbers.
    These numbers encode everything the model "knows" at that point -- syntax, semantics,
    safety judgments, world knowledge -- all superimposed on top of each other. You cannot
    look at dimension 1742 and say "this is the refusal dimension." The representation is
    *distributed*.

    A **Sparse Autoencoder** learns to decompose that dense vector into a much larger
    dictionary of features. For example, the 1M-wide SAE maps the 3840-dim input onto
    1,048,576 possible features, but for any given input only a handful (~57 on average)
    will have nonzero activations. Each feature ideally captures one interpretable concept:
    "the user is asking about explosives," "the model is about to apologize," etc.

    Gemma Scope 2 (McDougall et al., 2025) provides a pre-trained 1M-wide SAE
    for Gemma 3 12B. Crucially, it uses **Matryoshka training**: features are ordered by
    importance within the single SAE, so we can take the first 16k, 65k, or 262k features
    as progressively coarser sub-dictionaries (prefix slices of the same 1M SAE, not
    separate models):

    | Prefix slice | Features | Analogy |
    |--------------|----------|---------|
    | First 16k    | 16,384   | Broad strokes -- "this is a refusal" |
    | First 64k    | 65,536   | Finer distinctions -- "this is a *safety* refusal" |
    | First 256k   | 262,144  | Detailed -- "refusal about *synthesizing chemicals*" |
    | Full 1M      | 1,048,576 | Very fine-grained, but many features become redundant |

    Wider prefix slices can represent more fine-grained concepts, but they cost more compute
    and may split single concepts across multiple features. The question is: **where is the
    sweet spot?**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we measured

    We evaluated each dictionary width on four metrics. These tell us different things about
    how well the SAE captures refusal behaviour:

    1. **Relevant feature count** -- How many SAE features are specifically related to refusal?
       We identify these via contrastive scoring: a feature is "relevant" if it activates
       significantly more on refusal prompts than on matched compliant prompts. More relevant
       features means the SAE is decomposing refusal into finer sub-concepts.

    2. **Specificity (precision)** -- Of all the features that *activate* on refusal prompts,
       what fraction are actually refusal-related? High specificity means the SAE is not
       "wasting" features on noise. Low specificity means many features fire on refusal
       prompts by coincidence (e.g., they encode common English words, not refusal itself).

    3. **Mean effect size |d|** (Cohen's d) -- How *strongly* do the relevant features
       distinguish refusing from complying? A Cohen's d of 0.8 is conventionally "large."
       Values above 1.5 mean the feature almost perfectly separates the two conditions.
       Higher effect sizes mean cleaner, more reliable features.

    4. **Domain diversity** -- We test 6 types of refusal (safety, legal, ethical, privacy,
       capability boundary, identity boundary). Domain diversity measures how many of these 6
       types are represented among the discovered features. A score of 6/6 means the SAE
       found features for every refusal type, not just the most common ones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    site_selector = mo.ui.dropdown(
        options={
            "Both sites": "both",
            "Last Prompt Token (LPT)": "last_prompt_token",
            "Mean Response Token (MRT)": "mean_response_token",
        },
        value="Both sites",
        label="Extraction site",
    )
    site_selector
    return (site_selector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Width-scaling curves

    The figure below shows all four metrics (one panel each) across the four SAE widths.
    **Blue** = Last Prompt Token (LPT) site, **orange** = Mean Response Token (MRT) site.

    **What are LPT and MRT?** We extract activations at two points:

    - **LPT (Last Prompt Token):** The activation at the very last token of the user's
      message, right before the model starts generating. This is where the model makes
      its "should I refuse?" decision -- it has seen the full request and is about to commit.
    - **MRT (Mean Response Token):** The average activation across all tokens of the model's
      generated response. This captures *how* the model refuses -- the linguistic strategy
      it uses throughout its reply.

    **What to look for:** The shaded grey region (262k to 1M) highlights where improvements
    plateau. The critical transition is **16k to 65k** -- MRT specificity jumps from 0.34
    to 0.52, and LPT effect size nearly doubles (0.83 to 1.52). Going from 262k to 1M adds very few new relevant
    features and barely changes the other metrics. This suggests **65k-262k is the
    practical sweet spot** for refusal feature discovery.
    """)
    return


@app.cell(hide_code=True)
def _(
    SITES,
    SITE_COLORS,
    SITE_SHORT,
    WIDTHS,
    WIDTH_LABELS,
    go,
    make_subplots,
    metrics_by,
    site_selector,
):
    _metrics_info = [
        ("relevant_count", "Relevant features"),
        ("specificity", "Specificity (precision)"),
        ("mean_effect_size", "Mean |d|"),
        ("domain_diversity", "Domain diversity"),
    ]

    _sites = SITES if site_selector.value == "both" else [site_selector.value]

    fig_scaling = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[m[1] for m in _metrics_info],
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )

    for _mi, (_key, _label) in enumerate(_metrics_info):
        _row, _col = _mi // 2 + 1, _mi % 2 + 1
        for _site in _sites:
            _vals = [metrics_by.get((_site, w), {}).get(_key, 0) for w in WIDTHS]
            fig_scaling.add_trace(
                go.Scatter(
                    x=WIDTH_LABELS,
                    y=_vals,
                    mode="lines+markers",
                    marker=dict(size=8, color=SITE_COLORS[_site]),
                    line=dict(color=SITE_COLORS[_site], width=2),
                    name=SITE_SHORT[_site],
                    showlegend=(_mi == 0),
                    hovertemplate=f"{SITE_SHORT[_site]}<br>Width: %{{x}}<br>{_label}: %{{y:.2f}}<extra></extra>",
                ),
                row=_row,
                col=_col,
            )

        # Plateau shading
        fig_scaling.add_vrect(
            x0=2.5,
            x1=3.5,
            fillcolor="#E8E8E8",
            opacity=0.3,
            line_width=0,
            row=_row,
            col=_col,
        )

    fig_scaling.update_layout(
        height=500,
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
    )
    fig_scaling
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature genealogy: survival vs. emergence

    When we widen the SAE dictionary from 16k to 65k to 262k to 1M, what happens to the
    features we already found? There are two possibilities:

    - **Additive growth:** The 16k features persist unchanged, and the wider dictionary
      simply adds new, finer-grained features alongside them.
    - **Reorganization:** The wider dictionary reshuffles its feature basis, splitting or
      merging old features into new configurations. Features that existed at 16k may
      disappear or change meaning.

    We can test this because all four "widths" are **prefix slices of the same 1M SAE**
    (Matryoshka training orders features by importance). Feature at index `i` in the 16k
    slice is literally the same feature at index `i` in the 65k/262k/1M slices. So we
    can check: if feature `i` was refusal-relevant in the 16k slice, is it *still*
    refusal-relevant when we include more features around it in the wider slices?

    In the chart below, **blue** = features that survived from the 16k baseline, **orange** =
    newly emerged features that were not present at 16k.

    **The finding:** Only **28-41%** of 16k features survive to 1M. This means the dictionary
    fundamentally *reorganizes* as it widens. Wider SAEs do not simply add features on top --
    they rearrange what existing feature slots represent. This has practical implications:
    you cannot assume that a feature important at one width remains important (or even exists)
    at another width.
    """)
    return


@app.cell(hide_code=True)
def _(
    SITES,
    SITE_COLORS,
    SITE_SHORT,
    WIDTHS,
    WIDTH_LABELS,
    go,
    make_subplots,
    metrics_by,
    site_selector,
):
    _sites = SITES if site_selector.value == "both" else [site_selector.value]
    _n_sites = len(_sites)

    fig_gene = make_subplots(
        rows=1,
        cols=_n_sites,
        subplot_titles=[SITE_SHORT[s] for s in _sites],
        horizontal_spacing=0.12,
    )

    for _si, _site in enumerate(_sites):
        _baseline = metrics_by.get((_site, WIDTHS[0]), {}).get("relevant_count", 0)
        for _wi, (_w, _wl) in enumerate(zip(WIDTHS, WIDTH_LABELS)):
            _m = metrics_by.get((_site, _w), {})
            _total = _m.get("relevant_count", 0)
            _retained = _m.get("retention_from_16k") or (_total if _wi == 0 else 0)
            _new = max(_total - _retained, 0)

            fig_gene.add_trace(
                go.Bar(
                    y=[_wl],
                    x=[_retained],
                    orientation="h",
                    marker_color=SITE_COLORS[_site],
                    opacity=0.8,
                    name="Survived from 16k" if _wi == 1 and _si == 0 else None,
                    showlegend=(_wi == 1 and _si == 0),
                    hovertemplate=f"{_wl}: {_retained} survived<extra></extra>",
                ),
                row=1,
                col=_si + 1,
            )
            fig_gene.add_trace(
                go.Bar(
                    y=[_wl],
                    x=[_new],
                    orientation="h",
                    marker_color="#E69F00",
                    opacity=0.8,
                    name="Newly emerged" if _wi == 1 and _si == 0 else None,
                    showlegend=(_wi == 1 and _si == 0),
                    hovertemplate=f"{_wl}: {_new} new<extra></extra>",
                ),
                row=1,
                col=_si + 1,
            )

        fig_gene.update_xaxes(title_text="Feature count", row=1, col=_si + 1)

    fig_gene.update_layout(
        barmode="stack",
        height=300,
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    fig_gene
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Domain specificity: where does specialization happen?

    Our 240 contrastive pairs span 6 refusal domains, representing three distinct mechanisms:

    - **Value-based refusal** (safety, legal, ethical, privacy) -- the model *can* answer
      but *should not*. This is trained alignment behaviour.
    - **Identity honesty** (identity_boundary) -- the model should not claim false things
      about itself ("I am not a human").
    - **Capability acknowledgment** (capability_boundary) -- the model literally *cannot*
      do the action (e.g., "browse the web for you") because it lacks the tools.

    The heatmaps below show what fraction of refusal features at each width belong to each
    domain. The numbers inside cells are raw feature counts; the color intensity shows the
    proportion.

    **What to look for -- the key difference between LPT and MRT:**

    - **Last Prompt Token (LPT):** Dominated by Safety and Legal features across all widths.
      This suggests the model's "should I refuse?" decision is encoded in a relatively
      **domain-general** signal. The model uses a small number of broad features to flag
      harmful requests, regardless of the specific domain.

    - **Mean Response Token (MRT):** Shows a much more balanced distribution across domains.
      Ethical, Privacy, and Capability features each have substantial representation. This
      means the model's actual refusal *response* -- the words it generates -- uses
      **domain-specific** features. It refuses an ethics question differently from a
      capability question.

    **One interpretation:** This pattern is consistent with the model using a shared,
    domain-general signal to *decide* whether to refuse (captured at the last prompt token),
    while *executing* the refusal differently depending on the domain (captured across
    the response tokens). This is a hypothesis, not a proven mechanism — but the data
    consistently points in this direction across all four widths.
    """)
    return


@app.cell(hide_code=True)
def _(
    SITES,
    SITE_SHORT,
    WIDTHS,
    WIDTH_LABELS,
    go,
    make_subplots,
    metrics_by,
    np,
    site_selector,
):
    _sites = SITES if site_selector.value == "both" else [site_selector.value]
    _n_sites = len(_sites)
    _sub_types = [
        "safety",
        "legal",
        "ethical",
        "privacy",
        "capability_boundary",
        "identity_boundary",
    ]
    _sub_labels = ["Safety", "Legal", "Ethical", "Privacy", "Capability", "Identity"]

    fig_domain = make_subplots(
        rows=1,
        cols=_n_sites,
        subplot_titles=[SITE_SHORT[s] for s in _sites],
        horizontal_spacing=0.12,
    )

    for _si, _site in enumerate(_sites):
        _mat = np.zeros((len(_sub_types), len(WIDTHS)))
        for _wi, _w in enumerate(WIDTHS):
            _m = metrics_by.get((_site, _w), {})
            _counts = _m.get("sub_type_counts", {})
            _total = sum(_counts.values()) or 1
            for _ti, _st in enumerate(_sub_types):
                _mat[_ti, _wi] = _counts.get(_st, 0) / _total

        # Build text with raw counts
        _text = np.zeros_like(_mat, dtype=object)
        for _wi, _w in enumerate(WIDTHS):
            _counts = metrics_by.get((_site, _w), {}).get("sub_type_counts", {})
            for _ti, _st in enumerate(_sub_types):
                _text[_ti, _wi] = str(_counts.get(_st, 0))

        fig_domain.add_trace(
            go.Heatmap(
                z=_mat,
                x=WIDTH_LABELS,
                y=_sub_labels,
                colorscale=[[0, "#faf8f5"], [1, "#457b9d"]],
                zmin=0,
                zmax=1,
                text=_text,
                texttemplate="%{text}",
                textfont=dict(size=11),
                hovertemplate="%{y} @ %{x}: %{z:.0%} (%{text} features)<extra></extra>",
                showscale=(_si == _n_sites - 1),
            ),
            row=1,
            col=_si + 1,
        )

    fig_domain.update_layout(
        height=350,
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
        margin=dict(l=100, r=20, t=40, b=40),
    )
    fig_domain
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature catalogue

    The table below lists all **102 refusal-relevant features** discovered across all four
    dictionary widths and both extraction sites.

    **How each feature got here:** Every feature in this table passed a three-stage pipeline:

    1. **Contrastive scoring (Milestone 4):** We ran 240 matched prompt pairs (one that
       triggers refusal, one that does not) through Gemma 3 12B and collected SAE feature
       activations. Features that activate significantly more on refusal prompts than on
       compliant prompts were flagged as candidates.

    2. **Causal validation (Milestone 5):** We *steered* the model by artificially amplifying
       or suppressing each candidate feature during generation. If amplifying a feature makes
       the model refuse a normally-compliant prompt (or suppressing it makes it comply with
       a normally-refused prompt), the feature has a *causal* role in refusal -- it is not
       just a correlate.

    3. **Feature judging (Milestone 6):** An LLM-based classifier examined each
       feature's activation pattern and description to determine whether it is
       genuinely refusal-relevant, and if so, which refusal sub-type(s) it belongs to
       (safety, legal, ethical, privacy, capability, identity).

    **Column guide:**

    - **ID:** The SAE feature index (consistent across Matryoshka widths for features < 16k).
    - **Description:** Auto-generated label from Neuronpedia or the interpretability pipeline.
    - **Sub-types:** Which of the 6 refusal domains this feature is associated with.
    - **Max |d|:** The largest Cohen's d effect size across all tested widths. Values above
      0.8 are "large effects"; above 1.5 means near-perfect separation.
    - **Widths:** Which dictionary widths this feature appears in (relevant to survival analysis).
    - **Site:** LPT (decision point) or MRT (response execution).
    - **Cosine:** Cosine similarity of the feature's decoder direction to the mean refusal
      direction. Higher values mean the feature aligns with the overall "refusal axis."

    Use the filters below to explore by domain and minimum effect size.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    subtype_filter = mo.ui.dropdown(
        options={
            "All": "all",
            "Safety": "safety",
            "Legal": "legal",
            "Ethical": "ethical",
            "Privacy": "privacy",
            "Capability": "capability_boundary",
            "Identity": "identity_boundary",
        },
        value="All",
        label="Sub-type",
    )
    min_d_slider = mo.ui.slider(
        start=0.0, stop=3.0, step=0.1, value=0.5, label="Min Cohen's d"
    )
    mo.hstack([subtype_filter, min_d_slider])
    return min_d_slider, subtype_filter


@app.cell(hide_code=True)
def _(feature_catalogue, min_d_slider, mo, subtype_filter):
    filtered_features = []
    for _f in feature_catalogue:
        _max_d = (
            max(_f.get("cohens_d_by_width", {}).values())
            if _f.get("cohens_d_by_width")
            else 0
        )
        if _max_d < min_d_slider.value:
            continue
        _subs = _f.get("sub_types", [])
        if subtype_filter.value != "all" and subtype_filter.value not in _subs:
            continue
        filtered_features.append(
            {
                "ID": _f["feature_id"],
                "Description": _f.get("description", "")[:80],
                "Sub-types": ", ".join(_subs),
                "Max |d|": f"{_max_d:.2f}",
                "Widths": ", ".join(_f.get("widths_present", [])),
                "Site": _f.get("site", ""),
                "Cosine": f"{_f.get('cosine_sim') or 0:.3f}",
            }
        )

    mo.md(
        f"**{len(filtered_features)}** features match filters"
    ) if filtered_features else mo.md("*No features match*")
    return (filtered_features,)


@app.cell(hide_code=True)
def _(filtered_features, mo):
    mo.ui.table(filtered_features, selection=None, page_size=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    **Methodology summary.** Gemma 3 12B activations at layer 41, decoded by Gemma Scope 2
    SAEs at four Matryoshka widths (16k, 64k, 256k, 1M). Dataset: 240 contrastive prompt
    pairs spanning 6 refusal domains (safety, legal, ethical, privacy, capability boundary,
    identity boundary). Pipeline stages: contrastive scoring (M4), causal steering validation
    (M5), automated LLM judging (M6), width-scaling analysis (M7). Full results and
    reproducibility details: `findings/reports/rq1_poc_results.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
