# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.0.0",
#     "numpy>=1.26.0",
#     "plotly>=5.18.0",
#     "scipy>=1.12.0",
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
    # Is Refusal One Thing or Many Things?

    ## Looking inside a language model to understand *how* it says no

    When you ask a language model to help you build a bomb, it refuses.
    When you ask it to browse the web (something it literally cannot do), it also refuses.
    Both responses look similar on the surface --- the model declines your request.
    But are they the same process inside the model?

    This notebook answers that question by reading the model's internal activations
    --- the numerical representations the model builds as it processes your prompt.
    We find that **refusal is not one thing.** The model uses geometrically distinct
    mechanisms for different types of refusal, and we can surgically suppress one
    while leaving the others intact.

    This matters for AI safety: it means we could, in principle, build a system
    that relaxes capability limitations (so an AI agent can execute code or use tools)
    while keeping safety refusal fully intact.

    **Methodology.** We adapt the *activation capping* framework from
    Lu et al. (2026, *The Assistant Axis*, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)),
    who used it to stabilize the default assistant persona along a single axis.
    We extend their geometric approach to per-domain refusal directions
    to refusal in Gemma 3 12B, using 128 contrastive prompt pairs across 6 domains.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json

    import numpy as np
    import plotly.graph_objects as go
    import torch
    import torch.nn.functional as F
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    return F, go, json, leaves_list, linkage, np, squareform, torch


@app.cell(hide_code=True)
def _(mo):
    ROOT = mo.notebook_dir().parent
    DATA = ROOT / "data"
    return DATA, ROOT


@app.cell(hide_code=True)
def _(DATA, json, np, torch):
    retained = json.load(open(DATA / "retained_pairs.json"))
    manifest = json.load(open(DATA / "responses_manifest.json", encoding="utf-8"))

    pair_domain = {}
    for _e in manifest:
        if _e["pair_id"] in set(retained) and _e["condition"] == "positive":
            pair_domain[_e["pair_id"]] = _e["domain"]

    DOMAIN_COLORS = {
        "safety": "#D55E00",
        "ethical": "#E69F00",
        "legal": "#CC79A7",
        "privacy": "#0072B2",
        "capability_boundary": "#009E73",
        "identity_boundary": "#56B4E9",
    }
    DOMAIN_SHORT = {
        "safety": "Safety",
        "ethical": "Ethical",
        "legal": "Legal",
        "privacy": "Privacy",
        "capability_boundary": "Capability",
        "identity_boundary": "Identity",
    }
    domain_labels = [pair_domain.get(pid, "unknown") for pid in retained]
    domains_present = sorted(set(pair_domain.values()))

    sites_data = {}
    for _site in ["mean_response_token", "last_prompt_token"]:
        _pos = torch.stack(
            [
                torch.load(
                    DATA / "activations" / _site / "positive" / f"pair_{pid}.pt",
                    weights_only=True,
                )
                for pid in retained
            ]
        ).float()
        _neg = torch.stack(
            [
                torch.load(
                    DATA / "activations" / _site / "negative" / f"pair_{pid}.pt",
                    weights_only=True,
                )
                for pid in retained
            ]
        ).float()
        _mean_axis = torch.load(
            DATA / f"refusal_direction_{_site}.pt", weights_only=True
        ).float()

        _diffs = (_pos - _neg).numpy()
        _mean = _diffs.mean(axis=0)
        _diffs_centered = _diffs - _mean
        _, _S, _Vt = np.linalg.svd(_diffs_centered, full_matrices=False)
        _var_exp = (_S**2) / (_S**2).sum()

        _axis_in_full = _mean_axis.numpy()
        _axis_unit = _axis_in_full / (np.linalg.norm(_axis_in_full) + 1e-8)
        _projections = _diffs @ _axis_unit

        _domain_dirs = {}
        for _d in domains_present:
            _idx = [_i for _i, pid in enumerate(retained) if pair_domain.get(pid) == _d]
            if len(_idx) >= 2:
                _domain_dirs[_d] = _pos[_idx].mean(0) - _neg[_idx].mean(0)

        sites_data[_site] = {
            "diffs": _diffs,
            "diffs_centered": _diffs_centered,
            "Vt": _Vt,
            "var_exp": _var_exp,
            "projections": _projections,
            "mean_axis": _mean_axis,
            "domain_dirs": _domain_dirs,
        }
    return (
        DOMAIN_COLORS,
        DOMAIN_SHORT,
        domain_labels,
        domains_present,
        pair_domain,
        retained,
        sites_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is an activation?

    When a language model processes text, each layer transforms the input into a
    **hidden state** --- a vector of 3,840 numbers (for Gemma 3 12B). This vector
    encodes everything the model "knows" at that position: the meaning of the current
    token, the context of the conversation, and critically, the model's emerging
    *intent* --- whether it plans to comply or refuse.

    We can extract these vectors and compare them across different prompts. If we
    give the model a harmful prompt and a benign rephrasing of the same topic, the
    difference between their activation vectors tells us what the model represents
    differently --- the "refusal signal."

    ## The refusal direction

    Here is the key idea, borrowed from Lu et al. (2026): if we collect many such
    pairs (harmful prompt vs. benign rephrasing) and average the activation
    differences, we get a single vector --- the **refusal direction**. Think of it as
    a compass needle in 3,840-dimensional space that points from "comply" to "refuse."

    Any individual prompt's activation can be *projected* onto this direction to
    measure how much refusal signal it carries. A large positive projection means
    strong refusal; near-zero means the model sees nothing to refuse.

    ## Where we read the activations

    We extract activations at **two sites**, and the dropdown below lets you switch
    between them:

    - **Last Prompt Token** --- This is the model's *decision point*. It has read your
      entire prompt but has not yet generated any response. The activation here
      captures the model's **intent**: does it plan to refuse or comply?

    - **Mean Response Token** --- This is the average activation across all tokens of
      the model's actual response. It captures **behavior**: what the model is
      actually doing as it generates its answer.

    Both sites show similar structure, but last-prompt-token is arguably more
    interesting because it reveals the decision *before* it manifests in text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    site_selector = mo.ui.dropdown(
        options={
            "Mean Response Token": "mean_response_token",
            "Last Prompt Token": "last_prompt_token",
        },
        value="Mean Response Token",
        label="Extraction site",
    )
    site_selector
    return (site_selector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The 3D figure: seeing refusal in activation space

    The figure below is a window into the model's 3,840-dimensional activation space,
    compressed down to 3 dimensions using PCA (principal component analysis) so we
    can actually look at it.

    **How to read it:**

    - **Each dot** is one of our 128 contrastive pairs. Specifically, it is the
      *difference vector* (refused activation minus complied activation) for that pair,
      projected into the top 3 principal components.

    - **Color** encodes how strongly each pair projects onto the mean refusal direction.
      Blue dots have a strong refusal signal; red dots are closer to the compliance end.

    - **The dashed line** is the mean refusal direction itself --- the "compass needle"
      described above, projected into this same 3D view.

    - **Diamonds** are domain centroids --- the average position of all pairs from each
      refusal type (Safety, Ethical, Legal, Privacy, Capability, Identity).

    **What to look for:** If refusal were a single, unified mechanism, all the diamonds
    would sit neatly along the dashed line and cluster together. They don't. The
    value-based domains (Safety, Ethical, Legal, Privacy) *do* cluster along the axis ---
    they share a common refusal mechanism. But **Capability** (green diamond) sits off
    to the side, nearly perpendicular to the refusal axis. The model represents
    "I can't do this" in a fundamentally different geometric direction than "I won't do this."

    Drag to rotate the figure and explore the structure from different angles.
    """)
    return


@app.cell(hide_code=True)
def _(
    DOMAIN_COLORS,
    DOMAIN_SHORT,
    domain_labels,
    go,
    np,
    pair_domain,
    retained,
    site_selector,
    sites_data,
):
    _sd = sites_data[site_selector.value]
    _diffs_c = _sd["diffs_centered"]
    _Vt = _sd["Vt"]
    _var_exp = _sd["var_exp"]
    _projs = _sd["projections"]

    _pc3 = _diffs_c @ _Vt[:3].T
    _axis_in_pc = _sd["diffs"].mean(axis=0) @ _Vt[:3].T
    _axis_unit_pc = _axis_in_pc / (np.linalg.norm(_axis_in_pc) + 1e-8)
    _proj_norm = (_projs - _projs.min()) / (_projs.max() - _projs.min() + 1e-8)

    _hover = [
        f"Pair {pid}<br>Domain: {DOMAIN_SHORT.get(pair_domain.get(pid, ''), '?')}<br>Axis proj: {_projs[_i]:.1f}"
        for _i, pid in enumerate(retained)
    ]

    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=_pc3[:, 0],
            y=_pc3[:, 1],
            z=_pc3[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=_proj_norm,
                colorscale=[[0, "#e63946"], [0.5, "#d4d0ca"], [1, "#457b9d"]],
                opacity=0.6,
                colorbar=dict(title="Refusal<br>proj.", len=0.4, x=1.02),
            ),
            text=_hover,
            hoverinfo="text",
            name="Pairs",
        )
    )

    for _d in sorted(set(domain_labels)):
        _idx = [_i for _i, _l in enumerate(domain_labels) if _l == _d]
        if len(_idx) < 2:
            continue
        _c = _pc3[_idx].mean(axis=0)
        fig3d.add_trace(
            go.Scatter3d(
                x=[_c[0]],
                y=[_c[1]],
                z=[_c[2]],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=DOMAIN_COLORS.get(_d, "gray"),
                    symbol="diamond",
                    line=dict(width=1.5, color="white"),
                ),
                text=[DOMAIN_SHORT.get(_d, _d)],
                textposition="top center",
                textfont=dict(size=11, color=DOMAIN_COLORS.get(_d, "gray")),
                showlegend=False,
            )
        )

    _r = np.max(np.abs(_pc3)) * 1.0
    _s, _e = -_r * _axis_unit_pc, _r * _axis_unit_pc
    fig3d.add_trace(
        go.Scatter3d(
            x=[_s[0], _e[0]],
            y=[_s[1], _e[1]],
            z=[_s[2], _e[2]],
            mode="lines+text",
            line=dict(color="#457b9d", width=4, dash="dash"),
            text=["", "Refusal Axis"],
            textposition="top center",
            textfont=dict(size=11, color="#457b9d"),
            showlegend=False,
        )
    )

    fig3d.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({_var_exp[0] * 100:.1f}%)",
            yaxis_title=f"PC2 ({_var_exp[1] * 100:.1f}%)",
            zaxis_title=f"PC3 ({_var_exp[2] * 100:.1f}%)",
            bgcolor="#faf8f5",
        ),
        paper_bgcolor="#faf8f5",
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
        title=dict(text="Drag to rotate", font=dict(size=11, color="#999")),
    )
    fig3d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The independence matrix: can we cap one type of refusal without touching others?

    Seeing that refusal types occupy different directions is suggestive, but the real
    test is *causal*: if we suppress the refusal signal along one direction, does it
    selectively reduce only that type of refusal?

    **How activation capping works.** Following Lu et al. (2026), we take the
    per-domain refusal direction (e.g., the "safety refusal" direction) and, at
    inference time, *clamp* each token's projection onto that direction so it cannot
    exceed a threshold tau. This is like putting a ceiling on one specific signal
    in the model's internal representation. The threshold tau is calibrated from
    benign prompts (the p50 of their projections), so we are only removing the
    *excess* refusal signal beyond what benign prompts naturally carry.

    **How to read the matrix below:**

    - **Each row** is a capping direction --- which refusal signal we are suppressing.
      "Overall (single-axis)" uses the mean refusal direction (one cap for everything).
      The other rows use per-domain directions.

    - **Each column** is a test prompt category --- which prompts we evaluated after capping.

    - **The numbers** are the change in refusal score (negative means less refusal).
      A perfect selective cap would show a large negative number on the diagonal
      (strong effect on its target domain) and zeros everywhere else (no spillover).

    **Key findings:**

    - **Safety capping is highly selective.** The safety row shows -31.6 on safety
      prompts but less than 1.5 points of change on capability, privacy, or benign.
      You can turn down safety refusal without touching anything else.

    - **Overall capping is blunt.** The single-axis approach reduces safety refusal
      by -25.0 but also knocks out privacy refusal by -15.9. This is the cost of
      treating refusal as one thing when it is not.

    - **Capability and privacy capping show no selective effect** at this threshold
      and layer. This is itself informative --- not all domain directions are equally
      cappable. The report discusses possible reasons (weak baseline refusal,
      wrong layer for these domains).
    """)
    return


@app.cell(hide_code=True)
def _(DATA, go, json, np):
    # Structure: analysis[cap_dir][prompt_domain][tau_percentile] -> {mean_delta, ...}
    _analysis = json.load(open(DATA / "capping_results" / "analysis.json"))

    _cap_dirs = ["overall_refusal", "safety", "capability_boundary", "privacy"]
    _prompt_doms = ["safety", "capability_boundary", "privacy", "benign"]
    _cap_labels = [
        "Overall (single-axis)",
        "Safety only",
        "Capability only",
        "Privacy only",
    ]
    _prompt_labels = ["Safety", "Capability", "Privacy", "Benign"]
    _tau = "50"  # p50 sweet spot

    _mat = np.zeros((len(_cap_dirs), len(_prompt_doms)))
    for _i, _cd in enumerate(_cap_dirs):
        for _j, _pd in enumerate(_prompt_doms):
            _cell = _analysis.get(_cd, {}).get(_pd, {}).get(_tau, {})
            _mat[_i, _j] = _cell.get("mean_delta", 0.0)

    fig_ind = go.Figure(
        go.Heatmap(
            z=_mat,
            x=_prompt_labels,
            y=_cap_labels,
            colorscale=[[0, "#457b9d"], [0.5, "#f5f0eb"], [1, "#e63946"]],
            zmid=0,
            text=np.round(_mat, 1),
            texttemplate="%{text}",
            textfont=dict(size=15),
            hovertemplate="Cap: %{y}<br>Prompts: %{x}<br>Refusal delta: %{z:.1f}<extra></extra>",
        )
    )
    fig_ind.update_layout(
        xaxis_title="Test prompt domain",
        yaxis_title="Capping direction",
        height=350,
        margin=dict(l=150, r=40, t=30, b=60),
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
        yaxis_autorange="reversed",
        title=dict(
            text="Independence Matrix (refusal score change at tau=p50)",
            font=dict(size=12, color="#666"),
        ),
    )
    fig_ind
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The punchline: what survives scrutiny

    We subjected every claim to systematic falsification testing (10 tests,
    1000-iteration permutation nulls, bootstrap CIs). Here is what survived.

    **What is robust:**

    - The refusal space is genuinely structured --- 11 PCA dimensions for 70% variance,
      versus 80 for random vectors in the same space (p < 0.001).
    - The safety direction is specific, not noise --- it captures 60.9× more projection
      variance than random unit vectors.
    - Safety capping works selectively: -31.6 on target, <1.5 spillover.

    **What is weaker than it looks:**

    - The pairwise cosine between safety and capability (point estimate: 0.14) has a wide
      95% bootstrap CI of [-0.316, 0.639]. We cannot confidently say they are "near-orthogonal"
      --- only that capability loads *less* on the mean refusal axis than safety does (0.38 vs 0.91).
    - The overall domain decomposition is borderline distinguishable from random group splits
      (permutation test p=0.054). Suggestive, not definitive.
    - τ=p50 was found by sweeping 7 values across 112 cells. It is an exploratory finding,
      not a pre-registered result.

    **What we honestly don't know:**

    - Whether the privacy spillover (-15.9) is statistically significant (n=9, no CI available).
    - Whether capability capping would work at a different layer.
    - Whether these results generalize to other models or larger test sets.

    ---

    *128 contrastive pairs across 6 domains, Gemma 3 12B,
    layer 41 activations (deepest layer with a 1M SAE in Gemma Scope 2, at 85% depth).
    Capping applied at layer 36, tau calibrated from benign prompt projections at p50 (exploratory).
    Methodology adapted from Lu et al. (2026, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)).
    Falsification tests: `refusal_axis_falsification.py`.*
    """)
    return


if __name__ == "__main__":
    app.run()
