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


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Refusal Axis in Domain Space

    Interactive exploration of refusal geometry in Gemma 3 12B.
    Extends the Assistant Axis methodology (Lu et al. 2026) to refusal behavior,
    decomposing a single refusal direction into per-domain sub-directions.

    **Key finding:** Value-based refusal (safety/ethical/legal) aligns strongly with the mean
    refusal axis (cosine 0.89–0.91), while capability boundary has lower alignment (0.38) —
    but the safety-capability pairwise cosine has a wide bootstrap CI ([-0.316, 0.639]),
    so the separation is directional, not definitive. Domain-selective capping along the
    safety direction reduces safety refusal by 31.6 points with <1.5 spillover (robust: 60.9× above random).
    """)
    return


@app.cell
def _():
    import json

    import numpy as np
    import plotly.graph_objects as go
    import torch
    import torch.nn.functional as F
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    return F, go, json, leaves_list, linkage, np, squareform, torch


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parent
    DATA = ROOT / "data"
    return DATA, ROOT


@app.cell
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

    domains_present = sorted(set(pair_domain.values()))
    domain_labels = [pair_domain.get(pid, "unknown") for pid in retained]

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
            "pos": _pos,
            "neg": _neg,
            "mean_axis": _mean_axis,
            "diffs": _diffs,
            "diffs_centered": _diffs_centered,
            "Vt": _Vt,
            "var_exp": _var_exp,
            "projections": _projections,
            "domain_dirs": _domain_dirs,
            "pos_proj": (_pos.numpy() @ _axis_unit).tolist(),
            "neg_proj": (_neg.numpy() @ _axis_unit).tolist(),
        }
    return (
        DOMAIN_COLORS,
        DOMAIN_SHORT,
        domain_labels,
        pair_domain,
        retained,
        sites_data,
    )


@app.cell
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


# ── 3D Hero Figure ──────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("## The Refusal Axis in Domain Space")
    return


@app.cell
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
                colorbar=dict(title="Refusal<br>projection", len=0.5),
            ),
            text=_hover,
            hoverinfo="text",
            name="Prompt pairs",
        )
    )

    for _d in sorted(set(domain_labels)):
        _idx = [_i for _i, _l in enumerate(domain_labels) if _l == _d]
        if len(_idx) < 2:
            continue
        _centroid = _pc3[_idx].mean(axis=0)
        fig3d.add_trace(
            go.Scatter3d(
                x=[_centroid[0]],
                y=[_centroid[1]],
                z=[_centroid[2]],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=DOMAIN_COLORS.get(_d, "gray"),
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
                text=[DOMAIN_SHORT.get(_d, _d)],
                textposition="top center",
                textfont=dict(size=11, color=DOMAIN_COLORS.get(_d, "gray")),
                name=DOMAIN_SHORT.get(_d, _d),
                showlegend=False,
            )
        )

    _line_range = np.max(np.abs(_pc3)) * 1.0
    _start = -_line_range * _axis_unit_pc
    _end = _line_range * _axis_unit_pc
    fig3d.add_trace(
        go.Scatter3d(
            x=[_start[0], _end[0]],
            y=[_start[1], _end[1]],
            z=[_start[2], _end[2]],
            mode="lines+text",
            line=dict(color="#457b9d", width=4, dash="dash"),
            text=["", "The Refusal Axis"],
            textposition="top center",
            textfont=dict(size=12, color="#457b9d"),
            name="Refusal Axis",
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
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        title="Drag to rotate — hover for details",
    )
    fig3d
    return


# ── Domain Loading ──────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Domain Loading on Refusal Axis

    Cosine similarity of each domain's refusal direction with the mean refusal axis.
    High loading = aligned with "standard" refusal. Low loading = distinct mechanism.
    """)
    return


@app.cell
def _(DOMAIN_COLORS, DOMAIN_SHORT, F, go, site_selector, sites_data):
    _sd = sites_data[site_selector.value]
    _mean_ax = _sd["mean_axis"]
    _d_dirs = _sd["domain_dirs"]

    _loadings = {}
    for _d, _d_dir in _d_dirs.items():
        _cos = F.cosine_similarity(_d_dir.unsqueeze(0), _mean_ax.unsqueeze(0)).item()
        _loadings[DOMAIN_SHORT.get(_d, _d)] = _cos

    _sorted = sorted(_loadings.items(), key=lambda x: x[1], reverse=True)
    _names = [x[0] for x in _sorted]
    _vals = [x[1] for x in _sorted]
    _short_to_full = {v: k for k, v in DOMAIN_SHORT.items()}
    _colors = [DOMAIN_COLORS.get(_short_to_full.get(n, ""), "#888") for n in _names]

    fig_load = go.Figure(
        go.Bar(
            x=_vals,
            y=_names,
            orientation="h",
            marker_color=_colors,
            text=[f"{v:.2f}" for v in _vals],
            textposition="outside",
        )
    )
    fig_load.update_layout(
        xaxis_title="Cosine with mean refusal axis",
        xaxis_range=[0, 1.05],
        height=300,
        margin=dict(l=100, r=40, t=20, b=40),
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
    )
    fig_load
    return


# ── Cross-domain Cosine Matrix ─────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cross-Domain Cosine Similarity

    Pairwise cosine between domain refusal directions. Hover for exact values.
    Value-based domains (Safety/Ethical/Legal) cluster tightly. Capability appears separated,
    but clustering structure depends on linkage method (Ward vs complete vs single).
    Bootstrap CIs for small-n domains are wide — interpret with caution.
    """)
    return


@app.cell
def _(
    DOMAIN_SHORT, F, go, leaves_list, linkage, np, site_selector, sites_data, squareform
):
    _sd = sites_data[site_selector.value]
    _d_dirs = _sd["domain_dirs"]

    _ordered = sorted(_d_dirs.keys())
    _n = len(_ordered)
    _cos_mat = np.zeros((_n, _n))
    for _i, _d1 in enumerate(_ordered):
        for _j, _d2 in enumerate(_ordered):
            _cos_mat[_i, _j] = F.cosine_similarity(
                _d_dirs[_d1].unsqueeze(0), _d_dirs[_d2].unsqueeze(0)
            ).item()

    _dist = np.clip(1 - _cos_mat, 0, 2)
    np.fill_diagonal(_dist, 0)
    _dist = (_dist + _dist.T) / 2
    _Z = linkage(squareform(_dist), method="ward")
    _order = leaves_list(_Z)

    _labels = [DOMAIN_SHORT.get(_ordered[_k], _ordered[_k]) for _k in _order]
    _mat_ord = _cos_mat[np.ix_(_order, _order)]

    fig_cos = go.Figure(
        go.Heatmap(
            z=_mat_ord,
            x=_labels,
            y=_labels,
            colorscale=[[0, "#e63946"], [0.5, "#f5f0eb"], [1, "#457b9d"]],
            zmin=-0.2,
            zmax=1.0,
            text=np.round(_mat_ord, 2),
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig_cos.update_layout(
        height=450,
        width=550,
        margin=dict(l=80, r=40, t=20, b=80),
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
        yaxis_autorange="reversed",
    )
    fig_cos
    return


# ── PCA Dimensionality ─────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Refusal Space Dimensionality

    How many dimensions does refusal live in? PCA variance explained
    and 2D scatter colored by domain.
    """)
    return


@app.cell
def _(mo):
    n_pcs_slider = mo.ui.slider(
        start=5, stop=50, step=1, value=20, label="Show top N principal components"
    )
    n_pcs_slider
    return (n_pcs_slider,)


@app.cell
def _(go, n_pcs_slider, np, site_selector, sites_data):
    _sd = sites_data[site_selector.value]
    _var_exp = _sd["var_exp"]
    _n_show = n_pcs_slider.value

    _cumvar = np.cumsum(_var_exp[:_n_show]) * 100
    _indiv = _var_exp[:_n_show] * 100

    fig_var = go.Figure()
    fig_var.add_trace(
        go.Bar(
            x=list(range(1, _n_show + 1)),
            y=_indiv,
            name="Individual",
            marker_color="#457b9d",
            opacity=0.7,
        )
    )
    fig_var.add_trace(
        go.Scatter(
            x=list(range(1, _n_show + 1)),
            y=_cumvar,
            name="Cumulative",
            line=dict(color="#e63946", width=2),
            mode="lines+markers",
            marker=dict(size=4),
        )
    )

    for _thresh in [70, 90, 95]:
        _pc_idx = int(np.searchsorted(_cumvar, _thresh))
        if _pc_idx < _n_show:
            fig_var.add_hline(
                y=_thresh,
                line_dash="dot",
                line_color="#999",
                annotation_text=f"{_thresh}% @ PC{_pc_idx + 1}",
            )

    fig_var.update_layout(
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        height=350,
        margin=dict(l=60, r=20, t=20, b=40),
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
        legend=dict(x=0.7, y=0.95),
    )
    fig_var
    return


@app.cell
def _(DOMAIN_COLORS, DOMAIN_SHORT, domain_labels, go, np, site_selector, sites_data):
    _sd = sites_data[site_selector.value]
    _pc_scores = _sd["diffs_centered"] @ _sd["Vt"][:2].T
    _ve = _sd["var_exp"]

    fig_scatter = go.Figure()
    for _d in sorted(set(domain_labels)):
        _idx = [_i for _i, _l in enumerate(domain_labels) if _l == _d]
        _pts = _pc_scores[_idx]
        fig_scatter.add_trace(
            go.Scatter(
                x=_pts[:, 0],
                y=_pts[:, 1],
                mode="markers",
                marker=dict(size=7, color=DOMAIN_COLORS.get(_d, "#888"), opacity=0.7),
                name=DOMAIN_SHORT.get(_d, _d),
                hovertemplate=f"{DOMAIN_SHORT.get(_d, _d)}<br>PC1: %{{x:.1f}}<br>PC2: %{{y:.1f}}<extra></extra>",
            )
        )

        if len(_idx) >= 3:
            _mean_pt = _pts.mean(axis=0)
            _cov = np.cov(_pts.T)
            _eigvals, _eigvecs = np.linalg.eigh(_cov)
            _t = np.linspace(0, 2 * np.pi, 50)
            _ell = (
                np.column_stack([np.cos(_t), np.sin(_t)])
                @ np.diag(np.sqrt(_eigvals))
                @ _eigvecs.T
            )
            _ell += _mean_pt
            fig_scatter.add_trace(
                go.Scatter(
                    x=_ell[:, 0],
                    y=_ell[:, 1],
                    mode="lines",
                    line=dict(color=DOMAIN_COLORS.get(_d, "#888"), width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig_scatter.update_layout(
        xaxis_title=f"PC1 ({_ve[0] * 100:.1f}%)",
        yaxis_title=f"PC2 ({_ve[1] * 100:.1f}%)",
        height=500,
        margin=dict(l=60, r=20, t=20, b=40),
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
    )
    fig_scatter
    return


# ── Axis Strip ──────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Refusal Axis Projection Strip

    Every prompt projected onto the mean refusal axis.
    Positive (harmful) prompts should cluster right (blue), negative (benign) left (red).
    """)
    return


@app.cell
def _(
    DOMAIN_COLORS,
    DOMAIN_SHORT,
    domain_labels,
    go,
    np,
    retained,
    site_selector,
    sites_data,
):
    _sd = sites_data[site_selector.value]

    fig_strip = go.Figure()

    _neg_p = np.array(_sd["neg_proj"])
    fig_strip.add_trace(
        go.Scatter(
            x=_neg_p,
            y=np.random.default_rng(42).uniform(-0.3, -0.05, len(_neg_p)),
            mode="markers",
            marker=dict(size=5, color="#aaaaaa", opacity=0.5, symbol="square"),
            name="Benign",
            hovertemplate="Benign pair %{text}<br>Proj: %{x:.1f}<extra></extra>",
            text=[str(pid) for pid in retained],
        )
    )

    _pos_p = np.array(_sd["pos_proj"])
    for _d in sorted(set(domain_labels)):
        _idx = [_i for _i, _l in enumerate(domain_labels) if _l == _d]
        fig_strip.add_trace(
            go.Scatter(
                x=_pos_p[_idx],
                y=np.random.default_rng(hash(_d) % 2**31).uniform(0.05, 0.3, len(_idx)),
                mode="markers",
                marker=dict(size=6, color=DOMAIN_COLORS.get(_d, "#888"), opacity=0.7),
                name=DOMAIN_SHORT.get(_d, _d),
                hovertemplate=f"{DOMAIN_SHORT.get(_d, _d)} pair %{{text}}<br>Proj: %{{x:.1f}}<extra></extra>",
                text=[str(retained[_i]) for _i in _idx],
            )
        )

    _boundary = (np.mean(_pos_p) + np.mean(_neg_p)) / 2
    fig_strip.add_vline(
        x=_boundary,
        line_dash="dash",
        line_color="#999",
        annotation_text=f"Boundary ({_boundary:.0f})",
    )

    fig_strip.update_layout(
        xaxis_title="Projection onto refusal axis",
        yaxis_visible=False,
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
        legend=dict(orientation="h", y=1.15),
    )
    fig_strip
    return


# ── Independence Matrix ─────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Domain-Selective Capping — Independence Matrix

    Each cell: mean refusal score change vs uncapped baseline.
    **Diagonal** = target effect (should be large negative).
    **Off-diagonal** = spillover (should be near zero).
    """)
    return


@app.cell
def _(DATA, go, json, np):
    # Structure: analysis[cap_dir][prompt_domain][tau_percentile] -> {mean_delta, ...}
    _analysis = json.load(open(DATA / "capping_results" / "analysis.json"))

    _cap_dirs = ["overall_refusal", "safety", "capability_boundary", "privacy"]
    _prompt_doms = ["safety", "capability_boundary", "privacy", "benign"]
    _cap_labels = ["Overall", "Safety", "Capability", "Privacy"]
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
            textfont=dict(size=14),
            hovertemplate="Cap: %{y}<br>Prompts: %{x}<br>Delta: %{z:.1f}<extra></extra>",
        )
    )
    fig_ind.update_layout(
        xaxis_title="Prompt domain",
        yaxis_title="Capping direction",
        height=350,
        width=500,
        margin=dict(l=100, r=40, t=20, b=60),
        paper_bgcolor="#faf8f5",
        plot_bgcolor="#faf8f5",
        yaxis_autorange="reversed",
    )
    fig_ind
    return


# ── Footer ──────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    *Data: 128 retained contrastive pairs, Gemma 3 12B layer 41 activations.
    Methodology: [refusal_axis_methodology.md](../findings/reports/refusal_axis_methodology.md).
    Evidence-pinned to Lu et al. (2026, arXiv:2601.10387).*
    """)
    return


if __name__ == "__main__":
    app.run()
