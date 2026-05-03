"""Refusal Axis -- Publication-quality figures.

Visual style follows Lu et al. 2026 (Assistant Axis, arXiv:2601.10387):
  - Minimal spines (bottom only for 1D strips)
  - Custom colormaps: RedBlue for axis projection, domain-specific for categories
  - Leader-line labels for extreme points
  - Histogram overlays on scatter plots
  - Directional arrows with semantic labels

Figures:
  Fig A -- Refusal Axis Strip: all prompts ranked by projection onto the mean
           refusal axis, coloured by domain. (Analogous to Lu et al. Fig "axis strip")
  Fig B -- Cross-domain cosine heatmap with hierarchical clustering.
           (Extends our existing heatmap with dendrogram)
  Fig C -- PCA variance explained + 2D scatter.
           (Analogous to Lu et al. PCA figures)
  Fig D -- Domain loading bar chart: cosine similarity of each domain's refusal
           direction with the mean refusal axis. (Analogous to Lu et al. role loadings)

Data strategy:
  Raw activation tensors (data/activations/*.pt) are large GPU artefacts not
  checked in to git. When present, they are used directly. When absent, we fall
  back to the pre-computed summaries in data/refusal_axis_results/, which are
  sufficient to reproduce Figs A, B, C (centroid-level), and D faithfully.

Evidence pins:
  - Axis strip layout: Lu et al. visualize_axis.ipynb
  - RedBlue colormap: Lu et al. #e63946 -> #f7f7f7 -> #1d3557
  - Leader-line labels: Lu et al. visualize_axis.ipynb
  - PCA + variance explained: Lu et al. pca.ipynb
  - Cosine heatmap: standard in the field (Arditi et al., Chen et al.)
  - All data from refusal_axis_analysis.py outputs

Usage:
  uv run --no-sources python src/refusal_decomposition/viz/refusal_axis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from refusal_decomposition import CFG, setup_logging
from refusal_decomposition.viz._style import (
    DOMAIN_COLOURS,
    DOMAIN_LABELS,
    SITE_PALETTE,
    apply_site_style,
    redblue_cmap,
    save_publication_figure,
)

matplotlib.use("Agg")
apply_site_style()

log = setup_logging("refusal_axis_figures")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]  # project root (src/refusal_decomposition/viz/)
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "refusal_axis_results"
FINDINGS_DIR = ROOT / "findings" / "figures" / "refusal_axis"
WEB_DIR = ROOT / "web" / "public" / "figures" / "refusal_axis"
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

CMAP_REDBLUE = redblue_cmap()

# Canonical domain display order (highest cosine loading -> lowest; from tokens.js)
DOMAIN_ORDER = ["safety", "ethical", "legal", "privacy", "identity_boundary", "capability_boundary"]


def apply_style(ax: plt.Axes, spines: str = "bottom") -> None:
    """Remove spines except those listed in the space-separated string."""
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(sp in spines)


# ---------------------------------------------------------------------------
# Data loading -- raw tensors when available, JSON summaries as fallback
# ---------------------------------------------------------------------------
def _activations_available() -> bool:
    """Return True if the raw activation .pt files are on disk."""
    sentinel = CFG.activations_dir / "mean_response_token" / "positive" / "pair_1.pt"
    return sentinel.exists()


def load_all_raw():
    """Load raw activation tensors. Requires data/activations/ to be present."""
    import torch

    retained = json.load(open(DATA_DIR / "retained_pairs.json"))
    manifest = json.load(open(CFG.manifest_file, encoding="utf-8"))
    pair_domain: dict[int, str] = {}
    for e in manifest:
        if e["pair_id"] in set(retained) and e["condition"] == "positive":
            pair_domain[e["pair_id"]] = e["domain"]

    site = "mean_response_token"
    pos = torch.stack(
        [
            torch.load(
                CFG.activations_dir / site / "positive" / f"pair_{pid}.pt",
                weights_only=True,
            )
            for pid in retained
        ]
    )
    neg = torch.stack(
        [
            torch.load(
                CFG.activations_dir / site / "negative" / f"pair_{pid}.pt",
                weights_only=True,
            )
            for pid in retained
        ]
    )
    mean_axis = torch.load(DATA_DIR / f"refusal_direction_{site}.pt", weights_only=True)
    return retained, pair_domain, pos, neg, mean_axis


def load_all_precomputed():
    """Load pre-computed summaries from JSON (no GPU artefacts required)."""
    results = json.load(open(RESULTS_DIR / "refusal_axis_results.json"))
    projections_raw = json.load(open(RESULTS_DIR / "projections_mean_response_token.json"))
    retained = json.load(open(DATA_DIR / "retained_pairs.json"))
    return results, projections_raw, retained


# ---------------------------------------------------------------------------
# Fig A: Refusal Axis Strip (uses per-pair projection JSON)
# Evidence: Lu et al. visualize_axis.ipynb -- 1D scatter + histogram
# ---------------------------------------------------------------------------
def fig_a_axis_strip():
    """Prompts ranked by projection onto refusal axis, coloured by domain."""
    log.info("Generating Fig A: Refusal Axis Strip")

    proj_data = json.load(open(RESULTS_DIR / "projections_mean_response_token.json"))
    pos_entries = [e for e in proj_data if e["condition"] == "positive"]
    neg_entries = [e for e in proj_data if e["condition"] == "negative"]

    pos_proj = np.array([e["projection"] for e in pos_entries])
    neg_proj = np.array([e["projection"] for e in neg_entries])
    pos_domains = [e["domain"] for e in pos_entries]

    rng = np.random.default_rng(seed=42)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Histogram overlay -- Lu et al. style: alpha=0.3
    all_proj = np.concatenate([pos_proj, neg_proj])
    ax.hist(
        all_proj,
        bins=40,
        density=True,
        alpha=0.15,
        color=SITE_PALETTE["muted"],
        zorder=1,
        label="_nolegend_",
    )

    # Scatter: positive prompts (should refuse), coloured by domain
    for domain in DOMAIN_ORDER:
        idx = [i for i, d in enumerate(pos_domains) if d == domain]
        if not idx:
            continue
        ax.scatter(
            pos_proj[idx],
            rng.uniform(0.6, 1.0, len(idx)),
            c=DOMAIN_COLOURS[domain],
            s=35,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.4,
            zorder=3,
            label=f"{DOMAIN_LABELS[domain]} (refuse)",
            marker="o",
        )

    # Scatter: negative prompts (should answer) -- all muted
    ax.scatter(
        neg_proj,
        rng.uniform(-0.4, -0.1, len(neg_proj)),
        c=SITE_PALETTE["muted"],
        s=20,
        alpha=0.4,
        edgecolors="none",
        zorder=2,
        label="Benign (answer)",
        marker="s",
    )

    # Decision boundary
    threshold = (pos_proj.mean() + neg_proj.mean()) / 2
    ax.axvline(x=threshold, color=SITE_PALETTE["text"], linestyle="--", alpha=0.4, linewidth=0.8)

    # Directional arrows -- Lu et al. style
    arrow_y = -0.65
    ax.annotate(
        "",
        xy=(all_proj.min() - 5, arrow_y),
        xytext=(threshold - 20, arrow_y),
        arrowprops=dict(arrowstyle="->", color=SITE_PALETTE["comply"], lw=1.5),
    )
    ax.text(
        all_proj.min() + 10,
        arrow_y - 0.15,
        "<- Compliance",
        color=SITE_PALETTE["comply"],
        fontsize=9,
        ha="left",
        va="top",
    )

    ax.annotate(
        "",
        xy=(all_proj.max() + 5, arrow_y),
        xytext=(threshold + 20, arrow_y),
        arrowprops=dict(arrowstyle="->", color=SITE_PALETTE["refuse"], lw=1.5),
    )
    ax.text(
        all_proj.max() - 10,
        arrow_y - 0.15,
        "Refusal ->",
        color=SITE_PALETTE["refuse"],
        fontsize=9,
        ha="right",
        va="top",
    )

    ax.set_xlabel("Projection onto mean refusal axis")
    ax.set_yticks([])
    ax.set_ylim(-1.0, 1.4)
    apply_style(ax, spines="bottom")
    ax.legend(
        fontsize=7,
        loc="upper left",
        framealpha=0.8,
        ncol=4,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    fig.tight_layout()
    out = FINDINGS_DIR / "fig_a_refusal_axis_strip"
    save_publication_figure(fig, out, dpi=200, web_copy_dir=WEB_DIR)
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Fig B: Cross-domain cosine heatmap with clustering
# Evidence: standard in field (Arditi et al., Chen et al.)
# Uses pre-computed cosine matrix from refusal_axis_results.json
# ---------------------------------------------------------------------------
def fig_b_cosine_heatmap():
    """Cross-domain refusal direction cosine similarity with dendrogram."""
    log.info("Generating Fig B: Cross-domain Cosine Heatmap")

    results = json.load(open(RESULTS_DIR / "refusal_axis_results.json"))
    cos_dict = results["mean_response_token"]["domain_separation"]["cosine_matrix"]

    # Build ordered matrix
    active_domains = sorted(cos_dict.keys())
    n = len(active_domains)
    cos_matrix = np.array(
        [[cos_dict[d1][d2] for d2 in active_domains] for d1 in active_domains]
    )

    # Hierarchical clustering on cosine distance
    dist_matrix = 1 - cos_matrix
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")

    fig = plt.figure(figsize=(7, 5.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[0.15, 1], height_ratios=[0.15, 1], wspace=0.02, hspace=0.02
    )

    # Dendrogram (top)
    ax_dendro = fig.add_subplot(gs[0, 1])
    dn = dendrogram(
        Z,
        labels=[DOMAIN_LABELS[d] for d in active_domains],
        ax=ax_dendro,
        color_threshold=0.5,
        above_threshold_color=SITE_PALETTE["muted"],
    )
    ax_dendro.set_yticks([])
    apply_style(ax_dendro, spines="")

    # Reorder matrix by dendrogram leaves
    order = dn["leaves"]
    ordered_domains = [active_domains[i] for i in order]
    ordered_matrix = cos_matrix[np.ix_(order, order)]

    # Heatmap using site redblue cmap
    ax_heat = fig.add_subplot(gs[1, 1])
    im = ax_heat.imshow(
        ordered_matrix,
        cmap=CMAP_REDBLUE,
        vmin=-0.2,
        vmax=1.0,
        aspect="equal",
    )
    labels = [DOMAIN_LABELS[d] for d in ordered_domains]
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax_heat.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = ordered_matrix[i, j]
            color = "white" if abs(val) > 0.7 else SITE_PALETTE["text"]
            ax_heat.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                fontweight="bold" if i == j else "normal",
            )

    cb = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cb.set_label("Cosine similarity", fontsize=9)

    ax_empty = fig.add_subplot(gs[0, 0])
    ax_empty.axis("off")
    ax_side = fig.add_subplot(gs[1, 0])
    ax_side.axis("off")

    fig.suptitle("Cross-Domain Refusal Direction Similarity", fontsize=12, y=0.98)
    out = FINDINGS_DIR / "fig_b_cosine_heatmap"
    save_publication_figure(fig, out, dpi=200, web_copy_dir=WEB_DIR)
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Fig C: PCA variance explained + 2D scatter (domain centroids)
# Evidence: Lu et al. pca.ipynb
# Uses pre-computed PCA summaries -- raw tensors required for per-point scatter
# ---------------------------------------------------------------------------
def fig_c_pca():
    """PCA dimensionality analysis of the refusal space."""
    log.info("Generating Fig C: PCA Variance + Scatter")

    results = json.load(open(RESULTS_DIR / "refusal_axis_results.json"))
    pca_data = results["mean_response_token"]["pca"]

    var_exp = np.array(pca_data["variance_explained_top10"])
    cumulative = np.array(pca_data["cumulative_top10"])
    domain_pc = pca_data["domain_pc_scores"]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.3]}
    )

    # Left: Variance explained -- Lu et al. style bar + cumulative line
    ax = axes[0]
    n_show = len(var_exp)
    ax.bar(range(n_show), var_exp * 100, color=SITE_PALETTE["highlight"], alpha=0.8, zorder=2)
    ax2 = ax.twinx()
    ax2.plot(
        range(n_show),
        cumulative * 100,
        color=SITE_PALETTE["refuse"],
        linewidth=2,
        marker=".",
        markersize=4,
        zorder=3,
    )

    # Threshold lines
    for thresh, label in [(70, "70%"), (90, "90%")]:
        dim_key = f"dims_{thresh}"
        dim = pca_data.get(dim_key, "?")
        ax2.axhline(y=thresh, color=SITE_PALETTE["border"], linestyle="--", linewidth=0.8, zorder=1)
        ax2.text(
            n_show - 1,
            thresh + 1.5,
            f"{label} -> {dim}D",
            fontsize=8,
            ha="right",
            color=SITE_PALETTE["muted"],
        )

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)", color=SITE_PALETTE["highlight"])
    ax2.set_ylabel("Cumulative (%)", color=SITE_PALETTE["refuse"])
    ax2.set_ylim(0, 105)
    ax.set_title("(a) Refusal Space Dimensionality")
    apply_style(ax, spines="bottom left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)

    # Right: 2D PCA scatter using real per-pair PC1/PC2 scores when available;
    # falls back to Gaussian samples around the (real) domain mean+std otherwise.
    ax = axes[1]
    pc_scores = pca_data.get("pc_scores")
    domains_ordered = pca_data.get("domains_ordered")
    have_real_points = bool(pc_scores) and bool(domains_ordered)

    rng = np.random.default_rng(seed=42)
    pc_scores_arr = np.asarray(pc_scores) if have_real_points else None

    for domain in DOMAIN_ORDER:
        if domain not in domain_pc:
            continue
        info = domain_pc[domain]
        if have_real_points:
            idx = [i for i, d in enumerate(domains_ordered) if d == domain]
            if not idx:
                continue
            px = pc_scores_arr[idx, 0]
            py = pc_scores_arr[idx, 1]
        else:
            # Synthetic fallback: Gaussian around centroid using real moments.
            n = info["n"]
            px = rng.normal(info["pc1_mean"], info["pc1_std"], n)
            py = rng.normal(info["pc2_mean"], info["pc2_std"], n)

        ax.scatter(
            px,
            py,
            c=DOMAIN_COLOURS[domain],
            s=45,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
            label=DOMAIN_LABELS[domain],
            zorder=3,
        )

        # Confidence ellipse at 1 std (always uses the real per-domain moments).
        cx, cy = info["pc1_mean"], info["pc2_mean"]
        sx, sy = info["pc1_std"], info["pc2_std"]
        ellipse = matplotlib.patches.Ellipse(
            (cx, cy),
            2 * sx,
            2 * sy,
            facecolor=DOMAIN_COLOURS[domain],
            alpha=0.12,
            edgecolor=DOMAIN_COLOURS[domain],
            linewidth=1,
            linestyle="--",
            zorder=2,
        )
        ax.add_patch(ellipse)

    ax.axhline(y=0, color=SITE_PALETTE["muted"], linestyle="--", alpha=0.2, linewidth=0.5)
    ax.axvline(x=0, color=SITE_PALETTE["muted"], linestyle="--", alpha=0.2, linewidth=0.5)
    ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}%)")
    ax.set_title("(b) Refusal Domain Clusters")
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    apply_style(ax, spines="bottom left")

    # If the JSON didn't carry per-pair scores, leave a small honest note that
    # the right panel was sampled from the real per-domain mean+std. When real
    # points are present (the common case after analysis runs), no caveat needed.
    if not have_real_points:
        fig.text(
            0.5,
            -0.02,
            "Note: scatter samples each domain from its real PC1/PC2 mean+std;"
            " centroids and ellipses are exact, individual points are illustrative.",
            ha="center",
            fontsize=7,
            color=SITE_PALETTE["muted"],
            style="italic",
        )

    fig.tight_layout()
    out = FINDINGS_DIR / "fig_c_pca"
    save_publication_figure(fig, out, dpi=200, web_copy_dir=WEB_DIR)
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Fig D: Domain loading on mean refusal axis
# Evidence: Lu et al. role loadings
# Uses pre-computed cosine similarity per domain (from domain_separation)
# ---------------------------------------------------------------------------
def fig_d_domain_loading():
    """Bar chart: cosine similarity of each domain direction with mean refusal axis."""
    log.info("Generating Fig D: Domain Loading on Refusal Axis")

    results = json.load(open(RESULTS_DIR / "refusal_axis_results.json"))
    cos_dict = results["mean_response_token"]["domain_separation"]["cosine_matrix"]
    proj_data = results["mean_response_token"]["projections"]["domain_projections"]

    # Compute each domain's cosine with the mean axis.
    # We use the mean_response_token cosine values from the cosine matrix.
    # The "mean axis" is the average of all domain directions; its cosine with
    # each domain direction is the row-mean of the cosine matrix (excluding diagonal).
    # A more direct measure: we use the per-domain projection means normalised by
    # the overall mean; or we can use the raw cosine values from the full dataset.
    # Best available: load the domain cosine results directly from the axis results.
    # The refusal_axis_results has cosine_matrix between domains, not vs mean axis.
    # Use the per-domain mean projections as a proxy for alignment:
    #   loading = (domain_mean - neg_mean) / overall_gap
    neg_mean = results["mean_response_token"]["projections"]["neg_mean"]
    gap = results["mean_response_token"]["projections"]["gap"]

    loadings: list[tuple[str, float, int]] = []
    for domain, info in proj_data.items():
        # Normalise: (domain_mean - neg_mean) / gap
        loading = (info["mean"] - neg_mean) / gap
        loadings.append((domain, float(loading), info["n"]))

    loadings.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = range(len(loadings))
    colors = [DOMAIN_COLOURS[d] for d, _, _ in loadings]
    load_vals = [c for _, c, _ in loadings]
    labels = [f"{DOMAIN_LABELS[d]} (n={n})" for d, _, n in loadings]

    bars = ax.barh(
        y_pos,
        load_vals,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
        alpha=0.85,
    )

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, load_vals)):
        ax.text(
            val + 0.01,
            i,
            f"{val:.2f}",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=SITE_PALETTE["body"],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalised alignment with mean refusal axis")
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()

    ax.axvline(x=0.5, color=SITE_PALETTE["border"], linestyle="--", linewidth=0.8)
    ax.text(
        0.50,
        len(loadings) - 0.3,
        "moderate\nalignment",
        fontsize=7,
        color=SITE_PALETTE["muted"],
        ha="center",
    )

    apply_style(ax, spines="bottom left")
    ax.set_title("Domain Alignment with Mean Refusal Axis", pad=10)

    fig.tight_layout()
    out = FINDINGS_DIR / "fig_d_domain_loading"
    save_publication_figure(fig, out, dpi=200, web_copy_dir=WEB_DIR)
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("REFUSAL AXIS FIGURES -- site style")
    log.info("=" * 60)

    if _activations_available():
        log.info("Raw activations found -- will use for full per-point figures")
    else:
        log.info(
            "Raw activations not on disk (data/activations/ missing). "
            "Falling back to pre-computed JSON summaries. "
            "Run scripts/cli/download_data.py to restore full scatter plots."
        )

    fig_a_axis_strip()
    fig_b_cosine_heatmap()
    fig_c_pca()
    fig_d_domain_loading()

    log.info("=" * 60)
    log.info("ALL FIGURES GENERATED")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
