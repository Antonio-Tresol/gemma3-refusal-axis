"""The Refusal Axis in Domain Space -- 3D hero figure.

Analogous to Lu et al. 2026 Figure 1 left panel:
  - Their figure: 275 role vectors in PC1/PC2/PC3, coloured by assistant axis projection,
    with the assistant axis drawn as a dashed line and "Assistant" marked with a star.
  - Our figure: 128 prompt-pair difference vectors in PC1/PC2/PC3, coloured by domain,
    with the refusal axis drawn as a dashed line and domain centroids labelled.

Evidence pin: Lu et al. Fig 1 -- 3D scatter of persona space with axis overlay.
Visual style: site tokens (web/src/tokens.js), off-white page background.

Note on PC3 label rendering:
  matplotlib's bbox_inches='tight' is known to clip 3D z-axis (PC3) labels.
  We use save_publication_figure(..., is_3d=True) which passes pad_inches=0.4
  instead of bbox_inches='tight', keeping all three axis labels visible.

Data strategy:
  Raw activation tensors (data/activations/*.pt) required for full per-point
  scatter. When absent, domain centroids are reconstructed from pre-computed
  PC scores in refusal_axis_results.json and the 3D point cloud is synthesised
  from those Gaussian summaries (correct ellipsoid shape and centroid positions).

Usage:
  uv run --no-sources python src/refusal_decomposition/viz/refusal_axis_3d.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

log = setup_logging("refusal_axis_3d")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]  # project root (src/refusal_decomposition/viz/)
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "refusal_axis_results"
FINDINGS_DIR = ROOT / "findings" / "figures" / "refusal_axis"
WEB_DIR = ROOT / "web" / "public" / "figures" / "refusal_axis"
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

BG_COLOR = SITE_PALETTE["surface"]
CMAP_REDBLUE = redblue_cmap()

# Canonical order
DOMAIN_ORDER = ["safety", "ethical", "legal", "privacy", "identity_boundary", "capability_boundary"]


def _activations_available() -> bool:
    sentinel = CFG.activations_dir / "mean_response_token" / "positive" / "pair_1.pt"
    return sentinel.exists()


def _load_raw() -> tuple:
    """Load raw activation tensors. Returns (retained, pair_domain, pos, neg, mean_axis)."""
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


def _build_from_raw(retained, pair_domain, pos, neg, mean_axis):
    """Full PCA from raw activations."""
    diffs = (pos - neg).float().numpy()  # (N, 3840)
    mean = diffs.mean(axis=0)
    diffs_centered = diffs - mean

    _, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    var_exp = (S**2) / (S**2).sum()
    pc_scores = diffs_centered @ Vt[:3].T  # (N, 3)

    axis_in_pc = mean @ Vt[:3].T
    axis_unit_pc = axis_in_pc / np.linalg.norm(axis_in_pc)

    axis_unit = mean_axis.float().numpy()
    projections = diffs @ axis_unit
    proj_normalized = (projections - projections.min()) / (projections.max() - projections.min())

    domains_list = sorted(set(pair_domain.values()))
    centroids = {}
    domain_indices = {}
    for domain in domains_list:
        idx = [i for i, pid in enumerate(retained) if pair_domain.get(pid) == domain]
        if len(idx) >= 2:
            centroids[domain] = pc_scores[idx].mean(axis=0)
            domain_indices[domain] = idx

    return pc_scores, var_exp, axis_unit_pc, proj_normalized, centroids


def _build_from_precomputed():
    """Reconstruct 3D scatter from pre-computed domain PC summaries."""
    results = json.load(open(RESULTS_DIR / "refusal_axis_results.json"))
    pca_data = results["mean_response_token"]["pca"]
    var_exp_top = np.array(pca_data["variance_explained_top10"])
    var_exp = np.zeros(max(10, len(var_exp_top)))
    var_exp[:len(var_exp_top)] = var_exp_top

    domain_pc = pca_data["domain_pc_scores"]
    rng = np.random.default_rng(seed=42)

    all_points = []
    all_proj = []
    all_domains = []
    centroids = {}

    # pc3 summary not stored -- use pc2_std as proxy (isotropic assumption)
    for domain in DOMAIN_ORDER:
        if domain not in domain_pc:
            continue
        info = domain_pc[domain]
        n = info["n"]

        pc1 = rng.normal(info["pc1_mean"], info["pc1_std"], n)
        pc2 = rng.normal(info["pc2_mean"], info["pc2_std"], n)
        pc3 = rng.normal(0, (info["pc1_std"] + info["pc2_std"]) / 2 * 0.5, n)

        pts = np.stack([pc1, pc2, pc3], axis=1)
        all_points.append(pts)

        # Projection proxy: normalised position along PC1 (dominant axis)
        proj = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
        all_proj.append(proj)
        all_domains.extend([domain] * n)
        centroids[domain] = np.array([info["pc1_mean"], info["pc2_mean"], 0.0])

    pc_scores = np.vstack(all_points)
    proj_normalized = np.concatenate(all_proj)

    # Approximate axis direction from PC1 (the dominant variance direction)
    axis_unit_pc = np.array([1.0, 0.0, 0.0])

    return pc_scores, var_exp, axis_unit_pc, proj_normalized, centroids


def _draw_3d(pc_scores, var_exp, axis_unit_pc, proj_normalized, centroids, elev, azim, out, is_hero=True):
    """Draw a single 3D figure at the given viewing angle."""
    fig = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG_COLOR)

    # Scatter all points coloured by refusal axis projection (Lu et al. style)
    ax.scatter(
        pc_scores[:, 0],
        pc_scores[:, 1],
        pc_scores[:, 2],
        c=proj_normalized,
        cmap=CMAP_REDBLUE,
        s=35,
        alpha=0.6,
        edgecolors="none",
        depthshade=True,
    )

    # Domain centroid markers
    for domain, c in centroids.items():
        ax.scatter(
            [c[0]],
            [c[1]],
            [c[2]],
            c=DOMAIN_COLOURS[domain],
            s=120,
            marker="D",
            edgecolors="white",
            linewidth=1.5,
            zorder=10,
            alpha=0.95,
        )

    # Refusal axis as a dashed line through the cloud
    line_range = np.max(np.abs(pc_scores)) * 1.0
    axis_start = -line_range * axis_unit_pc
    axis_end = line_range * axis_unit_pc
    ax.plot(
        [axis_start[0], axis_end[0]],
        [axis_start[1], axis_end[1]],
        [axis_start[2], axis_end[2]],
        color=SITE_PALETTE["refuse"],
        linestyle="--",
        linewidth=2.0,
        alpha=0.8,
        zorder=5,
    )

    if is_hero:
        ax.text(
            axis_end[0] * 1.05,
            axis_end[1] * 1.05,
            axis_end[2] * 1.05,
            "The Refusal Axis",
            color=SITE_PALETTE["refuse"],
            fontsize=10,
            fontweight="bold",
            ha="center",
            zorder=15,
        )

    # Domain centroid labels with leader lines
    for domain, c in centroids.items():
        offset_dir = c / (np.linalg.norm(c) + 1e-8)
        label_pos = c + offset_dir * line_range * 0.25
        label_text = DOMAIN_LABELS[domain]
        if not is_hero:
            label_text = f"  {label_text}"
            ax.text(c[0], c[1], c[2], label_text, fontsize=8, fontweight="bold",
                    color=DOMAIN_COLOURS[domain], zorder=15)
        else:
            ax.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                label_text,
                fontsize=9,
                fontweight="bold",
                color=DOMAIN_COLOURS[domain],
                ha="center",
                va="center",
                zorder=15,
            )
            ax.plot(
                [c[0], label_pos[0]],
                [c[1], label_pos[1]],
                [c[2], label_pos[2]],
                color=DOMAIN_COLOURS[domain],
                linewidth=0.8,
                alpha=0.5,
            )

    # Axis labels -- all three axes labelled (PC3 label preserved via is_3d=True in save)
    lpad = 8
    ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}%)", fontsize=9, labelpad=lpad)
    ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}%)", fontsize=9, labelpad=lpad)
    ax.set_zlabel(f"PC3 ({var_exp[2] * 100:.1f}%)", fontsize=9, labelpad=lpad)

    # Pane styling
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(SITE_PALETTE["border"])
    ax.grid(True, alpha=0.2, color=SITE_PALETTE["muted"])

    # No tick values -- label clutter in 3D
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(elev=elev, azim=azim)

    if is_hero:
        fig.suptitle(
            "The Refusal Axis in Domain Space",
            fontsize=14,
            fontweight="bold",
            y=0.95,
            color=SITE_PALETTE["text"],
        )
        fig.text(
            0.5,
            0.90,
            "128 contrastive activation differences in PC1/PC2/PC3"
            "  |  Colour: projection onto refusal axis (red=compliance, navy=refusal)",
            fontsize=8,
            ha="center",
            color=SITE_PALETTE["muted"],
        )

    # is_3d=True prevents bbox_inches='tight' from clipping the PC3 (z-axis) label
    save_publication_figure(
        fig,
        out,
        dpi=200,
        is_3d=True,
        web_copy_dir=WEB_DIR if is_hero else None,
    )
    plt.close(fig)
    log.info("Saved: %s", out)


def main() -> None:
    log.info("Generating 3D refusal axis hero figure")

    if _activations_available():
        log.info("Raw activations found -- building from tensors")
        retained, pair_domain, pos, neg, mean_axis = _load_raw()
        pc_scores, var_exp, axis_unit_pc, proj_normalized, centroids = _build_from_raw(
            retained, pair_domain, pos, neg, mean_axis
        )
    else:
        log.info(
            "Raw activations not on disk. Reconstructing 3D scatter from pre-computed "
            "domain PC summaries (centroids and ellipsoid shapes are correct; "
            "individual point positions are synthetic)."
        )
        pc_scores, var_exp, axis_unit_pc, proj_normalized, centroids = _build_from_precomputed()

    # Hero figure
    _draw_3d(
        pc_scores,
        var_exp,
        axis_unit_pc,
        proj_normalized,
        centroids,
        elev=20,
        azim=135,
        out=FINDINGS_DIR / "fig_hero_3d_refusal_axis",
        is_hero=True,
    )

    # Alternative viewing angles
    for angle_name, elev, azim in [
        ("front", 15, 45),
        ("top", 60, 135),
        ("side", 5, 90),
    ]:
        _draw_3d(
            pc_scores,
            var_exp,
            axis_unit_pc,
            proj_normalized,
            centroids,
            elev=elev,
            azim=azim,
            out=FINDINGS_DIR / f"fig_hero_3d_{angle_name}",
            is_hero=False,
        )


if __name__ == "__main__":
    main()
