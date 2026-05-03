"""
Milestone 7 -- Figure 3: Domain-Specificity Emergence.

Iteration 5 (final): Clean normalised heatmap with mechanism separators.
Shows fraction of relevant features tagged with each sub-type, per width.
Horizontal lines separate value-based / identity / capability mechanisms.

Reads data/milestone_7_results/width_metrics.json and produces:
  - findings/figures/sae_width_scaling/fig3_domain_specificity.png (200 dpi)
  - findings/figures/sae_width_scaling/fig3_domain_specificity.pdf
  - web/public/figures/sae_width_scaling/fig3_domain_specificity.png (copied)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_PATH = ROOT / "data" / "milestone_7_results" / "width_metrics.json"
OUT_DIR = ROOT / "findings" / "figures" / "sae_width_scaling"
WEB_DIR = ROOT / "web" / "public" / "figures" / "sae_width_scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared site style
# ---------------------------------------------------------------------------
from refusal_decomposition.viz._style import (  # noqa: E402
    DOMAIN_COLOURS,
    SITE_PALETTE,
    apply_site_style,
    redblue_cmap,
    save_publication_figure,
)

matplotlib.use("Agg")
apply_site_style()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(DATA_PATH) as f:
    raw = json.load(f)

sites = ["last_prompt_token", "mean_response_token"]
site_labels = {
    "last_prompt_token": "(a) Last Prompt Token",
    "mean_response_token": "(b) Mean Response Token",
}
width_labels = ["16k", "65k", "262k", "1M"]

# Grouped by mechanism type (value-based, then identity, then capability)
sub_type_order = [
    "safety",
    "legal",
    "ethical",
    "privacy",
    "identity_boundary",
    "capability_boundary",
]
sub_type_display = {
    "safety": "Safety",
    "legal": "Legal",
    "ethical": "Ethical",
    "privacy": "Privacy",
    "identity_boundary": "Identity",
    "capability_boundary": "Capability",
}

# Separator lines between mechanism groups (after Privacy, after Identity)
GROUP_BOUNDARIES = [3.5, 4.5]

# Build normalised matrices
matrices_raw: dict[str, np.ndarray] = {}
matrices_norm: dict[str, np.ndarray] = {}
relevant_counts: dict[str, list[int]] = {}

for site in sites:
    site_data = [e for e in raw if e["site"] == site]
    mat = np.zeros((len(sub_type_order), len(width_labels)))
    counts = []
    for j, wl in enumerate(width_labels):
        entry = next(e for e in site_data if e["width"] == wl)
        n = entry["relevant_count"]
        counts.append(n)
        for i, st in enumerate(sub_type_order):
            mat[i, j] = entry["sub_type_counts"].get(st, 0)
    matrices_raw[site] = mat.copy()
    relevant_counts[site] = counts
    norm = mat.copy()
    for j in range(norm.shape[1]):
        if counts[j] > 0:
            norm[:, j] /= counts[j]
    matrices_norm[site] = norm

# ---------------------------------------------------------------------------
# Figure: side-by-side normalised heatmaps
# Use the site RedBlue cmap re-anchored to 0->1 (YlOrRd is replaced by site cmap)
# ---------------------------------------------------------------------------
cmap = redblue_cmap()

fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), constrained_layout=True, sharey=True)

for ax_idx, (ax, site) in enumerate(zip(axes, sites)):
    mat_norm = matrices_norm[site]
    mat_raw = matrices_raw[site]

    im = ax.imshow(mat_norm, aspect="auto", cmap=cmap, vmin=0, vmax=1.0)

    # Annotate cells with raw count
    for i in range(mat_norm.shape[0]):
        for j in range(mat_norm.shape[1]):
            raw_val = int(mat_raw[i, j])
            pct = mat_norm[i, j]
            color = "white" if pct > 0.55 else SITE_PALETTE["text"]
            ax.text(
                j,
                i,
                str(raw_val),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    # Mechanism-group separator lines
    for boundary in GROUP_BOUNDARIES:
        ax.axhline(y=boundary, color=SITE_PALETTE["body"], linewidth=1.2)

    ax.set_xticks(range(len(width_labels)))
    xlabels = [
        f"{wl}\n(n={relevant_counts[site][j]})" for j, wl in enumerate(width_labels)
    ]
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_xlabel("SAE width")
    ax.set_title(site_labels[site], fontweight="bold", fontsize=10)

# Y-axis on left panel only
axes[0].set_yticks(range(len(sub_type_order)))
axes[0].set_yticklabels([sub_type_display[st] for st in sub_type_order], fontsize=8.5)
plt.setp(axes[1].get_yticklabels(), visible=False)

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
cbar.set_label("Fraction of features", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.suptitle(
    "Domain-Specificity Emergence Across SAE Widths",
    fontsize=12,
    fontweight="bold",
)

fig.text(
    0.5,
    -0.01,
    "Rows grouped: value-based refusal (Safety-Privacy)"
    " | identity honesty | capability acknowledgment."
    "  Colour = fraction; numbers = raw counts.",
    ha="center",
    va="top",
    fontsize=7,
    fontstyle="italic",
    color=SITE_PALETTE["muted"],
)

out = OUT_DIR / "fig3_domain_specificity"
save_publication_figure(fig, out, dpi=200, web_copy_dir=WEB_DIR)
plt.close(fig)

print(f"Saved to {out}.png")
