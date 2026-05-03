"""
Milestone 7 -- Figure 1: Width-Scaling Curves.

Reads data/milestone_7_results/width_metrics.json and produces:
  - findings/figures/sae_width_scaling/fig1_width_scaling.png (200 dpi)
  - findings/figures/sae_width_scaling/fig1_width_scaling.pdf
  - web/public/figures/sae_width_scaling/fig1_width_scaling.png (copied)

Two extraction sites (LPT, MRT) x four SAE widths (16k-1M).
2x2 subplot grid: Relevant Count, Specificity, Mean |d|, Domain Diversity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    SITE_PALETTE,
    apply_site_style,
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
    "last_prompt_token": "Last Prompt Token (LPT)",
    "mean_response_token": "Mean Response Token (MRT)",
}

data: dict[str, dict[str, list]] = {
    s: {
        "width": [],
        "relevant_count": [],
        "specificity": [],
        "mean_effect_size": [],
        "domain_diversity": [],
    }
    for s in sites
}

for entry in raw:
    s = entry["site"]
    data[s]["width"].append(entry["width_int"])
    data[s]["relevant_count"].append(entry["relevant_count"])
    data[s]["specificity"].append(entry["specificity"])
    data[s]["mean_effect_size"].append(entry["mean_effect_size"])
    data[s]["domain_diversity"].append(entry["domain_diversity"])

# ---------------------------------------------------------------------------
# Two site colours from the site palette (Okabe-Ito, colourblind-safe)
# ---------------------------------------------------------------------------
OI_BLUE = SITE_PALETTE["refuse"]    # navy -- LPT
OI_ORANGE = SITE_PALETTE["orange"]  # clay-orange -- MRT

# ---------------------------------------------------------------------------
# Metrics to plot
# ---------------------------------------------------------------------------
metrics = [
    ("relevant_count", "Relevant features", "Count"),
    ("specificity", "Specificity", "Precision"),
    ("mean_effect_size", "Mean effect size |d|", "Effect size"),
    ("domain_diversity", "Domain diversity", "Domains (of 6)"),
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

# Per-panel label positions: (x, y) in axes coords.
LABEL_POS = [
    (0.04, 0.93),  # (a) top-left
    (0.04, 0.93),  # (b) top-left
    (0.04, 0.93),  # (c) top-left
    (0.04, 0.12),  # (d) bottom-left (MRT starts high)
]

# ---------------------------------------------------------------------------
# Figure: 2x2 grid, shared x-axis per column
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)
axes_flat = axes.ravel()

widths = np.array(data[sites[0]]["width"])  # same for both sites

for idx, (key, title, ylabel) in enumerate(metrics):
    ax = axes_flat[idx]

    # Horizontal grid lines only -- x has only 4 discrete ticks
    ax.yaxis.grid(True, linewidth=0.35, alpha=0.3, color=SITE_PALETTE["border"], zorder=0)
    ax.xaxis.grid(False)

    # Subtle plateau shading (262k-1M region)
    ax.axvspan(
        widths[2],
        widths[3] * 1.15,
        alpha=0.05,
        color=SITE_PALETTE["muted"],
        zorder=0,
    )

    for site, color, marker in [
        ("last_prompt_token", OI_BLUE, "o"),
        ("mean_response_token", OI_ORANGE, "s"),
    ]:
        vals = data[site][key]
        ax.plot(
            widths,
            vals,
            color=color,
            marker=marker,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.8,
            label=site_labels[site],
            zorder=3,
            clip_on=False,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(["16k", "65k", "262k", "1M"])
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Panel label
    lx, ly = LABEL_POS[idx]
    ax.text(
        lx,
        ly,
        PANEL_LABELS[idx],
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top" if ly > 0.5 else "bottom",
        ha="left",
        color=SITE_PALETTE["text"],
    )

    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)

    # Y-axis limits per metric
    if key == "domain_diversity":
        ax.set_ylim(-0.3, 6.8)
        ax.set_yticks([0, 2, 4, 6])

    if key == "specificity":
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if key == "relevant_count":
        ax.set_ylim(0, 50)

    if key == "mean_effect_size":
        ax.set_ylim(0, 1.8)

# x-axis labels only on bottom row
for ax in axes[1, :]:
    ax.set_xlabel("SAE width")

# Single legend in panel (a)
axes_flat[0].legend(loc="lower right", borderpad=0.5)

# Save to findings/ and copy to web/
out = OUT_DIR / "fig1_width_scaling"
save_publication_figure(fig, out, dpi=200, web_copy_dir=WEB_DIR)
plt.close(fig)

print(f"Saved to {out}.png")
