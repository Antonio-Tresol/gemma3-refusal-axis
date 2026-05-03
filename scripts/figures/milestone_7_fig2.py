"""
Milestone 7 -- Figure 2: Feature Genealogy.

Shows how the set of refusal-relevant features reorganises as SAE width
grows from 16k to 1M. At each width the relevant-feature set is
decomposed into:
  (a) features that already appeared at 16k ("core survivors"),
  (b) features that first appeared at a larger width ("emerged").

Annotations show the retention rate from 16k and the number of 16k
features lost, quantifying the *replacement* story.

Reads  data/milestone_7_results/width_metrics.json
Writes findings/figures/sae_width_scaling/fig2_feature_genealogy.png  (200 dpi)
       findings/figures/sae_width_scaling/fig2_feature_genealogy.pdf
       web/public/figures/sae_width_scaling/fig2_feature_genealogy.png (copied)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
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
width_order = ["16k", "65k", "262k", "1M"]


def build_site_data(site: str) -> dict:
    """Compute per-width decomposition for one site."""
    entries = [e for e in raw if e["site"] == site]
    entries.sort(key=lambda e: e["width_int"])

    base_ids = set(entries[0]["feature_ids"])  # 16k baseline

    result: dict[str, dict] = {}
    for e in entries:
        ids = set(e["feature_ids"])
        survived = ids & base_ids
        emerged = ids - base_ids
        lost_from_16k = base_ids - ids

        result[e["width"]] = {
            "total": len(ids),
            "survived": len(survived),
            "emerged": len(emerged),
            "lost_from_16k": len(lost_from_16k),
            "base_count": len(base_ids),
            "pct_retained": len(survived) / len(base_ids) * 100 if base_ids else 0,
        }

    return result


lpt = build_site_data("last_prompt_token")
mrt = build_site_data("mean_response_token")

# ---------------------------------------------------------------------------
# Palette for the stacked bars -- using site tokens
# ---------------------------------------------------------------------------
COLOR_SURVIVED = SITE_PALETTE["refuse"]     # navy -- core survivors
COLOR_EMERGED = SITE_PALETTE["orange"]      # clay-orange -- new features
COLOR_BASELINE = SITE_PALETTE["highlight"]  # teal-blue -- 16k baseline row

# ---------------------------------------------------------------------------
# Figure: two-panel horizontal stacked bar chart
# ---------------------------------------------------------------------------
fig, (ax_lpt, ax_mrt) = plt.subplots(
    1, 2, figsize=(8.0, 3.5), sharey=True, constrained_layout=True
)

bar_height = 0.50
y_pos = np.arange(len(width_order))


def draw_panel(
    ax: plt.Axes,
    site_data: dict,
    title: str,
    show_ylabel: bool = True,
) -> None:
    """Draw one panel of the figure."""
    survived_vals = [site_data[w]["survived"] for w in width_order]
    emerged_vals = [site_data[w]["emerged"] for w in width_order]
    totals = [site_data[w]["total"] for w in width_order]
    pct_retained = [site_data[w]["pct_retained"] for w in width_order]

    max_total = max(totals)

    for i, w in enumerate(width_order):
        surv = survived_vals[i]
        emrg = emerged_vals[i]

        if w == "16k":
            # Baseline row: single colour, no stacking
            ax.barh(
                y_pos[i],
                surv,
                height=bar_height,
                color=COLOR_BASELINE,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.text(
                surv / 2,
                y_pos[i],
                f"{surv} (baseline)",
                va="center",
                ha="center",
                fontsize=7.5,
                color=SITE_PALETTE["surface"],
                fontweight="bold",
            )
        else:
            # Survived segment
            ax.barh(
                y_pos[i],
                surv,
                height=bar_height,
                color=COLOR_SURVIVED,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            # Emerged segment
            ax.barh(
                y_pos[i],
                emrg,
                height=bar_height,
                left=surv,
                color=COLOR_EMERGED,
                edgecolor="white",
                linewidth=0.5,
                hatch="//",
                zorder=3,
            )

            if surv >= 4:
                ax.text(
                    surv / 2,
                    y_pos[i],
                    str(surv),
                    va="center",
                    ha="center",
                    fontsize=7.5,
                    color="white",
                    fontweight="bold",
                )
            if emrg >= 4:
                ax.text(
                    surv + emrg / 2,
                    y_pos[i],
                    str(emrg),
                    va="center",
                    ha="center",
                    fontsize=7.5,
                    color=SITE_PALETTE["text"],
                    fontweight="bold",
                )

            total = totals[i]
            pct = pct_retained[i]
            ax.text(
                total + 0.5,
                y_pos[i],
                f"{total}  ({pct:.0f}% of 16k kept)",
                va="center",
                ha="left",
                fontsize=6.5,
                color=SITE_PALETTE["muted"],
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(width_order)
    ax.invert_yaxis()
    ax.set_xlabel("Number of relevant features")
    ax.set_title(title, fontweight="bold", fontsize=9.5)
    ax.set_xlim(0, max_total + max_total * 0.65)

    if show_ylabel:
        ax.set_ylabel("SAE width")

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.axhline(y=0.5, color=SITE_PALETTE["border"], linewidth=0.5, linestyle="--", zorder=1)


draw_panel(ax_lpt, lpt, site_labels["last_prompt_token"], show_ylabel=True)
draw_panel(ax_mrt, mrt, site_labels["mean_response_token"], show_ylabel=False)

# Shared legend at bottom
legend_baseline = mpatches.Patch(
    facecolor=COLOR_BASELINE, edgecolor="white", label="16k baseline"
)
legend_survived = mpatches.Patch(
    facecolor=COLOR_SURVIVED, edgecolor="white", label="Survived from 16k"
)
legend_emerged = mpatches.Patch(
    facecolor=COLOR_EMERGED,
    edgecolor="white",
    hatch="//",
    label="Emerged at larger width",
)
fig.legend(
    handles=[legend_baseline, legend_survived, legend_emerged],
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.06),
    frameon=True,
    fontsize=8,
)

fig.suptitle(
    "Feature Genealogy: Dictionary Reorganisation Across Widths",
    fontsize=11,
    fontweight="bold",
)

out = OUT_DIR / "fig2_feature_genealogy"
save_publication_figure(fig, out, dpi=200, web_copy_dir=WEB_DIR)
plt.close(fig)

print(f"Saved: {out}.png")
