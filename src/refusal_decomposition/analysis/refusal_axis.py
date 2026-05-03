"""Refusal Axis Analysis — Geometric decomposition of the refusal direction.

Extends the Assistant Axis methodology (Lu et al., 2026, arXiv:2601.10387)
to refusal behavior in Gemma 3 12B. Uses existing data from Milestones 2-3
(no new generations needed).

Key idea: Lu et al. find a single "Assistant Axis" separating default
assistant behavior from role-playing. We have the analogous "Refusal Axis"
separating refusing responses from compliant ones. We analyze its geometry.

Evidence pins:
  - Axis formula: mean(positive) - mean(negative) — Lu et al. (2026, arXiv:2601.10387) Sec 3.1,
    also Arditi et al. (2024, arXiv:2406.11717) for the refusal-specific case
  - Projection: normalized dot product — Lu et al. (2026, arXiv:2601.10387) assistant_axis/axis.py
  - PCA on activation space — Lu et al. (2026, arXiv:2601.10387) pipeline/5_axis.py
  - Mean over assistant response tokens — Lu et al. (2026, arXiv:2601.10387) internals/activations.py

Analyses:
  1. Project all retained-pair activations onto the refusal axis
  2. Per-domain projections — do domains separate along the axis?
  3. PCA on the activation differences — is refusal truly 1D?
  4. Cross-layer analysis — which layers carry the refusal signal?

Usage:
  uv run python refusal_axis_analysis.py
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from refusal_decomposition import CFG, setup_logging

matplotlib.use("Agg")

log = setup_logging("refusal_axis")

DATA_DIR = CFG.data_dir
FINDINGS_DIR = Path("findings") / "figures" / "refusal_axis"
ROOT = Path(__file__).resolve().parent

# Okabe-Ito colorblind-safe palette
COLORS = {
    "safety": "#D55E00",
    "legal": "#CC79A7",
    "ethical": "#E69F00",
    "privacy": "#0072B2",
    "capability_boundary": "#009E73",
    "identity_boundary": "#56B4E9",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_retained_data() -> tuple[
    list[int],  # retained pair IDs
    dict[int, str],  # pair_id -> domain
    dict[int, int],  # pair_id -> positive refusal score
    dict[int, int],  # pair_id -> negative refusal score
]:
    """Load retained pairs and their metadata."""
    retained_ids: list[int] = json.load(open(DATA_DIR / "retained_pairs.json"))

    # Domain lookup from manifest
    with open(CFG.manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    pair_domain: dict[int, str] = {}
    for entry in manifest:
        if entry["pair_id"] in set(retained_ids) and entry["condition"] == "positive":
            pair_domain[entry["pair_id"]] = entry["domain"]

    # Refusal scores
    with open(DATA_DIR / "refusal_scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    pos_scores = {
        s["pair_id"]: s["refusal_score"] for s in scores if s["condition"] == "positive"
    }
    neg_scores = {
        s["pair_id"]: s["refusal_score"] for s in scores if s["condition"] == "negative"
    }

    return retained_ids, pair_domain, pos_scores, neg_scores


def load_activations(
    site: str, retained_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load positive and negative activations for a site.

    Returns: (pos_acts, neg_acts) each shape (N, d_model)
    """
    pos_acts = torch.stack(
        [
            torch.load(
                CFG.activations_dir / site / "positive" / f"pair_{pid}.pt",
                weights_only=True,
            )
            for pid in retained_ids
        ]
    )
    neg_acts = torch.stack(
        [
            torch.load(
                CFG.activations_dir / site / "negative" / f"pair_{pid}.pt",
                weights_only=True,
            )
            for pid in retained_ids
        ]
    )
    return pos_acts, neg_acts


# ---------------------------------------------------------------------------
# Analysis 1: Project onto refusal axis
# Evidence: Lu et al. (2026, arXiv:2601.10387) assistant_axis/axis.py — project() function
# Projection = act @ (axis / ||axis||) = ||act|| * cos(act, axis)
# ---------------------------------------------------------------------------
def analysis_1_projections(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    refusal_dir: torch.Tensor,
    retained_ids: list[int],
    pair_domain: dict[int, str],
    site: str,
) -> dict:
    """Project all activations onto the refusal axis."""
    log.info("=" * 60)
    log.info("Analysis 1: Projections onto refusal axis (%s)", site)
    log.info("=" * 60)

    # Normalize axis — Lu et al. (2026, arXiv:2601.10387) axis.py: ax = ax / (ax.norm() + 1e-8)
    axis_unit = refusal_dir / (refusal_dir.norm() + 1e-8)

    pos_proj = (pos_acts.float() @ axis_unit.float()).numpy()  # (N,)
    neg_proj = (neg_acts.float() @ axis_unit.float()).numpy()  # (N,)

    log.info(
        "Positive (should refuse) projections: mean=%.1f std=%.1f",
        pos_proj.mean(),
        pos_proj.std(),
    )
    log.info(
        "Negative (should answer) projections: mean=%.1f std=%.1f",
        neg_proj.mean(),
        neg_proj.std(),
    )

    # Separation
    gap = pos_proj.mean() - neg_proj.mean()
    pooled_std = np.sqrt((pos_proj.std() ** 2 + neg_proj.std() ** 2) / 2)
    cohens_d = gap / pooled_std if pooled_std > 0 else 0
    log.info("Gap: %.1f, Cohen's d: %.2f", gap, cohens_d)

    # Classification accuracy (simple threshold at midpoint)
    threshold = (pos_proj.mean() + neg_proj.mean()) / 2
    correct_pos = (pos_proj > threshold).sum()
    correct_neg = (neg_proj <= threshold).sum()
    accuracy = (correct_pos + correct_neg) / (len(pos_proj) + len(neg_proj))
    log.info("Linear separability (midpoint threshold): %.1f%%", accuracy * 100)

    # Per-domain projections
    log.info("Per-domain positive projections:")
    domain_projections: dict[str, list[float]] = defaultdict(list)
    for i, pid in enumerate(retained_ids):
        domain = pair_domain.get(pid, "unknown")
        domain_projections[domain].append(float(pos_proj[i]))

    for domain in sorted(domain_projections):
        vals = domain_projections[domain]
        log.info(
            "  %-25s n=%2d mean=%7.1f std=%6.1f",
            domain,
            len(vals),
            np.mean(vals),
            np.std(vals),
        )

    return {
        "site": site,
        "pos_mean": float(pos_proj.mean()),
        "neg_mean": float(neg_proj.mean()),
        "gap": float(gap),
        "cohens_d": round(cohens_d, 3),
        "accuracy": round(float(accuracy), 3),
        "domain_projections": {
            d: {"mean": round(np.mean(v), 1), "std": round(np.std(v), 1), "n": len(v)}
            for d, v in sorted(domain_projections.items())
        },
    }


# ---------------------------------------------------------------------------
# Analysis 2: Per-domain separation
# Novel analysis — not in Lu et al. (2026, arXiv:2601.10387). They have roles; we have refusal domains.
# ---------------------------------------------------------------------------
def analysis_2_domain_separation(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    retained_ids: list[int],
    pair_domain: dict[int, str],
    site: str,
) -> dict:
    """Compute per-domain refusal directions and cross-domain similarities."""
    log.info("=" * 60)
    log.info("Analysis 2: Per-domain refusal directions (%s)", site)
    log.info("=" * 60)

    domains = sorted(set(pair_domain.values()))
    domain_directions: dict[str, torch.Tensor] = {}

    for domain in domains:
        idx = [
            i for i, pid in enumerate(retained_ids) if pair_domain.get(pid) == domain
        ]
        if len(idx) < 3:
            log.info("  %-25s n=%d — too few samples, skipping", domain, len(idx))
            continue

        idx_t = torch.tensor(idx)
        d_pos = pos_acts[idx_t].mean(0)  # (d_model,)
        d_neg = neg_acts[idx_t].mean(0)  # (d_model,)
        direction = d_pos - d_neg
        domain_directions[domain] = direction

        log.info("  %-25s n=%2d norm=%.1f", domain, len(idx), direction.norm().item())

    # Cross-domain cosine similarity matrix
    # Evidence: If all domains share the same refusal direction, cosine ≈ 1.
    # If domains have distinct mechanisms, cosine < 1.
    # This extends Joad et al. (2026, arXiv:2602.02132)'s "same control knob" question.
    log.info("Cross-domain cosine similarity:")
    domain_list = sorted(domain_directions.keys())
    cos_matrix: dict[str, dict[str, float]] = {}
    for d1 in domain_list:
        cos_matrix[d1] = {}
        for d2 in domain_list:
            cos = torch.nn.functional.cosine_similarity(
                domain_directions[d1].unsqueeze(0),
                domain_directions[d2].unsqueeze(0),
            ).item()
            cos_matrix[d1][d2] = round(cos, 3)
        log.info("  %-20s %s", d1, {d2: cos_matrix[d1][d2] for d2 in domain_list})

    return {"site": site, "cosine_matrix": cos_matrix}


# ---------------------------------------------------------------------------
# Analysis 3: PCA — Is refusal 1D or multi-dimensional?
# Evidence: Lu et al. (2026, arXiv:2601.10387) pipeline/5_axis.py — PCA on role vectors.
# They find 4-19 components explain 70% of variance.
# We ask: how many components explain the refusal activation differences?
# ---------------------------------------------------------------------------
def analysis_3_pca(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    retained_ids: list[int],
    pair_domain: dict[int, str],
    site: str,
) -> dict:
    """PCA on activation differences to measure refusal dimensionality."""
    log.info("=" * 60)
    log.info("Analysis 3: PCA on refusal space (%s)", site)
    log.info("=" * 60)

    # Compute per-pair difference vectors
    # Evidence: Analogous to Lu et al. (2026, arXiv:2601.10387) computing per-role vectors, then PCA
    diffs = (pos_acts - neg_acts).float().numpy()  # (N, d_model)

    # Mean-center — Lu et al. (2026, arXiv:2601.10387) pca.py uses MeanScaler
    mean = diffs.mean(axis=0)
    diffs_centered = diffs - mean

    # SVD (more numerically stable than covariance for high-dim)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    variance_explained = (S**2) / (S**2).sum()
    cumulative = np.cumsum(variance_explained)

    # Report
    for k in [1, 2, 3, 5, 10, 20]:
        if k <= len(cumulative):
            log.info(
                "  Top %2d components: %.1f%% variance", k, cumulative[k - 1] * 100
            )

    # Elbow point
    dims_70 = int(np.searchsorted(cumulative, 0.70)) + 1
    dims_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    log.info("  Dimensions for 70%%: %d", dims_70)
    log.info("  Dimensions for 90%%: %d", dims_90)
    log.info("  Dimensions for 95%%: %d", dims_95)

    # PC1 alignment with refusal direction
    refusal_dir = torch.from_numpy(mean).float()
    refusal_unit = refusal_dir / refusal_dir.norm()
    pc1 = torch.from_numpy(Vt[0]).float()
    pc1_alignment = torch.nn.functional.cosine_similarity(
        refusal_unit.unsqueeze(0), pc1.unsqueeze(0)
    ).item()
    log.info("  PC1 alignment with refusal direction: %.3f", pc1_alignment)

    # Project per-domain onto PC1 and PC2
    # Evidence: Lu et al. (2026, arXiv:2601.10387) project roles onto PC1 to get "assistant axis loading"
    pc_scores = diffs_centered @ Vt[:2].T  # (N, 2)
    domain_pc: dict[str, dict] = {}
    for domain in sorted(set(pair_domain.values())):
        idx = [
            i for i, pid in enumerate(retained_ids) if pair_domain.get(pid) == domain
        ]
        if len(idx) < 3:
            continue
        d_scores = pc_scores[idx]
        domain_pc[domain] = {
            "pc1_mean": round(float(d_scores[:, 0].mean()), 2),
            "pc1_std": round(float(d_scores[:, 0].std()), 2),
            "pc2_mean": round(float(d_scores[:, 1].mean()), 2),
            "pc2_std": round(float(d_scores[:, 1].std()), 2),
            "n": len(idx),
        }
        log.info(
            "  %-20s PC1=%6.1f±%.1f  PC2=%6.1f±%.1f",
            domain,
            d_scores[:, 0].mean(),
            d_scores[:, 0].std(),
            d_scores[:, 1].mean(),
            d_scores[:, 1].std(),
        )

    return {
        "site": site,
        "variance_explained_top10": [
            round(float(v), 4) for v in variance_explained[:10]
        ],
        "cumulative_top10": [round(float(c), 4) for c in cumulative[:10]],
        "dims_70": dims_70,
        "dims_90": dims_90,
        "dims_95": dims_95,
        "pc1_alignment_with_refusal": round(pc1_alignment, 3),
        "domain_pc_scores": domain_pc,
        "pc_scores": pc_scores.tolist(),
        "domains_ordered": [pair_domain.get(pid, "unknown") for pid in retained_ids],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_projection_distributions(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    refusal_dir: torch.Tensor,
    retained_ids: list[int],
    pair_domain: dict[int, str],
    site: str,
) -> None:
    """Plot projection distributions: overall and per-domain."""
    axis_unit = refusal_dir / (refusal_dir.norm() + 1e-8)
    pos_proj = (pos_acts.float() @ axis_unit.float()).numpy()
    neg_proj = (neg_acts.float() @ axis_unit.float()).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: overall distribution
    ax = axes[0]
    ax.hist(
        neg_proj, bins=20, alpha=0.7, color="#0072B2", label="Benign (should answer)"
    )
    ax.hist(
        pos_proj, bins=20, alpha=0.7, color="#D55E00", label="Harmful (should refuse)"
    )
    ax.axvline(
        x=(pos_proj.mean() + neg_proj.mean()) / 2,
        color="grey",
        linestyle="--",
        alpha=0.5,
        label="Decision boundary",
    )
    ax.set_xlabel("Projection onto refusal axis")
    ax.set_ylabel("Count")
    ax.set_title("(a) Overall separation")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: per-domain positive projections
    ax = axes[1]
    domains = sorted(set(pair_domain.values()))
    domain_data = []
    domain_labels = []
    domain_colors = []
    for domain in domains:
        idx = [
            i for i, pid in enumerate(retained_ids) if pair_domain.get(pid) == domain
        ]
        if len(idx) < 3:
            continue
        domain_data.append([pos_proj[i] for i in idx])
        domain_labels.append(domain.replace("_", "\n"))
        domain_colors.append(COLORS.get(domain, "#999999"))

    bp = ax.boxplot(domain_data, labels=domain_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], domain_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Projection onto refusal axis")
    ax.set_title("(b) Per-domain (harmful prompts)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)

    fig.suptitle(f"Refusal Axis Projections — {site}", fontsize=11, y=1.02)
    fig.tight_layout()

    out = FINDINGS_DIR / f"refusal_axis_projections_{site}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


def plot_pca_scatter(pca_result: dict, site: str) -> None:
    """Plot PCA scatter of activation differences colored by domain."""
    pc_scores = np.array(pca_result["pc_scores"])
    domains = pca_result["domains_ordered"]
    var_exp = pca_result["variance_explained_top10"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for domain in sorted(set(domains)):
        idx = [i for i, d in enumerate(domains) if d == domain]
        if len(idx) < 3:
            continue
        color = COLORS.get(domain, "#999999")
        ax.scatter(
            pc_scores[idx, 0],
            pc_scores[idx, 1],
            c=color,
            label=domain,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}% variance)")
    ax.set_title(f"PCA of Refusal Activation Differences — {site}")
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    ax.axhline(y=0, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="grey", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FINDINGS_DIR / f"refusal_axis_pca_{site}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


def plot_cosine_matrix(cos_result: dict, site: str) -> None:
    """Plot cross-domain cosine similarity heatmap."""
    matrix = cos_result["cosine_matrix"]
    domains = sorted(matrix.keys())
    n = len(domains)

    data = np.array([[matrix[d1][d2] for d2 in domains] for d1 in domains])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [d.replace("_boundary", "").replace("_", "\n") for d in domains]
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if data[i, j] > 0.8 else "black"
            ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
    ax.set_title(f"Cross-Domain Refusal Direction Similarity — {site}", fontsize=10)
    fig.tight_layout()

    out = FINDINGS_DIR / f"refusal_axis_cosine_{site}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("REFUSAL AXIS ANALYSIS")
    log.info("Extending Lu et al. (2026, arXiv:2601.10387) (Assistant Axis) to refusal behavior")
    log.info("=" * 60)

    retained_ids, pair_domain, pos_scores, neg_scores = load_retained_data()
    log.info("Retained pairs: %d", len(retained_ids))
    log.info(
        "Domains: %s",
        dict(
            defaultdict(
                int,
                {
                    d: sum(1 for p in retained_ids if pair_domain.get(p) == d)
                    for d in set(pair_domain.values())
                },
            )
        ),
    )

    all_results: dict = {}

    for site in ("last_prompt_token", "mean_response_token"):
        log.info("\n%s", "=" * 60)
        log.info("SITE: %s", site)
        log.info("=" * 60)

        pos_acts, neg_acts = load_activations(site, retained_ids)
        log.info("Loaded: pos=%s neg=%s", tuple(pos_acts.shape), tuple(neg_acts.shape))

        # Load refusal direction (already computed in M3, recomputed with retained pairs)
        refusal_dir = torch.load(
            DATA_DIR / f"refusal_direction_{site}.pt", weights_only=True
        )

        # Run analyses
        proj_result = analysis_1_projections(
            pos_acts, neg_acts, refusal_dir, retained_ids, pair_domain, site
        )
        domain_result = analysis_2_domain_separation(
            pos_acts, neg_acts, retained_ids, pair_domain, site
        )
        pca_result = analysis_3_pca(pos_acts, neg_acts, retained_ids, pair_domain, site)

        all_results[site] = {
            "projections": proj_result,
            "domain_separation": domain_result,
            "pca": pca_result,
        }

        # Figures
        plot_projection_distributions(
            pos_acts, neg_acts, refusal_dir, retained_ids, pair_domain, site
        )
        plot_pca_scatter(pca_result, site)
        plot_cosine_matrix(domain_result, site)

    # Save all results
    results_path = FINDINGS_DIR / "refusal_axis_results.json"

    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=make_serializable)
    log.info("\nResults saved: %s", results_path)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for site in ("last_prompt_token", "mean_response_token"):
        r = all_results[site]
        log.info("%s:", site)
        log.info(
            "  Refusal axis separability: d=%.2f, accuracy=%.1f%%",
            r["projections"]["cohens_d"],
            r["projections"]["accuracy"] * 100,
        )
        log.info(
            "  PCA dims for 70%%: %d, for 90%%: %d",
            r["pca"]["dims_70"],
            r["pca"]["dims_90"],
        )
        log.info(
            "  PC1-refusal alignment: %.3f", r["pca"]["pc1_alignment_with_refusal"]
        )


if __name__ == "__main__":
    main()
