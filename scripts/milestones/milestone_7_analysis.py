"""Milestone 7 Part A: Compute metrics across site × width.

Pure CPU computation from M4/M5/M6 outputs. No GPU, no LLM calls.

Metrics (per site × width, following plan Sec. Milestone 7):
  1. Relevant feature count — top-50 features judged refusal-relevant
  2. Specificity — relevant_count / 50
  3. Mean effect size — mean |Cohen's d| of relevant features
  4. Domain diversity — distinct sub-types among relevant features
  5. Feature retention from 16k — fraction of 16k relevant features still in top-50
  6. New relevant features — features with IDs >= previous width boundary
  7. Dual-validation rate — fraction identified by both Method 1 and Method 2

Usage:
  uv run python milestone_7_analysis.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from refusal_decomposition import CFG, setup_logging

log = setup_logging("milestone_7")

M4_DIR = CFG.data_dir / "milestone_4_results"
M5_DIR = CFG.data_dir / "milestone_5_results"
M6_DIR = CFG.data_dir / "milestone_6_results"
M7_DIR = CFG.data_dir / "milestone_7_results"

WIDTHS = {"16k": 16_384, "65k": 65_536, "262k": 262_144, "1M": 1_048_576}
WIDTH_ORDER = ["16k", "65k", "262k", "1M"]
SITES = ["last_prompt_token", "mean_response_token"]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass
class WidthMetrics:
    site: str
    width: str
    width_int: int
    relevant_count: int
    specificity: float
    mean_effect_size: float
    domain_diversity: int
    sub_type_counts: dict[str, int]
    retention_from_16k: float | None  # None for 16k itself
    new_relevant_count: int
    dual_validation_rate: float
    feature_ids: list[int]  # for downstream use


@dataclass
class FeatureCatalogEntry:
    feature_id: int
    description: str
    sub_types: list[str]
    confidence: str
    cohens_d_by_width: dict[str, float]
    cosine_sim: float | None
    causal_passed: bool | None
    causal_best_layer: int | None
    causal_best_coeff: float | None
    widths_present: list[str]
    site: str


# ---------------------------------------------------------------------------
# Load upstream data
# ---------------------------------------------------------------------------
def load_judgments() -> dict[int, dict]:
    """Load M6 judgments as {feature_id: judgment_dict}."""
    with open(M6_DIR / "judgments.json", encoding="utf-8") as f:
        return {j["feature_id"]: j for j in json.load(f)}


def load_contrastive(site: str, width_label: str) -> list[dict]:
    """Load M4 contrastive top-50 for a site×width."""
    path = M4_DIR / f"contrastive_{site}_{width_label}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def load_cosine(site: str) -> set[int]:
    """Load M4 cosine top-50 feature IDs."""
    path = M4_DIR / f"cosine_{site}.json"
    if path.exists():
        with open(path) as f:
            return {e["feature_id"] for e in json.load(f)}
    return set()


def load_causal() -> dict[int, dict]:
    """Load M5 causal results."""
    path = M5_DIR / "analysis.json"
    if path.exists():
        with open(path) as f:
            return {r["feature_id"]: r for r in json.load(f)}
    return {}


def load_dossiers() -> dict[int, dict]:
    """Load M6 dossiers."""
    with open(M6_DIR / "dossiers.json", encoding="utf-8") as f:
        return {d["feature_id"]: d for d in json.load(f)}


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def compute_all_metrics() -> list[WidthMetrics]:
    """Compute all 7 metrics for each site × width combination."""
    M7_DIR.mkdir(parents=True, exist_ok=True)

    judgments = load_judgments()
    causal_results = load_causal()

    all_metrics: list[WidthMetrics] = []

    for site in SITES:
        cosine_ids = load_cosine(site)

        # Track 16k relevant features for retention metric
        relevant_at_16k: set[int] = set()

        prev_width_int = 0

        for width_label in WIDTH_ORDER:
            width_int = WIDTHS[width_label]

            # Load contrastive top-50
            contrastive = load_contrastive(site, width_label)
            top50_ids = {e["feature_id"] for e in contrastive}
            top50_d = {e["feature_id"]: e["cohens_d"] for e in contrastive}

            # Which are relevant?
            relevant_ids = [
                fid
                for fid in top50_ids
                if fid in judgments and judgments[fid].get("is_refusal_relevant", False)
            ]
            relevant_set = set(relevant_ids)

            # 1. Relevant count
            relevant_count = len(relevant_ids)

            # 2. Specificity
            specificity = (
                relevant_count / 50
                if len(contrastive) >= 50
                else (relevant_count / len(contrastive) if contrastive else 0)
            )

            # 3. Mean effect size (|Cohen's d| of relevant features)
            relevant_d_values = [
                abs(top50_d[fid]) for fid in relevant_ids if fid in top50_d
            ]
            mean_effect = (
                sum(relevant_d_values) / len(relevant_d_values)
                if relevant_d_values
                else 0
            )

            # 4. Domain diversity
            all_subtypes: list[str] = []
            for fid in relevant_ids:
                j = judgments.get(fid, {})
                all_subtypes.extend(j.get("sub_types", []))
            sub_type_counts = dict(Counter(all_subtypes))
            domain_diversity = len(sub_type_counts)

            # 5. Feature retention from 16k
            if width_label == "16k":
                relevant_at_16k = relevant_set
                retention = None
            else:
                if relevant_at_16k:
                    retained = relevant_at_16k & relevant_set
                    retention = len(retained) / len(relevant_at_16k)
                else:
                    retention = 0.0

            # 6. New relevant features (IDs >= previous width boundary)
            new_relevant = [fid for fid in relevant_ids if fid >= prev_width_int]
            new_relevant_count = len(new_relevant) if width_label != "16k" else 0

            # 7. Dual-validation rate
            cosine_in_width = {fid for fid in cosine_ids if fid < width_int}
            dual_validated = relevant_set & cosine_in_width & top50_ids
            dual_rate = len(dual_validated) / len(relevant_ids) if relevant_ids else 0

            metrics = WidthMetrics(
                site=site,
                width=width_label,
                width_int=width_int,
                relevant_count=relevant_count,
                specificity=round(specificity, 3),
                mean_effect_size=round(mean_effect, 4),
                domain_diversity=domain_diversity,
                sub_type_counts=sub_type_counts,
                retention_from_16k=round(retention, 3)
                if retention is not None
                else None,
                new_relevant_count=new_relevant_count,
                dual_validation_rate=round(dual_rate, 3),
                feature_ids=sorted(relevant_ids),
            )
            all_metrics.append(metrics)

            log.info(
                "%s/%s: relevant=%d/%d spec=%.2f |d|=%.3f diversity=%d retention=%s new=%d dual=%.2f",
                site,
                width_label,
                relevant_count,
                len(contrastive),
                specificity,
                mean_effect,
                domain_diversity,
                f"{retention:.2f}" if retention is not None else "n/a",
                new_relevant_count,
                dual_rate,
            )

            prev_width_int = width_int

    # Save metrics
    metrics_path = M7_DIR / "width_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)
    log.info("Metrics saved: %s", metrics_path)

    return all_metrics


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------
def build_catalogue() -> list[FeatureCatalogEntry]:
    """Build full catalogue of all relevant features."""
    judgments = load_judgments()
    dossiers = load_dossiers()
    causal = load_causal()

    catalogue: list[FeatureCatalogEntry] = []
    for fid, j in sorted(judgments.items()):
        if not j.get("is_refusal_relevant", False):
            continue

        d = dossiers.get(fid, {})
        c = causal.get(fid, {})

        # Determine primary site
        site = "mean_response_token"  # default
        for s in SITES:
            for wl in WIDTH_ORDER:
                contrastive = load_contrastive(s, wl)
                if any(e["feature_id"] == fid for e in contrastive):
                    site = s
                    break

        entry = FeatureCatalogEntry(
            feature_id=fid,
            description=j.get("description", ""),
            sub_types=j.get("sub_types", []),
            confidence=j.get("confidence", ""),
            cohens_d_by_width=d.get("cohens_d_by_width", {}),
            cosine_sim=d.get("cosine_sim"),
            causal_passed=c.get("passed") if c else None,
            causal_best_layer=c.get("best_layer") if c else None,
            causal_best_coeff=c.get("best_coeff") if c else None,
            widths_present=d.get("widths_present", []),
            site=site,
        )
        catalogue.append(entry)

    cat_path = M7_DIR / "feature_catalogue.json"
    with open(cat_path, "w") as f:
        json.dump([asdict(e) for e in catalogue], f, indent=2, ensure_ascii=False)
    log.info(
        "Feature catalogue: %d relevant features saved to %s", len(catalogue), cat_path
    )

    return catalogue


# ---------------------------------------------------------------------------
# Narrative test (plan validation gate)
# ---------------------------------------------------------------------------
def narrative_test(metrics: list[WidthMetrics]) -> str:
    """Generate the one-paragraph narrative test from the plan."""
    # Get metrics for the two extremes
    lpt_16k = next(
        (m for m in metrics if m.site == "last_prompt_token" and m.width == "16k"), None
    )
    lpt_1m = next(
        (m for m in metrics if m.site == "last_prompt_token" and m.width == "1M"), None
    )
    mrt_16k = next(
        (m for m in metrics if m.site == "mean_response_token" and m.width == "16k"),
        None,
    )
    mrt_1m = next(
        (m for m in metrics if m.site == "mean_response_token" and m.width == "1M"),
        None,
    )

    paragraphs = []

    for site_name, m16k, m1m in [
        ("last-prompt-token", lpt_16k, lpt_1m),
        ("mean-response-token", mrt_16k, mrt_1m),
    ]:
        if not m16k or not m1m:
            continue

        div_change = (
            "increased"
            if m1m.domain_diversity > m16k.domain_diversity
            else (
                "decreased"
                if m1m.domain_diversity < m16k.domain_diversity
                else "stayed constant"
            )
        )
        supports = (
            "supports"
            if (
                m1m.relevant_count > m16k.relevant_count
                or m1m.domain_diversity > m16k.domain_diversity
            )
            else "does not clearly support"
        )

        para = (
            f"At the {site_name} site: At 16k width, we identified {m16k.relevant_count} "
            f"refusal features with mean |d| of {m16k.mean_effect_size:.3f}. "
            f"At 1M width, we identified {m1m.relevant_count} features with mean |d| of "
            f"{m1m.mean_effect_size:.3f}. Domain diversity {div_change} from "
            f"{m16k.domain_diversity} to {m1m.domain_diversity} sub-types. "
            f"{m1m.new_relevant_count} new relevant features emerged at IDs >= 262k. "
            f"This {supports} the decomposition hypothesis."
        )
        paragraphs.append(para)

    narrative = "\n\n".join(paragraphs)
    log.info("NARRATIVE TEST:\n%s", narrative)

    narrative_path = M7_DIR / "narrative_test.txt"
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write(narrative)

    return narrative


# ---------------------------------------------------------------------------
# Site comparison
# ---------------------------------------------------------------------------
def site_comparison(metrics: list[WidthMetrics]) -> str:
    """One-paragraph extraction site comparison."""
    lpt = [m for m in metrics if m.site == "last_prompt_token"]
    mrt = [m for m in metrics if m.site == "mean_response_token"]

    lpt_total_relevant = sum(m.relevant_count for m in lpt)
    mrt_total_relevant = sum(m.relevant_count for m in mrt)
    lpt_mean_d = sum(m.mean_effect_size for m in lpt) / len(lpt)
    mrt_mean_d = sum(m.mean_effect_size for m in mrt) / len(mrt)
    lpt_max_diversity = max(m.domain_diversity for m in lpt)
    mrt_max_diversity = max(m.domain_diversity for m in mrt)

    comparison = (
        f"Extraction site comparison: The last-prompt-token site yielded {lpt_total_relevant} "
        f"total relevant features across widths (mean |d| = {lpt_mean_d:.3f}, "
        f"max domain diversity = {lpt_max_diversity}), while mean-response-token yielded "
        f"{mrt_total_relevant} (mean |d| = {mrt_mean_d:.3f}, max diversity = {mrt_max_diversity}). "
    )

    if lpt_mean_d > mrt_mean_d and mrt_max_diversity > lpt_max_diversity:
        comparison += (
            "The last-prompt-token site captures a stronger overall refusal signal (higher effect sizes), "
            "while mean-response-token captures more domain-specific variation (higher diversity). "
            "This aligns with the intuition that the last prompt token encodes intent to refuse, "
            "while response tokens encode how the refusal is expressed."
        )
    elif lpt_mean_d > mrt_mean_d:
        comparison += "The last-prompt-token site captures stronger effects overall. "
    else:
        comparison += (
            "The mean-response-token site captures stronger or comparable effects. "
        )

    log.info("SITE COMPARISON:\n%s", comparison)

    comp_path = M7_DIR / "site_comparison.txt"
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(comparison)

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("MILESTONE 7 PART A: COMPUTE METRICS")
    log.info("=" * 60)

    metrics = compute_all_metrics()
    catalogue = build_catalogue()
    narrative = narrative_test(metrics)
    comparison = site_comparison(metrics)

    # Validation gate checks
    log.info("=" * 60)
    log.info("VALIDATION GATE")
    log.info("=" * 60)

    # All metrics computable
    log.info("All metrics computed: %d site×width combinations", len(metrics))

    # Non-trivial pattern check
    for site in SITES:
        site_metrics = [m for m in metrics if m.site == site]
        relevants = [m.relevant_count for m in site_metrics]
        diversities = [m.domain_diversity for m in site_metrics]
        effects = [m.mean_effect_size for m in site_metrics]

        # Check for monotonic trends
        is_relevant_increasing = all(
            relevants[i] <= relevants[i + 1] for i in range(len(relevants) - 1)
        )
        is_diversity_increasing = all(
            diversities[i] <= diversities[i + 1] for i in range(len(diversities) - 1)
        )

        log.info("%s:", site)
        log.info(
            "  Relevant counts across widths: %s %s",
            relevants,
            "(monotonic)" if is_relevant_increasing else "",
        )
        log.info(
            "  Diversity across widths: %s %s",
            diversities,
            "(monotonic)" if is_diversity_increasing else "",
        )
        log.info("  Effect sizes across widths: %s", effects)

    # Narrative test passed if we could fill in all blanks
    log.info(
        "Narrative test: %s",
        "PASS" if "identified" in narrative and "sub-types" in narrative else "FAIL",
    )

    log.info("=" * 60)
    log.info("PART A COMPLETE — metrics ready for visualization")
    log.info("Next: Part B — iterative figure design")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
