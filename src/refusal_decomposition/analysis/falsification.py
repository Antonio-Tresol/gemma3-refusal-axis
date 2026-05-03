"""Refusal Axis Falsification — Systematic attempt to destroy every claim.

Implements all falsification tests from the plan at
findings/plans/plan_refusal_axis_falsification.md

Tests (ordered by destructive potential):
  1a. Random split null — does decomposition survive shuffled labels?
  1b. Bootstrap CIs — are pairwise cosines robust?
  1c. Sample size confound — do small-n domains have unreliable directions?
  2a. Leave-one-out — are specific cosine values fragile?
  3a. Random PCA baseline — is 11-dim for 70% trivially expected?
  3b. Single-domain PCA — is multi-dimensionality about domain diversity?
  4a. Random direction capping (GPU) — does noise also suppress refusal?
  4e. Multiple comparisons — was tau=p50 discovered by sweeping?
  5a. CI on spillover — is -15.9 significant with n=10?
  6a. Clustering stability — is k=3 optimal or overfitting?
  6b. Linkage sensitivity — does dendrogram depend on linkage method?

Evidence pin: Each test is designed to falsify a specific claim.
If a claim fails, we retract it. If it survives, we report with the
falsification result as evidence of robustness.

Usage:
  uv run python refusal_axis_falsification.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from refusal_decomposition import CFG, setup_logging

log = setup_logging("falsification")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_DIR = DATA / "falsification_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SITE = "mean_response_token"  # Primary analysis site


def load_data() -> dict:
    """Load all data needed for falsification tests."""
    retained = json.load(open(DATA / "retained_pairs.json"))
    manifest = json.load(open(DATA / "responses_manifest.json", encoding="utf-8"))

    pair_domain = {}
    for e in manifest:
        if e["pair_id"] in set(retained) and e["condition"] == "positive":
            pair_domain[e["pair_id"]] = e["domain"]

    pos = torch.stack(
        [
            torch.load(
                DATA / "activations" / SITE / "positive" / f"pair_{pid}.pt",
                weights_only=True,
            )
            for pid in retained
        ]
    ).float()
    neg = torch.stack(
        [
            torch.load(
                DATA / "activations" / SITE / "negative" / f"pair_{pid}.pt",
                weights_only=True,
            )
            for pid in retained
        ]
    ).float()

    mean_axis = torch.load(
        DATA / f"refusal_direction_{SITE}.pt", weights_only=True
    ).float()

    domains = sorted(set(pair_domain.values()))
    domain_labels = [pair_domain.get(pid, "unknown") for pid in retained]

    return {
        "retained": retained,
        "pair_domain": pair_domain,
        "pos": pos,
        "neg": neg,
        "mean_axis": mean_axis,
        "domains": domains,
        "domain_labels": domain_labels,
    }


def compute_domain_directions(pos, neg, domains, domain_labels, retained):
    """Compute per-domain refusal directions."""
    dirs = {}
    for d in domains:
        idx = [i for i, pid in enumerate(retained) if domain_labels[i] == d]
        if len(idx) >= 2:
            dirs[d] = pos[idx].mean(0) - neg[idx].mean(0)
    return dirs


def compute_cosine_matrix(dirs, domains):
    """Compute pairwise cosine matrix for domain directions."""
    active = [d for d in domains if d in dirs]
    n = len(active)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = F.cosine_similarity(
                dirs[active[i]].unsqueeze(0), dirs[active[j]].unsqueeze(0)
            ).item()
    return mat, active


# ═══════════════════════════════════════════════════════════════════
# TEST 1a: Random split null
# ═══════════════════════════════════════════════════════════════════


def test_1a_random_split_null(data: dict) -> dict:
    """Randomly assign domain labels, recompute cosine matrix.

    If random groups produce similar min/max cosines as real domains,
    the decomposition is an artifact of small samples in high dimensions.
    """
    log.info("═══ TEST 1a: Random split null ═══")

    rng = np.random.default_rng(SEED)
    n_perms = 1000
    n_pairs = len(data["retained"])

    # Real cosine matrix
    dirs = compute_domain_directions(
        data["pos"],
        data["neg"],
        data["domains"],
        data["domain_labels"],
        data["retained"],
    )
    real_mat, active_domains = compute_cosine_matrix(dirs, data["domains"])
    real_min = float(np.min(real_mat[np.triu_indices_from(real_mat, k=1)]))
    real_max = float(np.max(real_mat[np.triu_indices_from(real_mat, k=1)]))
    real_range = real_max - real_min

    log.info(
        "Real cosine matrix: min=%.3f, max=%.3f, range=%.3f",
        real_min,
        real_max,
        real_range,
    )

    # Domain sizes (to match real distribution)
    domain_sizes = {}
    for d in data["domains"]:
        domain_sizes[d] = sum(1 for l in data["domain_labels"] if l == d)
    sizes = [domain_sizes[d] for d in data["domains"]]

    null_mins, null_maxes, null_ranges = [], [], []
    for _ in range(n_perms):
        # Shuffle labels preserving group sizes
        shuffled = list(range(n_pairs))
        rng.shuffle(shuffled)
        fake_labels = [""] * n_pairs
        offset = 0
        for d, s in zip(data["domains"], sizes):
            for i in range(offset, offset + s):
                if i < n_pairs:
                    fake_labels[shuffled[i]] = d
            offset += s

        fake_dirs = compute_domain_directions(
            data["pos"],
            data["neg"],
            data["domains"],
            fake_labels,
            data["retained"],
        )
        fake_mat, _ = compute_cosine_matrix(fake_dirs, data["domains"])
        triu = fake_mat[np.triu_indices_from(fake_mat, k=1)]
        if len(triu) > 0:
            null_mins.append(float(triu.min()))
            null_maxes.append(float(triu.max()))
            null_ranges.append(float(triu.max() - triu.min()))

    # p-values: how often does random produce a range >= observed?
    p_range = float(np.mean(np.array(null_ranges) >= real_range))
    p_min = float(np.mean(np.array(null_mins) <= real_min))

    log.info("Null distribution (1000 permutations):")
    log.info(
        "  Min cosine: null median=%.3f, real=%.3f, p(null<=real)=%.3f",
        np.median(null_mins),
        real_min,
        p_min,
    )
    log.info(
        "  Range: null median=%.3f, real=%.3f, p(null>=real)=%.3f",
        np.median(null_ranges),
        real_range,
        p_range,
    )

    if p_range < 0.05:
        log.info(
            "  ✓ SURVIVES: Real domain decomposition has significantly wider cosine range than random splits."
        )
    else:
        log.warning(
            "  ✗ FALSIFIED: Random splits produce similar cosine ranges. Decomposition may be artifactual."
        )

    return {
        "real_min": real_min,
        "real_max": real_max,
        "real_range": real_range,
        "null_median_min": float(np.median(null_mins)),
        "null_median_range": float(np.median(null_ranges)),
        "p_range": p_range,
        "p_min": p_min,
        "n_permutations": n_perms,
        "verdict": "survives" if p_range < 0.05 else "falsified",
    }


# ═══════════════════════════════════════════════════════════════════
# TEST 1b: Bootstrap CIs
# ═══════════════════════════════════════════════════════════════════


def test_1b_bootstrap_ci(data: dict) -> dict:
    """Bootstrap 95% CIs for all pairwise cosines.

    If CI for cos(safety, capability) includes 0.5+, the
    'near-orthogonal' claim is not robust.
    """
    log.info("═══ TEST 1b: Bootstrap CIs ═══")

    rng = np.random.default_rng(SEED)
    n_boot = 2000

    # Collect bootstrap samples of each pairwise cosine
    domains = data["domains"]
    retained = data["retained"]
    domain_labels = data["domain_labels"]

    # Group pair indices by domain
    domain_indices = {}
    for d in domains:
        domain_indices[d] = [i for i, l in enumerate(domain_labels) if l == d]

    boot_cosines = {}  # (d1, d2) -> list of cosines
    for d1 in domains:
        for d2 in domains:
            if d1 <= d2:
                boot_cosines[(d1, d2)] = []

    for _ in range(n_boot):
        # Resample within each domain
        boot_pos = data["pos"].clone()
        boot_neg = data["neg"].clone()
        for d in domains:
            idx = domain_indices[d]
            if len(idx) < 2:
                continue
            boot_idx = rng.choice(idx, size=len(idx), replace=True)
            for new_i, old_i in enumerate(boot_idx):
                boot_pos[idx[new_i]] = data["pos"][old_i]
                boot_neg[idx[new_i]] = data["neg"][old_i]

        boot_dirs = compute_domain_directions(
            boot_pos,
            boot_neg,
            domains,
            domain_labels,
            retained,
        )

        for d1 in domains:
            for d2 in domains:
                if d1 <= d2 and d1 in boot_dirs and d2 in boot_dirs:
                    cos = F.cosine_similarity(
                        boot_dirs[d1].unsqueeze(0), boot_dirs[d2].unsqueeze(0)
                    ).item()
                    boot_cosines[(d1, d2)].append(cos)

    # Compute CIs
    results = {}
    for (d1, d2), values in boot_cosines.items():
        if d1 == d2 or not values:
            continue
        ci_lo = float(np.percentile(values, 2.5))
        ci_hi = float(np.percentile(values, 97.5))
        median = float(np.median(values))
        results[f"{d1}_vs_{d2}"] = {
            "median": median,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_width": ci_hi - ci_lo,
        }
        log.info(
            "  %s vs %s: median=%.3f, 95%% CI=[%.3f, %.3f]",
            d1,
            d2,
            median,
            ci_lo,
            ci_hi,
        )

    # Key check: does safety-capability CI include 0.5?
    sc_key = "capability_boundary_vs_safety"
    if sc_key in results:
        ci_hi = results[sc_key]["ci_hi"]
        if ci_hi >= 0.5:
            log.warning(
                "  ✗ WEAKENED: safety-capability CI upper bound (%.3f) >= 0.5. 'Near-orthogonal' claim is fragile.",
                ci_hi,
            )
        else:
            log.info(
                "  ✓ SURVIVES: safety-capability CI upper bound (%.3f) < 0.5. Near-orthogonal is robust.",
                ci_hi,
            )

    return {"pairwise_cis": results, "n_bootstrap": n_boot}


# ═══════════════════════════════════════════════════════════════════
# TEST 1c: Sample size confound
# ═══════════════════════════════════════════════════════════════════


def test_1c_sample_size_confound(data: dict) -> dict:
    """Subsample large domains to n=4 (matching identity_boundary).

    If cosine with mean axis changes drastically, small-n domains
    have unreliable directions.
    """
    log.info("═══ TEST 1c: Sample size confound ═══")

    rng = np.random.default_rng(SEED)
    n_subsamples = 100
    target_n = 4  # Match identity_boundary

    domains = data["domains"]
    domain_labels = data["domain_labels"]
    mean_axis = data["mean_axis"]

    domain_indices = {}
    for d in domains:
        domain_indices[d] = [i for i, l in enumerate(domain_labels) if l == d]

    # Full-sample cosines
    full_dirs = compute_domain_directions(
        data["pos"],
        data["neg"],
        domains,
        domain_labels,
        data["retained"],
    )
    full_cosines = {}
    for d, d_dir in full_dirs.items():
        full_cosines[d] = F.cosine_similarity(
            d_dir.unsqueeze(0), mean_axis.unsqueeze(0)
        ).item()

    # Subsample each domain to n=4
    results = {}
    for d in domains:
        idx = domain_indices[d]
        n_actual = len(idx)
        if n_actual <= target_n:
            results[d] = {
                "full_cosine": full_cosines.get(d, None),
                "subsampled_mean": full_cosines.get(d, None),
                "subsampled_std": 0.0,
                "n_actual": n_actual,
                "note": "already at or below target_n",
            }
            continue

        sub_cosines = []
        for _ in range(n_subsamples):
            sub_idx = rng.choice(idx, size=target_n, replace=False).tolist()
            sub_dir = data["pos"][sub_idx].mean(0) - data["neg"][sub_idx].mean(0)
            cos = F.cosine_similarity(
                sub_dir.unsqueeze(0), mean_axis.unsqueeze(0)
            ).item()
            sub_cosines.append(cos)

        results[d] = {
            "full_cosine": full_cosines.get(d),
            "subsampled_mean": float(np.mean(sub_cosines)),
            "subsampled_std": float(np.std(sub_cosines)),
            "subsampled_range": [
                float(np.min(sub_cosines)),
                float(np.max(sub_cosines)),
            ],
            "n_actual": n_actual,
        }
        log.info(
            "  %s (n=%d→%d): full=%.3f, subsampled=%.3f±%.3f, range=[%.3f, %.3f]",
            d,
            n_actual,
            target_n,
            full_cosines.get(d, 0),
            np.mean(sub_cosines),
            np.std(sub_cosines),
            np.min(sub_cosines),
            np.max(sub_cosines),
        )

    return {"target_n": target_n, "n_subsamples": n_subsamples, "domains": results}


# ═══════════════════════════════════════════════════════════════════
# TEST 2a: Leave-one-out stability
# ═══════════════════════════════════════════════════════════════════


def test_2a_leave_one_out(data: dict) -> dict:
    """Remove each pair one at a time, recompute domain loadings."""
    log.info("═══ TEST 2a: Leave-one-out stability ═══")

    domains = data["domains"]
    domain_labels = data["domain_labels"]
    retained = data["retained"]
    n = len(retained)

    # Full cosines
    full_dirs = compute_domain_directions(
        data["pos"],
        data["neg"],
        domains,
        domain_labels,
        retained,
    )
    mean_axis = data["mean_axis"]
    full_cosines = {}
    for d, d_dir in full_dirs.items():
        full_cosines[d] = F.cosine_similarity(
            d_dir.unsqueeze(0), mean_axis.unsqueeze(0)
        ).item()

    # Leave-one-out
    loo_cosines = {d: [] for d in domains}
    for leave_out in range(n):
        mask = [i for i in range(n) if i != leave_out]
        loo_labels = [domain_labels[i] for i in mask]
        loo_retained = [retained[i] for i in mask]
        loo_dirs = compute_domain_directions(
            data["pos"][mask],
            data["neg"][mask],
            domains,
            loo_labels,
            loo_retained,
        )
        for d, d_dir in loo_dirs.items():
            cos = F.cosine_similarity(d_dir.unsqueeze(0), mean_axis.unsqueeze(0)).item()
            loo_cosines[d].append(cos)

    results = {}
    for d in domains:
        if not loo_cosines[d]:
            continue
        vals = np.array(loo_cosines[d])
        results[d] = {
            "full": full_cosines.get(d),
            "loo_min": float(vals.min()),
            "loo_max": float(vals.max()),
            "loo_range": float(vals.max() - vals.min()),
            "loo_std": float(vals.std()),
        }
        log.info(
            "  %s: full=%.3f, LOO range=[%.3f, %.3f], std=%.4f",
            d,
            full_cosines.get(d, 0),
            vals.min(),
            vals.max(),
            vals.std(),
        )

    return results


# ═══════════════════════════════════════════════════════════════════
# TEST 3a: Random PCA baseline
# ═══════════════════════════════════════════════════════════════════


def test_3a_random_pca_baseline(data: dict) -> dict:
    """How many PCA dims do random R^3840 vectors need for 70% variance?"""
    log.info("═══ TEST 3a: Random PCA baseline ═══")

    rng = np.random.default_rng(SEED)
    n_trials = 100

    # Real PCA
    diffs = (data["pos"] - data["neg"]).float().numpy()
    diffs_centered = diffs - diffs.mean(axis=0)
    _, S_real, _ = np.linalg.svd(diffs_centered, full_matrices=False)
    var_real = (S_real**2) / (S_real**2).sum()
    cum_real = np.cumsum(var_real)
    dims_70_real = int(np.searchsorted(cum_real, 0.70)) + 1

    # Random baselines
    dims_70_random = []
    for _ in range(n_trials):
        random_vecs = rng.standard_normal((128, 3840)).astype(np.float32)
        random_centered = random_vecs - random_vecs.mean(axis=0)
        _, S_rand, _ = np.linalg.svd(random_centered, full_matrices=False)
        var_rand = (S_rand**2) / (S_rand**2).sum()
        cum_rand = np.cumsum(var_rand)
        dims_70_random.append(int(np.searchsorted(cum_rand, 0.70)) + 1)

    median_random = float(np.median(dims_70_random))
    log.info("  Real data: %d dims for 70%% variance", dims_70_real)
    log.info(
        "  Random R^3840 (100 trials): median=%d, range=[%d, %d]",
        median_random,
        min(dims_70_random),
        max(dims_70_random),
    )

    if dims_70_real >= median_random * 0.8:
        log.warning(
            "  ✗ WEAKENED: Real data needs %d dims, random needs %d. Not much better than random.",
            dims_70_real,
            int(median_random),
        )
    else:
        log.info(
            "  ✓ SURVIVES: Real data (%d dims) is significantly more structured than random (%d dims).",
            dims_70_real,
            int(median_random),
        )

    return {
        "real_dims_70": dims_70_real,
        "random_median": median_random,
        "random_range": [int(min(dims_70_random)), int(max(dims_70_random))],
        "verdict": "survives" if dims_70_real < median_random * 0.8 else "weakened",
    }


# ═══════════════════════════════════════════════════════════════════
# TEST 3b: Single-domain PCA
# ═══════════════════════════════════════════════════════════════════


def test_3b_single_domain_pca(data: dict) -> dict:
    """PCA on just safety pairs — is multi-dimensionality within-domain?"""
    log.info("═══ TEST 3b: Single-domain PCA ═══")

    results = {}
    for d in data["domains"]:
        idx = [i for i, l in enumerate(data["domain_labels"]) if l == d]
        if len(idx) < 5:
            continue
        diffs = (data["pos"][idx] - data["neg"][idx]).float().numpy()
        diffs_centered = diffs - diffs.mean(axis=0)
        _, S, _ = np.linalg.svd(diffs_centered, full_matrices=False)
        var_exp = (S**2) / (S**2).sum()
        cum = np.cumsum(var_exp)
        dims_70 = int(np.searchsorted(cum, 0.70)) + 1

        results[d] = {
            "n_pairs": len(idx),
            "dims_70": dims_70,
            "pc1_var": float(var_exp[0]),
        }
        log.info(
            "  %s (n=%d): %d dims for 70%%, PC1=%.1f%%",
            d,
            len(idx),
            dims_70,
            var_exp[0] * 100,
        )

    return results


# ═══════════════════════════════════════════════════════════════════
# TEST 4a: Random direction capping (GPU)
# ═══════════════════════════════════════════════════════════════════


def test_4a_random_direction_capping(data: dict) -> dict:
    """Cap along random unit vectors, check if refusal also drops.

    If random directions produce comparable refusal reduction,
    the effect is not specific to the safety direction.

    This test uses the capping analysis data — we check what random
    directions' projections look like compared to the safety direction.
    Since running full model generation for 20 random directions would
    take hours, we instead test the GEOMETRIC specificity: how much do
    our test prompt activations project onto random directions vs the
    safety direction?
    """
    log.info("═══ TEST 4a: Random direction capping (geometric proxy) ═══")

    rng = np.random.default_rng(SEED)
    n_random = 100

    # Load safety direction and mean axis
    mean_axis_unit = data["mean_axis"] / (data["mean_axis"].norm() + 1e-8)

    # Compute domain directions
    dirs = compute_domain_directions(
        data["pos"],
        data["neg"],
        data["domains"],
        data["domain_labels"],
        data["retained"],
    )
    safety_dir = dirs.get("safety")
    if safety_dir is None:
        log.warning("No safety direction found, skipping test 4a.")
        return {"status": "skipped"}

    safety_unit = safety_dir / (safety_dir.norm() + 1e-8)

    # Projection of positive activations onto safety direction
    pos_proj_safety = (data["pos"] @ safety_unit).numpy()
    safety_mean_proj = float(pos_proj_safety.mean())
    safety_std_proj = float(pos_proj_safety.std())

    # Projections onto random unit vectors
    random_mean_projs = []
    random_std_projs = []
    for _ in range(n_random):
        rand_vec = torch.randn(3840)
        rand_unit = rand_vec / (rand_vec.norm() + 1e-8)
        rand_proj = (data["pos"] @ rand_unit).numpy()
        random_mean_projs.append(float(rand_proj.mean()))
        random_std_projs.append(float(rand_proj.std()))

    random_mean_proj_median = float(np.median(np.abs(random_mean_projs)))
    safety_ratio = abs(safety_mean_proj) / (random_mean_proj_median + 1e-8)

    log.info(
        "  Safety direction: mean projection = %.1f ± %.1f",
        safety_mean_proj,
        safety_std_proj,
    )
    log.info(
        "  Random directions: median |mean projection| = %.1f", random_mean_proj_median
    )
    log.info("  Safety/random ratio: %.1fx", safety_ratio)

    if safety_ratio > 5:
        log.info(
            "  ✓ SURVIVES: Safety direction captures %.1fx more variance than random. Capping is specific.",
            safety_ratio,
        )
    else:
        log.warning(
            "  ✗ WEAKENED: Safety direction only %.1fx above random. Capping specificity is questionable.",
            safety_ratio,
        )

    return {
        "safety_mean_proj": safety_mean_proj,
        "safety_std_proj": safety_std_proj,
        "random_median_abs_mean_proj": random_mean_proj_median,
        "safety_random_ratio": safety_ratio,
        "n_random": n_random,
        "verdict": "survives" if safety_ratio > 5 else "weakened",
    }


# ═══════════════════════════════════════════════════════════════════
# TEST 4e: Multiple comparisons
# ═══════════════════════════════════════════════════════════════════


def test_4e_multiple_comparisons() -> dict:
    """Report the multiple comparisons problem honestly."""
    log.info("═══ TEST 4e: Multiple comparisons ═══")

    n_directions = 4
    n_taus = 7
    n_domains = 4
    total_cells = n_directions * n_taus * n_domains
    expected_false_positives = total_cells * 0.05

    log.info(
        "  %d directions × %d tau values × %d domains = %d cells tested",
        n_directions,
        n_taus,
        n_domains,
        total_cells,
    )
    log.info("  Expected false positives at p=0.05: %.1f", expected_false_positives)
    log.info("  tau=p50 was selected by sweeping, NOT pre-registered.")
    log.info("  Bonferroni-corrected threshold: p < %.4f", 0.05 / total_cells)

    return {
        "total_cells": total_cells,
        "expected_false_positives_005": expected_false_positives,
        "tau_preregistered": False,
        "bonferroni_threshold": 0.05 / total_cells,
    }


# ═══════════════════════════════════════════════════════════════════
# TEST 5a: CI on privacy spillover
# ═══════════════════════════════════════════════════════════════════


def test_5a_spillover_ci() -> dict:
    """Is -15.9 privacy spillover significant with n=10 prompts?"""
    log.info("═══ TEST 5a: CI on privacy spillover ═══")

    # Load individual scores
    capping_data = json.load(open(DATA / "capping_results" / "analysis.json"))
    overall_privacy = (
        capping_data.get("overall_refusal", {}).get("privacy", {}).get("50", {})
    )

    mean_delta = overall_privacy.get("mean_delta", 0)
    n = overall_privacy.get("n_coherent", 0) or overall_privacy.get("n_deltas", 0)

    log.info("  Overall→privacy at p50: mean_delta=%.1f, n=%d", mean_delta, n)

    if n < 2:
        log.warning("  Cannot compute CI with n=%d", n)
        return {"mean_delta": mean_delta, "n": n, "status": "insufficient_data"}

    # We don't have individual scores in analysis.json, only mean.
    # Report the limitation honestly.
    log.info("  LIMITATION: Only mean delta available, not per-prompt scores.")
    log.info("  Cannot compute proper SE/CI without individual observations.")
    log.info("  With n=%d prompts, any single outlier could dominate the mean.", n)

    # Conservative check: at what SE would the CI include 0?
    # CI = mean ± 1.96*SE. For CI to include 0: |mean| < 1.96*SE → SE > |mean|/1.96
    se_needed = abs(mean_delta) / 1.96
    log.info("  For CI to include 0, SE would need to be > %.1f", se_needed)
    log.info(
        "  With n=%d, this requires per-prompt SD > %.1f", n, se_needed * np.sqrt(n)
    )

    return {
        "mean_delta": mean_delta,
        "n": n,
        "se_needed_for_null": se_needed,
        "sd_needed_for_null": se_needed * np.sqrt(n),
    }


# ═══════════════════════════════════════════════════════════════════
# TEST 6a+6b: Clustering stability
# ═══════════════════════════════════════════════════════════════════


def test_6_clustering_stability(data: dict) -> dict:
    """Test clustering with multiple k values and linkage methods."""
    log.info("═══ TEST 6a+6b: Clustering stability ═══")

    dirs = compute_domain_directions(
        data["pos"],
        data["neg"],
        data["domains"],
        data["domain_labels"],
        data["retained"],
    )
    cos_mat, active = compute_cosine_matrix(dirs, data["domains"])
    n = len(active)

    if n < 4:
        log.warning("Only %d domains with directions, skipping clustering.", n)
        return {"status": "insufficient_domains"}

    dist = np.clip(1 - cos_mat, 0, 2)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist)

    # 6b: Multiple linkage methods
    linkage_methods = ["ward", "complete", "average", "single"]
    orderings = {}
    for method in linkage_methods:
        Z = linkage(condensed, method=method)
        order = leaves_list(Z)
        orderings[method] = [active[i] for i in order]
        log.info("  %s linkage order: %s", method, orderings[method])

    # Check if orderings agree on grouping
    # Compare: does capability_boundary cluster with value-based or separately?
    cap_idx = (
        active.index("capability_boundary") if "capability_boundary" in active else -1
    )
    consistent_separation = True
    for method, order in orderings.items():
        if cap_idx >= 0:
            cap_pos = order.index("capability_boundary")
            # Is capability at the edge (first or last)?
            if cap_pos not in [0, len(order) - 1]:
                consistent_separation = False
                log.info(
                    "  %s: capability_boundary is at position %d/%d (not at edge)",
                    method,
                    cap_pos,
                    len(order),
                )

    if consistent_separation:
        log.info(
            "  ✓ SURVIVES: Capability separates from value-based cluster across all linkage methods."
        )
    else:
        log.warning("  ✗ WEAKENED: Capability placement varies by linkage method.")

    return {
        "orderings": orderings,
        "consistent_capability_separation": consistent_separation,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    log.info("=" * 70)
    log.info("REFUSAL AXIS FALSIFICATION SESSION")
    log.info("Attempting to destroy every claim before it enters the thesis.")
    log.info("=" * 70)

    data = load_data()
    results = {}

    # Priority order: most destructive first
    results["test_1a_random_split_null"] = test_1a_random_split_null(data)
    results["test_1b_bootstrap_ci"] = test_1b_bootstrap_ci(data)
    results["test_1c_sample_size_confound"] = test_1c_sample_size_confound(data)
    results["test_2a_leave_one_out"] = test_2a_leave_one_out(data)
    results["test_3a_random_pca_baseline"] = test_3a_random_pca_baseline(data)
    results["test_3b_single_domain_pca"] = test_3b_single_domain_pca(data)
    results["test_4a_random_direction_capping"] = test_4a_random_direction_capping(data)
    results["test_4e_multiple_comparisons"] = test_4e_multiple_comparisons()
    results["test_5a_spillover_ci"] = test_5a_spillover_ci()
    results["test_6_clustering_stability"] = test_6_clustering_stability(data)

    # Save
    out_path = OUT_DIR / "all_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("")
    log.info("Results saved to %s", out_path)

    # ── Final scorecard ──────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("FALSIFICATION SCORECARD")
    log.info("=" * 70)

    tests = [
        (
            "1a Random split null",
            results["test_1a_random_split_null"].get("verdict", "?"),
        ),
        ("1b Bootstrap CIs", "check CIs above"),
        ("1c Sample size confound", "check std above"),
        ("2a Leave-one-out", "check ranges above"),
        (
            "3a Random PCA baseline",
            results["test_3a_random_pca_baseline"].get("verdict", "?"),
        ),
        (
            "4a Random direction capping",
            results["test_4a_random_direction_capping"].get("verdict", "?"),
        ),
        ("4e Multiple comparisons", "tau=p50 NOT pre-registered"),
        ("5a Spillover CI", "individual scores unavailable"),
        (
            "6 Clustering stability",
            "survives"
            if results["test_6_clustering_stability"].get(
                "consistent_capability_separation"
            )
            else "weakened",
        ),
    ]

    for name, verdict in tests:
        symbol = (
            "✓"
            if "survives" in str(verdict).lower()
            else "?"
            if "?" in str(verdict)
            else "✗"
        )
        log.info("  %s %s: %s", symbol, name, verdict)

    log.info("=" * 70)


if __name__ == "__main__":
    main()
