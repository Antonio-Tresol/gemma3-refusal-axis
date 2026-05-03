"""Feature Hierarchy Analysis — Methods 1-3.

Tests whether broad refusal features at 16k width decompose into finer
sub-features at 65k width, or whether the dictionary genuinely reorganizes.

Methods:
  1. Decoder cosine similarity (CPU) — geometric relatedness of decoder vectors
  2. Co-activation analysis (CPU) — do parent/child features fire on the same prompts?
  3. Hierarchy R² (CPU) — can a parent's decoder be reconstructed from children?

Evidence pins:
  - Method 1: Decoder cosine is standard for SAE feature matching
    (Bricken et al. (2023, Transformer Circuits), Bussmann et al. (2025, arXiv:2503.17547), Luo et al. (2026, arXiv:2602.11881))
  - Method 2: Co-activation distinguishes absorption from splitting
    (Chanin et al. (2024, arXiv:2409.14507), "Feature Splitting and Absorption")
  - Method 3: Parent = weighted sum of children is the structural alignment
    metric from Luo et al. (2026, arXiv:2602.11881), HSAE Sec 3.2
  - GS2 constrains decoder columns to unit norm (paper line 174)
  - Single shared w_dec across Matryoshka prefix widths (M0 bootstrap)

Falsification gates:
  - Method 1: NO parent has cos > 0.5 with ANY child → H1 falsified
  - Method 2: Geometric matches don't co-activate → similarity is artifact
  - Method 3: R² < 0.3 for most groupings → no hierarchical structure

Data sources:
  - Decoder vectors: loaded from 1M SAE (GPU, then freed)
  - Co-activation: data/encoded/{site}_{condition}.pt (per-prompt 1M encodings)
    Because Matryoshka prefix-slicing shares feature IDs, feature i at 16k
    IS feature i at 1M — we can use the 1M encodings directly.
  - Feature IDs per width: data/milestone_7_results/width_metrics.json

GPU pattern: Load SAE → extract decoder rows → del SAE → CPU analysis.

Usage:
  uv run python feature_hierarchy_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from refusal_decomposition import CFG, setup_logging

log = setup_logging("feature_hierarchy")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_DIR = DATA / "hierarchy_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_sets() -> dict:
    """Load feature ID sets per site per width from width_metrics.json."""
    metrics = json.load(open(DATA / "milestone_7_results" / "width_metrics.json"))
    result = {}
    for m in metrics:
        site = m["site"]
        width = m["width"]
        result[(site, width)] = set(m["feature_ids"])
    return result


def extract_decoder_vectors(feature_ids: list[int]) -> torch.Tensor:
    """Load 1M SAE, extract decoder rows for given feature IDs, free SAE.

    Returns tensor of shape (len(feature_ids), d_model) on CPU, float32.
    """
    from refusal_decomposition.model import load_sae

    log.info("Loading 1M SAE to extract %d decoder vectors...", len(feature_ids))
    sae = load_sae(
        width=1_048_576,
        width_label="1m",
        device="cuda",
    )

    # Verify unit norm (falsification check 1)
    sample_norms = sae.w_dec[feature_ids[:5]].norm(dim=1)
    log.info("Decoder vector norms (sample): %s", sample_norms.tolist())
    if not torch.allclose(sample_norms, torch.ones_like(sample_norms), atol=0.01):
        log.warning(
            "FALSIFICATION: Decoder vectors are NOT unit norm! Cosine analysis may be unreliable."
        )

    # Extract just the rows we need
    vectors = sae.w_dec[feature_ids].detach().cpu().float()
    log.info("Extracted decoder vectors: shape %s", vectors.shape)

    # Free GPU
    del sae
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    vram = torch.cuda.memory_allocated() / 1e9
    log.info("VRAM after cleanup: %.2f GB", vram)

    return vectors


def compute_cosine_matrix(
    parent_vecs: torch.Tensor, child_vecs: torch.Tensor
) -> np.ndarray:
    """Compute cosine similarity matrix between parent and child decoder vectors.

    Args:
        parent_vecs: shape (n_parents, d_model)
        child_vecs: shape (n_children, d_model)

    Returns:
        cos_matrix: shape (n_parents, n_children), values in [-1, 1]
    """
    # Normalize (should already be ~unit norm, but be safe)
    p_norm = parent_vecs / (parent_vecs.norm(dim=1, keepdim=True) + 1e-8)
    c_norm = child_vecs / (child_vecs.norm(dim=1, keepdim=True) + 1e-8)
    cos = (p_norm @ c_norm.T).numpy()
    return cos


def load_coactivation_vectors(
    site: str, feature_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Load per-prompt SAE activations for specific features from 1M encodings.

    Because Matryoshka prefix-slicing means feature i at 16k IS feature i
    at 1M, we can extract co-activation patterns from the 1M encoded data.

    Returns:
        pos_acts: shape (n_prompts, n_features) — activations on positive (refusal) prompts
        neg_acts: shape (n_prompts, n_features) — activations on negative (benign) prompts

    Data source: data/encoded/{site}_{positive|negative}.pt
    Evidence pin: Chanin et al. (2024, arXiv:2409.14507) use co-activation to distinguish
    absorption from genuine splitting.
    """
    encoded_dir = DATA / "encoded"
    pos_path = encoded_dir / f"{site}_positive.pt"
    neg_path = encoded_dir / f"{site}_negative.pt"

    log.info("Loading encoded activations from %s", encoded_dir)

    # These are shape (n_prompts, 1048576) — sparse, mostly zeros
    # We only need specific feature columns
    pos_full = torch.load(pos_path, weights_only=True)
    pos_acts = pos_full[:, feature_ids].float().numpy()
    del pos_full

    neg_full = torch.load(neg_path, weights_only=True)
    neg_acts = neg_full[:, feature_ids].float().numpy()
    del neg_full

    log.info(
        "Loaded co-activation data: pos=%s, neg=%s", pos_acts.shape, neg_acts.shape
    )
    return pos_acts, neg_acts


def method2_coactivation(
    site: str,
    parent_ids: list[int],
    child_ids: list[int],
) -> dict:
    """Method 2: Co-activation analysis.

    For each disappeared parent feature, check whether candidate children
    (from Method 1 decoder cosine) activate on the SAME prompts.

    Evidence pin: Chanin et al. (2024, arXiv:2409.14507) — if geometrically similar features
    don't co-activate, the similarity is an artifact of high-dimensional space.

    Even without high decoder cosine, co-activation can reveal functional
    relationships that geometry misses (e.g., if the SAE reparameterizes
    the same concept with a rotated decoder direction).
    """
    site_short = "LPT" if site == "last_prompt_token" else "MRT"
    log.info("─── %s co-activation analysis ───", site_short)

    all_ids = sorted(set(parent_ids) | set(child_ids))
    id_to_col = {fid: i for i, fid in enumerate(all_ids)}

    pos_acts, neg_acts = load_coactivation_vectors(site, all_ids)

    # Binary activation: feature is "active" if activation > 0
    pos_binary = (pos_acts > 0).astype(float)

    # For each parent, compute Jaccard similarity with each child
    # Jaccard = |intersection| / |union| of prompts where each fires
    parent_results = []
    for pid in parent_ids:
        p_col = id_to_col[pid]
        p_active = pos_binary[:, p_col]  # (n_prompts,)
        p_count = p_active.sum()

        if p_count == 0:
            parent_results.append(
                {
                    "parent_id": pid,
                    "parent_activation_rate": 0.0,
                    "top_coactivations": [],
                    "max_jaccard": 0.0,
                }
            )
            continue

        child_matches = []
        for cid in child_ids:
            c_col = id_to_col[cid]
            c_active = pos_binary[:, c_col]
            c_count = c_active.sum()

            intersection = (p_active * c_active).sum()
            union = ((p_active + c_active) > 0).sum()

            jaccard = float(intersection / union) if union > 0 else 0.0
            # Also compute: what fraction of parent's prompts does child cover?
            coverage = float(intersection / p_count) if p_count > 0 else 0.0

            child_matches.append(
                {
                    "child_id": cid,
                    "jaccard": jaccard,
                    "coverage": coverage,
                    "child_activation_rate": float(c_count / len(pos_binary)),
                }
            )

        child_matches.sort(key=lambda x: x["jaccard"], reverse=True)

        best = child_matches[0]
        if best["jaccard"] > 0.2:
            log.info(
                "  Feature %d → top co-activation: %d (jaccard=%.3f, coverage=%.3f)",
                pid,
                best["child_id"],
                best["jaccard"],
                best["coverage"],
            )

        parent_results.append(
            {
                "parent_id": pid,
                "parent_activation_rate": float(p_count / len(pos_binary)),
                "top_coactivations": child_matches[:5],
                "max_jaccard": best["jaccard"],
            }
        )

    # Summary
    n_jaccard_03 = sum(1 for r in parent_results if r["max_jaccard"] > 0.3)
    n_jaccard_05 = sum(1 for r in parent_results if r["max_jaccard"] > 0.5)
    mean_max_jaccard = float(np.mean([r["max_jaccard"] for r in parent_results]))

    log.info("")
    log.info("─── %s CO-ACTIVATION SUMMARY ───", site_short)
    log.info("Parents with max Jaccard > 0.5: %d / %d", n_jaccard_05, len(parent_ids))
    log.info("Parents with max Jaccard > 0.3: %d / %d", n_jaccard_03, len(parent_ids))
    log.info("Mean max Jaccard: %.3f", mean_max_jaccard)

    if n_jaccard_05 == 0 and n_jaccard_03 == 0:
        log.info("CONSISTENT with Method 1: no co-activation signal either.")
    elif n_jaccard_05 > 0:
        log.warning(
            "CONTRADICTS Method 1: %d parents co-activate with children despite low decoder cosine!",
            n_jaccard_05,
        )

    return {
        "parent_results": parent_results,
        "summary": {
            "n_parents": len(parent_ids),
            "n_jaccard_05": n_jaccard_05,
            "n_jaccard_03": n_jaccard_03,
            "mean_max_jaccard": mean_max_jaccard,
        },
    }


def method3_hierarchy_r2(
    parent_ids: list[int],
    child_ids: list[int],
    all_vectors: torch.Tensor,
    id_to_idx: dict[int, int],
    site: str,
) -> dict:
    """Method 3: Hierarchy R² — can a parent be reconstructed from children?

    For each parent, fit: parent_vec ≈ Σ αᵢ · childᵢ_vec
    using least-squares. R² measures how much of the parent's direction
    is explained by the child subspace.

    Evidence pin: Luo et al. (2026, arXiv:2602.11881) (HSAE) Sec 3.2 — structural constraint
    loss encourages parent decoder = weighted sum of children decoders.
    We apply their evaluation metric, not their training method.

    Falsification: R² < 0.3 for most parents → children don't span the
    parent's direction, no hierarchical structure.
    """
    site_short = "LPT" if site == "last_prompt_token" else "MRT"
    log.info("─── %s hierarchy R² analysis ───", site_short)

    if len(child_ids) < 2:
        log.info("Too few children (%d) for R² analysis, skipping.", len(child_ids))
        return {"parent_results": [], "summary": {"n_parents": 0}}

    # Build child matrix: shape (n_children, d_model)
    child_matrix = all_vectors[
        [id_to_idx[c] for c in child_ids]
    ].numpy()  # (n_children, 3840)

    parent_results = []
    for pid in parent_ids:
        parent_vec = all_vectors[id_to_idx[pid]].numpy()  # (3840,)

        # Least squares: parent_vec ≈ child_matrix.T @ alpha
        # Solve: (child_matrix @ child_matrix.T) @ alpha = child_matrix @ parent_vec
        # Using numpy lstsq for numerical stability
        alpha, residuals, rank, sv = np.linalg.lstsq(
            child_matrix.T, parent_vec, rcond=None
        )

        # Compute R²
        predicted = child_matrix.T @ alpha
        ss_res = np.sum((parent_vec - predicted) ** 2)
        ss_tot = np.sum((parent_vec - parent_vec.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Also report which children have largest weights
        top_children_idx = np.argsort(np.abs(alpha))[::-1][:5]
        top_children = [
            {"child_id": child_ids[i], "weight": float(alpha[i])}
            for i in top_children_idx
        ]

        if r2 > 0.3:
            log.info(
                "  Feature %d: R²=%.3f (top child: %d, weight=%.3f)",
                pid,
                r2,
                top_children[0]["child_id"],
                top_children[0]["weight"],
            )

        parent_results.append(
            {
                "parent_id": pid,
                "r2": float(r2),
                "top_children": top_children,
            }
        )

    # Summary
    r2_values = [r["r2"] for r in parent_results]
    n_r2_03 = sum(1 for r2 in r2_values if r2 > 0.3)
    n_r2_05 = sum(1 for r2 in r2_values if r2 > 0.5)
    mean_r2 = float(np.mean(r2_values))

    log.info("")
    log.info("─── %s R² SUMMARY ───", site_short)
    log.info("Parents with R² > 0.5: %d / %d", n_r2_05, len(parent_ids))
    log.info("Parents with R² > 0.3: %d / %d", n_r2_03, len(parent_ids))
    log.info("Mean R²: %.3f", mean_r2)

    # NOTE: With enough children (23 for LPT), R² can be high by chance
    # because the children span a high-dimensional subspace. We report the
    # ratio children/d_model as a sanity check.
    ratio = len(child_ids) / 3840
    log.info("Children/d_model ratio: %d/3840 = %.4f", len(child_ids), ratio)
    if ratio > 0.01:
        log.warning(
            "CAUTION: %d children span %.1f%% of d_model dimensions. "
            "R² values may be inflated by chance alignment.",
            len(child_ids),
            ratio * 100,
        )

    return {
        "parent_results": parent_results,
        "summary": {
            "n_parents": len(parent_ids),
            "n_r2_05": n_r2_05,
            "n_r2_03": n_r2_03,
            "mean_r2": mean_r2,
            "children_d_model_ratio": ratio,
        },
    }


def robustness_checks(
    results_by_site: dict,
    feature_sets: dict,
    all_vectors: torch.Tensor,
    id_to_idx: dict[int, int],
) -> dict:
    """Robustness checks to falsify or qualify the main findings.

    Check 1: Null distribution for Jaccard.
      Shuffle feature labels 1000 times, recompute max Jaccard per parent.
      If observed Jaccard is within the null distribution, co-activation
      is expected by chance (base-rate confound).

    Check 2: Base-rate analysis for high-Jaccard pairs.
      If both features in a high-Jaccard pair fire on >90% of positive
      prompts, their co-activation is trivially expected.

    Check 3: Threshold sensitivity — "disappeared" vs "below threshold."
      Load contrastive scores at 65k for disappeared features. If they
      still have nonzero Cohen's d at 65k, they didn't truly vanish.

    Check 4: Other width transitions (65k→262k, 262k→1M).
      Same decoder cosine analysis for subsequent width transitions.
      If hierarchy emerges at a later transition, the 16k→65k null
      doesn't generalize.

    Check 5: Power analysis.
      With n_parents and n_prompts, what minimum Jaccard is detectable?
    """
    log.info("")
    log.info("=" * 60)
    log.info("ROBUSTNESS CHECKS")
    log.info("=" * 60)

    checks = {}

    # ── Check 1 & 2: Null distribution + base rate for high-Jaccard pairs ─
    for site, info in results_by_site.items():
        site_short = "LPT" if site == "last_prompt_token" else "MRT"
        log.info("")
        log.info("─── %s: Check 1 & 2 (null distribution + base rates) ───", site_short)

        all_ids = sorted(set(info["disappeared"]) | set(info["new_at_65k"]))
        id_to_col = {fid: i for i, fid in enumerate(all_ids)}

        pos_acts, _ = load_coactivation_vectors(site, all_ids)
        pos_binary = (pos_acts > 0).astype(float)

        # Check 2: Base rates for the high-Jaccard pairs from Method 2
        high_jaccard_pairs = []
        for pr in info["method2"]["parent_results"]:
            if pr["max_jaccard"] > 0.3:
                best_child = pr["top_coactivations"][0]
                pid = pr["parent_id"]
                cid = best_child["child_id"]
                p_rate = pr["parent_activation_rate"]
                c_rate = best_child["child_activation_rate"]
                jaccard = best_child["jaccard"]

                # Expected Jaccard under independence:
                # If P(A)=p, P(B)=q, independent, then
                # E[Jaccard] = p*q / (p + q - p*q)
                expected_jaccard = (p_rate * c_rate) / (
                    p_rate + c_rate - p_rate * c_rate + 1e-8
                )

                log.info(
                    "  Pair %d→%d: Jaccard=%.3f, parent_rate=%.3f, child_rate=%.3f, "
                    "expected_if_independent=%.3f",
                    pid,
                    cid,
                    jaccard,
                    p_rate,
                    c_rate,
                    expected_jaccard,
                )

                high_jaccard_pairs.append(
                    {
                        "parent_id": pid,
                        "child_id": cid,
                        "observed_jaccard": jaccard,
                        "parent_activation_rate": p_rate,
                        "child_activation_rate": c_rate,
                        "expected_jaccard_independent": expected_jaccard,
                        "ratio_observed_expected": jaccard / (expected_jaccard + 1e-8),
                    }
                )

        # Check 1: Null distribution via permutation
        # For each parent, shuffle child labels 1000 times, compute max Jaccard
        rng = np.random.default_rng(42)
        n_permutations = 1000
        parent_ids = info["disappeared"]
        child_ids = info["new_at_65k"]

        null_max_jaccards = []  # per parent, distribution of max Jaccard under null
        observed_max_jaccards = []

        for pid in parent_ids:
            p_col = id_to_col[pid]
            p_active = pos_binary[:, p_col]
            p_count = p_active.sum()
            if p_count == 0:
                continue

            # Observed max Jaccard for this parent
            obs_max = 0.0
            for cid in child_ids:
                c_col = id_to_col[cid]
                c_active = pos_binary[:, c_col]
                intersection = (p_active * c_active).sum()
                union = ((p_active + c_active) > 0).sum()
                j = intersection / union if union > 0 else 0
                obs_max = max(obs_max, j)
            observed_max_jaccards.append(obs_max)

            # Null: shuffle the parent's activation vector, recompute
            null_maxes = []
            for _ in range(n_permutations):
                shuffled = rng.permutation(p_active)
                perm_max = 0.0
                for cid in child_ids:
                    c_col = id_to_col[cid]
                    c_active = pos_binary[:, c_col]
                    intersection = (shuffled * c_active).sum()
                    union = ((shuffled + c_active) > 0).sum()
                    j = intersection / union if union > 0 else 0
                    perm_max = max(perm_max, j)
                null_maxes.append(perm_max)
            null_max_jaccards.append(null_maxes)

        # Compute p-values: fraction of null samples >= observed
        p_values = []
        for obs, nulls in zip(observed_max_jaccards, null_max_jaccards):
            p = (np.array(nulls) >= obs).mean()
            p_values.append(p)

        n_significant = sum(1 for p in p_values if p < 0.05)
        log.info("")
        log.info("  Permutation test (1000 shuffles, seed=42):")
        log.info("    Parents with p < 0.05: %d / %d", n_significant, len(p_values))
        log.info(
            "    Median p-value: %.3f",
            np.median(p_values) if p_values else float("nan"),
        )

        # Report specific p-values for high-Jaccard pairs
        for pair_info in high_jaccard_pairs:
            pid = pair_info["parent_id"]
            pid_idx = parent_ids.index(pid) if pid in parent_ids else -1
            if pid_idx >= 0 and pid_idx < len(p_values):
                pair_info["permutation_p_value"] = float(p_values[pid_idx])
                log.info(
                    "    Pair %d→%d: observed=%.3f, p=%.3f %s",
                    pair_info["parent_id"],
                    pair_info["child_id"],
                    pair_info["observed_jaccard"],
                    p_values[pid_idx],
                    "***" if p_values[pid_idx] < 0.05 else "(not significant)",
                )

        checks[site] = {
            "high_jaccard_pairs": high_jaccard_pairs,
            "permutation_test": {
                "n_permutations": n_permutations,
                "n_significant_005": n_significant,
                "n_tested": len(p_values),
                "median_p": float(np.median(p_values)) if p_values else None,
                "p_values": [float(p) for p in p_values],
            },
        }

    # ── Check 3: Threshold sensitivity ───────────────────────────────
    log.info("")
    log.info("─── Check 3: Threshold sensitivity ───")
    log.info("Did 'disappeared' features truly vanish or just fall below threshold?")

    for site, info in results_by_site.items():
        site_short = "LPT" if site == "last_prompt_token" else "MRT"

        # Load contrastive scores at 65k for disappeared features
        width_label = "65k"
        contrastive_file = (
            DATA / "milestone_4_results" / f"contrastive_{site}_{width_label}.json"
        )
        if not contrastive_file.exists():
            log.warning(
                "  %s: contrastive file not found at %s", site_short, contrastive_file
            )
            continue

        contrastive_65k = json.load(open(contrastive_file))
        scores_by_id = {e["feature_id"]: e for e in contrastive_65k}

        still_active = 0
        truly_gone = 0
        threshold_info = []
        for pid in info["disappeared"]:
            if pid in scores_by_id:
                d = scores_by_id[pid]["cohens_d"]
                rate = scores_by_id[pid].get("activation_rate_pos", 0)
                still_active += 1
                threshold_info.append(
                    {
                        "feature_id": pid,
                        "cohens_d_at_65k": d,
                        "activation_rate_pos": rate,
                    }
                )
                log.info(
                    "  %s feature %d: still in 65k contrastive data, d=%.3f, rate=%.3f",
                    site_short,
                    pid,
                    d,
                    rate,
                )
            else:
                truly_gone += 1

        log.info(
            "  %s: %d/%d disappeared features still have contrastive scores at 65k",
            site_short,
            still_active,
            len(info["disappeared"]),
        )
        log.info(
            "  %s: %d/%d truly absent from 65k contrastive data",
            site_short,
            truly_gone,
            len(info["disappeared"]),
        )

        checks.setdefault(site, {})["threshold_sensitivity"] = {
            "still_in_contrastive_data": still_active,
            "truly_absent": truly_gone,
            "total_disappeared": len(info["disappeared"]),
            "details": threshold_info,
        }

    # ── Check 4: Other width transitions ─────────────────────────────
    log.info("")
    log.info("─── Check 4: Other width transitions (65k→262k, 262k→1M) ───")

    width_pairs = [("65k", "262k"), ("262k", "1M")]
    for w_from, w_to in width_pairs:
        for site in ["last_prompt_token", "mean_response_token"]:
            site_short = "LPT" if site == "last_prompt_token" else "MRT"
            ids_from = feature_sets.get((site, w_from), set())
            ids_to = feature_sets.get((site, w_to), set())

            disappeared = sorted(ids_from - ids_to)
            new = sorted(ids_to - ids_from)
            survived = sorted(ids_from & ids_to)

            if not disappeared or not new:
                log.info(
                    "  %s %s→%s: %d disappeared, %d new — nothing to compare",
                    site_short,
                    w_from,
                    w_to,
                    len(disappeared),
                    len(new),
                )
                continue

            # Compute decoder cosine for this transition
            trans_ids = sorted(set(disappeared) | set(new))
            # Check if we already have these vectors
            missing = [i for i in trans_ids if i not in id_to_idx]
            if missing:
                log.info(
                    "  %s %s→%s: need %d additional decoder vectors, skipping (would require GPU)",
                    site_short,
                    w_from,
                    w_to,
                    len(missing),
                )
                checks.setdefault(site, {})[f"transition_{w_from}_{w_to}"] = {
                    "disappeared": len(disappeared),
                    "new": len(new),
                    "survived": len(survived),
                    "status": "skipped_missing_vectors",
                }
                continue

            p_vecs = all_vectors[[id_to_idx[i] for i in disappeared]]
            c_vecs = all_vectors[[id_to_idx[i] for i in new]]
            cos = compute_cosine_matrix(p_vecs, c_vecs)

            max_cos = float(cos.max())
            n_above_03 = int((cos.max(axis=1) > 0.3).sum())
            n_above_05 = int((cos.max(axis=1) > 0.5).sum())

            log.info(
                "  %s %s→%s: %d disappeared, %d new, max_cos=%.3f, %d>0.3, %d>0.5",
                site_short,
                w_from,
                w_to,
                len(disappeared),
                len(new),
                max_cos,
                n_above_03,
                n_above_05,
            )

            checks.setdefault(site, {})[f"transition_{w_from}_{w_to}"] = {
                "disappeared": len(disappeared),
                "new": len(new),
                "survived": len(survived),
                "max_cosine": max_cos,
                "n_above_03": n_above_03,
                "n_above_05": n_above_05,
            }

    # ── Check 5: Power analysis ──────────────────────────────────────
    log.info("")
    log.info("─── Check 5: Power analysis ───")
    for site, info in results_by_site.items():
        site_short = "LPT" if site == "last_prompt_token" else "MRT"
        n_parents = len(info["disappeared"])
        n_children = len(info["new_at_65k"])
        n_prompts = 104  # from data/encoded

        log.info(
            "  %s: %d parents × %d children × %d prompts",
            site_short,
            n_parents,
            n_children,
            n_prompts,
        )
        log.info(
            "    Decoder cosine: %d×%d = %d pairwise comparisons in R^3840",
            n_parents,
            n_children,
            n_parents * n_children,
        )
        log.info(
            "    Co-activation: binary vectors of length %d, Jaccard on %d-element sets",
            n_prompts,
            n_prompts,
        )

        # Expected max cosine between random unit vectors in R^3840
        # For d-dim unit vectors, E[|cos|] ≈ sqrt(2/(π*d))
        expected_random_cos = np.sqrt(2 / (np.pi * 3840))
        # With n_children comparisons, expected max ≈ E[|cos|] * sqrt(2*ln(n_children))
        expected_max_cos = expected_random_cos * np.sqrt(2 * np.log(max(n_children, 2)))
        log.info(
            "    Expected max |cos| between random unit vectors: %.3f (with %d comparisons)",
            expected_max_cos,
            n_children,
        )
        log.info("    Observed max cos: %.3f", info["summary"]["max_cosine_overall"])
        if info["summary"]["max_cosine_overall"] <= expected_max_cos * 1.5:
            log.info(
                "    → Observed max cos is within ~1.5× of random expectation. NO signal above noise."
            )
        else:
            log.info(
                "    → Observed max cos exceeds random expectation. Possible signal."
            )

        checks.setdefault(site, {})["power_analysis"] = {
            "n_parents": n_parents,
            "n_children": n_children,
            "n_prompts": n_prompts,
            "expected_max_random_cos": float(expected_max_cos),
            "observed_max_cos": info["summary"]["max_cosine_overall"],
        }

    return checks


def main():
    log.info("=" * 60)
    log.info("FEATURE HIERARCHY ANALYSIS — Method 1: Decoder Cosine")
    log.info("=" * 60)

    # ── Step 1: Identify disappeared and new features ────────────────
    feature_sets = load_feature_sets()

    results_by_site = {}

    for site in ["last_prompt_token", "mean_response_token"]:
        site_short = "LPT" if site == "last_prompt_token" else "MRT"
        log.info("")
        log.info("─── %s ───", site_short)

        ids_16k = feature_sets.get((site, "16k"), set())
        ids_65k = feature_sets.get((site, "65k"), set())

        disappeared = sorted(ids_16k - ids_65k)
        new_at_65k = sorted(ids_65k - ids_16k)
        survived = sorted(ids_16k & ids_65k)

        log.info("16k features: %d", len(ids_16k))
        log.info("65k features: %d", len(ids_65k))
        log.info("Survived (16k ∩ 65k): %d", len(survived))
        log.info("Disappeared (16k only): %d", len(disappeared))
        log.info("New at 65k: %d", len(new_at_65k))

        if not disappeared or not new_at_65k:
            log.info("Nothing to compare for %s, skipping.", site_short)
            continue

        results_by_site[site] = {
            "disappeared": disappeared,
            "new_at_65k": new_at_65k,
            "survived": survived,
        }

    # ── Step 2: Extract decoder vectors (one GPU load) ───────────────
    # Collect IDs from primary 16k→65k analysis
    all_ids = set()
    for info in results_by_site.values():
        all_ids.update(info["disappeared"])
        all_ids.update(info["new_at_65k"])
        all_ids.update(info["survived"])

    # Also collect IDs for other width transitions (65k→262k, 262k→1M)
    # so robustness Check 4 doesn't need a second GPU load
    for w_from, w_to in [("65k", "262k"), ("262k", "1M")]:
        for site in ["last_prompt_token", "mean_response_token"]:
            ids_from = feature_sets.get((site, w_from), set())
            ids_to = feature_sets.get((site, w_to), set())
            all_ids.update(ids_from - ids_to)  # disappeared
            all_ids.update(ids_to - ids_from)  # new

    all_ids_sorted = sorted(all_ids)
    id_to_idx = {fid: i for i, fid in enumerate(all_ids_sorted)}

    all_vectors = extract_decoder_vectors(all_ids_sorted)

    # ── Step 3: Compute cosine similarities ──────────────────────────
    for site, info in results_by_site.items():
        site_short = "LPT" if site == "last_prompt_token" else "MRT"
        log.info("")
        log.info("─── %s cosine analysis ───", site_short)

        parent_ids = info["disappeared"]
        child_ids = info["new_at_65k"]

        parent_vecs = all_vectors[[id_to_idx[i] for i in parent_ids]]
        child_vecs = all_vectors[[id_to_idx[i] for i in child_ids]]

        cos_matrix = compute_cosine_matrix(parent_vecs, child_vecs)

        # ── Step 4: Analyze results ──────────────────────────────────
        # For each parent, find top matches
        parent_matches = []
        for pi, pid in enumerate(parent_ids):
            row = cos_matrix[pi]
            top_indices = np.argsort(row)[::-1][:5]
            top_matches = [
                {"child_id": child_ids[ci], "cosine": float(row[ci])}
                for ci in top_indices
            ]
            parent_matches.append(
                {
                    "parent_id": pid,
                    "top_matches": top_matches,
                    "max_cosine": float(row.max()),
                    "n_above_0.5": int((row > 0.5).sum()),
                    "n_above_0.3": int((row > 0.3).sum()),
                }
            )

            if row.max() > 0.3:
                log.info(
                    "  Feature %d → top match: %d (cos=%.3f), %d matches >0.3",
                    pid,
                    child_ids[top_indices[0]],
                    row[top_indices[0]],
                    (row > 0.3).sum(),
                )

        # ── Falsification check ──────────────────────────────────────
        n_with_match_05 = sum(1 for m in parent_matches if m["max_cosine"] > 0.5)
        n_with_match_03 = sum(1 for m in parent_matches if m["max_cosine"] > 0.3)
        n_with_multi_05 = sum(1 for m in parent_matches if m["n_above_0.5"] >= 2)

        log.info("")
        log.info("─── %s SUMMARY ───", site_short)
        log.info(
            "Parents with max cos > 0.5: %d / %d", n_with_match_05, len(parent_ids)
        )
        log.info(
            "Parents with max cos > 0.3: %d / %d", n_with_match_03, len(parent_ids)
        )
        log.info(
            "Parents with 2+ children cos > 0.5: %d / %d",
            n_with_multi_05,
            len(parent_ids),
        )

        if n_with_match_05 == 0:
            log.warning(
                "FALSIFICATION: No parent has cos > 0.5 with any child. "
                "H1 (hierarchical decomposition) is NOT supported for %s.",
                site_short,
            )
        elif n_with_multi_05 >= 5:
            log.info(
                "SUCCESS: %d parents have 2+ children with cos > 0.5. "
                "H1 is supported — evidence of hierarchical splitting.",
                n_with_multi_05,
            )
        else:
            log.info(
                "MIXED: Some parents have matches but not strong evidence of widespread splitting."
            )

        info["parent_matches"] = parent_matches
        info["cos_matrix"] = cos_matrix.tolist()
        info["summary"] = {
            "n_parents": len(parent_ids),
            "n_children": len(child_ids),
            "n_with_match_05": n_with_match_05,
            "n_with_match_03": n_with_match_03,
            "n_with_multi_05": n_with_multi_05,
            "max_cosine_overall": float(cos_matrix.max()),
            "mean_max_cosine": float(
                np.array([m["max_cosine"] for m in parent_matches]).mean()
            ),
        }

    # ── Step 5: Method 2 — Co-activation ────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("METHOD 2: Co-activation Analysis")
    log.info("=" * 60)

    for site, info in results_by_site.items():
        m2_result = method2_coactivation(
            site=site,
            parent_ids=info["disappeared"],
            child_ids=info["new_at_65k"],
        )
        info["method2"] = m2_result

    # ── Step 6: Method 3 — Hierarchy R² ──────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("METHOD 3: Hierarchy R²")
    log.info("=" * 60)

    for site, info in results_by_site.items():
        m3_result = method3_hierarchy_r2(
            parent_ids=info["disappeared"],
            child_ids=info["new_at_65k"],
            all_vectors=all_vectors,
            id_to_idx=id_to_idx,
            site=site,
        )
        info["method3"] = m3_result

    # ── Step 7: Robustness checks ──────────────────────────────────────
    robustness = robustness_checks(
        results_by_site=results_by_site,
        feature_sets=feature_sets,
        all_vectors=all_vectors,
        id_to_idx=id_to_idx,
    )

    # ── Step 8: Save all results ─────────────────────────────────────
    output = {}
    for site, info in results_by_site.items():
        output[site] = {
            "disappeared": info["disappeared"],
            "new_at_65k": info["new_at_65k"],
            "survived": info["survived"],
            "method1_decoder_cosine": {
                "parent_matches": info["parent_matches"],
                "summary": info["summary"],
            },
            "method2_coactivation": {
                "parent_results": info["method2"]["parent_results"],
                "summary": info["method2"]["summary"],
            },
            "method3_hierarchy_r2": {
                "parent_results": info["method3"]["parent_results"],
                "summary": info["method3"]["summary"],
            },
            "robustness": robustness.get(site, {}),
        }

    out_path = OUT_DIR / "all_methods.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("")
    log.info("Results saved to %s", out_path)

    # ── Final verdict across all methods ─────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("FINAL VERDICT — ALL THREE METHODS")
    log.info("=" * 60)
    for site, info in results_by_site.items():
        site_short = "LPT" if site == "last_prompt_token" else "MRT"
        s1 = info["summary"]
        s2 = info["method2"]["summary"]
        s3 = info["method3"]["summary"]
        log.info("  %s:", site_short)
        log.info(
            "    M1 decoder cosine: %d/%d >0.5, max=%.3f, mean_max=%.3f",
            s1["n_with_match_05"],
            s1["n_parents"],
            s1["max_cosine_overall"],
            s1["mean_max_cosine"],
        )
        log.info(
            "    M2 co-activation:  %d/%d Jaccard>0.5, %d >0.3, mean_max=%.3f",
            s2["n_jaccard_05"],
            s2["n_parents"],
            s2["n_jaccard_03"],
            s2["mean_max_jaccard"],
        )
        log.info(
            "    M3 hierarchy R²:   %d/%d >0.5, %d >0.3, mean=%.3f (children/d=%.4f)",
            s3["n_r2_05"],
            s3["n_parents"],
            s3["n_r2_03"],
            s3["mean_r2"],
            s3.get("children_d_model_ratio", 0),
        )

    # Cross-method consistency check
    log.info("")
    all_consistent = True
    for site, info in results_by_site.items():
        site_short = "LPT" if site == "last_prompt_token" else "MRT"
        s1 = info["summary"]
        s2 = info["method2"]["summary"]
        m1_null = s1["n_with_match_05"] == 0
        m2_null = s2["n_jaccard_05"] == 0
        if m1_null and m2_null:
            log.info("  %s: CONSISTENT — both M1 and M2 show no hierarchy.", site_short)
        elif m1_null and not m2_null:
            log.warning(
                "  %s: CONTRADICTION — M1 (geometry) sees no hierarchy, "
                "but M2 (behavior) finds co-activation! Investigate further.",
                site_short,
            )
            all_consistent = False
        elif not m1_null and m2_null:
            log.warning(
                "  %s: CONTRADICTION — M1 (geometry) finds matches, "
                "but M2 (behavior) shows no co-activation! Likely artifact.",
                site_short,
            )
            all_consistent = False
        else:
            log.info(
                "  %s: CONSISTENT — both M1 and M2 find hierarchy signals.", site_short
            )

    if all_consistent:
        log.info("ALL METHODS CONSISTENT — result is robust.")
    else:
        log.warning("METHODS DISAGREE — further investigation needed.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
