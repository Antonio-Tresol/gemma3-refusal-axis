"""Milestone 4: SAE Encoding + Contrastive Scoring.

GPU Phase B: Load 1M SAE, encode retained-pair activations, identify
candidate refusal features via two methods:
  Method 1 — Per-feature contrastive scoring (SAILS Stage 1)
  Method 2 — Alignment direction decomposition (Chen et al. Appendix M)

Then cross-reference, track across widths, and run domain sub-analysis.

Usage:
  uv run python milestone_4_encode_score.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from refusal_decomposition import (
    CFG,
    ProgressTracker,
    clear_vram,
    fmt_time,
    load_sae,
    setup_logging,
    vram_report,
)

log = setup_logging("milestone_4")

RETAINED_FILE = CFG.data_dir / "retained_pairs.json"
ENCODED_DIR = CFG.data_dir / "encoded"
RESULTS_DIR = CFG.data_dir / "milestone_4_results"

WIDTHS = {
    "16k": 16_384,
    "65k": 65_536,
    "262k": 262_144,
    "1M": 1_048_576,
}
SITES = ("last_prompt_token", "mean_response_token")
TOP_K = 50


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass
class FeatureScore:
    feature_id: int
    cohens_d: float
    contrastive_score: float
    activation_rate_pos: float
    activation_rate_neg: float


@dataclass
class CosineScore:
    feature_id: int
    cosine_sim: float


# ---------------------------------------------------------------------------
# Phase 1: SAE Encoding
# ---------------------------------------------------------------------------
def load_activations(
    site: str, condition: str, retained_ids: list[int]
) -> torch.Tensor:
    """Load activation .pt files for retained pairs. Returns (N, d_model)."""
    acts = []
    for pair_id in retained_ids:
        path = CFG.activation_path(site, condition, pair_id)
        acts.append(torch.load(path, weights_only=True))
    return torch.stack(acts)  # (N, 3840)


def encode_activations(sae, retained_ids: list[int]) -> None:
    """Encode all activations through SAE and save."""
    ENCODED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("PHASE 1: SAE Encoding (%d retained pairs)", len(retained_ids))
    log.info("=" * 60)

    for site in SITES:
        for condition in ("positive", "negative"):
            out_path = ENCODED_DIR / f"{site}_{condition}.pt"
            if out_path.exists():
                log.info("Already encoded: %s", out_path)
                continue

            t0 = time.time()
            acts = load_activations(site, condition, retained_ids)
            log.info("%s/%s: loaded %s activations", site, condition, tuple(acts.shape))

            # Encode on GPU in batches to avoid OOM
            acts_gpu = acts.to(device=CFG.device, dtype=torch.bfloat16)
            with torch.no_grad():
                encoded = sae.encode(acts_gpu)  # (N, 1M)
            encoded_cpu = encoded.cpu().to(torch.float32)
            del acts_gpu, encoded
            torch.cuda.empty_cache()

            torch.save(encoded_cpu, out_path)
            elapsed = time.time() - t0

            # L0 stats
            l0_per_sample = (encoded_cpu != 0).sum(dim=1).float()
            log.info(
                "  Saved %s | shape=%s | L0 mean=%.1f min=%d max=%d | %.1fs",
                out_path,
                tuple(encoded_cpu.shape),
                l0_per_sample.mean().item(),
                int(l0_per_sample.min().item()),
                int(l0_per_sample.max().item()),
                elapsed,
            )


# ---------------------------------------------------------------------------
# Phase 2: FVU sanity check
# ---------------------------------------------------------------------------
def fvu_check(sae, retained_ids: list[int]) -> None:
    """Spot-check reconstruction quality on a few activations."""
    log.info("=" * 60)
    log.info("FVU Sanity Check")
    log.info("=" * 60)

    site = SITES[0]
    acts = load_activations(site, "positive", retained_ids[:5])
    acts_gpu = acts.to(device=CFG.device, dtype=torch.bfloat16)

    with torch.no_grad():
        recon, _ = sae(acts_gpu)

    residual = acts_gpu.float() - recon.float()
    total_var = acts_gpu.float() - acts_gpu.float().mean(dim=1, keepdim=True)
    fvu = (residual**2).sum(dim=1) / (total_var**2).sum(dim=1)

    for i, f in enumerate(fvu):
        log.info("  Sample %d: FVU=%.4f", i, f.item())

    mean_fvu = fvu.mean().item()
    log.info("  Mean FVU: %.4f (expect < 0.10)", mean_fvu)
    if mean_fvu > 0.10:
        log.warning("FVU %.4f > 0.10 — reconstruction quality is low", mean_fvu)


# ---------------------------------------------------------------------------
# Phase 3: Method 1 — Per-feature contrastive scoring
# ---------------------------------------------------------------------------
def contrastive_scoring(retained_ids: list[int]) -> None:
    """Compute Cohen's d per feature at each width x site."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("PHASE 3: Method 1 — Contrastive Scoring")
    log.info("=" * 60)

    for site in SITES:
        pos = torch.load(ENCODED_DIR / f"{site}_positive.pt", weights_only=True)
        neg = torch.load(ENCODED_DIR / f"{site}_negative.pt", weights_only=True)
        log.info("%s: pos=%s neg=%s", site, tuple(pos.shape), tuple(neg.shape))

        for width_label, width in WIDTHS.items():
            t0 = time.time()
            pos_w = pos[:, :width]  # (N, W)
            neg_w = neg[:, :width]  # (N, W)

            mean_pos = pos_w.mean(dim=0)  # (W,)
            mean_neg = neg_w.mean(dim=0)  # (W,)
            var_pos = pos_w.var(dim=0)  # (W,)
            var_neg = neg_w.var(dim=0)  # (W,)

            contrastive = mean_pos - mean_neg  # (W,)
            pooled_std = ((var_pos + var_neg) / 2).sqrt().clamp(min=1e-8)  # (W,)
            cohens_d = contrastive / pooled_std  # (W,)

            # Activation rates
            act_rate_pos = (pos_w > 0).float().mean(dim=0)  # (W,)
            act_rate_neg = (neg_w > 0).float().mean(dim=0)  # (W,)

            # Top 50 by |Cohen's d|
            abs_d = cohens_d.abs()
            top_indices = abs_d.topk(TOP_K).indices.tolist()

            results = []
            for idx in top_indices:
                results.append(
                    FeatureScore(
                        feature_id=idx,
                        cohens_d=round(cohens_d[idx].item(), 4),
                        contrastive_score=round(contrastive[idx].item(), 4),
                        activation_rate_pos=round(act_rate_pos[idx].item(), 4),
                        activation_rate_neg=round(act_rate_neg[idx].item(), 4),
                    )
                )

            out_path = RESULTS_DIR / f"contrastive_{site}_{width_label}.json"
            with open(out_path, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)

            max_d = abs_d.max().item()
            log.info(
                "  %s/%s: max|d|=%.3f top3=%s | %.1fs",
                site,
                width_label,
                max_d,
                [(r.feature_id, r.cohens_d) for r in results[:3]],
                time.time() - t0,
            )


# ---------------------------------------------------------------------------
# Phase 4: Method 2 — Alignment direction decomposition
# ---------------------------------------------------------------------------
def alignment_decomposition(sae) -> None:
    """Cosine similarity between refusal direction and SAE decoder vectors."""
    log.info("=" * 60)
    log.info("PHASE 4: Method 2 — Alignment Direction Decomposition")
    log.info("=" * 60)

    w_dec = sae.w_dec.data.cpu().to(torch.float32)  # (1M, d_model)

    for site in SITES:
        refusal_dir = torch.load(
            CFG.refusal_direction_path(site), weights_only=True
        )  # (d_model,)

        # Cosine similarity with each decoder row
        # cos_sim = (w_dec @ refusal_dir) / (||w_dec_row|| * ||refusal_dir||)
        # refusal_dir is already unit norm
        dec_norms = w_dec.norm(dim=1).clamp(min=1e-8)  # (1M,)
        cos_sims = (w_dec @ refusal_dir) / dec_norms  # (1M,)

        # Top 50 by |cosine similarity|
        abs_cos = cos_sims.abs()
        top_indices = abs_cos.topk(TOP_K).indices.tolist()

        results = []
        for idx in top_indices:
            results.append(
                CosineScore(
                    feature_id=idx,
                    cosine_sim=round(cos_sims[idx].item(), 4),
                )
            )

        out_path = RESULTS_DIR / f"cosine_{site}.json"
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        log.info(
            "  %s: top3=%s",
            site,
            [(r.feature_id, r.cosine_sim) for r in results[:3]],
        )


# ---------------------------------------------------------------------------
# Phase 5: Cross-referencing + Width tracking + Domain analysis
# ---------------------------------------------------------------------------
def cross_reference_and_analyze(retained_ids: list[int]) -> None:
    """Cross-reference methods, track features across widths, domain sub-analysis."""
    log.info("=" * 60)
    log.info("PHASE 5: Cross-reference, Width Tracking, Domain Analysis")
    log.info("=" * 60)

    # Load manifest for domain info
    with open(CFG.manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    retained_set = set(retained_ids)
    # Map pair_id -> domain
    pair_domain: dict[int, str] = {}
    for entry in manifest:
        if entry["pair_id"] in retained_set and entry["condition"] == "positive":
            pair_domain[entry["pair_id"]] = entry["domain"]

    domains = sorted(set(pair_domain.values()))

    for site in SITES:
        log.info("--- %s ---", site)

        # Load cosine scores for this site
        with open(RESULTS_DIR / f"cosine_{site}.json") as f:
            cosine_top = {s["feature_id"] for s in json.load(f)}

        width_top_sets: dict[str, set[int]] = {}
        all_dual_validated: set[int] = set()

        for width_label, width in WIDTHS.items():
            with open(RESULTS_DIR / f"contrastive_{site}_{width_label}.json") as f:
                contrastive_top = {s["feature_id"] for s in json.load(f)}

            # Restrict cosine top to features within this width
            cosine_in_width = {fid for fid in cosine_top if fid < width}
            dual = contrastive_top & cosine_in_width
            all_dual_validated |= dual

            width_top_sets[width_label] = contrastive_top

            log.info(
                "  %s: contrastive=%d cosine_in_width=%d dual_validated=%d",
                width_label,
                len(contrastive_top),
                len(cosine_in_width),
                len(dual),
            )

        # Width tracking: feature persistence and emergence
        prev_label = None
        for width_label in WIDTHS:
            if prev_label is not None:
                prev_set = width_top_sets[prev_label]
                curr_set = width_top_sets[width_label]
                prev_width = WIDTHS[prev_label]

                persisted = prev_set & curr_set
                new_features = {fid for fid in curr_set if fid >= prev_width}
                dropped = prev_set - curr_set

                log.info(
                    "  %s->%s: persisted=%d new(id>=%d)=%d dropped=%d",
                    prev_label,
                    width_label,
                    len(persisted),
                    prev_width,
                    len(new_features),
                    len(dropped),
                )
            prev_label = width_label

        # Domain sub-analysis at 1M width
        log.info("  Domain specificity at 1M:")
        pos_encoded = torch.load(ENCODED_DIR / f"{site}_positive.pt", weights_only=True)
        neg_encoded = torch.load(ENCODED_DIR / f"{site}_negative.pt", weights_only=True)

        # Map retained pair indices to domains
        pair_indices_by_domain: dict[str, list[int]] = {d: [] for d in domains}
        for i, pid in enumerate(retained_ids):
            if pid in pair_domain:
                pair_indices_by_domain[pair_domain[pid]].append(i)

        # Load 1M contrastive features
        with open(RESULTS_DIR / f"contrastive_{site}_1M.json") as f:
            top_features_1m = json.load(f)

        domain_profiles: list[dict] = []
        for feat in top_features_1m[:TOP_K]:
            fid = feat["feature_id"]
            profile: dict = {"feature_id": fid, "overall_d": feat["cohens_d"]}

            specific_domains: list[str] = []
            for domain in domains:
                idx = pair_indices_by_domain[domain]
                if len(idx) < 3:
                    profile[domain] = None
                    continue
                idx_t = torch.tensor(idx)
                p = pos_encoded[idx_t, fid]
                n = neg_encoded[idx_t, fid]
                mean_diff = p.mean() - n.mean()
                pooled = ((p.var() + n.var()) / 2).sqrt().clamp(min=1e-8)
                d_val = (mean_diff / pooled).item()
                profile[domain] = round(d_val, 3)
                if abs(d_val) > 0.3:
                    specific_domains.append(domain)

            profile["specific_to"] = specific_domains
            n_specific = len(specific_domains)
            if n_specific <= 2 and n_specific > 0:
                profile["type"] = "domain_specific"
            elif n_specific > 2:
                profile["type"] = "general"
            else:
                profile["type"] = "weak"

            domain_profiles.append(profile)

        out_path = RESULTS_DIR / f"domain_profiles_{site}.json"
        with open(out_path, "w") as f:
            json.dump(domain_profiles, f, indent=2)

        n_general = sum(1 for p in domain_profiles if p["type"] == "general")
        n_specific = sum(1 for p in domain_profiles if p["type"] == "domain_specific")
        n_weak = sum(1 for p in domain_profiles if p["type"] == "weak")
        log.info(
            "  general=%d domain_specific=%d weak=%d", n_general, n_specific, n_weak
        )

    # Save all dual-validated feature IDs
    dual_file = RESULTS_DIR / "dual_validated_features.json"
    with open(dual_file, "w") as f:
        json.dump(sorted(all_dual_validated), f, indent=2)
    log.info(
        "Dual-validated features: %d saved to %s", len(all_dual_validated), dual_file
    )


# ---------------------------------------------------------------------------
# Phase 6: Save decoder vectors for candidate features
# ---------------------------------------------------------------------------
def save_decoder_vectors(sae) -> None:
    """Save decoder vectors for all candidate features."""
    log.info("=" * 60)
    log.info("PHASE 6: Save decoder vectors")
    log.info("=" * 60)

    # Collect all candidate feature IDs across all result files
    all_feature_ids: set[int] = set()
    for f in RESULTS_DIR.glob("contrastive_*.json"):
        with open(f) as fh:
            for entry in json.load(fh):
                all_feature_ids.add(entry["feature_id"])
    for f in RESULTS_DIR.glob("cosine_*.json"):
        with open(f) as fh:
            for entry in json.load(fh):
                all_feature_ids.add(entry["feature_id"])

    log.info("Candidate features: %d unique", len(all_feature_ids))

    w_dec = sae.w_dec.data.cpu().to(torch.float32)  # (1M, d_model)
    decoder_vectors = {fid: w_dec[fid] for fid in sorted(all_feature_ids)}

    out_path = RESULTS_DIR / "decoder_vectors.pt"
    torch.save(decoder_vectors, out_path)
    log.info(
        "Saved: %s (%d vectors, each shape %s)",
        out_path,
        len(decoder_vectors),
        (CFG.d_model,),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate() -> None:
    """Run milestone 4 validation gate."""
    log.info("=" * 60)
    log.info("VALIDATION: Milestone 4 Gate")
    log.info("=" * 60)

    # Signal strength
    max_d_overall = 0.0
    for f in RESULTS_DIR.glob("contrastive_*.json"):
        with open(f) as fh:
            for entry in json.load(fh):
                max_d_overall = max(max_d_overall, abs(entry["cohens_d"]))
    log.info("Max |Cohen's d| across all: %.3f (expect > 0.5)", max_d_overall)
    if max_d_overall < 0.3:
        log.error("Max |d| < 0.3 — SAE features not picking up refusal signal")
    elif max_d_overall < 0.5:
        log.warning("Max |d| < 0.5 — signal present but moderate")

    # Method agreement
    for site in SITES:
        with open(RESULTS_DIR / f"cosine_{site}.json") as f:
            cosine_ids = {e["feature_id"] for e in json.load(f)}
        for width_label in WIDTHS:
            with open(RESULTS_DIR / f"contrastive_{site}_{width_label}.json") as f:
                contrastive_ids = {e["feature_id"] for e in json.load(f)}
            overlap = cosine_ids & {
                fid for fid in contrastive_ids if fid < WIDTHS[width_label]
            }
            log.info("  %s/%s: method overlap=%d", site, width_label, len(overlap))

    # Width variation
    for site in SITES:
        with open(RESULTS_DIR / f"contrastive_{site}_16k.json") as f:
            set_16k = {e["feature_id"] for e in json.load(f)}
        with open(RESULTS_DIR / f"contrastive_{site}_1M.json") as f:
            set_1m = {e["feature_id"] for e in json.load(f)}
        diff = set_1m - set_16k
        log.info("  %s: features in 1M top50 but not 16k top50: %d", site, len(diff))

    # Both sites produce results
    for site in SITES:
        with open(RESULTS_DIR / f"contrastive_{site}_1M.json") as f:
            top = json.load(f)
        max_d_site = max(abs(e["cohens_d"]) for e in top)
        log.info("  %s max|d| at 1M: %.3f", site, max_d_site)

    # Domain sub-analysis
    for site in SITES:
        dp_path = RESULTS_DIR / f"domain_profiles_{site}.json"
        if dp_path.exists():
            with open(dp_path) as f:
                profiles = json.load(f)
            n_specific = sum(1 for p in profiles if p["type"] == "domain_specific")
            log.info("  %s: %d domain-specific features found", site, n_specific)

    # Decoder vectors
    dv_path = RESULTS_DIR / "decoder_vectors.pt"
    if dv_path.exists():
        dv = torch.load(dv_path, weights_only=False)
        log.info(
            "  Decoder vectors: %d features, shape %s each", len(dv), (CFG.d_model,)
        )
    else:
        log.error("  Decoder vectors not found!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log.info("=" * 60)
    log.info("MILESTONE 4: SAE ENCODING + CONTRASTIVE SCORING")
    log.info("=" * 60)

    # Load retained pairs
    with open(RETAINED_FILE, encoding="utf-8") as f:
        retained_ids: list[int] = json.load(f)
    log.info("Retained pairs: %d", len(retained_ids))

    # Load SAE
    log.info("Loading 1M SAE...")
    sae_t0 = time.time()
    vram_report("before SAE load", log)
    sae = load_sae(width=CFG.sae_width_1m, width_label="1m", log=log)
    vram_report("after SAE load", log)
    log.info("SAE loaded in %s", fmt_time(time.time() - sae_t0))

    # Phase 1: Encode
    encode_activations(sae, retained_ids)

    # FVU check
    fvu_check(sae, retained_ids)

    # Phase 3: Method 1
    contrastive_scoring(retained_ids)

    # Phase 4: Method 2
    alignment_decomposition(sae)

    # Phase 5: Cross-reference + domain analysis
    cross_reference_and_analyze(retained_ids)

    # Phase 6: Save decoder vectors
    save_decoder_vectors(sae)

    # Unload SAE
    log.info("Unloading SAE...")
    del sae
    clear_vram()
    vram_report("after SAE unload", log)

    # Validation
    validate()

    log.info("=" * 60)
    log.info("ALL DONE in %s", fmt_time(time.time() - t0))
    log.info("Next: Milestone 5 — Causal Steering Validation")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
