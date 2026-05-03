"""Milestone 6: Feature Judging — Automated interpretability of candidate features.

Pipeline (following Bills et al. 2023 → Templeton et al. 2024 → Gemma Scope 2 Sec 4.3):
  Phase 1 — Build dossiers: max/min activating prompts, statistics (CPU)
  Phase 2 — Automated judging: LLM describes each feature from examples (Agent SDK, Sonnet 4.6)
  Phase 3 — Detection score: LLM predicts which examples activate from description alone
           (Gemma Scope 2 Sec 4.3 validation method)
  Phase 4 — Validation gate

Design decisions (evidence-pinned):
  - Max-activating examples as primary evidence — Bills et al. 2023, standard since
  - Detection score for description validation — Gemma Scope 2 Sec 4.3
  - Domain labels WITHHELD from initial judging prompt to prevent anchoring — novel, justified
    by concern that showing domain profile would make the judge parrot labels rather than
    independently classify from activation patterns
  - Single judging pass (not 3) — descriptions are qualitative, no meaningful "median";
    detection score validates instead
  - Sub-type classification — novel contribution of this thesis

Usage:
  uv run python milestone_6_judging.py
  uv run python milestone_6_judging.py --phase dossier
  uv run python milestone_6_judging.py --phase judge
  uv run python milestone_6_judging.py --phase detect
  uv run python milestone_6_judging.py --phase validate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anyio
import torch
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from refusal_decomposition import CFG, fmt_time, setup_logging

log = setup_logging("milestone_6")

M4_DIR = CFG.data_dir / "milestone_4_results"
M5_DIR = CFG.data_dir / "milestone_5_results"
M6_DIR = CFG.data_dir / "milestone_6_results"
ENCODED_DIR = CFG.data_dir / "encoded"

TOP_EXAMPLES = 5  # max/min activating prompts per feature


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass
class ActivatingExample:
    pair_id: int
    condition: str
    domain: str
    prompt_text: str
    activation_value: float


@dataclass
class FeatureDossier:
    feature_id: int
    widths_present: list[str]  # which width top-50s it appears in
    cohens_d_by_width: dict[str, float]
    cosine_sim: float | None
    activation_rate_pos: float
    activation_rate_neg: float
    causal_result: dict | None  # from M5 if tested
    max_activating: list[dict]  # top-K examples
    min_activating: list[dict]  # bottom-K examples (lowest nonzero)
    # Domain profile is built but NOT shown to the judge initially — prevents anchoring
    domain_profile: dict[str, float]  # per-domain Cohen's d


@dataclass
class JudgmentEntry:
    feature_id: int
    description: str
    is_refusal_relevant: bool
    sub_types: list[str]
    confidence: str  # high/medium/low
    rationale: str


@dataclass
class DetectionResult:
    feature_id: int
    accuracy: float  # fraction correctly classified
    n_tested: int


# ---------------------------------------------------------------------------
# Phase 1: Build Dossiers
# ---------------------------------------------------------------------------
def build_dossiers() -> list[FeatureDossier]:
    """Build a dossier for each unique candidate feature from M4 data."""
    M6_DIR.mkdir(parents=True, exist_ok=True)

    dossier_path = M6_DIR / "dossiers.json"
    if dossier_path.exists():
        log.info("Dossiers already built, loading from %s", dossier_path)
        with open(dossier_path, encoding="utf-8") as f:
            return [FeatureDossier(**d) for d in json.load(f)]

    log.info("=" * 60)
    log.info("PHASE 1: Build Feature Dossiers")
    log.info("=" * 60)

    # Load manifest for prompt text + domain
    with open(CFG.manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    retained_ids = json.load(open(CFG.data_dir / "retained_pairs.json"))
    retained_set = set(retained_ids)

    # Build lookup: (pair_id, condition) -> manifest entry
    manifest_lookup: dict[tuple[int, str], dict] = {}
    for entry in manifest:
        if entry["pair_id"] in retained_set:
            manifest_lookup[(entry["pair_id"], entry["condition"])] = entry

    # Collect all unique candidate feature IDs
    all_feature_ids: set[int] = set()
    for f in M4_DIR.glob("contrastive_*.json"):
        with open(f) as fh:
            for entry in json.load(fh):
                all_feature_ids.add(entry["feature_id"])
    for f in M4_DIR.glob("cosine_*.json"):
        with open(f) as fh:
            for entry in json.load(fh):
                all_feature_ids.add(entry["feature_id"])

    log.info("Unique candidate features: %d", len(all_feature_ids))

    # Load contrastive results for width presence + Cohen's d
    widths = {"16k": 16384, "65k": 65536, "262k": 262144, "1M": 1048576}
    # We use mean_response_token as primary site — captures response behavior
    # (Chen et al. Sec 2.2: extract from response tokens where behavior is expressed)
    site = "mean_response_token"

    contrastive_by_width: dict[str, dict[int, dict]] = {}
    for wl in widths:
        path = M4_DIR / f"contrastive_{site}_{wl}.json"
        if path.exists():
            with open(path) as f:
                contrastive_by_width[wl] = {e["feature_id"]: e for e in json.load(f)}

    # Also load last_prompt_token for features only found there
    for wl in widths:
        path = M4_DIR / f"contrastive_last_prompt_token_{wl}.json"
        if path.exists():
            with open(path) as f:
                for e in json.load(f):
                    if e["feature_id"] not in contrastive_by_width.get(wl, {}):
                        contrastive_by_width.setdefault(wl, {})[e["feature_id"]] = e

    # Cosine similarity lookup
    cosine_lookup: dict[int, float] = {}
    for site_name in ("mean_response_token", "last_prompt_token"):
        path = M4_DIR / f"cosine_{site_name}.json"
        if path.exists():
            with open(path) as f:
                for e in json.load(f):
                    if e["feature_id"] not in cosine_lookup:
                        cosine_lookup[e["feature_id"]] = e["cosine_sim"]

    # Load encoded features for max-activating examples
    # Using mean_response_token site as primary
    pos_encoded = torch.load(
        ENCODED_DIR / "mean_response_token_positive.pt", weights_only=True
    )
    neg_encoded = torch.load(
        ENCODED_DIR / "mean_response_token_negative.pt", weights_only=True
    )

    # Domain profiles
    domain_profiles: dict[int, dict] = {}
    dp_path = M4_DIR / "domain_profiles_mean_response_token.json"
    if dp_path.exists():
        with open(dp_path) as f:
            for p in json.load(f):
                fid = p["feature_id"]
                domain_profiles[fid] = {
                    k: v
                    for k, v in p.items()
                    if k not in ("feature_id", "overall_d", "specific_to", "type")
                    and v is not None
                }

    # M5 causal results
    causal_lookup: dict[int, dict] = {}
    m5_analysis = M5_DIR / "analysis.json"
    if m5_analysis.exists():
        with open(m5_analysis) as f:
            for r in json.load(f):
                causal_lookup[r["feature_id"]] = {
                    "passed": r["passed_causal"],
                    "max_refusal_increase_benign": r["max_refusal_increase_benign"],
                    "max_refusal_decrease_harmful": r["max_refusal_decrease_harmful"],
                    "best_layer": r["best_layer"],
                    "best_coeff": r["best_coeff"],
                }

    # Build dossiers
    dossiers: list[FeatureDossier] = []
    for fid in sorted(all_feature_ids):
        # Width presence
        widths_present = [
            wl for wl, data in contrastive_by_width.items() if fid in data
        ]

        # Cohen's d per width
        d_by_width = {}
        for wl in widths_present:
            if fid in contrastive_by_width.get(wl, {}):
                d_by_width[wl] = contrastive_by_width[wl][fid]["cohens_d"]

        # Activation rates from any available contrastive entry
        act_pos = 0.0
        act_neg = 0.0
        for wl in widths_present:
            entry = contrastive_by_width.get(wl, {}).get(fid)
            if entry:
                act_pos = entry.get("activation_rate_pos", 0.0)
                act_neg = entry.get("activation_rate_neg", 0.0)
                break

        # Max/min activating examples
        # Check if feature is within the encoded tensor width (1M)
        if fid < pos_encoded.shape[1]:
            pos_acts = pos_encoded[:, fid]  # (N,)
            neg_acts = neg_encoded[:, fid]  # (N,)
            all_acts = torch.cat([pos_acts, neg_acts])
            all_conditions = ["positive"] * len(pos_acts) + ["negative"] * len(neg_acts)
            all_pair_ids = retained_ids + retained_ids

            # Top-K max activating
            topk_vals, topk_idx = all_acts.topk(min(TOP_EXAMPLES, len(all_acts)))
            max_examples = []
            for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
                pid = all_pair_ids[idx]
                cond = all_conditions[idx]
                m_entry = manifest_lookup.get((pid, cond), {})
                max_examples.append(
                    {
                        "pair_id": pid,
                        "condition": cond,
                        "domain": m_entry.get("domain", ""),
                        "prompt_text": m_entry.get("prompt_text", ""),
                        "activation_value": round(val, 4),
                    }
                )

            # Bottom-K min activating (nonzero)
            nonzero_mask = all_acts > 0
            if nonzero_mask.sum() > 0:
                nonzero_acts = all_acts.clone()
                nonzero_acts[~nonzero_mask] = float("inf")
                botk_vals, botk_idx = nonzero_acts.topk(
                    min(TOP_EXAMPLES, int(nonzero_mask.sum())), largest=False
                )
                min_examples = []
                for val, idx in zip(botk_vals.tolist(), botk_idx.tolist()):
                    if val == float("inf"):
                        continue
                    pid = all_pair_ids[idx]
                    cond = all_conditions[idx]
                    m_entry = manifest_lookup.get((pid, cond), {})
                    min_examples.append(
                        {
                            "pair_id": pid,
                            "condition": cond,
                            "domain": m_entry.get("domain", ""),
                            "prompt_text": m_entry.get("prompt_text", ""),
                            "activation_value": round(val, 4),
                        }
                    )
            else:
                min_examples = []
        else:
            max_examples = []
            min_examples = []

        dossier = FeatureDossier(
            feature_id=fid,
            widths_present=widths_present,
            cohens_d_by_width=d_by_width,
            cosine_sim=cosine_lookup.get(fid),
            activation_rate_pos=act_pos,
            activation_rate_neg=act_neg,
            causal_result=causal_lookup.get(fid),
            max_activating=max_examples,
            min_activating=min_examples,
            domain_profile=domain_profiles.get(fid, {}),
        )
        dossiers.append(dossier)

    # Save
    with open(dossier_path, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in dossiers], f, indent=2, ensure_ascii=False)
    log.info("Built %d dossiers, saved to %s", len(dossiers), dossier_path)

    return dossiers


# ---------------------------------------------------------------------------
# Phase 2: Automated Judging
# ---------------------------------------------------------------------------

# Judging prompt — evidence-pinned:
# - Max-activating examples as primary evidence: Bills et al. 2023
# - Structured 5-part output: adapted from plan Sec. Milestone 6
# - Domain labels WITHHELD: novel choice to prevent anchoring
JUDGING_PROMPT = """You are an expert in mechanistic interpretability of language models. You are examining a feature (latent) extracted from a Sparse Autoencoder trained on a language model's residual stream.

Your task: Based on the evidence below, determine what this feature captures and whether it is relevant to refusal behavior (the model declining requests).

## Evidence

Feature ID: {feature_id}
Activation rate on refusal prompts: {act_pos:.0%}
Activation rate on benign prompts: {act_neg:.0%}
{causal_line}

### Top-{n_max} MAXIMALLY activating prompts (highest activation):
{max_examples}

### Top-{n_min} MINIMALLY activating prompts (lowest nonzero activation):
{min_examples}

## Instructions

Examine the patterns in what makes this feature activate strongly vs weakly. Then provide:

1. DESCRIPTION: One sentence describing what this feature captures.
2. REFUSAL_RELEVANT: Yes or No — is this feature relevant to the model refusing or declining requests?
3. SUB_TYPES: If relevant, which refusal sub-type(s) does it relate to? Choose from: safety, ethical, legal, privacy, identity_boundary, capability_boundary, general_refusal. List all that apply, comma-separated. If not relevant, write "none".
4. CONFIDENCE: high, medium, or low.
5. RATIONALE: One sentence explaining your judgment.

Format EXACTLY as:
DESCRIPTION: <text>
REFUSAL_RELEVANT: <Yes or No>
SUB_TYPES: <comma-separated list or "none">
CONFIDENCE: <high/medium/low>
RATIONALE: <text>
"""


def format_examples(examples: list[dict], label: str) -> str:
    if not examples:
        return f"(no {label} examples available)"
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(
            f"{i}. [activation={ex['activation_value']:.2f}] "
            f'[{ex["condition"]}] "{ex["prompt_text"][:200]}"'
        )
    return "\n".join(lines)


def parse_judgment(text: str) -> JudgmentEntry | None:
    """Parse structured judgment from LLM output."""
    fields: dict[str, str] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        for key in (
            "DESCRIPTION",
            "REFUSAL_RELEVANT",
            "SUB_TYPES",
            "CONFIDENCE",
            "RATIONALE",
        ):
            if line.upper().startswith(key + ":"):
                fields[key.lower()] = line.split(":", 1)[1].strip()

    if "description" not in fields:
        return None

    is_relevant = fields.get("refusal_relevant", "no").strip().lower().startswith("y")
    sub_types_raw = fields.get("sub_types", "none").strip()
    sub_types = (
        [s.strip() for s in sub_types_raw.split(",") if s.strip() != "none"]
        if is_relevant
        else []
    )

    return JudgmentEntry(
        feature_id=0,  # filled by caller
        description=fields.get("description", ""),
        is_refusal_relevant=is_relevant,
        sub_types=sub_types,
        confidence=fields.get("confidence", "low").strip().lower(),
        rationale=fields.get("rationale", ""),
    )


async def judge_one(dossier: FeatureDossier) -> JudgmentEntry | None:
    """Judge a single feature using Sonnet 4.6."""
    causal_line = ""
    if dossier.causal_result:
        cr = dossier.causal_result
        causal_line = (
            f"Causal steering test: {'PASSED' if cr['passed'] else 'FAILED'} "
            f"(refusal increase on benign: +{cr['max_refusal_increase_benign']}, "
            f"refusal decrease on harmful: -{cr['max_refusal_decrease_harmful']})"
        )

    prompt = JUDGING_PROMPT.format(
        feature_id=dossier.feature_id,
        act_pos=dossier.activation_rate_pos,
        act_neg=dossier.activation_rate_neg,
        causal_line=causal_line,
        n_max=len(dossier.max_activating),
        max_examples=format_examples(dossier.max_activating, "max"),
        n_min=len(dossier.min_activating),
        min_examples=format_examples(dossier.min_activating, "min"),
    )

    for attempt in range(3):
        try:
            result_text = ""
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(max_turns=1, model="sonnet"),
            ):
                if isinstance(message, ResultMessage):
                    result_text = message.result.strip()

            judgment = parse_judgment(result_text)
            if judgment:
                judgment.feature_id = dossier.feature_id
                return judgment
            log.warning("Could not parse judgment for feature %d", dossier.feature_id)
            return None
        except Exception as e:
            if attempt < 2:
                log.warning(
                    "Judge failed for feature %d (attempt %d): %s",
                    dossier.feature_id,
                    attempt + 1,
                    e,
                )
                await anyio.sleep(2**attempt)
            else:
                log.error(
                    "Judge failed for feature %d after 3 attempts: %s",
                    dossier.feature_id,
                    e,
                )
                return None


async def run_judging(
    dossiers: list[FeatureDossier], concurrency: int
) -> list[JudgmentEntry]:
    """Judge all features. Resume-safe."""
    judgments_path = M6_DIR / "judgments.json"

    if judgments_path.exists():
        with open(judgments_path, encoding="utf-8") as f:
            existing = [JudgmentEntry(**d) for d in json.load(f)]
        done_ids = {j.feature_id for j in existing}
    else:
        existing = []
        done_ids = set()

    remaining = [d for d in dossiers if d.feature_id not in done_ids]

    log.info("=" * 60)
    log.info("PHASE 2: Automated Judging (Sonnet 4.6)")
    log.info(
        "Total: %d  Done: %d  Remaining: %d",
        len(dossiers),
        len(done_ids),
        len(remaining),
    )
    log.info("=" * 60)

    if not remaining:
        log.info("All features already judged")
        return existing

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    times: list[float] = []

    async def process_one(dossier: FeatureDossier) -> None:
        nonlocal completed
        async with semaphore:
            t0 = time.time()
            judgment = await judge_one(dossier)

            if judgment:
                existing.append(judgment)
                with open(judgments_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [asdict(j) for j in existing], f, indent=2, ensure_ascii=False
                    )

            elapsed = time.time() - t0
            completed += 1
            times.append(elapsed)
            avg = sum(times) / len(times)
            eta = avg * (len(remaining) - completed) / max(concurrency, 1)

            status = (
                "relevant"
                if (judgment and judgment.is_refusal_relevant)
                else "not_relevant"
            )
            log.info(
                "[%d/%d] feature %7d | %s | %s | %.1fs | ETA %s",
                len(done_ids) + completed,
                len(dossiers),
                dossier.feature_id,
                status,
                judgment.description[:60] if judgment else "FAILED",
                elapsed,
                fmt_time(eta),
            )

    async with anyio.create_task_group() as tg:
        for dossier in remaining:
            tg.start_soon(process_one, dossier)

    return existing


# ---------------------------------------------------------------------------
# Phase 3: Detection Score
# Evidence: Gemma Scope 2 Sec 4.3 — "present this explanation along with a
# randomly ordered list of sequences (some which cause the feature to fire,
# some which don't) and ask the model to classify which ones fire"
# ---------------------------------------------------------------------------
DETECTION_PROMPT = """A feature in a neural network has been described as:
"{description}"

Below are {n} prompts. Some of them would activate this feature, some would not.
For each prompt, predict YES (would activate) or NO (would not activate).

{examples}

Reply with ONLY a list of predictions, one per line:
1. YES or NO
2. YES or NO
...
"""


async def detect_one(
    dossier: FeatureDossier,
    judgment: JudgmentEntry,
    pos_encoded: torch.Tensor,
    neg_encoded: torch.Tensor,
    retained_ids: list[int],
    manifest_lookup: dict[tuple[int, str], dict],
) -> DetectionResult | None:
    """Test if the description can predict which prompts activate the feature."""
    fid = dossier.feature_id
    if fid >= pos_encoded.shape[1]:
        return None

    # Get activations for this feature across all prompts
    pos_acts = pos_encoded[:, fid]
    neg_acts = neg_encoded[:, fid]
    all_acts = torch.cat([pos_acts, neg_acts])
    all_conditions = ["positive"] * len(pos_acts) + ["negative"] * len(neg_acts)
    all_pair_ids = retained_ids + retained_ids

    # Select 5 that activate (>0) and 5 that don't (==0)
    active_idx = (all_acts > 0).nonzero(as_tuple=True)[0]
    inactive_idx = (all_acts == 0).nonzero(as_tuple=True)[0]

    if len(active_idx) < 3 or len(inactive_idx) < 3:
        return None  # not enough examples for a meaningful test

    # Sample 5 of each (or fewer if not available)
    import random

    rng = random.Random(fid)  # deterministic per feature
    n_each = min(5, len(active_idx), len(inactive_idx))
    active_sample = rng.sample(active_idx.tolist(), n_each)
    inactive_sample = rng.sample(inactive_idx.tolist(), n_each)

    # Shuffle together
    test_items = [(idx, True) for idx in active_sample] + [
        (idx, False) for idx in inactive_sample
    ]
    rng.shuffle(test_items)

    # Format examples
    example_lines = []
    ground_truth = []
    for i, (idx, is_active) in enumerate(test_items, 1):
        pid = all_pair_ids[idx]
        cond = all_conditions[idx]
        m_entry = manifest_lookup.get((pid, cond), {})
        prompt_text = m_entry.get("prompt_text", "")[:200]
        example_lines.append(f'{i}. "{prompt_text}"')
        ground_truth.append(is_active)

    prompt = DETECTION_PROMPT.format(
        description=judgment.description,
        n=len(test_items),
        examples="\n".join(example_lines),
    )

    for attempt in range(3):
        try:
            result_text = ""
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(max_turns=1, model="sonnet"),
            ):
                if isinstance(message, ResultMessage):
                    result_text = message.result.strip()

            # Parse YES/NO predictions
            predictions = []
            for line in result_text.split("\n"):
                line = line.strip().upper()
                if "YES" in line:
                    predictions.append(True)
                elif "NO" in line:
                    predictions.append(False)

            if len(predictions) != len(ground_truth):
                log.warning(
                    "Detection for feature %d: got %d predictions, expected %d",
                    fid,
                    len(predictions),
                    len(ground_truth),
                )
                if len(predictions) < len(ground_truth):
                    return None

            correct = sum(
                p == g for p, g in zip(predictions[: len(ground_truth)], ground_truth)
            )
            accuracy = correct / len(ground_truth)

            return DetectionResult(
                feature_id=fid,
                accuracy=round(accuracy, 3),
                n_tested=len(ground_truth),
            )
        except Exception as e:
            if attempt < 2:
                await anyio.sleep(2**attempt)
            else:
                log.error("Detection failed for feature %d: %s", fid, e)
                return None


async def run_detection(
    dossiers: list[FeatureDossier],
    judgments: list[JudgmentEntry],
    concurrency: int,
) -> list[DetectionResult]:
    """Run detection score on all judged features."""
    detection_path = M6_DIR / "detection_scores.json"

    if detection_path.exists():
        with open(detection_path, encoding="utf-8") as f:
            existing = [DetectionResult(**d) for d in json.load(f)]
        done_ids = {d.feature_id for d in existing}
    else:
        existing = []
        done_ids = set()

    judgment_lookup = {j.feature_id: j for j in judgments}
    dossier_lookup = {d.feature_id: d for d in dossiers}
    remaining_fids = [j.feature_id for j in judgments if j.feature_id not in done_ids]

    log.info("=" * 60)
    log.info("PHASE 3: Detection Score (Gemma Scope 2 Sec 4.3)")
    log.info(
        "Total: %d  Done: %d  Remaining: %d",
        len(judgments),
        len(done_ids),
        len(remaining_fids),
    )
    log.info("=" * 60)

    if not remaining_fids:
        log.info("All detection scores computed")
        return existing

    # Load data
    pos_encoded = torch.load(
        ENCODED_DIR / "mean_response_token_positive.pt", weights_only=True
    )
    neg_encoded = torch.load(
        ENCODED_DIR / "mean_response_token_negative.pt", weights_only=True
    )
    retained_ids = json.load(open(CFG.data_dir / "retained_pairs.json"))
    with open(CFG.manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_lookup = {
        (e["pair_id"], e["condition"]): e
        for e in manifest
        if e["pair_id"] in set(retained_ids)
    }

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    times: list[float] = []

    async def process_one(fid: int) -> None:
        nonlocal completed
        async with semaphore:
            t0 = time.time()
            result = await detect_one(
                dossier_lookup[fid],
                judgment_lookup[fid],
                pos_encoded,
                neg_encoded,
                retained_ids,
                manifest_lookup,
            )

            if result:
                existing.append(result)
                with open(detection_path, "w", encoding="utf-8") as f:
                    json.dump([asdict(d) for d in existing], f, indent=2)

            elapsed = time.time() - t0
            completed += 1
            times.append(elapsed)
            avg = sum(times) / len(times)
            eta = avg * (len(remaining_fids) - completed) / max(concurrency, 1)

            log.info(
                "[%d/%d] feature %7d | acc=%.2f | %.1fs | ETA %s",
                len(done_ids) + completed,
                len(judgments),
                fid,
                result.accuracy if result else -1,
                elapsed,
                fmt_time(eta),
            )

    async with anyio.create_task_group() as tg:
        for fid in remaining_fids:
            tg.start_soon(process_one, fid)

    return existing


# ---------------------------------------------------------------------------
# Phase 4: Validation Gate
# ---------------------------------------------------------------------------
def run_validation(
    dossiers: list[FeatureDossier],
    judgments: list[JudgmentEntry],
    detection_results: list[DetectionResult],
) -> None:
    """Check milestone 6 validation gate."""
    log.info("=" * 60)
    log.info("VALIDATION: Milestone 6 Gate")
    log.info("=" * 60)

    # All features judged
    log.info("Features judged: %d/%d", len(judgments), len(dossiers))

    # Relevant count
    relevant = [j for j in judgments if j.is_refusal_relevant]
    not_relevant = [j for j in judgments if not j.is_refusal_relevant]
    log.info("Refusal-relevant: %d  Not relevant: %d", len(relevant), len(not_relevant))

    if len(relevant) < 5:
        log.error("Fewer than 5 refusal-relevant features — strong negative finding")
    elif len(relevant) < 10:
        log.warning("Fewer than 10 refusal-relevant features")

    # Sub-type distribution
    from collections import Counter

    all_subtypes: list[str] = []
    for j in relevant:
        all_subtypes.extend(j.sub_types)
    subtype_counts = Counter(all_subtypes)
    log.info("Sub-type distribution among relevant features:")
    for st, count in subtype_counts.most_common():
        log.info("  %-25s %d", st, count)
    n_distinct = len(subtype_counts)
    log.info("Distinct sub-types: %d (expect >= 2 for decomposition)", n_distinct)

    # Detection scores
    if detection_results:
        accs = [d.accuracy for d in detection_results]
        mean_acc = sum(accs) / len(accs)
        log.info(
            "Detection scores: mean=%.2f, min=%.2f, max=%.2f (n=%d)",
            mean_acc,
            min(accs),
            max(accs),
            len(accs),
        )
        good_descriptions = sum(1 for a in accs if a >= 0.7)
        log.info("Features with detection >= 0.7: %d/%d", good_descriptions, len(accs))

    # Confidence distribution
    conf_counts = Counter(j.confidence for j in judgments)
    log.info("Confidence: %s", dict(conf_counts))

    # Sample relevant descriptions for qualitative review
    log.info("Sample relevant feature descriptions:")
    for j in relevant[:10]:
        log.info(
            "  feature %7d: %s [%s] (%s)",
            j.feature_id,
            j.description[:80],
            ",".join(j.sub_types),
            j.confidence,
        )

    log.info("Sample NOT-relevant feature descriptions:")
    for j in not_relevant[:5]:
        log.info(
            "  feature %7d: %s (%s)", j.feature_id, j.description[:80], j.confidence
        )

    # Cross-reference with domain profiles (now that judging is done independently)
    log.info("Cross-referencing judgments with domain profiles:")
    dossier_lookup = {d.feature_id: d for d in dossiers}
    agreement = 0
    total_checked = 0
    for j in relevant:
        d = dossier_lookup.get(j.feature_id)
        if not d or not d.domain_profile:
            continue
        # Which domains have |d| > 0.3 in the profile?
        profile_domains = [
            dom for dom, val in d.domain_profile.items() if abs(val) > 0.3
        ]
        if not profile_domains and "general_refusal" in j.sub_types:
            agreement += 1
        elif set(profile_domains) & set(j.sub_types):
            agreement += 1
        total_checked += 1
    if total_checked:
        log.info(
            "  Judge vs domain-profile agreement: %d/%d (%.0f%%)",
            agreement,
            total_checked,
            100 * agreement / total_checked,
        )

    # Save summary
    summary = {
        "total_features": len(dossiers),
        "total_judged": len(judgments),
        "relevant": len(relevant),
        "not_relevant": len(not_relevant),
        "distinct_subtypes": n_distinct,
        "subtype_counts": dict(subtype_counts),
        "mean_detection_accuracy": round(mean_acc, 3) if detection_results else None,
    }
    with open(M6_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", M6_DIR / "summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["dossier", "judge", "detect", "validate", "all"],
        default="all",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("MILESTONE 6: FEATURE JUDGING")
    log.info("=" * 60)

    dossiers = build_dossiers()

    judgments: list[JudgmentEntry] = []
    detection_results: list[DetectionResult] = []

    if args.phase in ("judge", "all"):
        judgments = anyio.run(run_judging, dossiers, args.concurrency)

    if args.phase in ("detect", "all"):
        if not judgments:
            jpath = M6_DIR / "judgments.json"
            if jpath.exists():
                with open(jpath, encoding="utf-8") as f:
                    judgments = [JudgmentEntry(**d) for d in json.load(f)]
        detection_results = anyio.run(
            run_detection, dossiers, judgments, args.concurrency
        )

    if args.phase in ("validate", "all"):
        if not judgments:
            jpath = M6_DIR / "judgments.json"
            if jpath.exists():
                with open(jpath, encoding="utf-8") as f:
                    judgments = [JudgmentEntry(**d) for d in json.load(f)]
        if not detection_results:
            dpath = M6_DIR / "detection_scores.json"
            if dpath.exists():
                with open(dpath, encoding="utf-8") as f:
                    detection_results = [DetectionResult(**d) for d in json.load(f)]
        run_validation(dossiers, judgments, detection_results)

    log.info("=" * 60)
    log.info("DONE in %s", fmt_time(time.time() - t0))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
