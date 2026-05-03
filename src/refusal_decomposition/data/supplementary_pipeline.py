"""Pipeline for supplementary pairs: merge → generate → score → extract → reanalyze.

Orchestrates the full pipeline for supplementary contrastive pairs:
  1. Merge supplementary pairs into the main dataset
  2. Generate responses (GPU) — reuses M2-3 pipeline with resume
  3. Score for refusal (Agent SDK) — reuses M2 scoring with resume
  4. Extract activations (GPU) — reuses M2-3 pipeline with resume
  5. Recompute refusal directions with expanded retained set
  6. Rerun refusal axis analysis

All steps are resume-safe. Re-running skips completed items.

Usage:
  uv run python supplementary_pipeline.py --phase merge
  uv run python supplementary_pipeline.py --phase generate   # GPU
  uv run python supplementary_pipeline.py --phase score
  uv run python supplementary_pipeline.py --phase extract    # GPU
  uv run python supplementary_pipeline.py --phase recompute
  uv run python supplementary_pipeline.py --phase all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from refusal_decomposition import CFG, clear_vram, fmt_time, setup_logging, vram_report

log = setup_logging("supplementary_pipeline")

DATA_DIR = CFG.data_dir
SUPP_FILE = DATA_DIR / "supplementary_pairs.json"
MAIN_PAIRS_FILE = DATA_DIR / "contrastive_pairs.json"
MANIFEST_FILE = CFG.manifest_file


def phase_merge() -> None:
    """Merge supplementary pairs into the main dataset."""
    log.info("=" * 60)
    log.info("PHASE: Merge supplementary pairs")
    log.info("=" * 60)

    with open(MAIN_PAIRS_FILE, encoding="utf-8") as f:
        main_pairs = json.load(f)
    with open(SUPP_FILE, encoding="utf-8") as f:
        supp_pairs = json.load(f)

    existing_ids = {p["pair_id"] for p in main_pairs}
    new_pairs = [p for p in supp_pairs if p["pair_id"] not in existing_ids]

    if not new_pairs:
        log.info("All supplementary pairs already in main dataset")
        return

    merged = main_pairs + new_pairs
    with open(MAIN_PAIRS_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    log.info("Merged %d new pairs (total now %d)", len(new_pairs), len(merged))
    from collections import Counter

    domains = Counter(p["domain"] for p in merged)
    log.info("Domain distribution: %s", dict(domains))


def phase_generate() -> None:
    """Generate responses for new pairs. Reuses M2-3 pipeline with resume."""
    log.info("=" * 60)
    log.info("PHASE: Generate responses (GPU)")
    log.info("=" * 60)

    # Import and run the generation function from M2-3
    # The M2-3 script has resume support — it skips existing manifest entries
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "m23", "scripts/milestones/milestone_2_3_generate_extract.py"
    )
    m23 = importlib.util.module_from_spec(spec)

    # We need to run just the generation part
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_decomposition import HF_TOKEN

    with open(MAIN_PAIRS_FILE, encoding="utf-8") as f:
        all_pairs = json.load(f)

    # Check what's already in the manifest
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            manifest = json.load(f)
        done_keys = {(e["pair_id"], e["condition"]) for e in manifest}
    else:
        manifest = []
        done_keys = set()

    remaining = []
    for p in all_pairs:
        for cond in ("positive", "negative"):
            if (p["pair_id"], cond) not in done_keys:
                remaining.append((p, cond))

    if not remaining:
        log.info("All responses already generated")
        return

    log.info("%d new responses to generate", len(remaining))

    # Load model
    log.info("Loading model...")
    vram_report("before model load", log)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        token=HF_TOKEN,
    )
    model.eval()
    vram_report("after model load", log)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    from refusal_decomposition import ProgressTracker, get_model_layers

    tracker = ProgressTracker(len(remaining))

    for pair, condition in remaining:
        t0 = time.time()
        prompt_text = pair["positive"] if condition == "positive" else pair["negative"]

        messages = [{"role": "user", "content": prompt_text}]
        tokenized = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        )
        input_ids = tokenized["input_ids"].to(CFG.device)
        attention_mask = tokenized["attention_mask"].to(CFG.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            gen_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
            )

        response_tokens = gen_output[0][prompt_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        entry = {
            "pair_id": pair["pair_id"],
            "domain": pair["domain"],
            "sub_topic": pair.get("sub_topic", ""),
            "condition": condition,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "prompt_token_len": prompt_len,
            "response_token_len": len(response_tokens),
        }
        manifest.append(entry)

        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        tracker.tick(time.time() - t0)
        tracker.log_item(
            log,
            f"pair {pair['pair_id']:3d} {condition:8s} | {len(response_tokens):3d} tok",
        )

    # Also extract activations while model is loaded
    log.info("Extracting activations for new responses...")
    layers = get_model_layers(model)
    cache: dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        cache["activations"] = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    handle = layers[CFG.layer].register_forward_hook(hook_fn)

    for site in ("last_prompt_token", "mean_response_token"):
        for cond in ("positive", "negative"):
            (CFG.activations_dir / site / cond).mkdir(parents=True, exist_ok=True)

    new_entries = [
        e for e in manifest if (e["pair_id"], e["condition"]) not in done_keys
    ]
    tracker2 = ProgressTracker(len(new_entries))

    for entry in new_entries:
        t0 = time.time()
        lpt_path = (
            CFG.activations_dir
            / "last_prompt_token"
            / entry["condition"]
            / f"pair_{entry['pair_id']}.pt"
        )
        mrt_path = (
            CFG.activations_dir
            / "mean_response_token"
            / entry["condition"]
            / f"pair_{entry['pair_id']}.pt"
        )

        if lpt_path.exists() and mrt_path.exists():
            tracker2.tick(0.0)
            continue

        full_tokenized = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": entry["prompt_text"]},
                {"role": "assistant", "content": entry["response_text"]},
            ],
            return_tensors="pt",
            return_dict=True,
        )
        full_ids = full_tokenized["input_ids"].to(CFG.device)
        full_mask = full_tokenized["attention_mask"].to(CFG.device)

        prompt_tokenized = tokenizer.apply_chat_template(
            [{"role": "user", "content": entry["prompt_text"]}],
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        prompt_len = prompt_tokenized["input_ids"].shape[1]

        cache.clear()
        with torch.no_grad():
            _ = model(full_ids, attention_mask=full_mask)

        acts = cache["activations"].squeeze(0)
        last_prompt_act = acts[prompt_len - 1].cpu().to(torch.float32)
        response_acts = acts[prompt_len:]
        mean_response_act = (
            response_acts.mean(dim=0).cpu().to(torch.float32)
            if response_acts.shape[0] > 0
            else last_prompt_act.clone()
        )

        torch.save(last_prompt_act, lpt_path)
        torch.save(mean_response_act, mrt_path)

        tracker2.tick(time.time() - t0)
        tracker2.log_item(
            log, f"extract pair {entry['pair_id']:3d} {entry['condition']:8s}"
        )

    handle.remove()
    del model, tokenizer
    clear_vram()
    vram_report("after model unload", log)


def phase_score() -> None:
    """Score new responses. Reuses M2 scoring pipeline."""
    log.info("=" * 60)
    log.info("PHASE: Score responses")
    log.info("=" * 60)
    log.info("Run: uv run python scripts/milestones/milestone_2_score.py")
    log.info("It will detect new unscored responses and score them.")


def phase_recompute() -> None:
    """Recompute refusal directions and rerun axis analysis."""
    log.info("=" * 60)
    log.info("PHASE: Recompute with expanded data")
    log.info("=" * 60)

    # Reload retained pairs (need to re-filter with new scores)
    with open(DATA_DIR / "refusal_scores.json", encoding="utf-8") as f:
        scores = json.load(f)

    pos_scores = {
        s["pair_id"]: s["refusal_score"] for s in scores if s["condition"] == "positive"
    }
    neg_scores = {
        s["pair_id"]: s["refusal_score"] for s in scores if s["condition"] == "negative"
    }

    retained = sorted(
        pid
        for pid in pos_scores
        if pos_scores.get(pid, 0) > 50 and neg_scores.get(pid, 100) < 30
    )

    with open(DATA_DIR / "retained_pairs.json", "w") as f:
        json.dump(retained, f, indent=2)
    log.info("Retained pairs: %d (was 104)", len(retained))

    from collections import Counter

    with open(MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)
    pair_domain = {
        e["pair_id"]: e["domain"] for e in manifest if e["condition"] == "positive"
    }
    domain_counts = Counter(pair_domain[pid] for pid in retained if pid in pair_domain)
    log.info("Retained by domain: %s", dict(domain_counts))

    # Recompute refusal directions
    for site in ("last_prompt_token", "mean_response_token"):
        pos_acts = torch.stack(
            [
                torch.load(
                    CFG.activations_dir / site / "positive" / f"pair_{pid}.pt",
                    weights_only=True,
                )
                for pid in retained
            ]
        )
        neg_acts = torch.stack(
            [
                torch.load(
                    CFG.activations_dir / site / "negative" / f"pair_{pid}.pt",
                    weights_only=True,
                )
                for pid in retained
            ]
        )
        refusal_dir = pos_acts.mean(0) - neg_acts.mean(0)
        refusal_unit = refusal_dir / refusal_dir.norm()
        torch.save(refusal_unit, DATA_DIR / f"refusal_direction_{site}.pt")
        log.info("%s: direction norm=%.1f, saved", site, refusal_dir.norm().item())

    # Rerun refusal axis analysis
    log.info("Rerunning refusal axis analysis...")
    import subprocess

    result = subprocess.run(
        ["uv", "run", "python", "refusal_axis_analysis.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode == 0:
        log.info("Refusal axis analysis complete")
    else:
        log.error("Refusal axis analysis failed:\n%s", result.stderr[-500:])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["merge", "generate", "score", "extract", "recompute", "all"],
        default="all",
    )
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("SUPPLEMENTARY DATA PIPELINE")
    log.info("=" * 60)

    if args.phase in ("merge", "all"):
        phase_merge()
    if args.phase in ("generate", "all"):
        phase_generate()
    if args.phase in ("score", "all"):
        phase_score()
    if args.phase in ("recompute", "all"):
        phase_recompute()

    log.info("=" * 60)
    log.info("DONE in %s", fmt_time(time.time() - t0))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
