"""Milestones 2 & 3: Response Generation + Activation Extraction.

Single model load for both milestones:
  Phase 1 — Generate responses for all contrastive pairs.
  Phase 2 — Extract layer-41 activations at two sites.
  Phase 3 — Compute refusal directions (CPU).

Resume-safe: saves after every item. Re-run to pick up where it left off.

Usage:
  uv run python milestone_2_3_generate_extract.py
  uv run python milestone_2_3_generate_extract.py 2>&1 | tee logs/m2_3.log
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refusal_decomposition import (
    CFG,
    HF_TOKEN,
    ProgressTracker,
    clear_vram,
    fmt_time,
    get_model_layers,
    setup_logging,
    vram_report,
)

log = setup_logging("milestone_2_3")

# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
DO_SAMPLE = True
SEED = 42

MANIFEST_FILE = CFG.manifest_file
SITES = ("last_prompt_token", "mean_response_token")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class Condition(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class ContrastivePair:
    pair_id: int
    positive: str
    negative: str
    domain: str
    sub_topic: str
    subtle: bool

    @classmethod
    def from_dict(cls, d: dict) -> ContrastivePair:
        return cls(
            pair_id=d["pair_id"],
            positive=d["positive"],
            negative=d["negative"],
            domain=d["domain"],
            sub_topic=d.get("sub_topic", ""),
            subtle=d.get("subtle", False),
        )

    def prompt(self, condition: Condition) -> str:
        return self.positive if condition is Condition.POSITIVE else self.negative


@dataclass
class ResponseEntry:
    pair_id: int
    domain: str
    sub_topic: str
    condition: str
    prompt_text: str
    response_text: str
    prompt_token_len: int
    response_token_len: int

    @classmethod
    def from_dict(cls, d: dict) -> ResponseEntry:
        return cls(**d)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------
def load_manifest() -> list[ResponseEntry]:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            return [ResponseEntry.from_dict(d) for d in json.load(f)]
    return []


def save_manifest(entries: list[ResponseEntry]) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, indent=2, ensure_ascii=False)


def activation_path(site: str, condition: Condition, pair_id: int) -> Path:
    return CFG.activations_dir / site / condition.value / f"pair_{pair_id}.pt"


# ---------------------------------------------------------------------------
# Phase 1: Response Generation
# ---------------------------------------------------------------------------
def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[ContrastivePair],
) -> list[ResponseEntry]:
    """Generate responses for all prompts. Resume-safe."""
    manifest = load_manifest()
    done_keys: set[tuple[int, str]] = {(e.pair_id, e.condition) for e in manifest}
    total = len(pairs) * 2
    remaining = total - len(done_keys)

    log.info("=" * 60)
    log.info("PHASE 1: Response Generation")
    log.info("total=%d  done=%d  remaining=%d", total, len(done_keys), remaining)
    log.info("=" * 60)

    if remaining == 0:
        log.info("All responses already generated, skipping")
        return manifest

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    tracker = ProgressTracker(total, already_done=len(done_keys))

    for pair in pairs:
        for condition in Condition:
            if (pair.pair_id, condition.value) in done_keys:
                continue

            t0 = time.time()
            prompt_text = pair.prompt(condition)

            messages = [{"role": "user", "content": prompt_text}]
            tokenized = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )
            input_ids = tokenized["input_ids"].to(CFG.device)
            attention_mask = tokenized["attention_mask"].to(CFG.device)
            prompt_len: int = input_ids.shape[1]

            with torch.no_grad():
                gen_output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                )

            response_tokens = gen_output[0][prompt_len:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            entry = ResponseEntry(
                pair_id=pair.pair_id,
                domain=pair.domain,
                sub_topic=pair.sub_topic,
                condition=condition.value,
                prompt_text=prompt_text,
                response_text=response_text,
                prompt_token_len=prompt_len,
                response_token_len=len(response_tokens),
            )
            manifest.append(entry)
            save_manifest(manifest)

            tracker.tick(time.time() - t0)
            tracker.log_item(
                log,
                f"pair {pair.pair_id:3d} {condition.value:8s} | {len(response_tokens):3d} tok",
            )

    log.info(
        "Phase 1 complete: %d items in %s",
        tracker.done_this_run,
        fmt_time(tracker.phase_elapsed),
    )
    return manifest


# ---------------------------------------------------------------------------
# Phase 2: Activation Extraction
# ---------------------------------------------------------------------------
def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    manifest: list[ResponseEntry],
) -> None:
    """Extract layer-41 activations. Resume-safe: skips existing .pt files."""
    # Create output directories
    for site in SITES:
        for condition in Condition:
            activation_path(site, condition, 0).parent.mkdir(
                parents=True, exist_ok=True
            )

    # Determine remaining work
    to_extract: list[ResponseEntry] = []
    for entry in manifest:
        cond = Condition(entry.condition)
        lpt = activation_path("last_prompt_token", cond, entry.pair_id)
        mrt = activation_path("mean_response_token", cond, entry.pair_id)
        if not (lpt.exists() and mrt.exists()):
            to_extract.append(entry)

    total = len(manifest)
    already_done = total - len(to_extract)

    log.info("=" * 60)
    log.info("PHASE 2: Activation Extraction")
    log.info("total=%d  done=%d  remaining=%d", total, already_done, len(to_extract))
    log.info("=" * 60)

    if not to_extract:
        log.info("All activations already extracted, skipping")
        return

    # Hook setup
    cache: dict[str, torch.Tensor] = {}

    def hook_fn(
        module: torch.nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        cache["activations"] = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    layers = get_model_layers(model)
    handle = layers[CFG.layer].register_forward_hook(hook_fn)

    tracker = ProgressTracker(total, already_done=already_done)

    for entry in to_extract:
        t0 = time.time()
        cond = Condition(entry.condition)

        # Full sequence: prompt + response
        full_tokenized = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": entry.prompt_text},
                {"role": "assistant", "content": entry.response_text},
            ],
            return_tensors="pt",
            return_dict=True,
        )
        full_ids = full_tokenized["input_ids"].to(CFG.device)
        full_mask = full_tokenized["attention_mask"].to(CFG.device)

        # Prompt boundary
        prompt_tokenized = tokenizer.apply_chat_template(
            [{"role": "user", "content": entry.prompt_text}],
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        prompt_len: int = prompt_tokenized["input_ids"].shape[1]

        # Forward pass
        cache.clear()
        with torch.no_grad():
            _ = model(full_ids, attention_mask=full_mask)

        acts = cache["activations"].squeeze(0)  # (seq_len, d_model)

        # Last-prompt-token: (d_model,)
        last_prompt_act = acts[prompt_len - 1].cpu().to(torch.float32)
        assert last_prompt_act.shape == (CFG.d_model,)

        # Mean-response-token: (d_model,)
        response_acts = acts[prompt_len:]  # (response_len, d_model)
        if response_acts.shape[0] == 0:
            log.warning(
                "pair %d %s has empty response, copying last_prompt_token",
                entry.pair_id,
                cond.value,
            )
            mean_response_act = last_prompt_act.clone()
        else:
            mean_response_act = response_acts.mean(dim=0).cpu().to(torch.float32)
        assert mean_response_act.shape == (CFG.d_model,)

        # Save
        lpt_path = activation_path("last_prompt_token", cond, entry.pair_id)
        mrt_path = activation_path("mean_response_token", cond, entry.pair_id)
        torch.save(last_prompt_act, lpt_path)
        torch.save(mean_response_act, mrt_path)

        tracker.tick(time.time() - t0)
        tracker.log_item(
            log,
            f"pair {entry.pair_id:3d} {cond.value:8s} | resp_len={response_acts.shape[0]:3d}",
        )

    handle.remove()
    log.info(
        "Phase 2 complete: %d items in %s",
        tracker.done_this_run,
        fmt_time(tracker.phase_elapsed),
    )


# ---------------------------------------------------------------------------
# Phase 3: Refusal Directions (CPU)
# ---------------------------------------------------------------------------
def compute_refusal_directions() -> None:
    """Compute Arditi-style refusal direction per site: mean(pos) - mean(neg), unit-normed."""
    log.info("=" * 60)
    log.info("PHASE 3: Refusal Directions (CPU)")
    log.info("=" * 60)

    for site in SITES:
        pos_dir = CFG.activations_dir / site / Condition.POSITIVE.value
        neg_dir = CFG.activations_dir / site / Condition.NEGATIVE.value

        pos_files = sorted(pos_dir.glob("pair_*.pt"))
        neg_files = sorted(neg_dir.glob("pair_*.pt"))
        log.info("%s: %d positive, %d negative", site, len(pos_files), len(neg_files))

        pos_acts = torch.stack([torch.load(f, weights_only=True) for f in pos_files])
        neg_acts = torch.stack([torch.load(f, weights_only=True) for f in neg_files])

        mean_pos = pos_acts.mean(dim=0)  # (d_model,)
        mean_neg = neg_acts.mean(dim=0)  # (d_model,)

        refusal_dir = mean_pos - mean_neg
        norm = refusal_dir.norm().item()
        refusal_dir_unit = refusal_dir / refusal_dir.norm()

        out_path = CFG.data_dir / f"refusal_direction_{site}.pt"
        torch.save(refusal_dir_unit, out_path)

        cos_sim = torch.nn.functional.cosine_similarity(
            mean_pos.unsqueeze(0), mean_neg.unsqueeze(0)
        ).item()

        log.info(
            "  norm=%.4f  cos_sim(pos,neg)=%.4f  saved=%s", norm, cos_sim, out_path
        )

    # Cross-site comparison
    rd_lpt = torch.load(
        CFG.data_dir / "refusal_direction_last_prompt_token.pt", weights_only=True
    )
    rd_mrt = torch.load(
        CFG.data_dir / "refusal_direction_mean_response_token.pt", weights_only=True
    )
    cross_cos = torch.nn.functional.cosine_similarity(
        rd_lpt.unsqueeze(0), rd_mrt.unsqueeze(0)
    ).item()
    log.info("Cross-site cosine similarity: %.4f", cross_cos)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_activations() -> None:
    """Check all activation files: shape, NaN, Inf, disk usage."""
    log.info("=" * 60)
    log.info("VALIDATION: Activation files")
    log.info("=" * 60)

    total_files = 0
    for site in SITES:
        for condition in Condition:
            d = CFG.activations_dir / site / condition.value
            files = sorted(d.glob("pair_*.pt"))
            total_files += len(files)
            for f in files:
                t = torch.load(f, weights_only=True)
                assert t.shape == (CFG.d_model,), f"{f}: shape {t.shape}"
                assert not torch.isnan(t).any(), f"{f}: NaN"
                assert not torch.isinf(t).any(), f"{f}: Inf"
            log.info("  %s/%s: %d files OK", site, condition.value, len(files))

    total_bytes = sum(f.stat().st_size for f in CFG.activations_dir.rglob("*.pt"))
    log.info(
        "PASS: %d files, all shape (%d,), no NaN/Inf, %.1f MB on disk",
        total_files,
        CFG.d_model,
        total_bytes / (1024 * 1024),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log.info("=" * 60)
    log.info("MILESTONES 2 & 3: GENERATE + EXTRACT")
    log.info("=" * 60)

    # Load pairs
    with open(CFG.pairs_file, encoding="utf-8") as f:
        pairs = [ContrastivePair.from_dict(d) for d in json.load(f)]
    log.info("Loaded %d contrastive pairs (%d prompts)", len(pairs), len(pairs) * 2)

    # Load model
    log.info("Loading %s ...", CFG.model_id)
    model_t0 = time.time()
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
    log.info("Model loaded in %s", fmt_time(time.time() - model_t0))

    # Phase 1
    manifest = generate_responses(model, tokenizer, pairs)

    # Phase 2
    extract_activations(model, tokenizer, manifest)

    # Unload
    log.info("Unloading model...")
    del model
    del tokenizer
    clear_vram()
    vram_report("after model unload", log)

    # Phase 3
    compute_refusal_directions()

    # Validate
    validate_activations()

    log.info("=" * 60)
    log.info("ALL DONE in %s", fmt_time(time.time() - t0))
    log.info("Next: score responses for refusal (milestone 2 gate)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
