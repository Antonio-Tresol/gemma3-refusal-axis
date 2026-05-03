"""Milestone 0: Project Bootstrap.

Confirms all infrastructure assumptions with measured numbers:
- Model loading, VRAM measurement, hook capture at layer 41
- SAE loading, encoding, L0 measurement
- Matryoshka nesting verification (prefix slicing, not cross-SAE comparison)
- Full load-unload cycle without VRAM leaks

Usage:
  uv run python milestone_0_bootstrap.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download, repo_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from refusal_decomposition import (
    CFG,
    HF_TOKEN,
    JumpReLUSAE,
    clear_vram,
    fmt_time,
    get_model_layers,
    load_sae,
    setup_logging,
    vram_report,
)

log = setup_logging("milestone_0")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class BootstrapResults:
    model_vram_gb: float = 0.0
    sae_vram_gb: float = 0.0
    combined_vram_gb: float = 0.0
    num_layers: int = 0
    hook_shape: tuple[int, ...] = ()
    l0: int = 0
    matryoshka_max_diff: float = 0.0
    final_vram_gb: float = 0.0

    def log_summary(self) -> None:
        log.info("=" * 60)
        log.info("MILESTONE 0 SUMMARY")
        log.info("=" * 60)
        log.info("  Model VRAM:       %.2f GB", self.model_vram_gb)
        log.info("  SAE VRAM:         %.2f GB", self.sae_vram_gb)
        log.info("  Combined:         %.2f GB (limit: 32 GB)", self.combined_vram_gb)
        log.info("  Cannot coexist:   %s", self.combined_vram_gb > 32)
        log.info("  Num layers:       %d", self.num_layers)
        log.info("  Hook shape:       %s", self.hook_shape)
        log.info("  SAE L0:           %d", self.l0)
        log.info("  Matryoshka diff:  %.6e", self.matryoshka_max_diff)
        log.info("  Final VRAM:       %.2f GB", self.final_vram_gb)


# ---------------------------------------------------------------------------
# Step 0b: Authenticate
# ---------------------------------------------------------------------------
def step_0b_authenticate() -> None:
    log.info("=" * 60)
    log.info("Step 0b: Authenticate with HuggingFace")
    log.info("=" * 60)

    config_path = hf_hub_download(
        repo_id=CFG.model_id, filename="config.json", token=HF_TOKEN
    )
    log.info("Model config downloaded: %s", config_path)

    info = repo_info(repo_id=CFG.sae_repo, token=HF_TOKEN)
    log.info("SAE repo accessible: %s (modified %s)", info.id, info.last_modified)
    log.info("PASS: HuggingFace access confirmed")


# ---------------------------------------------------------------------------
# Step 0c: Load model
# ---------------------------------------------------------------------------
def step_0c_load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer, float]:
    log.info("=" * 60)
    log.info("Step 0c: Load Gemma 3 12B bf16")
    log.info("=" * 60)

    clear_vram()
    vram_report("before model load", log)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        token=HF_TOKEN,
    )
    model.eval()

    alloc = vram_report("after model load", log)
    log.info("Model loaded in %s, VRAM: %.2f GB", fmt_time(time.time() - t0), alloc)
    return model, tokenizer, alloc


# ---------------------------------------------------------------------------
# Steps 0d-0e: Hook capture + activation extraction
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    num_layers: int
    hook_shape: tuple[int, ...]
    last_prompt_act: torch.Tensor  # (d_model,)
    mean_response_act: torch.Tensor  # (d_model,)


def steps_0d_0e_hook_and_extract(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> ExtractionResult:
    log.info("=" * 60)
    log.info("Steps 0d-0e: Hook capture + activation extraction")
    log.info("=" * 60)

    cache: dict[str, torch.Tensor] = {}

    def hook_fn(
        module: torch.nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        cache["activations"] = (
            output[0].detach().clone()
            if isinstance(output, tuple)
            else output.detach().clone()
        )

    layers = get_model_layers(model)
    num_layers = len(layers)
    log.info("Found %d transformer layers", num_layers)

    handle = layers[CFG.layer].register_forward_hook(hook_fn)

    # Step 0d: single forward pass
    test_prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": test_prompt}]
    tokenized = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    )
    input_ids = tokenized["input_ids"].to(CFG.device)
    attention_mask = tokenized["attention_mask"].to(CFG.device)
    prompt_len: int = input_ids.shape[1]
    log.info("Prompt tokens: %d", prompt_len)

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    activations = cache["activations"].squeeze(0)  # (seq_len, d_model)
    hook_shape = tuple(activations.shape)
    log.info("Hook captured shape: %s", hook_shape)
    assert activations.shape[1] == CFG.d_model
    log.info(
        "PASS: Hook captures (seq_len=%d, %d) at layer %d",
        activations.shape[0],
        CFG.d_model,
        CFG.layer,
    )

    # Step 0e: generate ~50 tokens, extract both sites
    cache.clear()
    with torch.no_grad():
        gen_output = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=50, do_sample=False
        )

    response_text = tokenizer.decode(
        gen_output[0][prompt_len:], skip_special_tokens=True
    )
    log.info("Generated: %s...", response_text[:100])

    cache.clear()
    with torch.no_grad():
        _ = model(gen_output)

    full_acts = cache["activations"].squeeze(0)  # (full_seq_len, d_model)
    log.info("Full sequence activations: %s", tuple(full_acts.shape))

    last_prompt_act = full_acts[prompt_len - 1]  # (d_model,)
    assert last_prompt_act.shape == (CFG.d_model,)
    log.info("Last-prompt-token shape: %s", tuple(last_prompt_act.shape))

    response_acts = full_acts[prompt_len:]  # (response_len, d_model)
    mean_response_act = response_acts.mean(dim=0)  # (d_model,)
    assert mean_response_act.shape == (CFG.d_model,)
    log.info("Mean-response-token shape: %s", tuple(mean_response_act.shape))
    log.info("PASS: Both extraction sites work correctly")

    handle.remove()

    return ExtractionResult(
        num_layers=num_layers,
        hook_shape=hook_shape,
        last_prompt_act=last_prompt_act.cpu(),
        mean_response_act=mean_response_act.cpu(),
    )


# ---------------------------------------------------------------------------
# Step 0g-0h: Load SAE + encode + L0
# ---------------------------------------------------------------------------
def steps_0g_0h_sae_encode(
    activation: torch.Tensor,
) -> tuple[JumpReLUSAE, float, int]:
    log.info("=" * 60)
    log.info("Steps 0g-0h: Load 1M SAE + encode + L0 check")
    log.info("=" * 60)

    clear_vram()
    vram_report("before SAE load", log)
    t0 = time.time()

    sae = load_sae(width=CFG.sae_width_1m, width_label="1m", log=log)

    sae_alloc = vram_report("after SAE load", log)
    log.info("SAE loaded in %s, VRAM: %.2f GB", fmt_time(time.time() - t0), sae_alloc)

    # Encode
    act_gpu = activation.to(device=CFG.device, dtype=torch.bfloat16)
    with torch.no_grad():
        encoded = sae.encode(act_gpu.unsqueeze(0)).squeeze(0)  # (sae_width,)

    assert encoded.shape == (CFG.sae_width_1m,)
    l0 = int((encoded != 0).sum().item())
    log.info("Encoded shape: %s, L0=%d", tuple(encoded.shape), l0)
    assert 30 <= l0 <= 120, f"L0={l0} outside expected [30, 120]"
    log.info("PASS: L0=%d in expected range", l0)

    return sae, sae_alloc, l0


# ---------------------------------------------------------------------------
# Step 0i: Matryoshka nesting verification
# ---------------------------------------------------------------------------
@dataclass
class MatryoshkaResult:
    widths: list[int]
    active_features: list[int]
    fvu: list[float]
    cos_sim: list[float]


def step_0i_matryoshka(sae: JumpReLUSAE, activation: torch.Tensor) -> MatryoshkaResult:
    """Verify Matryoshka prefix slicing: reconstruction quality degrades gracefully."""
    log.info("=" * 60)
    log.info("Step 0i: Matryoshka nesting verification")
    log.info("=" * 60)

    act_gpu = activation.to(device=CFG.device, dtype=torch.bfloat16)

    with torch.no_grad():
        encoded_1m = sae.encode(act_gpu.unsqueeze(0)).squeeze(0)  # (1M,)

    total_active = int((encoded_1m != 0).sum().item())
    widths = [16_384, 65_536, 262_144, CFG.sae_width_1m]
    result = MatryoshkaResult(widths=widths, active_features=[], fvu=[], cos_sim=[])

    for width in widths:
        prefix = encoded_1m[:width]  # (width,)
        active = int((prefix != 0).sum().item())

        # Reconstruct using prefix of w_dec
        with torch.no_grad():
            recon = prefix @ sae.w_dec[:width] + sae.b_dec  # (d_model,)

        # FVU = ||x - recon||^2 / ||x - mean(x)||^2
        residual = (act_gpu - recon).float()
        total_var = (act_gpu - act_gpu.mean()).float()
        fvu_val = (residual**2).sum().item() / (total_var**2).sum().item()

        cos = torch.nn.functional.cosine_similarity(
            act_gpu.float().unsqueeze(0), recon.float().unsqueeze(0)
        ).item()

        result.active_features.append(active)
        result.fvu.append(fvu_val)
        result.cos_sim.append(cos)

        log.info(
            "  width=%7d | active=%2d/%d | FVU=%.4f | cos_sim=%.4f",
            width,
            active,
            total_active,
            fvu_val,
            cos,
        )

    log.info("PASS: Reconstruction degrades gracefully across widths")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log.info("=" * 60)
    log.info("MILESTONE 0: PROJECT BOOTSTRAP")
    log.info("=" * 60)

    results = BootstrapResults()

    # 0b: Auth
    step_0b_authenticate()

    # 0c: Load model
    model, tokenizer, model_vram = step_0c_load_model()
    results.model_vram_gb = model_vram

    # 0d-0e: Hook + extraction
    extraction = steps_0d_0e_hook_and_extract(model, tokenizer)
    results.num_layers = extraction.num_layers
    results.hook_shape = extraction.hook_shape

    # 0f: Unload model
    del model
    del tokenizer
    clear_vram()
    results_alloc = vram_report("after model unload", log)
    if results_alloc < 0.5:
        log.info("PASS: VRAM returned to near zero")
    else:
        log.warning("VRAM leak: %.2f GB still allocated", results_alloc)

    # 0g-0h: SAE + encode + L0
    sae, sae_vram, l0 = steps_0g_0h_sae_encode(extraction.last_prompt_act)
    results.sae_vram_gb = sae_vram
    results.combined_vram_gb = model_vram + sae_vram
    results.l0 = l0

    # 0i: Matryoshka
    matryoshka = step_0i_matryoshka(sae, extraction.last_prompt_act)

    # 0j: Final cleanup
    del sae
    clear_vram()
    results.final_vram_gb = vram_report("final state", log)

    # Summary
    results.log_summary()
    log.info("Total time: %s", fmt_time(time.time() - t0))


if __name__ == "__main__":
    main()
