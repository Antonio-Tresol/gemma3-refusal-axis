"""Shared infrastructure for all milestone scripts.

Provides:
  - Logging setup (console + file, timestamped)
  - VRAM utilities
  - Model layer navigation (Gemma 3 multimodal wrapper)
  - JumpReLU SAE (Gemma Scope 2 architecture)
  - SAE loading helper
  - Progress tracking with per-item timing and ETA
  - Project-wide config dataclass
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create a logger with console (INFO) + file (DEBUG) handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectConfig:
    model_id: str = "google/gemma-3-12b-it"
    sae_repo: str = "google/gemma-scope-2-12b-it"
    layer: int = 41
    d_model: int = 3840
    sae_width_1m: int = 1_048_576
    sae_width_16k: int = 16_384
    device: str = "cuda"
    data_dir: Path = Path("data")

    @property
    def pairs_file(self) -> Path:
        return self.data_dir / "contrastive_pairs.json"

    @property
    def manifest_file(self) -> Path:
        return self.data_dir / "responses_manifest.json"

    @property
    def activations_dir(self) -> Path:
        return self.data_dir / "activations"

    def activation_path(self, site: str, condition: str, pair_id: int) -> Path:
        return self.activations_dir / site / condition / f"pair_{pair_id}.pt"

    def refusal_direction_path(self, site: str) -> Path:
        return self.data_dir / f"refusal_direction_{site}.pt"

    def sae_path(self, width_label: str, l0: str = "medium") -> str:
        """HuggingFace path for a Gemma Scope 2 SAE."""
        return f"resid_post/layer_{self.layer}_width_{width_label}_l0_{l0}/params.safetensors"


CFG = ProjectConfig()


# ---------------------------------------------------------------------------
# VRAM utilities
# ---------------------------------------------------------------------------
def gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def vram_report(label: str, log: logging.Logger | None = None) -> float:
    """Log current VRAM usage. Returns allocated GB."""
    alloc = gb(torch.cuda.memory_allocated())
    res = gb(torch.cuda.memory_reserved())
    msg = "VRAM %-25s alloc=%.2f GB  reserved=%.2f GB"
    if log:
        log.info(msg, label, alloc, res)
    else:
        print(msg % (label, alloc, res))
    return alloc


def clear_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def unload_and_verify(
    label: str, log: logging.Logger, *objects_to_del: object
) -> float:
    """Delete references, clear VRAM, verify it returned near zero."""
    log.info("Unloading %s ...", label)
    for obj in objects_to_del:
        del obj
    clear_vram()
    alloc = vram_report(f"after unloading {label}", log)
    if alloc < 0.5:
        log.info("PASS: VRAM returned to near zero")
    else:
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        alloc = vram_report("after aggressive cleanup", log)
        if alloc < 0.5:
            log.info("PASS: VRAM cleared after aggressive cleanup")
        else:
            log.error("FAIL: VRAM leak of %.2f GB", alloc)
    return alloc


# ---------------------------------------------------------------------------
# Model layer navigation
# ---------------------------------------------------------------------------
def get_model_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Navigate Gemma 3's multimodal wrapper to reach transformer layers."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model"):
            lm = inner.language_model
            if hasattr(lm, "layers"):
                return lm.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
        if hasattr(inner, "layers"):
            return inner.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


# ---------------------------------------------------------------------------
# JumpReLU SAE
# ---------------------------------------------------------------------------
class JumpReLUSAE(torch.nn.Module):
    """JumpReLU Sparse Autoencoder matching Gemma Scope 2 architecture.

    Architecture:
      pre_acts = x @ w_enc + b_enc        # (batch, d_sae)
      mask     = pre_acts > threshold      # JumpReLU gate
      acts     = mask * relu(pre_acts)     # sparse features
      recon    = acts @ w_dec + b_dec       # reconstruction
    """

    def __init__(self, d_in: int, d_sae: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.w_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features. x: (..., d_in) -> (..., d_sae)"""
        pre_acts = x @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        return mask * torch.nn.functional.relu(pre_acts)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to reconstruction. acts: (..., d_sae) -> (..., d_in)"""
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, sparse_features)."""
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts


def load_sae(
    width: int,
    width_label: str,
    l0: str = "medium",
    device: str = CFG.device,
    dtype: torch.dtype = torch.bfloat16,
    log: logging.Logger | None = None,
) -> JumpReLUSAE:
    """Download and load a Gemma Scope 2 SAE from HuggingFace."""
    hf_path = CFG.sae_path(width_label, l0)
    if log:
        log.info("Downloading SAE: %s/%s", CFG.sae_repo, hf_path)

    local_path = hf_hub_download(
        repo_id=CFG.sae_repo,
        filename=hf_path,
        token=HF_TOKEN,
    )
    params = load_file(local_path)

    if log:
        log.debug("SAE params: %s", {k: tuple(v.shape) for k, v in params.items()})

    sae = JumpReLUSAE(d_in=CFG.d_model, d_sae=width)

    # Gemma Scope may use W_enc or w_enc — handle both
    key_map = {
        "W_enc": "w_enc",
        "W_dec": "w_dec",
        "threshold": "threshold",
        "b_enc": "b_enc",
        "b_dec": "b_dec",
    }
    state_dict: dict[str, torch.Tensor] = {}
    for gs_key, our_key in key_map.items():
        if gs_key in params:
            state_dict[our_key] = params[gs_key]
        elif gs_key.lower() in params:
            state_dict[our_key] = params[gs_key.lower()]

    if not state_dict:
        state_dict = dict(params)

    sae.load_state_dict(state_dict)
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()

    if log:
        log.info("SAE loaded: width=%d, device=%s, dtype=%s", width, device, dtype)

    return sae


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class ProgressTracker:
    """Tracks per-item timing, running average, and ETA."""

    def __init__(self, total: int, already_done: int = 0) -> None:
        self.total = total
        self.already_done = already_done
        self.done_this_run = 0
        self._times: list[float] = []
        self._phase_start = time.time()

    def tick(self, elapsed: float) -> None:
        self.done_this_run += 1
        self._times.append(elapsed)

    @property
    def completed(self) -> int:
        return self.already_done + self.done_this_run

    @property
    def remaining(self) -> int:
        return self.total - self.completed

    @property
    def avg_time(self) -> float:
        return sum(self._times) / len(self._times) if self._times else 0.0

    @property
    def eta(self) -> float:
        return self.avg_time * self.remaining

    @property
    def phase_elapsed(self) -> float:
        return time.time() - self._phase_start

    def log_item(self, log: logging.Logger, msg: str) -> None:
        log.info(
            "[%d/%d] %s | %.1fs | avg %.1fs | ETA %s",
            self.completed,
            self.total,
            msg,
            self._times[-1],
            self.avg_time,
            fmt_time(self.eta),
        )
