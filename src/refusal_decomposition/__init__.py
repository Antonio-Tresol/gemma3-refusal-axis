"""Mechanistic interpretability of refusal behaviour in Gemma 3 12B.

Top-level package re-exports the most-used utilities from `model` so existing
analysis code can `from refusal_decomposition import CFG, setup_logging, ...`.
"""

from refusal_decomposition.model import (  # noqa: F401
    CFG,
    HF_TOKEN,
    JumpReLUSAE,
    ProgressTracker,
    clear_vram,
    fmt_time,
    get_model_layers,
    load_sae,
    setup_logging,
    vram_report,
)

__all__ = [
    "CFG",
    "HF_TOKEN",
    "JumpReLUSAE",
    "ProgressTracker",
    "clear_vram",
    "fmt_time",
    "get_model_layers",
    "load_sae",
    "setup_logging",
    "vram_report",
]
