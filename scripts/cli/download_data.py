"""Download essential data from the private Hugging Face dataset.

Usage:
    uv run python download_data.py

Requires: HF_TOKEN env var or `huggingface-cli login`.
Downloads ~1.7 GB of data needed to run scripts and notebooks:
  - data/activations/        (per-pair activation tensors)
  - data/encoded/            (SAE encodings)
  - data/refusal_direction_*.pt (refusal axis vectors)
"""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

REPO_ID = "abotresol/gemma3-refusal-axis-data"
# Resolve to the project root regardless of where the script is called from.
# scripts/cli/download_data.py -> repo root is two parents up.
LOCAL_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main():
    log.info("Downloading from %s ...", REPO_ID)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(LOCAL_DIR),
    )

    # Verify key files exist
    checks = [
        LOCAL_DIR / "activations" / "mean_response_token" / "positive" / "pair_1.pt",
        LOCAL_DIR / "refusal_direction_mean_response_token.pt",
    ]
    for p in checks:
        if p.exists():
            log.info("  OK: %s", p.relative_to(LOCAL_DIR))
        else:
            log.warning("  MISSING: %s", p.relative_to(LOCAL_DIR))

    log.info("Done. Data is in %s", LOCAL_DIR)


if __name__ == "__main__":
    main()
