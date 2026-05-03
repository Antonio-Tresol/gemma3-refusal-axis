"""Upload untracked essential data to a private Hugging Face dataset.

Usage:
    uv run python upload_data.py

Requires: HF_TOKEN env var or `huggingface-cli login`.
Uploads ~1.7 GB of data that's gitignored but needed to run scripts:
  - data/activations/        (per-pair activation tensors, ~23 MB)
  - data/encoded/            (SAE encodings, ~1.7 GB)
  - data/refusal_direction_*.pt (refusal axis vectors, ~34 KB)

Does NOT upload:
  - data/milestone_4_results/decoder_vectors.pt (16 GB, regenerable from SAE)
  - JSON files already tracked by git
"""

import logging
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

REPO_ID = "abotresol/gemma3-refusal-axis-data"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATASET_CARD = DATA_DIR / "HF_README.md"

# Folders and files to upload (gitignored essentials only)
UPLOAD_PATTERNS = [
    "activations/**/*.pt",
    "encoded/*.pt",
    "refusal_direction_last_prompt_token.pt",
    "refusal_direction_mean_response_token.pt",
]


def main():
    api = HfApi()
    user = api.whoami()["name"]
    log.info("Authenticated as: %s", user)

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    log.info("Repo ready: %s (private)", REPO_ID)

    # Push the dataset card (README.md on HF) before uploading the tensors so it
    # shows up first when the user visits the dataset page mid-upload.
    if DATASET_CARD.exists():
        api.upload_file(
            path_or_fileobj=str(DATASET_CARD),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Update dataset card",
        )
        log.info("Dataset card uploaded as README.md")
    else:
        log.warning("Dataset card not found at %s; skipping README upload", DATASET_CARD)

    # Collect files to upload
    files_to_upload = []
    for pattern in UPLOAD_PATTERNS:
        matched = sorted(DATA_DIR.glob(pattern))
        files_to_upload.extend(matched)
        log.info("Pattern '%s': %d files", pattern, len(matched))

    log.info("Total files to upload: %d", len(files_to_upload))

    # Upload with flat structure (no 'data/' prefix in repo)
    # so download_data.py can use local_dir=data/ directly
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(DATA_DIR),
        path_in_repo=".",
        allow_patterns=[
            "activations/**/*.pt",
            "encoded/*.pt",
            "refusal_direction_*.pt",
        ],
        commit_message="Upload essential data artifacts (activations, encodings, refusal directions)",
    )

    log.info(
        "Done. %d files uploaded to https://huggingface.co/datasets/%s",
        len(files_to_upload),
        REPO_ID,
    )


if __name__ == "__main__":
    main()
