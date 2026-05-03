# The Refusal Axis in Gemma 3 12B

Geometric decomposition and domain-selective capping of refusal behaviour in **Gemma 3 12B** using **Gemma Scope 2** sparse autoencoders.

> **Web explainer:** https://antonio-tresol.github.io/gemma3-refusal-axis/
> **Dataset (HF):** https://huggingface.co/datasets/abotresol/gemma3-refusal-axis-data
> **An exploratory project on the interpretability of refusal behaviour.**

---

## Repository structure

```
.
├── README.md                          # This file
├── CITATION.cff                       # Academic citation metadata
├── LICENSE                            # MIT
├── pyproject.toml                     # uv-managed package
├── uv.lock
├── CLAUDE.md                          # Project instructions for Claude Code
│
├── docs/                              # GitHub Pages explainer (Jekyll)
│   ├── index.md                       # Main explainer
│   ├── methodology.md
│   ├── hierarchy.md
│   ├── falsification.md
│   ├── _config.yml
│   ├── _layouts/default.html
│   └── figures/
│
├── src/refusal_decomposition/         # Importable Python package
│   ├── __init__.py
│   ├── model.py                       # Model + SAE loading, logging, progress
│   ├── data/
│   │   ├── supplementary_pairs.py     # Contrastive pair generation
│   │   └── supplementary_pipeline.py  # End-to-end scoring pipeline
│   ├── analysis/
│   │   ├── refusal_axis.py            # Axis construction + per-domain directions
│   │   ├── domain_capping.py          # Activation capping per Lu et al.
│   │   ├── feature_hierarchy.py       # Matryoshka hierarchy tests (3 methods)
│   │   └── falsification.py           # 10 falsification tests
│   └── viz/
│       ├── refusal_axis.py            # 2D figures (Lu et al. style)
│       └── refusal_axis_3d.py         # 3D hero figure
│
├── scripts/
│   ├── cli/                           # User-facing entry points
│   │   ├── download_data.py           # Fetch HF dataset
│   │   └── upload_data.py             # Push HF dataset
│   ├── milestones/                    # Pipeline (M0-M7)
│   ├── figures/                       # Width-scaling publication figures
│   └── dataset/                       # Pair construction & validation
│
├── notebooks/                         # marimo interactive explorers
│   ├── refusal_axis_explorer.py
│   ├── sae_width_scaling_explorer.py
│   └── refusal_decomposition_demo.py
│
├── findings/                          # Reports + figures + literature
│   ├── reports/
│   ├── figures/
│   └── literature/
│
├── plans/                             # Planning docs + adversarial reviews
│
├── references/                        # BibTeX + reference papers + SAE code
│
├── data/                              # Gitignored; fetch from HF (3.3 GB)
│   └── README.md
│
└── .claude/                           # The Claude Code harness (see "AI research automation" below)
    ├── settings.json
    ├── hooks/
    │   ├── format-python.sh           # PostToolUse: ruff format + isort
    │   └── check-evidence-pinning.sh  # PreToolUse: warn on un-cited findings
    └── skills/
        ├── thesis-experiment/         # GPU pattern, tensor conventions
        ├── validate-claims/           # Convergence validation
        ├── falsify/                   # Falsification workflow
        ├── alphaxiv-paper-lookup/     # Paper retrieval via alphaxiv
        └── marimo-notebook/           # Notebook authoring
```

---

## Quick start

### 1. Install

```bash
uv sync
```

Requires Python 3.12+. PyTorch is pulled from the CUDA 12.8 index (`pytorch-cu128`); for CPU-only or different CUDA, edit `pyproject.toml`.

### 2. Fetch the data (3.3 GB from Hugging Face)

```bash
uv run python scripts/cli/download_data.py
```

This downloads activations, SAE encodings, and refusal direction vectors from [`abotresol/gemma3-refusal-axis-data`](https://huggingface.co/datasets/abotresol/gemma3-refusal-axis-data). No GPU needed for this step.

### 3. Reproduce the analyses

```bash
# Refusal axis: construct, per-domain directions, PCA, cosines
uv run python -m refusal_decomposition.analysis.refusal_axis

# Generate publication figures
uv run python -m refusal_decomposition.viz.refusal_axis
uv run python -m refusal_decomposition.viz.refusal_axis_3d

# Falsification (10 tests)
uv run python -m refusal_decomposition.analysis.falsification

# Feature hierarchy across Matryoshka widths
uv run python -m refusal_decomposition.analysis.feature_hierarchy

# Domain-selective capping (requires GPU + HF tokens for Gemma 3 + Gemma Scope 2)
uv run python -m refusal_decomposition.analysis.domain_capping
```

### 4. Interactive exploration (marimo)

```bash
uv run marimo edit notebooks/refusal_axis_explorer.py
```

### Hardware

Most analyses run on CPU. The capping experiment and any new activation extraction need a GPU with ~32 GB VRAM (RTX 5090 or equivalent). See [`findings/reports/milestone_0_bootstrap.md`](findings/reports/milestone_0_bootstrap.md) for measured VRAM use.

---

## AI research automation: the Claude harness

This project was developed in close collaboration with Claude Code. To make that workflow reproducible and reviewable, the entire harness is committed under `.claude/`:

- **`settings.json`**: auto-format on edit, evidence-pinning warnings on `findings/` writes.
- **`hooks/format-python.sh`**: runs `ruff format` and `ruff check --select I --fix` after every Python edit.
- **`hooks/check-evidence-pinning.sh`**: warns if a write to `findings/` lacks a citation, pushing toward the project's "every claim traces to source" rule.
- **`skills/thesis-experiment/`**: GPU memory pattern, tensor shape conventions, reproducibility checklist.
- **`skills/validate-claims/`**: convergence validation. Spawn parallel agents, check until zero mismatches between report and data.
- **`skills/falsify/`**: pre-register predictions, run falsification tests, update claims based on results.
- **`skills/alphaxiv-paper-lookup/`**: paper retrieval via alphaxiv.org structured overviews.
- **`skills/marimo-notebook/`**: notebook authoring conventions.

`CLAUDE.md` at the repo root encodes the project's "Beliefs": Do → Test → Critique → Validate → Deliver. Every claim should trace to code and data.

---

## Citation

If you reference this work:

```bibtex
@misc{badillaolivas2026refusalaxis,
  title  = {The Refusal Axis: Geometric Decomposition and Domain-Selective
            Capping in Gemma 3 12B},
  author = {Badilla-Olivas, Antonio},
  year   = {2026},
  url    = {https://github.com/Antonio-Tresol/gemma3-refusal-axis},
  note   = {Project on the interpretability of refusal in Gemma 3 12B.},
}
```

Full bibliography of cited works in `references/from-safety-prompts-project/relevant_references.bib`.

---

## Licence

MIT. See [`LICENSE`](LICENSE).

The Gemma 3 12B model and Gemma Scope 2 SAEs are subject to Google DeepMind's separate licences; obtain them via Hugging Face.

---
## Disclaimer

This is a first attempt at interpretability research. **Claude Code** (Anthropic's CLI agent) was used extensively throughout the project: writing analysis code, monitoring long-running experiments, computing statistics, drafting reports, designing figures, running falsification tests, and verifying claims. The full Claude harness used here (hooks, skills, subagent prompts, MCP servers) is included in this repository under `.claude/` for transparency and reproducibility.

Things may well be wrong. **Feedback is genuinely welcome**: open an issue, send corrections, suggest extensions. Honest critique is the most useful thing you can offer.

---
## Feedback

This is exploratory research, done by a first-time interp researcher with substantial AI assistance. Please open an issue if you spot:

- A claim that doesn't trace to data
- A misattributed citation
- A statistical or methodological error
- A way to make the falsification stronger
- An interesting follow-up experiment

Honest, specific critique is the most valuable contribution you can make.
