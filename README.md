# The Refusal Axis in Gemma 3 12B

Geometric decomposition and domain-selective capping of refusal behaviour in **Gemma 3 12B** using **Gemma Scope 2** sparse autoencoders.

> **Web explainer (canonical):** https://antonio-tresol.github.io/gemma3-refusal-axis/
> **Dataset (HF):** https://huggingface.co/datasets/abotresol/gemma3-refusal-axis-data
> **An exploratory project on the interpretability of refusal behaviour.**

The web explainer is the source of truth for everything this project claims: research questions, methodology, headline findings, falsification ledger, glossary, and references. Markdown files in this repo (`findings/`, `plans/`, working notes) may lag the explainer or be wrong, since they are not kept in sync. Treat them as scratch work, not as the published version.

---

## Reproducibility

```bash
# 1. Install (Python 3.12+, GPU optional except for capping/extraction)
uv sync

# 2. Fetch activations + SAE encodings + refusal directions (~3.3 GB)
uv run python scripts/cli/download_data.py

# 3. Re-run any analysis
uv run python -m refusal_decomposition.analysis.refusal_axis
uv run python -m refusal_decomposition.analysis.falsification
uv run python -m refusal_decomposition.analysis.feature_hierarchy
uv run python -m refusal_decomposition.analysis.domain_capping
```

Capping and any new activation extraction need a GPU with ~32 GB VRAM (RTX 5090 or equivalent); everything else runs on CPU. Layer choices, prompt sets, falsification thresholds, and statistical procedures are documented in the **Methodology** section of the explainer.

---

## AI research automation

This project was developed in close collaboration with Claude Code. The full harness (settings, hooks, skills) used to coordinate analysis, format edits on save, enforce evidence-pinning on `findings/` writes, and run falsification is committed under `.claude/` for transparency and reproducibility. `CLAUDE.md` at the repo root encodes the project's working rules.

---

## Citation

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
