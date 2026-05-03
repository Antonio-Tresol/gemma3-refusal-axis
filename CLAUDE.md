# CLAUDE.md · gemma3-refusal-axis

Mechanistic interpretability of refusal behaviour in Gemma 3 12B using Gemma Scope 2 SAEs. This is science: every claim must be defensible, every number reproducible, every result validated before reported.

## Beliefs (non-negotiable)

1. **Do → Test → Critique → Validate → Deliver.** Never deliver without validation. See `/validate-claims`.
2. **Claims trace to code and data.** No hand-waving. No post-hoc rationalisation. If data disagrees with the plan, the plan is wrong.
3. **Falsify before reporting.** Scientific claims survive falsification efforts. See `/falsify`.
4. **Evidence pin every decision.** `[Decision] · [Source, Section/Page]`. Enforced by hook on `findings/` writes.
5. **Minimal code, maximal clarity.** Typed tensor shapes, explicit seeds, no unnecessary dependencies.

## Environment

- Python 3.12, `uv`, RTX 5090 (32 GB VRAM)
- `torch` from `pytorch-cu128` index
- Formatting: `ruff format` + `ruff check --fix --select I` (enforced by post-edit hook)

## Key scientific distinction

The 6 refusal domains span three distinguishable directions (hypothesis under test):
1. **Value-based refusal** (safety, ethical, legal, privacy): model CAN but SHOULD NOT
2. **Identity honesty** (identity_boundary): truthfulness constraint
3. **Capability acknowledgment** (capability_boundary): factual limitation

## Project map

| What | Where |
|------|-------|
| Research plan (milestones 0-7) | `rq1_poc_plan_v5.md` |
| Findings & reports | `findings/reports/`, `findings/figures/` |
| Analysis plans | `findings/plans/` |
| Literature notes | `findings/literature/` |
| Notebooks (interactive) | `notebooks/*.py` (marimo) |
| Data (activations, encodings, results) | `data/` |
| References & BibTeX | `references/` |
| Technical bootstrap facts | `findings/reports/milestone_0_bootstrap.md` § Technical Notes |
| Past session transcripts | `.sessions/` |

## Skills (what Claude knows how to do)

| Skill | Invocation | Purpose |
|-------|-----------|---------|
| `thesis-experiment` | Auto (background) | GPU pattern, tensor conventions, evidence pinning, reproducibility |
| `validate-claims` | Auto or `/validate-claims` | Convergence validation: check numbers against data until zero mismatches |
| `falsify` | `/falsify [target]` | Plan and run falsification tests on claims |
| `research` | Auto or `/research` | Literature search via MCP servers + AlphaXiv |
| `marimo-notebook` | Auto or `/marimo-notebook` | Write marimo notebooks correctly |

## Hooks (what's enforced automatically)

| Hook | Event | What it does |
|------|-------|-------------|
| `format-python.sh` | PostToolUse (Edit/Write) | Runs ruff format + isort on .py files |
| `check-evidence-pinning.sh` | PreToolUse (Write/Edit) | Warns if findings/ files lack citations |
| Stop prompt | Stop | Checks if quantitative claims were validated before delivery |

## Memory

At conversation end, save genuinely valuable discoveries to memory. Not ephemeral task state, only things useful across conversations.
