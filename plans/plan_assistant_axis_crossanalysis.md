# Plan: Assistant Axis Cross-Analysis

**Status:** PLANNED, not yet executed
**Priority:** After domain-selective capping experiment
**Estimated GPU time:** ~12 hours

## Goal

Compute the Assistant Axis (Lu et al. (2026, "The Assistant Axis," arXiv:2601.10387)) on Gemma 3 12B using their exact prompts, then measure where our refusal axis sits relative to it. Answers: "Is refusal part of the assistant persona, or a separate direction?"

## Data available from their repo

All at https://github.com/safety-research/assistant-axis:

- `data/roles/role_list.json`: 275 roles with descriptions
- `data/roles/instructions/<role>.json`: 5 system prompts + 40 questions + eval_prompt per role
- `data/roles/instructions/default.json`: 5 neutral default system prompts
- `data/extraction_questions.jsonl`: 240 general questions
- Scoring rubric: 0-3 scale judge (in each role's `eval_prompt` field)

Pre-computed axes for other models at: `huggingface.co/datasets/lu-christina/assistant-axis-vectors`

## Pipeline

1. **Clone their data:** Download role list + instructions + extraction questions
2. **Generate responses:** For each of 275 roles × 5 system prompts × 240 questions = ~330K generations. Use all 240 extraction questions to match their methodology exactly.
3. **Score with judge:** Use Sonnet 4.6 (matching their GPT-4.1-mini judge) with their exact rubric. Keep only score=3 responses.
4. **Extract activations:** Mean over assistant response tokens at layer 24 (50% depth, matching their methodology of using middle layers). Also extract at layer 41 for comparison.
5. **Compute axis:** `assistant_axis = mean(default_vectors) - mean(role_vectors)`
6. **Cross-analysis:**
   - `cos(refusal_axis, assistant_axis)`: how aligned are they?
   - Per-domain: `cos(domain_refusal_direction, assistant_axis)`: which refusal types are "more assistant-like"?
   - PCA on the joint space of refusal + assistant directions

## Estimated compute

- 330K generations × 12s = ~46 days at 1 GPU. Need to reduce.
- **Realistic subset:** 30 roles × 240 questions × 5 prompts = ~36K generations ≈ 5 days. Still heavy.
- **Minimal viable:** 30 roles × 50 questions × 2 prompts = 3,000 generations ≈ 10 hours. Enough for a stable axis estimate.

## Decision

Do the minimal viable version first (10h GPU). If the cross-analysis shows interesting signal (cos between refusal and assistant axes is neither ~0 nor ~1), then scale up.

## Evidence pins

- Methodology: Lu et al. (2026) Sec 3.1 (axis computation)
- Prompts: their repo `data/roles/` and `data/extraction_questions.jsonl`
- Judge rubric: their repo, each `<role>.json` `eval_prompt` field
- Layer choice: Lu et al. (2026) use middle layer (~50% depth); also compare with our layer 41
