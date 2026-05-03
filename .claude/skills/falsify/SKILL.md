---
name: falsify
description: Plan and execute scientific falsification tests for claims. Creates a falsification plan, implements tests, runs them, and reports which claims survived.
disable-model-invocation: true
effort: max
---

# Falsification Protocol

Scientific claims survive falsification efforts. That is science. This skill systematically attempts to destroy every claim before it enters the thesis.

## Workflow

### Step 1: Identify claims
Read the target document ($ARGUMENTS or the most recent findings report). Extract every testable claim as a numbered list.

### Step 2: Design falsification tests
For each claim, design at least one test that could destroy it:

- **Statistical claims** → permutation null, bootstrap CI, effect size vs random baseline
- **"X is different from Y"** → is the difference distinguishable from noise? Random split null.
- **"X causes Y"** → random direction control (does a random intervention also produce the effect?)
- **"N dimensions needed"** → random baseline (how many dims do random vectors need?)
- **"Cluster structure"** → stability across methods (Ward vs complete vs single linkage)
- **Small-n claims** → subsample instability (what happens at n=4?)
- **Sweeping claims** → multiple comparisons correction. Was the "best" parameter pre-registered?
- **Scorer-based claims** → test-retest reliability of the scorer

### Step 3: Prioritize by destructive potential
Order tests by: which test, if it succeeds, destroys the most important claim?

### Step 4: Implement and run
Write the tests as a single reproducible script. Include:
- Clear logging of each test
- JSON output for programmatic validation
- Random seeds for reproducibility

### Step 5: Report
For each claim, state:
1. The claim
2. The falsification test(s) applied
3. The result: **Survives** / **Weakened** (qualified version) / **Failed** (retracted)
4. If weakened: the qualified version
5. If failed: the corrected finding

### Step 6: Update
Update the source document with qualified claims and a falsification scorecard.

## Lessons from this project

- Feature 9449→20318 "reparameterization" was a **base-rate artifact** caught by permutation test (both fire on 97% of prompts, expected Jaccard under independence = 0.944)
- "Near-orthogonal" safety-capability cosine had bootstrap CI [-0.316, 0.639] — crosses zero
- Random split null gave p=0.054 — borderline, not definitive
- Always check: is the "effect" just what you'd expect from base rates?

$ARGUMENTS
