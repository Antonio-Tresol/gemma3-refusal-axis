---
name: gemma-2-scope
description: Extract and analyze model activations using Gemma Scope 2 sparse autoencoders (SAEs).
---

# Gemma 2 Scope

This skill documents how to use the **Gemma Scope 2** sparse autoencoders (SAEs) integrated into `model_evaluation`. It allows you to decompose dense model activations into sparse, interpretable features (latents) to understand "what the model is thinking."

## Prerequisites

- **Local Package**: `model_evaluation` (specifically `model_evaluation.main_agent.gemma_scope_sae`)
- **Libraries**: `transformers`, `torch`, `huggingface_hub`
- **Models**: Gemma 3 family (1B, 4B, 12B, 27B)

## Core Capabilities

### 1. Load a Specific SAE

Use `load_gemma_scope_sae` to fetch pretrained SAE weights from HuggingFace.

```python
from model_evaluation.main_agent.gemma_scope_sae import load_gemma_scope_sae

# Load SAE for Gemma 3 12B, Instruction Tuned, Layer 41 (Legacy "resid_post" site)
sae, sae_config = load_gemma_scope_sae(
    model_size="12b",
    model_type="it",
    layer=41,  # Recommended layer for high-level concepts
    width="16k",
    l0_size="medium",
    device="cuda"
)
```

### 2. Extract Features from Text

Use `extract_sae_features` to run a prompt through the model and capture SAE features.

```python
from model_evaluation.main_agent.gemma_scope_sae import extract_sae_features

result = extract_sae_features(
    model=model,        # Your loaded HuggingFace model
    tokenizer=tokenizer,
    sae=sae,
    sae_config=sae_config,
    text="The quick brown fox jumps over the lazy dog",
    max_new_tokens=20,
    top_k=10            # Keep top 10 features per token
)

print(f"Generated: {result.answer}")
print(f"Average L0 (sparsity): {result.l0}")
print(f"FVU (reconstruction error): {result.fvu:.2%}")
```

### 3. Analyze Top Features

Inspect what features are active to interpret model behavior.

```python
from model_evaluation.main_agent.gemma_scope_sae import get_top_features_summary

# Get most active features across the whole sequence
top_features = get_top_features_summary(result=result)

for feat_idx, activation in list(top_features.items())[:5]:
    print(f"Feature {feat_idx}: {activation:.2f}")
```

### 4. Visualization

Render HTML visualizations to see exactly *where* features fire.

```python
from model_evaluation.main_agent.gemma_scope_sae import (
    visualize_token_activations,
    visualize_top_features_per_token
)

# 1. Heatmap for the whole sequence (total SAE activation)
visualize_token_activations(result=result)

# 2. Heatmap for a specific feature (e.g., Feature 12345)
visualize_token_activations(result=result, feature_idx=12345)

# 3. Table of top features per token
visualize_top_features_per_token(result=result, num_tokens=10)
```

## Architecture Details

- **JumpReLU**: The SAEs use a JumpReLU activation function (ReLU with a learned threshold) for better sparsity/fidelity trade-off.
- **Hooking**: The implementation uses PyTorch forward hooks (`register_forward_hook`) to intercept the residual stream (`resid_post`) without modifying the model architecture.
- **Normalization**: Activations are normalized before encoding, and the SAE weights are scaled to handle this internally.

## Common Workflows

1. **Hypothesis Testing**:
    - Generate text about a topic (e.g., "Physics").
    - Identify top features.
    - Run control text (e.g., "History") to see if those features deactivate.
2. **Steering (Advanced)**:
    - Identify a feature vector (`sae.w_dec[feature_idx]`).
    - Add a multiple of this vector to the residual stream during generation (custom hook required, see tutorial for example).

## References

For deep dives and runnable code, refer to the following resources included in this skill:

- **Technical Paper**: [Gemma Scope 2 Paper](references/gemmascope2_paper.md) - Details on training, architecture (JumpReLU), and evaluations.
- **Tutorial**: [Gemma Scope 2 Tutorial](scripts/tutorial_gemma_scope_2.py) - Comprehensive guide on manual hooking, activation steering, and analysis.
- **Visualization Tests**: [Wrapper Visualization](scripts/test_wrapper_visualization.py) - Examples of using the `GemmaWithSAE` wrapper and visualization tools.
