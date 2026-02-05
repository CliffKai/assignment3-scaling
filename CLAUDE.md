# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stanford CS336 Assignment 3: Scaling Laws for transformer language models. The project provides a transformer implementation and empirical data for studying how model performance scales with compute budget and parameter count.

## Build Commands

```bash
# Run any command in the project environment
uv run <command>

# Run Python scripts
uv run python <script.py>

# Add dependencies
uv add <package>

# Get Python interpreter path (for IDE configuration)
uv run which python
```

## Architecture

### Core Model (`cs336_scaling/model.py`)

**BasicsTransformerLM** - Main transformer language model class
- Pre-norm architecture (layer norm before sublayers)
- Causal masking for autoregressive generation
- No bias in layer norms or attention
- Key methods: `forward()`, `generate()`, `get_num_params()`, `from_pretrained()`

**TransformerBlock** - Single transformer layer
- Multi-head self-attention with causal masking (uses `nn.MultiheadAttention`)
- Feed-forward network with GELU activation
- Residual connections with optional dropout

### Data (`data/isoflops_curves.json`)

Empirical scaling data with 64 data points across compute budgets (6e18 to 3e21 FLOPs). Each entry contains `parameters`, `compute_budget`, and `final_loss`.

## Key Design Patterns

- Model config stored as dict in `__init__` for serialization
- `from_pretrained()` handles compiled model prefixes (`_orig_mod.`)
- Generation supports temperature scaling, top-k sampling, and EOS token stopping
