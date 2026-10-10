# Generative Recommenders in JAX

A JAX/Flax reimplementation of **HSTU (Hierarchical Sequential Transduction Units)**, the generative recommender architecture from Meta's *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations* (Zhai et al., ICML 2024).

The Python package is named `htsu` and uses `HTSU*` class names; those refer to the same HSTU encoder.

## Overview

Traditional deep learning recommendation models (DLRMs) score user-item pairs using large sets of hand-engineered features and tend to plateau as compute grows. Generative Recommenders (GRs) instead treat recommendation as a sequential transduction problem: a user's history of items and actions is modeled as a single sequence, and ranking and retrieval become next-token prediction tasks.

- **Ranking**: predicting the next action a user will take
- **Retrieval**: predicting the next item a user will interact with

HSTU is the encoder built for this setting. Its main design choices are:

- **Pointwise aggregated attention:** replaces softmax normalization, which suits non-stationary item vocabularies and preserves signal about the intensity of user preferences.
- **Fused gating layer:** an elementwise gate replaces the feed-forward block, cutting linear layers and activation memory.
- **Relative attention bias:** incorporates both positional and temporal information.
- **Stochastic Length:** randomly drops parts of long user histories during training to reduce attention cost without hurting quality.

This repo reimplements the HSTU encoder and generative training setup in JAX (Flax NNX + Optax) for functional, JIT-compiled training.

### What's implemented

- HSTU architecture with position and temporal biases
- Architectural variants:
  - SiLU-based attention (paper HSTU)
  - Softmax attention baseline
  - FFN ablation studies
  - Positional-only variants (no temporal bias)
- Synthetic dataset generation matching the paper's Appendix C
- Ranking and retrieval losses, metrics, and example training scripts
- Stochastic length sampling for efficient training

## Tech Stack

| Component | Tools |
|-----------|-------|
| Framework | JAX, Flax NNX |
| Optimization | Optax |
| Data | NumPy |
| Tooling | Python 3.9+, pytest |

## Installation

```bash
git clone https://github.com/carlychvn/generative-recommenders-jax.git
cd generative-recommenders-jax
pip install -e .
```

Optional extras: `pip install -e ".[dev]"` for tests/linting, or `pip install -e ".[cuda]"` for GPU JAX.

## Quick Start

```python
import jax
from flax import nnx
from htsu.layers import HTSULayer
from htsu.sequencing import interleave_items_and_actions, build_positions, build_timestamps
from htsu.losses import ranking_loss, retrieval_loss

d_model = 128
num_buckets = 32
max_distance = 128
rngs = nnx.Rngs(0)

layer = HTSULayer(d_model, num_buckets, max_distance, rngs)

# items, actions: (batch, seq_len, d_model)
# timestamps: (batch, seq_len)
sequence = interleave_items_and_actions(items, actions)
positions = build_positions(seq_len)
timestamps_interleaved = build_timestamps(timestamps)

output = layer(sequence, positions, timestamps_interleaved)

rank_loss = ranking_loss(output, target_actions, mask)
retr_loss = retrieval_loss(output, target_items, mask)
```

Generate synthetic data and train:

```python
from htsu.synthetic_data import generate_dataset

dataset = generate_dataset(
    num_records=1000,
    vocab_size=10000,
    num_categories=100,
    rng=jax.random.PRNGKey(42),
)
```

See `examples/train_synthetic.py` for a full training loop, and `examples/README.md` for the other scripts.

```bash
python examples/quickstart.py
python examples/train_synthetic.py --mode train --num-steps 3000
python examples/evaluate.py
```

## Architecture

1. **PointwiseProjection**: projects input and splits into U, Q, V, K via SiLU
2. **RelativeAttentionBias**: learnable position and time bias (T5-style bucketing)
3. **SpatialAggregation**: SiLU-based attention (`Q·K^T + bias`)
4. **PointwiseTransformation**: LayerNorm + gating + output projection

```python
from htsu.layers import (
    HTSULayer,                          # Full HSTU with position + time
    HTSULayerPositionalOnly,            # Position-only variant
    HTSULayerSoftmax,                   # Softmax attention baseline
    HTSULayerFFN,                       # FFN ablation
    VanillaTransformerBlock,            # Standard transformer comparison
)
```

## Project Structure

```
├── htsu/
│   ├── layers.py                  # HSTU layer implementations
│   ├── losses.py                  # Ranking and retrieval losses
│   ├── sequencing.py              # Sequence building utilities
│   ├── stochastic_length.py       # Stochastic length sampling
│   ├── synthetic_data.py          # Synthetic dataset generation
│   ├── metrics.py                 # Ranking / retrieval metrics
│   └── visualization.py           # Training log helpers
├── tests/
├── scripts/                       # Overfit, efficiency, comparison
└── examples/                      # quickstart, train_synthetic, evaluate
```

## Testing

```bash
pytest
pytest tests/test_layers.py
pytest --cov=htsu --cov-report=html
```

## Datasets and Evaluation

The paper's public-dataset benchmarks are MovieLens-1M, MovieLens-20M, and Amazon Books, compared against a SASRec baseline, with Hit Rate@K and NDCG@K over the full item corpus. This repo currently trains and evaluates on synthetic data; public-dataset training is not wired up yet.

Metrics available in `htsu.metrics` include Hit Rate@K, NDCG@K, MRR, recall, and precision.

## Stochastic Length Sampling

```python
from htsu.stochastic_length import stochastic_length_sample

subsampled_items, subsampled_actions, subsampled_positions = stochastic_length_sample(
    items, actions, positions, item_ids,
    target_length=128,
    rng=jax.random.PRNGKey(0),
)
```

## References

- Zhai et al. *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.* ICML 2024. [arXiv:2402.17152](https://arxiv.org/abs/2402.17152)
- Official implementation: [facebookresearch/generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
