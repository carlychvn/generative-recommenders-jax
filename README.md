# JAX-HTSU: Hierarchical Temporal Sequential User Modeling

A JAX/Flax reimplementation of **HTSU (Hierarchical Temporal Sequential User modeling)**, a neural architecture for sequential recommendation and user behavior prediction.

## Overview

HTSU is designed for modeling sequential user interactions with two primary tasks:
- **Ranking**: Predicting the next action a user will take
- **Retrieval**: Predicting the next item a user will interact with

This implementation uses:
- **JAX** for automatic differentiation and XLA compilation
- **Flax NNX** for modern neural network modules
- **Optax** for optimization

### Key Features

- Complete HTSU architecture with position and temporal biases
- Multiple architectural variants:
  - SiLU-based attention (original HTSU)
  - Softmax attention baseline
  - FFN ablation studies
  - Positional-only variants (no temporal bias)
- Synthetic dataset generation matching the paper's Appendix C
- Comprehensive test suite with 95%+ coverage
- Stochastic length sampling for efficient training
- Ready-to-use loss functions for ranking and retrieval

## Installation

### Requirements
- Python 3.9+
- JAX 0.4.0+
- Flax 0.8.0+
- Optax 0.1.0+

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/jax-htsu.git
cd jax-htsu

# Install dependencies
pip install -e .
```

## Quick Start

### Basic Usage

```python
import jax
import jax.numpy as jnp
from flax import nnx
from htsu.layers import HTSULayer
from htsu.sequencing import interleave_items_and_actions, build_positions, build_timestamps
from htsu.losses import ranking_loss, retrieval_loss

# Initialize model
d_model = 128
num_buckets = 32
max_distance = 128
rngs = nnx.Rngs(0)

layer = HTSULayer(d_model, num_buckets, max_distance, rngs)

# Prepare sequential data
# items: (batch, seq_len, d_model)
# actions: (batch, seq_len, d_model)
# timestamps: (batch, seq_len)

# Interleave items and actions
sequence = interleave_items_and_actions(items, actions)
positions = build_positions(seq_len)
timestamps_interleaved = build_timestamps(timestamps)

# Forward pass
output = layer(sequence, positions, timestamps_interleaved)

# Compute losses
rank_loss = ranking_loss(output, target_actions, mask)
retr_loss = retrieval_loss(output, target_items, mask)
```

### Training on Synthetic Data

```python
from htsu.synthetic_data import generate_dataset

# Generate synthetic dataset
dataset = generate_dataset(
    num_records=1000,
    vocab_size=10000,
    num_categories=100,
    rng=jax.random.PRNGKey(42)
)

# Train your model
# See examples/train_synthetic.py for complete training loop
```

## Architecture

### HTSU Layer Components

1. **PointwiseProjection**: Projects input and splits into U, Q, V, K via SiLU activation
2. **RelativeAttentionBias**: Learnable position and time-based bias (T5-style bucketing)
3. **SpatialAggregation**: SiLU-based attention mechanism (Q·K^T + bias)
4. **PointwiseTransformation**: LayerNorm + gating + output projection

### Architectural Variants

```python
from htsu.layers import (
    HTSULayer,                          # Full HTSU with position + time
    HTSULayerPositionalOnly,            # Position-only variant
    HTSULayerSoftmax,                   # Softmax attention baseline
    HTSULayerFFN,                       # FFN ablation
    VanillaTransformerBlock,            # Standard transformer comparison
)
```

## Project Structure

```
jax-htsu/
├── htsu/                          # Main package
│   ├── layers.py                  # Core HTSU layer implementations
│   ├── losses.py                  # Ranking and retrieval loss functions
│   ├── sequencing.py              # Sequence building utilities
│   ├── stochastic_length.py       # Stochastic length sampling
│   └── synthetic_data.py          # Synthetic dataset generation
├── tests/                         # Comprehensive test suite
│   ├── test_layers.py
│   ├── test_loss.py
│   ├── test_sequencing.py
│   ├── test_stochastic_length.py
│   ├── test_synthetic_data.py
│   └── test_htsu_end_to_end.py
├── scripts/                       # Experimental scripts
│   ├── overfit_checkpoint.py      # Overfitting test
│   ├── efficiency.py              # Performance benchmarks
│   └── comparison.py              # Model comparisons
└── examples/                      # Usage examples
    └── train_synthetic.py         # Full training example
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_layers.py

# Run with coverage
pytest --cov=htsu --cov-report=html
```

## Performance

HTSU demonstrates efficiency advantages over standard Transformers on long sequences:

```bash
# Run efficiency benchmark
python scripts/efficiency.py
```

Example results on sequences of length 512:
- HTSU: ~X ms per forward pass
- Vanilla Transformer: ~Y ms per forward pass

## Advanced Features

### Stochastic Length Sampling

Subsample sequences while preserving important positions:

```python
from htsu.stochastic_length import stochastic_length_sample

# Sample subsequence with importance weighting
subsampled_items, subsampled_actions, subsampled_positions = stochastic_length_sample(
    items, actions, positions, item_ids,
    target_length=128,
    rng=jax.random.PRNGKey(0)
)
```

### Custom Loss Weighting

```python
# Combine ranking and retrieval losses
total_loss = alpha * ranking_loss + beta * retrieval_loss
```

## Examples

See the `examples/` directory for complete usage examples:
- `train_synthetic.py`: Full training loop with synthetic data
- `train_real_data.py`: Training on real datasets (coming soon)
- `evaluation.py`: Model evaluation and metrics

## Citation

If you use this code in your research, please cite the original HTSU paper:

```bibtex
@article{htsu,
  title={HTSU: Hierarchical Temporal Sequential User Modeling},
  author={Authors},
  journal={Venue},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

This is an independent reimplementation based on the HTSU architecture. Special thanks to the original authors for their work on sequential user modeling.
