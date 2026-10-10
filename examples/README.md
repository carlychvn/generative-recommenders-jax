# HTSU Examples

This directory contains example scripts demonstrating various uses of the JAX-HTSU library.

## Available Examples

### 1. `quickstart.py` - Basic Usage

A minimal example showing the core concepts:
- Creating synthetic data
- Initializing HTSU models
- Running forward passes
- Computing losses for ranking and retrieval

**Run it:**
```bash
python examples/quickstart.py
```

**Best for:** First-time users wanting to understand basic HTSU usage.

---

### 2. `train_synthetic.py` - Full Training Pipeline

Complete training script with:
- Training loop with logging
- Checkpointing
- Metric tracking
- Model comparison mode

**Run it:**
```bash
# Standard training
python examples/train_synthetic.py --mode train --num-steps 3000

# Compare HTSU variants
python examples/train_synthetic.py --mode compare --num-steps 1000

# Custom configuration
python examples/train_synthetic.py \
    --mode train \
    --num-records 2000 \
    --vocab-size 20000 \
    --num-steps 5000 \
    --seed 123
```

**Best for:** Understanding full training workflows and experimenting with hyperparameters.

---

### 3. `evaluate.py` - Model Evaluation

Demonstrates comprehensive evaluation:
- Computing multiple metrics (accuracy, MRR, NDCG)
- Generating confusion matrices
- Visualizing results

**Run it:**
```bash
python examples/evaluate.py
```

**Best for:** Learning how to evaluate trained models and interpret results.

---

## Example Outputs

### Quickstart Output
```
================================================================================
HTSU Quickstart: Ranking Task
================================================================================

1. Generating synthetic data...
   Generated 10 records
   Sequence length: 47 events

2. Creating embeddings...
   Embedding dimension: 64

3. Building interleaved sequence...
   Interleaved sequence shape: (94, 64)

4. Initializing HTSU model...
   Model initialized

5. Running forward pass...
   Output shape: (94, 64)

6. Computing ranking loss...
   Loss: 2.3456

================================================================================
Ranking example complete!
================================================================================
```

### Training Output
```
================================================================================
Training HTSU on Ranking Task
================================================================================
Config: {'task': 'ranking', 'num_records': 1000, ...}

Generating synthetic dataset...
Generated 1000 records

Training...
--------------------------------------------------------------------------------
Step     0 (   0.1s) | loss: 2.3124
Step   100 (   5.2s) | loss: 1.8765
Step   200 (  10.3s) | loss: 1.5432
...
Step  3000 ( 150.7s) | loss: 0.4321
--------------------------------------------------------------------------------
Training complete! Logs saved to logs/ranking_1234567890
================================================================================
```

### Comparison Output
```
================================================================================
Comparing HTSU Variants
================================================================================

Training HTSU (SiLU)...
  Final loss (last 100 steps): 0.4123

Training HTSU (Softmax)...
  Final loss (last 100 steps): 0.4567

Training HTSU (FFN)...
  Final loss (last 100 steps): 0.5234

================================================================================
Comparison Results:
  HTSU (SiLU)        : 0.4123
  HTSU (Softmax)     : 0.4567
  HTSU (FFN)         : 0.5234
================================================================================
```

## Next Steps

After running these examples:

1. **Modify hyperparameters** in the training scripts to see their effects
2. **Implement your own dataset** by following the structure in `synthetic_data.py`
3. **Add custom metrics** in `htsu/metrics.py`
4. **Experiment with different HTSU variants** (positional-only, softmax, FFN)
5. **Visualize results** using the utilities in `htsu/visualization.py`

## Common Issues

### Out of Memory
If you run out of memory, try:
- Reducing `d_model` (embedding dimension)
- Reducing `vocab_size` or `num_records`
- Using shorter sequences

### Slow Training
To speed up training:
- Use GPU/TPU by installing `jax[cuda]` or `jax[tpu]`
- Reduce `num_steps`
- Use smaller `max_distance` and `num_buckets`
- Enable XLA compilation (automatic with JAX)

### NaN Losses
If you see NaN losses:
- Reduce learning rate
- Check input data for NaN/Inf values
- Try gradient clipping (add to optimizer)

## Additional Resources

- **Main README**: `../README.md` - Full documentation
- **Tests**: `../tests/` - More usage examples in test cases
- **Scripts**: `../scripts/` - Research experiment scripts
