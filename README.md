# Generative Recommenders in JAX

A JAX implementation of HSTU (Hierarchical Sequential Transduction Units), the generative recommender architecture from Meta's *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations* (Zhai et al., ICML 2024).

> **Work in progress.** Full documentation, training instructions, and results coming soon.

## Overview

Traditional deep learning recommendation models (DLRMs) score user-item pairs using large sets of hand-engineered features and tend to plateau as compute grows. Generative Recommenders (GRs) instead treat recommendation as a sequential transduction problem: a user's history of items and actions is modeled as a single sequence, and ranking and retrieval become next-token prediction tasks.

HSTU is the encoder built for this setting. Its main design choices are:

- **Pointwise aggregated attention:** replaces softmax normalization, which suits non-stationary item vocabularies and preserves signal about the intensity of user preferences.
- **Fused gating layer:** an elementwise gate replaces the feed-forward block, cutting linear layers and activation memory.
- **Relative attention bias:** incorporates both positional and temporal information.
- **Stochastic Length:** randomly drops parts of long user histories during training to reduce attention cost without hurting quality.

This repo reimplements the HSTU encoder and generative training setup in JAX for functional, JIT-compiled training.

## Tech Stack

| Component | Tools |
|-----------|-------|
| Framework | JAX, Flax |
| Optimization | Optax |
| Data | NumPy, pandas |
| Tooling | Python 3.11+, pytest |

## Datasets and Evaluation

Following the paper's public-dataset benchmarks: MovieLens-1M, MovieLens-20M, and Amazon Books, compared against a SASRec baseline. Metrics are Hit Rate@K and NDCG@K over the full item corpus.

## References

- Zhai et al. *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.* ICML 2024. [arXiv:2402.17152](https://arxiv.org/abs/2402.17152)
- Official implementation: [facebookresearch/generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
