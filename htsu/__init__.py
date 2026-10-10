"""
JAX-HTSU: Hierarchical Temporal Sequential User Modeling

A JAX/Flax reimplementation of HTSU for sequential recommendation and user behavior prediction.
"""

__version__ = "0.1.0"

# Core layer components
from htsu.layers import (
    # Main HTSU variants
    HTSULayer,
    HTSULayerPositionalOnly,
    HTSULayerSoftmax,
    HTSULayerSoftmaxPositionalOnly,
    HTSULayerFFN,
    HTSULayerFFNPositionalOnly,

    # Baseline
    VanillaTransformerBlock,

    # Individual components
    PointwiseProjection,
    RelativeAttentionBias,
    RelativeAttentionBiasPositionalOnly,
    SpatialAggregation,
    SpatialAggregationSoftmax,
    PointwiseTransformation,
    PointwiseTransformationFFN,

    # Task-specific heads
    ActionHead,
    ItemEmbedding,
    RetrievalDownProjection,

    # Utilities
    compute_relative_distance_bucket,
)

# Loss functions
from htsu.losses import (
    ranking_loss,
    retrieval_loss,
)

# Sequencing utilities
from htsu.sequencing import (
    interleave_items_and_actions,
    build_positions,
    build_timestamps,
    build_ranking_target_mask,
    build_retrieval_pairs,
    build_retrieval_positions,
    build_retrieval_timestamps,
)

# Stochastic length sampling
from htsu.stochastic_length import (
    compute_position_importance,
    stochastic_length_sample,
)

# Synthetic data generation
from htsu.synthetic_data import (
    generate_dataset,
    generate_record,
)

# Evaluation metrics
from htsu.metrics import (
    ranking_accuracy,
    mean_reciprocal_rank,
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    compute_all_ranking_metrics,
    compute_all_retrieval_metrics,
    compute_confusion_matrix,
)

# Visualization utilities
from htsu.visualization import (
    load_training_metrics,
    print_training_summary,
    print_ascii_plot,
    visualize_training_log,
    print_confusion_matrix,
    print_metric_comparison,
)

__all__ = [
    # Version
    "__version__",

    # Main HTSU layers
    "HTSULayer",
    "HTSULayerPositionalOnly",
    "HTSULayerSoftmax",
    "HTSULayerSoftmaxPositionalOnly",
    "HTSULayerFFN",
    "HTSULayerFFNPositionalOnly",
    "VanillaTransformerBlock",

    # Layer components
    "PointwiseProjection",
    "RelativeAttentionBias",
    "RelativeAttentionBiasPositionalOnly",
    "SpatialAggregation",
    "SpatialAggregationSoftmax",
    "PointwiseTransformation",
    "PointwiseTransformationFFN",

    # Task heads
    "ActionHead",
    "ItemEmbedding",
    "RetrievalDownProjection",

    # Utilities
    "compute_relative_distance_bucket",

    # Losses
    "ranking_loss",
    "retrieval_loss",

    # Sequencing
    "interleave_items_and_actions",
    "build_positions",
    "build_timestamps",
    "build_ranking_target_mask",
    "build_retrieval_pairs",
    "build_retrieval_positions",
    "build_retrieval_timestamps",

    # Stochastic sampling
    "compute_position_importance",
    "stochastic_length_sample",

    # Data generation
    "generate_dataset",
    "generate_record",

    # Metrics
    "ranking_accuracy",
    "mean_reciprocal_rank",
    "recall_at_k",
    "precision_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "compute_all_ranking_metrics",
    "compute_all_retrieval_metrics",
    "compute_confusion_matrix",

    # Visualization
    "load_training_metrics",
    "print_training_summary",
    "print_ascii_plot",
    "visualize_training_log",
    "print_confusion_matrix",
    "print_metric_comparison",
]
