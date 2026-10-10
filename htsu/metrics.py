"""Evaluation metrics for HTSU models.

Provides standard metrics for sequential recommendation:
- Ranking metrics: Accuracy, MRR (Mean Reciprocal Rank), NDCG
- Retrieval metrics: Recall@K, Precision@K, Hit Rate
"""

import jax
import jax.numpy as jnp
from typing import Dict


def ranking_accuracy(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute top-1 accuracy for ranking task.

    Args:
        predictions: (N, num_actions) logits
        targets: (N,) ground truth action indices

    Returns:
        Accuracy as float between 0 and 1
    """
    predicted_actions = jnp.argmax(predictions, axis=-1)
    correct = predicted_actions == targets
    return jnp.mean(correct).item()


def mean_reciprocal_rank(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        predictions: (N, num_actions) logits or scores
        targets: (N,) ground truth indices

    Returns:
        MRR score
    """
    # Get ranks of true targets
    sorted_indices = jnp.argsort(-predictions, axis=-1)  # Descending order
    ranks = jnp.where(sorted_indices == targets[:, None])[1] + 1  # 1-indexed ranks
    reciprocal_ranks = 1.0 / ranks
    return jnp.mean(reciprocal_ranks).item()


def recall_at_k(predictions: jnp.ndarray, targets: jnp.ndarray, k: int) -> float:
    """Compute Recall@K for retrieval task.

    Args:
        predictions: (N, num_items) similarity scores
        targets: (N,) ground truth item indices
        k: Number of top predictions to consider

    Returns:
        Recall@K score
    """
    top_k_indices = jnp.argsort(-predictions, axis=-1)[:, :k]
    hits = jnp.any(top_k_indices == targets[:, None], axis=-1)
    return jnp.mean(hits).item()


def precision_at_k(predictions: jnp.ndarray, targets: jnp.ndarray, k: int) -> float:
    """Compute Precision@K for retrieval task.

    Args:
        predictions: (N, num_items) similarity scores
        targets: (N,) ground truth item indices
        k: Number of top predictions to consider

    Returns:
        Precision@K score
    """
    top_k_indices = jnp.argsort(-predictions, axis=-1)[:, :k]
    hits = jnp.sum(top_k_indices == targets[:, None], axis=-1)
    return jnp.mean(hits / k).item()


def ndcg_at_k(predictions: jnp.ndarray, targets: jnp.ndarray, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain @ K.

    Args:
        predictions: (N, num_items) similarity scores
        targets: (N,) ground truth item indices
        k: Number of top predictions to consider

    Returns:
        NDCG@K score
    """
    top_k_indices = jnp.argsort(-predictions, axis=-1)[:, :k]

    # Binary relevance: 1 if in top-k and correct, 0 otherwise
    relevance = (top_k_indices == targets[:, None]).astype(jnp.float32)

    # Discount factors: 1/log2(rank + 1)
    positions = jnp.arange(1, k + 1)
    discounts = 1.0 / jnp.log2(positions + 1)

    # DCG
    dcg = jnp.sum(relevance * discounts[None, :], axis=-1)

    # IDCG (ideal DCG - assumes 1 relevant item at position 0)
    idcg = 1.0 / jnp.log2(2.0)  # 1/log2(1+1)

    # NDCG
    ndcg = dcg / idcg
    return jnp.mean(ndcg).item()


def hit_rate_at_k(predictions: jnp.ndarray, targets: jnp.ndarray, k: int) -> float:
    """Compute Hit Rate @ K (same as Recall@K for single target).

    Args:
        predictions: (N, num_items) similarity scores
        targets: (N,) ground truth item indices
        k: Number of top predictions to consider

    Returns:
        Hit Rate@K score
    """
    return recall_at_k(predictions, targets, k)


def compute_all_ranking_metrics(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict[str, float]:
    """Compute all ranking metrics.

    Args:
        predictions: (N, num_actions) logits
        targets: (N,) ground truth action indices

    Returns:
        Dictionary of metric names to values
    """
    return {
        "accuracy": ranking_accuracy(predictions, targets),
        "mrr": mean_reciprocal_rank(predictions, targets),
        "ndcg@5": ndcg_at_k(predictions, targets, 5),
        "ndcg@10": ndcg_at_k(predictions, targets, 10),
    }


def compute_all_retrieval_metrics(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict[str, float]:
    """Compute all retrieval metrics.

    Args:
        predictions: (N, num_items) similarity scores
        targets: (N,) ground truth item indices

    Returns:
        Dictionary of metric names to values
    """
    return {
        "recall@1": recall_at_k(predictions, targets, 1),
        "recall@5": recall_at_k(predictions, targets, 5),
        "recall@10": recall_at_k(predictions, targets, 10),
        "recall@20": recall_at_k(predictions, targets, 20),
        "ndcg@5": ndcg_at_k(predictions, targets, 5),
        "ndcg@10": ndcg_at_k(predictions, targets, 10),
        "mrr": mean_reciprocal_rank(predictions, targets),
    }


def compute_confusion_matrix(predictions: jnp.ndarray, targets: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """Compute confusion matrix for ranking predictions.

    Args:
        predictions: (N, num_actions) logits
        targets: (N,) ground truth action indices
        num_classes: Number of action classes

    Returns:
        (num_classes, num_classes) confusion matrix
    """
    predicted_actions = jnp.argmax(predictions, axis=-1)

    confusion = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)

    # Build confusion matrix
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            mask = (targets == true_class) & (predicted_actions == pred_class)
            confusion = confusion.at[true_class, pred_class].add(jnp.sum(mask))

    return confusion
