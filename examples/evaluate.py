"""Evaluation script for HTSU models.

Demonstrates how to evaluate trained models on test data with comprehensive metrics.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from htsu.layers import HTSULayer, ActionHead
from htsu.sequencing import (
    interleave_items_and_actions,
    build_positions,
    build_timestamps,
    build_ranking_target_mask,
)
from htsu.synthetic_data import generate_dataset
from htsu.metrics import (
    ranking_accuracy,
    mean_reciprocal_rank,
    ndcg_at_k,
    compute_all_ranking_metrics,
    compute_confusion_matrix,
)
from htsu.visualization import print_confusion_matrix, print_metric_comparison


def evaluate_ranking_model(model, dataset, d_model, seed=42):
    """Evaluate a ranking model on a dataset.

    Args:
        model: Trained HTSU model
        dataset: List of data records
        d_model: Model dimension
        seed: Random seed for embeddings

    Returns:
        Dictionary of evaluation metrics
    """
    all_predictions = []
    all_targets = []

    rng = jax.random.PRNGKey(seed)

    print("Evaluating on dataset...")
    for i, record in enumerate(dataset):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(dataset)}")

        # Extract data
        item_ids = record["item_ids"]
        timestamps = record["timestamps"]
        action_labels = record["action_labels"]

        # Create embeddings
        rng, embed_rng = jax.random.split(rng)
        item_embeddings = jax.random.normal(embed_rng, (len(item_ids), d_model))
        action_embeddings = jax.random.normal(embed_rng, (len(action_labels), d_model))

        # Build sequences
        sequence = interleave_items_and_actions(item_embeddings, action_embeddings)
        positions = build_positions(len(item_ids))
        timestamps_interleaved = build_timestamps(timestamps)
        target_mask = build_ranking_target_mask(len(item_ids))

        # Get predictions
        output = model.htsu(sequence, positions, timestamps_interleaved)
        extracted = output[target_mask]
        logits = model.action_head(extracted)

        all_predictions.append(logits)
        all_targets.append(action_labels)

    # Concatenate all predictions
    predictions = jnp.concatenate(all_predictions, axis=0)
    targets = jnp.concatenate(all_targets, axis=0)

    # Compute metrics
    metrics = compute_all_ranking_metrics(predictions, targets)

    return metrics, predictions, targets


def main():
    """Run evaluation example."""
    print("=" * 80)
    print("HTSU Model Evaluation Example")
    print("=" * 80)

    # Configuration
    num_records = 100
    vocab_size = 5000
    num_categories = 50
    num_actions = 10
    d_model = 128
    num_buckets = 32
    max_distance = 128
    seed = 42

    print("\nGenerating test dataset...")
    rng = jax.random.PRNGKey(seed)
    test_dataset = generate_dataset(num_records, vocab_size, num_categories, rng)
    print(f"Generated {len(test_dataset)} test records")

    print("\nInitializing model...")
    rngs = nnx.Rngs(seed)

    class HTSURankingModel(nnx.Module):
        def __init__(self):
            self.htsu = HTSULayer(d_model, num_buckets, max_distance, rngs)
            self.action_head = ActionHead(d_model, num_actions, rngs)

    model = HTSURankingModel()
    print("Model initialized (untrained - for demonstration)")

    # Evaluate
    print("\n" + "-" * 80)
    metrics, predictions, targets = evaluate_ranking_model(model, test_dataset, d_model, seed)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    for metric_name, value in metrics.items():
        print(f"  {metric_name:20s}: {value:.4f}")
    print("=" * 80)

    # Compute and display confusion matrix
    print("\nComputing confusion matrix...")
    confusion = compute_confusion_matrix(predictions, targets, num_actions)
    print_confusion_matrix(confusion, class_names=[f"Action {i}" for i in range(num_actions)])

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
