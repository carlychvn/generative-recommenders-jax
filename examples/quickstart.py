"""Quickstart example for HTSU.

A minimal example showing how to:
1. Create synthetic data
2. Initialize an HTSU model
3. Run a forward pass
4. Compute losses
"""

import jax
import jax.numpy as jnp
from flax import nnx

from htsu.layers import HTSULayer, ActionHead, ItemEmbedding, RetrievalDownProjection
from htsu.sequencing import (
    interleave_items_and_actions,
    build_positions,
    build_timestamps,
    build_ranking_target_mask,
    build_retrieval_pairs,
)
from htsu.losses import ranking_loss, retrieval_loss
from htsu.synthetic_data import generate_dataset


def quickstart_ranking():
    """Minimal example for ranking task."""
    print("=" * 80)
    print("HTSU Quickstart: Ranking Task")
    print("=" * 80)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    rng = jax.random.PRNGKey(42)
    dataset = generate_dataset(
        num_records=10,
        vocab_size=1000,
        num_categories=20,
        rng=rng
    )
    print(f"   Generated {len(dataset)} records")

    # Pick one record
    record = dataset[0]
    item_ids = record["item_ids"]
    timestamps = record["timestamps"]
    action_labels = record["action_labels"]
    print(f"   Sequence length: {len(item_ids)} events")

    # 2. Create embeddings (in practice, use learned embeddings)
    print("\n2. Creating embeddings...")
    d_model = 64
    rng, embed_rng = jax.random.split(rng)
    item_embeddings = jax.random.normal(embed_rng, (len(item_ids), d_model))
    action_embeddings = jax.random.normal(embed_rng, (len(action_labels), d_model))
    print(f"   Embedding dimension: {d_model}")

    # 3. Build interleaved sequence
    print("\n3. Building interleaved sequence...")
    sequence = interleave_items_and_actions(item_embeddings, action_embeddings)
    positions = build_positions(len(item_ids))
    timestamps_interleaved = build_timestamps(timestamps)
    target_mask = build_ranking_target_mask(len(item_ids))
    print(f"   Interleaved sequence shape: {sequence.shape}")

    # 4. Initialize model
    print("\n4. Initializing HTSU model...")
    num_actions = 10
    num_buckets = 16
    max_distance = 64
    rngs = nnx.Rngs(42)

    htsu_layer = HTSULayer(d_model, num_buckets, max_distance, rngs)
    action_head = ActionHead(d_model, num_actions, rngs)
    print("   Model initialized")

    # 5. Forward pass
    print("\n5. Running forward pass...")
    output = htsu_layer(sequence, positions, timestamps_interleaved)
    print(f"   Output shape: {output.shape}")

    # 6. Compute loss
    print("\n6. Computing ranking loss...")
    loss = ranking_loss(output, target_mask, action_labels, action_head)
    print(f"   Loss: {loss:.4f}")

    print("\n" + "=" * 80)
    print("Ranking example complete!")
    print("=" * 80)


def quickstart_retrieval():
    """Minimal example for retrieval task."""
    print("\n" + "=" * 80)
    print("HTSU Quickstart: Retrieval Task")
    print("=" * 80)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    rng = jax.random.PRNGKey(42)
    dataset = generate_dataset(
        num_records=10,
        vocab_size=1000,
        num_categories=20,
        rng=rng
    )

    record = dataset[0]
    item_ids = record["item_ids"]
    timestamps = record["timestamps"]
    print(f"   Sequence length: {len(item_ids)} events")

    # 2. Create embeddings
    print("\n2. Creating embeddings...")
    d_model = 64
    rng, embed_rng = jax.random.split(rng)
    item_embeddings = jax.random.normal(embed_rng, (len(item_ids), d_model))
    action_embeddings = jax.random.normal(embed_rng, (len(item_ids), d_model))

    # 3. Build pairs for retrieval
    print("\n3. Building retrieval pairs...")
    positions = jnp.arange(len(item_ids))

    # 4. Initialize model
    print("\n4. Initializing HTSU retrieval model...")
    num_items = 1000
    num_buckets = 16
    max_distance = 64
    rngs = nnx.Rngs(42)

    down_projection = RetrievalDownProjection(2 * d_model, d_model, rngs)
    htsu_layer = HTSULayer(d_model, num_buckets, max_distance, rngs)
    item_embedding = ItemEmbedding(num_items, d_model, rngs)
    print("   Model initialized")

    # 5. Forward pass
    print("\n5. Running forward pass...")
    pairs = build_retrieval_pairs(item_embeddings, action_embeddings)
    down_projected = down_projection(pairs)
    output = htsu_layer(down_projected, positions, timestamps)
    print(f"   Output shape: {output.shape}")

    # 6. Compute loss
    print("\n6. Computing retrieval loss...")
    positive_item_ids = item_ids  # Next items (simplified)
    loss = retrieval_loss(output, item_embedding, positive_item_ids)
    print(f"   Loss: {loss:.4f}")

    print("\n" + "=" * 80)
    print("Retrieval example complete!")
    print("=" * 80)


def main():
    """Run both quickstart examples."""
    print("\n" + "=" * 80)
    print("JAX-HTSU Quickstart Examples")
    print("=" * 80)
    print("\nThis script demonstrates basic usage of HTSU for both")
    print("ranking and retrieval tasks.\n")

    # Run ranking example
    quickstart_ranking()

    # Run retrieval example
    quickstart_retrieval()

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  - See examples/train_synthetic.py for full training example")
    print("  - See examples/evaluate.py for evaluation metrics")
    print("  - See README.md for detailed documentation")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
