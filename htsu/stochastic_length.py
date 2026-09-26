import jax
import jax.numpy as jnp


def compute_position_importance(item_ids):
    """
    Feature-weighted importance score per position: recency (later
    positions weighted higher) combined with inverse frequency within
    this sequence (rarer items weighted higher, to preserve diversity
    rather than just recency alone).

    item_ids: (seq_len,) int array

    Returns: (seq_len,) float array of importance weights (unnormalized)
    """
    seq_len = item_ids.shape[0]

    recency = jnp.arange(1, seq_len + 1) / seq_len

    matches = item_ids[:, None] == item_ids[None, :]
    freq = jnp.sum(matches, axis=-1).astype(jnp.float32)
    rarity = 1.0 / freq

    importance = recency * rarity
    return importance


def stochastic_length_sample(item_ids, positions, keep_ratio, rng):
    """
    Subsample a sequence down to keep_ratio of its original length,
    weighted by compute_position_importance, while preserving original
    position VALUES (not renumbering) so relative distances stay real.

    item_ids: (seq_len,) int array
    positions: (seq_len,) int array — original position indices
    keep_ratio: float in (0, 1]
    rng: PRNGKey

    Returns: (kept_item_ids, kept_positions), both length round(seq_len * keep_ratio)
    """
    seq_len = item_ids.shape[0]
    keep_len = max(1, int(seq_len * keep_ratio))

    importance = compute_position_importance(item_ids)
    probs = importance / jnp.sum(importance)

    sampled_indices = jax.random.choice(
        rng, seq_len, shape=(keep_len,), replace=False, p=probs
    )

    sorted_indices = jnp.sort(sampled_indices)

    kept_item_ids = item_ids[sorted_indices]
    kept_positions = positions[sorted_indices]

    return kept_item_ids, kept_positions
