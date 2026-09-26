import jax
import jax.numpy as jnp
from htsu.stochastic_length import compute_position_importance, stochastic_length_sample


def test_compute_position_importance_shape():
    item_ids = jnp.array([5, 5, 8, 2, 5, 9])
    importance = compute_position_importance(item_ids)
    assert importance.shape == (6,)
    assert jnp.all(importance > 0)


def test_compute_position_importance_favors_rare_items():
    item_ids = jnp.array([5, 5, 5, 2])
    importance = compute_position_importance(item_ids)
    assert importance[3] > importance[0]


def test_stochastic_length_sample_shape_and_order():
    item_ids = jnp.arange(20)
    positions = jnp.arange(20)
    rng = jax.random.PRNGKey(0)

    kept_items, kept_positions = stochastic_length_sample(item_ids, positions, keep_ratio=0.5, rng=rng)

    assert kept_items.shape == (10,)
    assert kept_positions.shape == (10,)
    assert jnp.all(jnp.diff(kept_positions) > 0)


def test_stochastic_length_sample_preserves_original_position_values():
    item_ids = jnp.arange(20)
    positions = jnp.arange(100, 120)
    rng = jax.random.PRNGKey(0)

    _, kept_positions = stochastic_length_sample(item_ids, positions, keep_ratio=0.3, rng=rng)

    for p in kept_positions:
        assert p in positions


if __name__ == "__main__":
    test_compute_position_importance_shape()
    test_compute_position_importance_favors_rare_items()
    test_stochastic_length_sample_shape_and_order()
    test_stochastic_length_sample_preserves_original_position_values()
    print("all checks ran")
