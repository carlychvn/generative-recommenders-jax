import jax.numpy as jnp
from htsu.sequencing import (
    interleave_items_and_actions,
    build_positions,
    build_timestamps,
    build_ranking_target_mask,
    build_retrieval_pairs,
    build_retrieval_timestamps,
    build_retrieval_positions
)


def test_interleave_shape_and_order():
    item_tokens = jnp.array([10, 20])
    action_tokens = jnp.array([1, 2])

    result = interleave_items_and_actions(item_tokens, action_tokens)

    expected = jnp.array([10, 1, 20, 2])
    assert result.shape == (4,)
    assert jnp.array_equal(result, expected)


def test_interleave_preserves_feature_dim():
    item_tokens = jnp.array([[1.0, 2.0], [5.0, 6.0]])
    action_tokens = jnp.array([[3.0, 4.0], [7.0, 8.0]])

    result = interleave_items_and_actions(item_tokens, action_tokens)

    expected = jnp.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])
    assert result.shape == (4, 2)
    assert jnp.array_equal(result, expected)


def test_positions_are_sequential():
    positions = build_positions(3)

    expected = jnp.array([0, 1, 2, 3, 4, 5])
    assert positions.shape == (6,)
    assert jnp.array_equal(positions, expected)


def test_timestamps_shared_within_pair():
    event_time = jnp.array([5.0, 12.0, 20.0])

    timestamps = build_timestamps(event_time)

    expected = jnp.array([5.0, 5.0, 12.0, 12.0, 20.0, 20.0])
    assert timestamps.shape == (6,)
    assert jnp.array_equal(timestamps, expected)

    for i in range(len(event_time)):
        assert timestamps[2 * i] == timestamps[2 * i + 1]

    assert jnp.all(jnp.diff(event_time) > 0)


def test_ranking_target_mask_marks_item_positions():
    mask = build_ranking_target_mask(3)

    expected = jnp.array([True, False, True, False, True, False])
    assert mask.shape == (6,)
    assert jnp.array_equal(mask, expected)

def test_build_retrieval_pairs_shape_and_values():
    item_tokens = jnp.array([[1.0, 2.0], [5.0, 6.0]])
    action_tokens = jnp.array([[3.0, 4.0], [7.0, 8.0]])

    result = build_retrieval_pairs(item_tokens, action_tokens)

    expected = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])
    assert result.shape == (2, 4)
    assert jnp.array_equal(result, expected)

def test_build_retrieval_positions():
    positions = build_retrieval_positions(3)
    assert jnp.array_equal(positions, jnp.array([0, 1, 2]))


def test_build_retrieval_timestamps():
    event_time = jnp.array([5.0, 12.0, 20.0])
    timestamps = build_retrieval_timestamps(event_time)
    assert jnp.array_equal(timestamps, event_time)


if __name__ == "__main__":
    test_interleave_shape_and_order()
    test_interleave_preserves_feature_dim()
    test_positions_are_sequential()
    test_timestamps_shared_within_pair()
    test_ranking_target_mask_marks_item_positions()
    test_build_retrieval_pairs_shape_and_values()
    test_build_retrieval_positions()
    test_build_retrieval_timestamps()
    print("all checks ran")
