from flax import nnx
import jax
import jax.numpy as jnp
from htsu.layers import PointwiseProjection, RelativeAttentionBias, SpatialAggregation, PointwiseTransformation, HTSULayer, ActionHead, ItemEmbedding, SpatialAggregationSoftmax, HTSULayerFFN

def test_pointwise_projection_shapes():
    rngs = nnx.Rngs(0)
    d_model = 8
    pointwise_proj = PointwiseProjection(d_model, 4 * d_model, rngs)

    batch, seq_len = 2, 5
    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))

    U, Q, V, K = pointwise_proj(x)
    expected_shape = (batch, seq_len, d_model)
    assert U.shape == expected_shape
    assert Q.shape == expected_shape
    assert V.shape == expected_shape
    assert K.shape == expected_shape


def test_pointwise_projection_no_nans():
    rngs = nnx.Rngs(0)
    d_model = 8
    pointwise_proj = PointwiseProjection(d_model, 4 * d_model, rngs)

    batch, seq_len = 2, 5
    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))

    U, Q, V, K = pointwise_proj(x)
    assert not jnp.any(jnp.isnan(U))
    assert not jnp.any(jnp.isnan(Q))
    assert not jnp.any(jnp.isnan(V))
    assert not jnp.any(jnp.isnan(K))


def test_relative_attention_bias_shape():
    rngs = nnx.Rngs(0)
    num_buckets = 8
    max_distance = 16
    seq_len = 5

    rab = RelativeAttentionBias(num_buckets, rngs, max_distance)

    positions = jnp.arange(seq_len)
    timestamps = jnp.arange(seq_len) * 10.0

    relative_attention_bias = rab(positions, timestamps)

    expected_shape = (seq_len, seq_len)
    assert relative_attention_bias.shape == expected_shape


def test_relative_attention_bias_varies_by_distance():
    rngs = nnx.Rngs(0)
    num_buckets = 8
    max_distance = 16
    seq_len = 5

    rab = RelativeAttentionBias(num_buckets, rngs, max_distance)

    positions = jnp.arange(seq_len)
    timestamps = jnp.arange(seq_len) * 10.0

    relative_attention_bias = rab(positions, timestamps)
    assert relative_attention_bias[4, 0] != relative_attention_bias[4, 3]


def _reference_relative_bucket(relative_distance, num_buckets, max_distance):
    """Ground-truth bucketing, written independently from RelativeAttentionBias's
    own implementation, to check our bucket math against (T5-style, symmetric)."""
    relative_distance = jnp.abs(relative_distance.astype(jnp.float32))
    max_exact = num_buckets // 2
    is_small = relative_distance < max_exact

    val_if_large = max_exact + (
        jnp.log(jnp.maximum(relative_distance, 1e-6) / max_exact)
        / jnp.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)

    return jnp.where(is_small, relative_distance.astype(jnp.int32), val_if_large)


def test_relative_attention_bias_bucket_matches_reference():
    rngs = nnx.Rngs(0)
    num_buckets = 8
    max_distance = 16

    rab = RelativeAttentionBias(num_buckets, rngs, max_distance)

    positions = jnp.arange(50)
    relative_position = positions[None, :] - positions[:, None]

    got = rab.relative_distance_bucket(relative_position, num_buckets, max_distance)
    expected = _reference_relative_bucket(relative_position, num_buckets, max_distance)

    assert jnp.array_equal(got, expected)


def test_relative_attention_bias_bucket_boundaries():
    rngs = nnx.Rngs(0)
    num_buckets = 8
    max_distance = 16
    max_exact = num_buckets // 2

    rab = RelativeAttentionBias(num_buckets, rngs, max_distance)

    small_distances = jnp.arange(max_exact)
    buckets = rab.relative_distance_bucket(small_distances, num_buckets, max_distance)
    assert jnp.array_equal(buckets, small_distances)

    far_distances = jnp.array([max_distance, max_distance * 10, max_distance * 100])
    far_buckets = rab.relative_distance_bucket(far_distances, num_buckets, max_distance)
    assert jnp.all(far_buckets == num_buckets - 1)


def test_spatial_aggregation_shape():
    batch, seq_len, d_model = 2, 5, 8

    Q = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    K = jax.random.normal(jax.random.PRNGKey(1), (batch, seq_len, d_model))
    V = jax.random.normal(jax.random.PRNGKey(2), (batch, seq_len, d_model))
    rab = jax.random.normal(jax.random.PRNGKey(3), (seq_len, seq_len))

    spatial_agg = SpatialAggregation()
    output = spatial_agg(Q, K, V, rab)

    assert output.shape == (batch, seq_len, d_model)


def test_spatial_aggregation_nan():
    batch, seq_len, d_model = 2, 5, 8

    Q = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    K = jax.random.normal(jax.random.PRNGKey(1), (batch, seq_len, d_model))
    V = jax.random.normal(jax.random.PRNGKey(2), (batch, seq_len, d_model))
    rab = jax.random.normal(jax.random.PRNGKey(3), (seq_len, seq_len))

    spatial_agg = SpatialAggregation()
    output = spatial_agg(Q, K, V, rab)

    assert not jnp.any(jnp.isnan(output))
    assert not jnp.any(jnp.isnan(Q))
    assert not jnp.any(jnp.isnan(K))
    assert not jnp.any(jnp.isnan(V))
    assert not jnp.any(jnp.isnan(rab))


def test_spatial_aggregation_casual_mask_blocks_future():
    seq_len, d_model = 5, 8

    Q = jax.random.normal(jax.random.PRNGKey(0), (seq_len, d_model))
    K = jax.random.normal(jax.random.PRNGKey(1), (seq_len, d_model))
    V = jax.random.normal(jax.random.PRNGKey(2), (seq_len, d_model))
    rab = jnp.zeros((seq_len, seq_len))

    spatial_agg = SpatialAggregation()
    out1 = spatial_agg(Q, K, V, rab)

    K2 = K.at[-1].add(100.0)
    V2 = V.at[-1].add(100.0)
    out2 = spatial_agg(Q, K2, V2, rab)

    assert jnp.allclose(out1[:-1], out2[:-1], atol=1e-5)
    assert not jnp.allclose(out1[-1], out2[-1], atol=1e-5)


def test_pointwise_transformation_shape():
    batch, seq_len, d_model = 2, 5, 8

    rngs = nnx.Rngs(0)
    pointwise_trans = PointwiseTransformation(d_model, rngs)

    attn_output = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    U = jax.random.normal(jax.random.PRNGKey(1), (batch, seq_len, d_model))

    output = pointwise_trans(attn_output, U)

    assert output.shape == (batch, seq_len, d_model)


def test_pointwise_transformation_no_nans():
    batch, seq_len, d_model = 2, 5, 8

    rngs = nnx.Rngs(0)
    pointwise_trans = PointwiseTransformation(d_model, rngs)

    attn_output = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    U = jax.random.normal(jax.random.PRNGKey(1), (batch, seq_len, d_model))

    output = pointwise_trans(attn_output, U)

    assert not jnp.any(jnp.isnan(output))


def test_hstu_layer_shape_and_nans():
    batch, seq_len, d_model = 2, 5, 8
    num_buckets, max_distance = 8, 16

    rngs = nnx.Rngs(0)
    layer = HTSULayer(d_model, num_buckets, max_distance, rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    positions = jnp.arange(seq_len)
    timestamps = jnp.arange(seq_len) * 10.0

    output = layer(x, positions, timestamps)

    assert output.shape == (batch, seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))


def test_hstu_layer_gradient_flow():
    batch, seq_len, d_model = 2, 5, 8
    num_buckets, max_distance = 8, 16

    rngs = nnx.Rngs(0)
    layer = HTSULayer(d_model, num_buckets, max_distance, rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    positions = jnp.arange(seq_len)
    timestamps = jnp.arange(seq_len) * 10.0

    def loss_fn(model, x, positions, timestamps):
        y_pred = model(x, positions, timestamps)
        return y_pred.sum()

    loss, grads = nnx.value_and_grad(loss_fn)(layer, x, positions, timestamps)
    assert jnp.any(grads['rab']['pos_bias_table'][...] != 0)


def test_hstu_layer_casual_consistency():
    seq_len, d_model = 6, 8
    num_buckets, max_distance = 8, 16

    rngs = nnx.Rngs(0)
    layer = HTSULayer(d_model, num_buckets, max_distance, rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, d_model))
    positions = jnp.arange(seq_len)
    timestamps = jnp.arange(seq_len) * 10.0

    full_output = layer(x, positions, timestamps)

    truncated_len = 4
    truncated_output = layer(
        x[:truncated_len], positions[:truncated_len], timestamps[:truncated_len]
    )

    assert jnp.allclose(
        full_output[:truncated_len], truncated_output, atol=1e-4
    )

def test_action_head_shape():
    rngs = nnx.Rngs(0)
    d_model, num_actions = 8, 4
    head = ActionHead(d_model, num_actions, rngs)

    n, d = 5, d_model
    x = jax.random.normal(jax.random.PRNGKey(0), (n, d))

    logits = head(x)
    assert logits.shape == (n, num_actions)

def test_item_embedding_shape():
    rngs = nnx.Rngs(0)
    num_items, d_model = 100, 8
    item_embed = ItemEmbedding(num_items, d_model, rngs)

    item_ids = jnp.array([3, 17, 42])
    embeddings = item_embed(item_ids)

    assert embeddings.shape == (3, d_model)

def test_spatial_aggregation_softmax_causal_mask_blocks_future():
    seq_len, d_model = 5, 8

    Q = jax.random.normal(jax.random.PRNGKey(0), (seq_len, d_model))
    K = jax.random.normal(jax.random.PRNGKey(1), (seq_len, d_model))
    V = jax.random.normal(jax.random.PRNGKey(2), (seq_len, d_model))
    rab = jnp.zeros((seq_len, seq_len))

    spatial_agg = SpatialAggregationSoftmax()
    out1 = spatial_agg(Q, K, V, rab)

    K2 = K.at[-1].add(100.0)
    V2 = V.at[-1].add(100.0)
    out2 = spatial_agg(Q, K2, V2, rab)

    assert jnp.allclose(out1[:-1], out2[:-1], atol=1e-4)
    assert not jnp.allclose(out1[-1], out2[-1], atol=1e-4)

def test_htsu_layer_ffn_shape_and_nans():
    batch, seq_len, d_model = 2, 5, 8
    num_buckets, max_distance = 8, 16

    rngs = nnx.Rngs(0)
    layer = HTSULayerFFN(d_model, num_buckets, max_distance, rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    positions = jnp.arange(seq_len)
    timestamps = jnp.arange(seq_len) * 10.0

    output = layer(x, positions, timestamps)

    assert output.shape == (batch, seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))

if __name__ == "__main__":
    test_pointwise_projection_shapes()
    test_pointwise_projection_no_nans()
    test_relative_attention_bias_shape()
    test_relative_attention_bias_varies_by_distance()
    test_relative_attention_bias_bucket_matches_reference()
    test_relative_attention_bias_bucket_boundaries()
    test_spatial_aggregation_shape()
    test_spatial_aggregation_nan()
    test_spatial_aggregation_casual_mask_blocks_future()
    test_pointwise_transformation_shape()
    test_pointwise_transformation_no_nans()
    test_hstu_layer_shape_and_nans()
    test_hstu_layer_gradient_flow()
    test_hstu_layer_casual_consistency()
    test_action_head_shape()
    test_item_embedding_shape()
    test_spatial_aggregation_softmax_causal_mask_blocks_future()
    test_htsu_layer_ffn_shape_and_nans()
    print("all checks ran")
