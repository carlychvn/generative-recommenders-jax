def test_relative_attention_bias_positional_only_shape():
    rngs = nnx.Rngs(0)
    num_buckets = 8
    max_distance = 16
    seq_len = 5

    rab = RelativeAttentionBiasPositionalOnly(num_buckets, rngs, max_distance)

    positions = jnp.arange(seq_len)
    bias = rab(positions)

    assert bias.shape == (seq_len, seq_len)


def test_relative_attention_bias_positional_only_has_no_time_table():
    rngs = nnx.Rngs(0)
    rab = RelativeAttentionBiasPositionalOnly(num_buckets=8, rngs=rngs, max_distance=16)

    assert not hasattr(rab, "time_bias_table")


def test_relative_attention_bias_positional_only_matches_pos_component():
    num_buckets = 8
    max_distance = 16
    seq_len = 5
    positions = jnp.arange(seq_len)

    relative_position = positions[None, :] - positions[:, None]
    expected_buckets = compute_relative_distance_bucket(relative_position, num_buckets, max_distance)

    rngs = nnx.Rngs(0)
    rab = RelativeAttentionBiasPositionalOnly(num_buckets, rngs, max_distance)
    bias = rab(positions)

    expected_bias = rab.pos_bias_table[expected_buckets]
    assert jnp.array_equal(bias, expected_bias)


def test_htsu_layer_positional_only_shape_and_nans():
    batch, seq_len, d_model = 2, 5, 8
    num_buckets, max_distance = 8, 16

    rngs = nnx.Rngs(0)
    layer = HTSULayerPositionalOnly(d_model, num_buckets, max_distance, rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    positions = jnp.arange(seq_len)

    output = layer(x, positions)

    assert output.shape == (batch, seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))


def test_htsu_layer_softmax_positional_only_shape_and_nans():
    batch, seq_len, d_model = 2, 5, 8
    num_buckets, max_distance = 8, 16

    rngs = nnx.Rngs(0)
    layer = HTSULayerSoftmaxPositionalOnly(d_model, num_buckets, max_distance, rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, d_model))
    positions = jnp.arange(seq_len)

    output = layer(x, positions)

    assert output.shape == (batch, seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))


def test_htsu_layer_positional_only_causal_consistency():
    seq_len, d_model = 6, 8
    num_buckets, max_distance = 8, 16

    rngs = nnx.Rngs(0)
    layer = HTSULayerPositionalOnly(d_model, num_buckets, max_distance, rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, d_model))
    positions = jnp.arange(seq_len)

    full_output = layer(x, positions)

    truncated_len = 4
    truncated_output = layer(x[:truncated_len], positions[:truncated_len])

    assert jnp.allclose(full_output[:truncated_len], truncated_output, atol=1e-4)
