from flax import nnx
import jax
import jax.numpy as jnp


"""
    Pointwise Projection:
    takes one linear layer that widens the input, squashes it with SiLU then chops
    the result into four equal pieces called U, Q, V, K
"""
class PointwiseProjection(nnx.Module):
    def __init__(self, din, dout, rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)

    def __call__(self, x):
        projected = self.linear(x)
        silu_projected = jax.nn.silu(projected)

        U, Q, V, K = jnp.split(silu_projected, 4, axis=-1)

        return U, Q, V, K


def compute_relative_distance_bucket(relative_distance, num_buckets, max_distance):
    """
    Shared bucketing logic, used by both RelativeAttentionBias and
    RelativeAttentionBiasPositionalOnly so there is exactly one
    already-tested implementation, not two that could drift apart.
    """
    relative_distance = relative_distance.astype(jnp.float32)
    relative_distance = jnp.abs(relative_distance)

    max_exact = num_buckets // 2
    is_small = relative_distance < max_exact

    relative_distance_if_large = max_exact + (
        jnp.log(jnp.maximum(relative_distance, 1e-6) / max_exact)
        / jnp.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    relative_distance_if_large = jnp.minimum(
        relative_distance_if_large, num_buckets - 1
    )

    relative_buckets = jnp.where(is_small, relative_distance.astype(jnp.int32), relative_distance_if_large)
    return relative_buckets


"""
    RelativeAttentionBias:
    for any two tokens i and j in your sequence, it needs to produce a single learned number
    (a bias) based on how far apart they are i-j produce a single learned number and add that
    raw attention score before the SiLU
"""
class RelativeAttentionBias(nnx.Module):

    """
        One function f(i - j) → bucket_id for position differences
        One function f_t(t_i - t_j) → bucket_id for time differences (log-scale bucketing)
        Two small learned tables (β and β^t), each mapping bucket_id → a bias number
        Add both bias numbers together to get the final rab(i,j) that gets added to Q·Kᵀ
    """
    def __init__(self, num_buckets, rngs, max_distance):
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.pos_bias_table = nnx.Param(rngs.params.uniform((num_buckets,)))
        self.time_bias_table = nnx.Param(rngs.params.uniform((num_buckets,)))

    def relative_distance_bucket(self, relative_distance, num_buckets, max_distance):

        relative_distance = relative_distance.astype(jnp.float32)
        relative_distance = jnp.abs(relative_distance)

        max_exact = num_buckets // 2
        is_small = relative_distance < max_exact

        relative_distance_if_large = max_exact + (
            jnp.log(jnp.maximum(relative_distance, 1e-6) / max_exact)
            / jnp.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        relative_distance_if_large = jnp.minimum(
            relative_distance_if_large, num_buckets - 1
        )

        relative_buckets = jnp.where(is_small, relative_distance.astype(jnp.int32), relative_distance_if_large)
        return relative_buckets

    def __call__(self, positions, timestamps):
        """
        positions:  (seq_len,) int array, e.g. jnp.arange(seq_len)
        timestamps: (seq_len,) array of event times (for the temporal bias)
        Returns: (seq_len, seq_len) bias matrix to add to Q @ K^T
        """
        relative_position = positions[None, :] - positions[:, None]
        relative_time = timestamps[None, :] - timestamps[:, None]

        pos_buckets = self.relative_distance_bucket(
            relative_position, self.num_buckets, self.max_distance
        )
        time_buckets = self.relative_distance_bucket(
            relative_time, self.num_buckets, self.max_distance
        )

        pos_bias = self.pos_bias_table[pos_buckets]
        time_bias = self.time_bias_table[time_buckets]

        return pos_bias + time_bias


"""
    RelativeAttentionBiasPositionalOnly
    Variant of RelativeAttentionBias with the temporal bias table removed
    entirely — used for the Appendix C synthetic dataset experiment, since
    that data has no timestamps (per the paper: "we always ablate rab^{p,t}
    for HSTU as this dataset does not have timestamps").
"""
class RelativeAttentionBiasPositionalOnly(nnx.Module):
    def __init__(self, num_buckets, rngs, max_distance):
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.pos_bias_table = nnx.Param(rngs.params.uniform((num_buckets,)))

    def __call__(self, positions):
        """
        positions: (seq_len,) int array

        Returns: (seq_len, seq_len) bias matrix — positional only
        """
        relative_position = positions[None, :] - positions[:, None]

        pos_buckets = compute_relative_distance_bucket(
            relative_position, self.num_buckets, self.max_distance
        )

        pos_bias = self.pos_bias_table[pos_buckets]
        return pos_bias


"""
    Spatial Aggregation:
    A(X)V(X) = SiLU(Q(X)K(X)ᵀ + rab) · V(X)
    1. Compute Q·Kᵀ — a matrix multiply between Q and K, transposed appropriately
    2. Add rab to that result (relying on broadcasting)
    3. Apply SiLU to the sum
    4. Multiply the result by V
"""
class SpatialAggregation(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, Q, K, V, rab):
        seq_len = Q.shape[-2]
        k_t = jnp.swapaxes(K, -2, -1)
        q_kt = jnp.matmul(Q, k_t)

        add_rab = q_kt + rab

        silu = jax.nn.silu(add_rab)

        casual_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        silu = jnp.where(casual_mask, silu, 0.0)

        final = jnp.matmul(silu, V)

        return final


"""
    Pointwise Transformation
    equation: LayerNorm(A(X)V(X)) ⊙ U(X) followed by an output linear projection
"""
class PointwiseTransformation(nnx.Module):
    def __init__(self, d_model, rngs):
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.linear = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, attn_output, U):
        normed = self.layer_norm(attn_output)
        gated = normed * U
        output = self.linear(gated)
        return output


"""
    HTSU Layer
"""
class HTSULayer(nnx.Module):
    def __init__(self, d_model, num_buckets, max_distance, rngs):
        self.pointwise_proj = PointwiseProjection(d_model, 4 * d_model, rngs)
        self.rab = RelativeAttentionBias(num_buckets, rngs, max_distance)
        self.spatial_agg = SpatialAggregation()
        self.pointwise_transformation = PointwiseTransformation(d_model, rngs)

    def __call__(self, x, positions, timestamps):
        U, Q, V, K = self.pointwise_proj(x)
        rab = self.rab(positions, timestamps)
        attention_output = self.spatial_agg(Q, K, V, rab)
        transformed = self.pointwise_transformation(attention_output, U)
        output = x + transformedx
        return output


"""
    HTSULayerPositionalOnly
"""
class HTSULayerPositionalOnly(nnx.Module):
    def __init__(self, d_model, num_buckets, max_distance, rngs):
        self.pointwise_proj = PointwiseProjection(d_model, 4 * d_model, rngs)
        self.rab = RelativeAttentionBiasPositionalOnly(num_buckets, rngs, max_distance)
        self.spatial_agg = SpatialAggregation()
        self.pointwise_transformation = PointwiseTransformation(d_model, rngs)

    def __call__(self, x, positions):
        U, Q, V, K = self.pointwise_proj(x)
        rab = self.rab(positions)
        attention_output = self.spatial_agg(Q, K, V, rab)
        transformed = self.pointwise_transformation(attention_output, U)
        output = x + transformed
        return output


"""
    ActionHead
    module used for mapping d_model from HTSULayer to map num_actions numbers,
    one score per possible action for loss calculation
"""
class ActionHead(nnx.Module):
    def __init__(self, d_model, num_actions, rngs):
        self.linear = nnx.Linear(d_model, num_actions, rngs=rngs)

    def __call__(self, x):
        res = self.linear(x)
        return res


"""
    RetrievalDownProjection
    Projects the concatenated (Φᵢ, aᵢ) retrieval pair — width 2*d_model —
    down to d_model, so the combined item+action representation can be
    fed into the same shared HTSULayer used for ranking.
"""
class RetrievalDownProjection(nnx.Module):
    def __init__(self, in_dim, d_model, rngs):
        self.linear = nnx.Linear(in_dim, d_model, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


"""
    ItemEmbedding
"""
class ItemEmbedding(nnx.Module):
    def __init__(self, num_items, d_model, rngs):
        self.embed = nnx.Embed(num_items, d_model, rngs=rngs)

    def __call__(self, item_ids):
        return self.embed(item_ids)


"""
    SpatialAggregationSoftmax
"""
class SpatialAggregationSoftmax(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, Q, K, V, rab):
        seq_len = Q.shape[-2]
        d_k = Q.shape[-1]

        k_t = jnp.swapaxes(K, -2, -1)
        q_kt = jnp.matmul(Q, k_t)

        scaled = q_kt / jnp.sqrt(d_k)

        add_rab = scaled + rab

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        masked = jnp.where(causal_mask, add_rab, -jnp.inf)

        attn_weights = jax.nn.softmax(masked, axis=-1)

        final = jnp.matmul(attn_weights, V)
        return final


"""
    HTSULayerSoftmax
"""
class HTSULayerSoftmax(nnx.Module):
    def __init__(self, d_model, num_buckets, max_distance, rngs):
        self.pointwise_proj = PointwiseProjection(d_model, 4 * d_model, rngs)
        self.rab = RelativeAttentionBias(num_buckets, rngs, max_distance)
        self.spatial_agg = SpatialAggregationSoftmax()
        self.pointwise_transformation = PointwiseTransformation(d_model, rngs)

    def __call__(self, x, positions, timestamps):
        U, Q, V, K = self.pointwise_proj(x)
        rab = self.rab(positions, timestamps)
        attention_output = self.spatial_agg(Q, K, V, rab)
        transformed = self.pointwise_transformation(attention_output, U)
        output = x + transformed
        return output


"""
    HTSULayerSoftmaxPositionalOnly
"""
class HTSULayerSoftmaxPositionalOnly(nnx.Module):
    def __init__(self, d_model, num_buckets, max_distance, rngs):
        self.pointwise_proj = PointwiseProjection(d_model, 4 * d_model, rngs)
        self.rab = RelativeAttentionBiasPositionalOnly(num_buckets, rngs, max_distance)
        self.spatial_agg = SpatialAggregationSoftmax()
        self.pointwise_transformation = PointwiseTransformation(d_model, rngs)

    def __call__(self, x, positions):
        U, Q, V, K = self.pointwise_proj(x)
        rab = self.rab(positions)
        attention_output = self.spatial_agg(Q, K, V, rab)
        transformed = self.pointwise_transformation(attention_output, U)
        output = x + transformed
        return output


"""
    PointwiseTransformationFFN
"""
class PointwiseTransformationFFN(nnx.Module):
    def __init__(self, d_model, rngs):
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn_in = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.ffn_out = nnx.Linear(4 * d_model, d_model, rngs=rngs)

    def __call__(self, attn_output):
        normed = self.layer_norm(attn_output)
        hidden = jax.nn.relu(self.ffn_in(normed))
        output = self.ffn_out(hidden)
        return output


"""
    HTSULayerFFN
"""
class HTSULayerFFN(nnx.Module):
    def __init__(self, d_model, num_buckets, max_distance, rngs):
        self.pointwise_proj = PointwiseProjection(d_model, 4 * d_model, rngs)
        self.rab = RelativeAttentionBias(num_buckets, rngs, max_distance)
        self.spatial_agg = SpatialAggregation()
        self.pointwise_transformation = PointwiseTransformationFFN(d_model, rngs)

    def __call__(self, x, positions, timestamps):
        U, Q, V, K = self.pointwise_proj(x)
        rab = self.rab(positions, timestamps)
        attention_output = self.spatial_agg(Q, K, V, rab)
        transformed = self.pointwise_transformation(attention_output)
        output = x + transformed
        return output


"""
    HTSULayerFFNPositionalOnly
"""
class HTSULayerFFNPositionalOnly(nnx.Module):
    def __init__(self, d_model, num_buckets, max_distance, rngs):
        self.pointwise_proj = PointwiseProjection(d_model, 4 * d_model, rngs)
        self.rab = RelativeAttentionBiasPositionalOnly(num_buckets, rngs, max_distance)
        self.spatial_agg = SpatialAggregation()
        self.pointwise_transformation = PointwiseTransformationFFN(d_model, rngs)

    def __call__(self, x, positions):
        U, Q, V, K = self.pointwise_proj(x)
        rab = self.rab(positions)
        attention_output = self.spatial_agg(Q, K, V, rab)
        transformed = self.pointwise_transformation(attention_output)
        output = x + transformed
        return output


"""
    VanillaTransformerBlock
"""
class VanillaTransformerBlock(nnx.Module):
    def __init__(self, d_model, rngs):
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn_in = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.ffn_out = nnx.Linear(4 * d_model, d_model, rngs=rngs)

    def __call__(self, x):
        seq_len, d_model = x.shape[-2], x.shape[-1]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        k_t = jnp.swapaxes(K, -2, -1)
        scores = jnp.matmul(Q, k_t) / jnp.sqrt(d_model)

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        masked = jnp.where(causal_mask, scores, -jnp.inf)
        attn_weights = jax.nn.softmax(masked, axis=-1)

        attn_output = jnp.matmul(attn_weights, V)
        normed = self.layer_norm(attn_output)
        hidden = jax.nn.relu(self.ffn_in(normed))
        transformed = self.ffn_out(hidden)

        return x + transformed
