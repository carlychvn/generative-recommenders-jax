import time
from flax import nnx
import jax
import jax.numpy as jnp

from htsu.layers import HTSULayerPositionalOnly, VanillaTransformerBlock
from htsu.stochastic_length import stochastic_length_sample

d_model = 16
num_buckets = 8
max_distance = 16


def time_forward_pass(fn, *args, num_warmup=3, num_trials=10):
    for _ in range(num_warmup):
        jax.block_until_ready(fn(*args))

    start = time.perf_counter()
    for _ in range(num_trials):
        jax.block_until_ready(fn(*args))
    end = time.perf_counter()

    return (end - start) / num_trials

for seq_len in [32, 128, 512]:
    rngs = nnx.Rngs(0)
    htsu_layer = HTSULayerPositionalOnly(d_model, num_buckets, max_distance, rngs)
    vanilla_block = VanillaTransformerBlock(d_model, rngs)

    x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, d_model))
    positions = jnp.arange(seq_len)

    htsu_fn = lambda x, positions: htsu_layer(x, positions)
    vanilla_fn = lambda x: vanilla_block(x)

    htsu_time = time_forward_pass(htsu_fn, x, positions)
    vanilla_time = time_forward_pass(vanilla_fn, x)

    print(f"seq_len={seq_len:4d}  |  HSTU: {htsu_time*1000:.3f} ms  |  Transformer: {vanilla_time*1000:.3f} ms")


print("\n--- Stochastic Length demo ---")
item_ids = jnp.arange(64)
positions = jnp.arange(64)
rng = jax.random.PRNGKey(0)

kept_items, kept_positions = stochastic_length_sample(item_ids, positions, keep_ratio=0.25, rng=rng)
print(f"original length: {item_ids.shape[0]}, kept length: {kept_items.shape[0]}")
print(f"kept positions (original indices preserved): {kept_positions}")