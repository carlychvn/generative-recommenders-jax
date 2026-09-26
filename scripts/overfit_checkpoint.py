from flax import nnx
import jax
import jax.numpy as jnp
import optax

from htsu.layers import HTSULayer, ActionHead, RetrievalDownProjection, ItemEmbedding
from htsu.sequencing import (
    interleave_items_and_actions, build_positions, build_timestamps, build_ranking_target_mask,
    build_retrieval_pairs, build_retrieval_positions, build_retrieval_timestamps,
)
from htsu.losses import ranking_loss, retrieval_loss

d_model = 4
num_events = 4
num_actions = 3
num_items = 50

item_tokens = jax.random.normal(jax.random.PRNGKey(0), (num_events, d_model))
action_tokens = jax.random.normal(jax.random.PRNGKey(1), (num_events, d_model))
event_time = jnp.array([0.0, 5.0, 12.0, 12.5])
positive_item_ids = jnp.array([7, 3, 19, 2])
action_labels = jnp.array([1, 0, 2, 1])

ranking_seq = interleave_items_and_actions(item_tokens, action_tokens)
ranking_positions = build_positions(num_events)
ranking_timestamps = build_timestamps(event_time)
ranking_target_mask = build_ranking_target_mask(num_events)

retrieval_seq = build_retrieval_pairs(item_tokens, action_tokens)
retrieval_positions = build_retrieval_positions(num_events)
retrieval_timestamps = build_retrieval_timestamps(event_time)

class HSTUModel(nnx.Module):
    def __init__(self, htsu_layer, action_head, retrieval_down_proj, item_embedding):
        self.htsu_layer = htsu_layer
        self.action_head = action_head
        self.retrieval_down_proj = retrieval_down_proj
        self.item_embedding = item_embedding


rngs = nnx.Rngs(0)

htsu_layer = HTSULayer(d_model, num_buckets=8, max_distance=16, rngs=rngs)
action_head = ActionHead(d_model, num_actions, rngs)
retrieval_down_proj = RetrievalDownProjection(in_dim=2 * d_model, d_model=d_model, rngs=rngs)
item_embedding = ItemEmbedding(num_items, d_model, rngs)

model = HSTUModel(htsu_layer, action_head, retrieval_down_proj, item_embedding)

def loss_fn(htsu_layer, action_head, retrieval_down_proj, item_embedding):
    ranking_output = htsu_layer(ranking_seq, ranking_positions, ranking_timestamps)
    retrieval_input = retrieval_down_proj(retrieval_seq)
    retrieval_output = htsu_layer(retrieval_input, retrieval_positions, retrieval_timestamps)

    rank_loss_value = ranking_loss(ranking_output, ranking_target_mask, action_labels, action_head)
    retr_loss_value = retrieval_loss(retrieval_output, item_embedding, positive_item_ids)

    return rank_loss_value + retr_loss_value, (rank_loss_value, retr_loss_value)


def train_loss_fn(model):
    return loss_fn(model.htsu_layer, model.action_head, model.retrieval_down_proj, model.item_embedding)


optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

for step in range(1000):
    loss, grads = nnx.value_and_grad(train_loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)

    if step % 50 == 0:
        print(step, loss)
