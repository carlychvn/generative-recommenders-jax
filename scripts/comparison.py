from flax import nnx
import jax
import jax.numpy as jnp
import optax

from htsu.layers import HTSULayerPositionalOnly, HTSULayerSoftmaxPositionalOnly, HTSULayerFFN, ItemEmbedding, HTSULayerFFNPositionalOnly
from htsu.synthetic_data import generate_dataset
from htsu.losses import retrieval_loss

num_records = 200
seq_len = 32
num_items = 20_000
num_categories = 100
d_model = 16
num_buckets = 8
max_distance = 16
num_steps = 300

rng = jax.random.PRNGKey(0)

print("generating synthetic dataset...")
dataset = generate_dataset(num_records, num_items, num_categories, seq_len, rng)
print("dataset shape:", dataset.shape)


class RetrievalModel(nnx.Module):
    def __init__(self, htsu_layer, item_embedding):
        self.htsu_layer = htsu_layer
        self.item_embedding = item_embedding


def loss_fn(model, item_ids, positions):
    item_tokens = model.item_embedding(item_ids)
    model_output = model.htsu_layer(item_tokens, positions)

    predictions = model_output[:-1]
    positive_item_ids = item_ids[1:]

    return retrieval_loss(predictions, model.item_embedding, positive_item_ids)


def overfit_one_record(model_name, htsu_layer_cls, rng, record, num_steps=300):
    """
    Diagnostic: can this pipeline drive loss to near-zero memorizing ONE
    repeated record? If not, something's wired wrong. If yes, the earlier
    flat-loss result was just a data-scale/step-count issue, not a bug.
    """
    rngs = nnx.Rngs(rng)
    htsu_layer = htsu_layer_cls(d_model, num_buckets, max_distance, rngs)
    item_embedding = ItemEmbedding(num_items, d_model, rngs)
    model = RetrievalModel(htsu_layer, item_embedding)

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    positions = jnp.arange(seq_len)

    def train_loss_fn(model):
        return loss_fn(model, record, positions)

    print(f"\noverfitting one record — {model_name}...")
    final_loss = None
    for step in range(num_steps):
        loss, grads = nnx.value_and_grad(train_loss_fn)(model)
        optimizer.update(model, grads)
        if step % 50 == 0:
            print(f"  step {step}: loss {loss:.4f}")
        final_loss = loss

    return final_loss


record = dataset[0]

def train_on_full_dataset(model_name, htsu_layer_cls, rng, dataset, num_steps=3000):
    rngs = nnx.Rngs(rng)
    htsu_layer = htsu_layer_cls(d_model, num_buckets, max_distance, rngs)
    item_embedding = ItemEmbedding(num_items, d_model, rngs)
    model = RetrievalModel(htsu_layer, item_embedding)

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    positions = jnp.arange(seq_len)

    print(f"\ntraining on full dataset — {model_name}...")
    for step in range(num_steps):
        record_idx = step % dataset.shape[0]
        item_ids = dataset[record_idx]

        def train_loss_fn(model):
            return loss_fn(model, item_ids, positions)

        loss, grads = nnx.value_and_grad(train_loss_fn)(model)
        optimizer.update(model, grads)

        if step % 500 == 0:
            print(f"  step {step}: loss {loss:.4f}")

    return loss


final_loss_hstu = train_on_full_dataset("HSTU (pointwise/SiLU)", HTSULayerPositionalOnly, jax.random.PRNGKey(1), dataset, num_steps=3000)
final_loss_softmax = train_on_full_dataset("Softmax baseline", HTSULayerSoftmaxPositionalOnly, jax.random.PRNGKey(1), dataset, num_steps=3000)

print("\n--- Day 3 comparison ---")
print(f"HSTU final loss:    {final_loss_hstu:.4f}")
print(f"Softmax final loss: {final_loss_softmax:.4f}")

final_loss_ffn_ablation = train_on_full_dataset(
    "HSTU with FFN instead of gating (Day 4 ablation)",
    HTSULayerFFNPositionalOnly, jax.random.PRNGKey(1), dataset, num_steps=3000
)

print("\n--- Day 4 ablation result ---")
print(f"HSTU (gating):        {final_loss_hstu:.4f}")
print(f"HSTU (FFN, ablated):  {final_loss_ffn_ablation:.4f}")
print(f"Softmax baseline:     {final_loss_softmax:.4f}")

if final_loss_ffn_ablation > final_loss_hstu:
    print("\nGating outperforms the FFN replacement — supports HSTU's design choice.")
else:
    print("\nFFN replacement matched or beat gating — worth noting as a genuine, explainable negative result.")
