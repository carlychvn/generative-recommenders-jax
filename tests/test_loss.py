import jax
import jax.numpy as jnp
from flax import nnx
from htsu.layers import ActionHead, ItemEmbedding
from htsu.losses import ranking_loss, retrieval_loss


def test_ranking_loss_shape_and_finite():
    rngs = nnx.Rngs(0)
    d_model, num_actions = 8, 4
    seq_len = 6

    action_head = ActionHead(d_model, num_actions, rngs)

    model_output = jax.random.normal(jax.random.PRNGKey(0), (seq_len, d_model))
    target_mask = jnp.array([True, False, True, False, True, False])
    action_labels = jnp.array([0, 2, 1])

    loss = ranking_loss(model_output, target_mask, action_labels, action_head)

    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_ranking_loss_extracts_correct_number_of_predictions():
    rngs = nnx.Rngs(0)
    d_model, num_actions = 8, 3
    seq_len = 4

    action_head = ActionHead(d_model, num_actions, rngs)

    model_output = jax.random.normal(jax.random.PRNGKey(1), (seq_len, d_model))
    target_mask = jnp.array([True, False, True, False])
    action_labels = jnp.array([1, 0])

    loss = ranking_loss(model_output, target_mask, action_labels, action_head)

    assert loss.shape == ()


def test_ranking_loss_low_when_logits_confidently_correct():
    rngs = nnx.Rngs(0)
    d_model, num_actions = 8, 4
    seq_len = 4

    action_head = ActionHead(d_model, num_actions, rngs)

    model_output = jax.random.normal(jax.random.PRNGKey(2), (seq_len, d_model))
    target_mask = jnp.array([True, False, True, False])

    extracted = model_output[target_mask]
    logits = action_head(extracted)
    predicted_classes = jnp.argmax(logits, axis=-1)

    matching_labels = predicted_classes
    wrong_labels = (predicted_classes + 1) % num_actions

    loss_matching = ranking_loss(model_output, target_mask, matching_labels, action_head)
    loss_wrong = ranking_loss(model_output, target_mask, wrong_labels, action_head)

    assert loss_matching < loss_wrong


def test_retrieval_loss_shape_and_finite():
    rngs = nnx.Rngs(0)
    num_items, d_model = 50, 8
    n = 4

    item_embedding = ItemEmbedding(num_items, d_model, rngs)

    model_output = jax.random.normal(jax.random.PRNGKey(0), (n, d_model))
    positive_item_ids = jnp.array([3, 17, 42, 9])

    loss = retrieval_loss(model_output, item_embedding, positive_item_ids)

    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_retrieval_loss_lower_when_predictions_match_positives():
    rngs = nnx.Rngs(0)
    num_items, d_model = 50, 8
    n = 4

    item_embedding = ItemEmbedding(num_items, d_model, rngs)

    positive_item_ids = jnp.array([3, 17, 42, 9])
    positive_embeddings = item_embedding(positive_item_ids)

    noise = jax.random.normal(jax.random.PRNGKey(1), (n, d_model)) * 0.01
    matching_output = positive_embeddings + noise

    random_output = jax.random.normal(jax.random.PRNGKey(2), (n, d_model))

    loss_matching = retrieval_loss(matching_output, item_embedding, positive_item_ids)
    loss_random = retrieval_loss(random_output, item_embedding, positive_item_ids)

    assert loss_matching < loss_random


def test_retrieval_loss_distinguishes_correct_item_from_others_in_batch():
    rngs = nnx.Rngs(0)
    num_items, d_model = 50, 8
    n = 3

    item_embedding = ItemEmbedding(num_items, d_model, rngs)

    positive_item_ids = jnp.array([5, 20, 35])
    positive_embeddings = item_embedding(positive_item_ids)

    correct_output = positive_embeddings
    loss_correct = retrieval_loss(correct_output, item_embedding, positive_item_ids)

    shuffled_output = jnp.roll(positive_embeddings, shift=1, axis=0)
    loss_shuffled = retrieval_loss(shuffled_output, item_embedding, positive_item_ids)

    assert loss_correct < loss_shuffled


if __name__ == "__main__":
    test_ranking_loss_shape_and_finite()
    test_ranking_loss_extracts_correct_number_of_predictions()
    test_ranking_loss_low_when_logits_confidently_correct()
    test_retrieval_loss_shape_and_finite()
    test_retrieval_loss_lower_when_predictions_match_positives()
    test_retrieval_loss_distinguishes_correct_item_from_others_in_batch()
    print("all checks ran")
