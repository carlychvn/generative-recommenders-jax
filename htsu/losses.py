import jax
import jax.numpy as jnp
import optax 
from htsu.layers import ActionHead

def ranking_loss(model_output, target_mask, action_labels, action_head):
    """
    Extract the model's predictions at item positions and compute
    cross-entropy loss against the true next-action labels.

    model_output:  (2N, d_model) or (2N, num_actions) — model output over the full interleaved sequence
    target_mask:   (2N,) bool — True at item positions (ranking prediction targets)
    action_labels: (N,) int — true action taken after each item, in event order

    Returns: scalar loss
    """
    extracted = model_output[target_mask]
    logits = action_head(extracted)

    logit_loss = optax.softmax_cross_entropy_with_integer_labels(logits, action_labels)
    loss = logit_loss.mean()

    return loss 

def retrieval_loss(model_output, item_embedding, positive_item_ids):
    """
    Contrastive retrieval loss. Treats each row's true next item as the
    correct match among every other item in the batch (in-batch negatives).

    model_output:       (N, d_model) — model's prediction vector per event
    item_embedding:      an already-constructed ItemEmbedding module
    positive_item_ids:   (N,) int — the true next item's ID for each event

    Returns: scalar loss
    """
    positive_embeddings = item_embedding(positive_item_ids)
    similarity_matrix = model_output @ positive_embeddings.T
    labels = jnp.arange(model_output.shape[0])
    per_example_loss = optax.softmax_cross_entropy_with_integer_labels(similarity_matrix, labels)
    loss = per_example_loss.mean()

    return loss