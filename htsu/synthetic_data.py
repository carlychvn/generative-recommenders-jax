import jax
import jax.numpy as jnp


def assign_items_to_categories(num_items, num_categories, rng):
    """
    One-time setup: randomly assign each of num_items item IDs to
    exactly one of num_categories categories.

    Returns: (num_items,) int array, category id per item
    """
    logits = jnp.zeros(num_categories)
    assignments = jax.random.categorical(rng, logits, shape=(num_items,))
    return assignments


def available_vocab_size(record_index, total_records, num_items):
    """
    Streaming vocabulary growth: 40% of items available initially,
    remaining 60% unlocked progressively at equal intervals.

    Returns: int, max item ID currently sampleable
    """
    progress = record_index / total_records
    fraction_unlocked = 0.4 + 0.6 * progress
    max_id = int(fraction_unlocked * num_items)
    return max_id


def sample_category_sequence(alpha, category_prior, seq_len, rng):
    """
    Dirichlet-process category sampling for one record's sequence.

    alpha: float, concentration parameter for this record
    category_prior: (5,) array — H_c, a probability distribution over
                     the 5 selected categories for this record
    seq_len: e.g. 128

    Returns: (seq_len,) int array of INDICES into category_prior (0-4)
    """
    num_selected = category_prior.shape[0]
    counts = jnp.zeros(num_selected)
    sequence = []

    key = rng
    for n in range(1, seq_len + 1):
        key, draw_key, choice_key = jax.random.split(key, 3)

        if n == 1:
            chosen = jax.random.choice(draw_key, num_selected, p=category_prior)
        else:
            new_prob = alpha / (alpha + n - 1)
            draw_new = jax.random.bernoulli(choice_key, p=new_prob)

            if draw_new:
                chosen = jax.random.choice(draw_key, num_selected, p=category_prior)
            else:
                reuse_probs = counts / jnp.sum(counts)
                chosen = jax.random.choice(draw_key, num_selected, p=reuse_probs)

        counts = counts.at[chosen].add(1)
        sequence.append(chosen)

    return jnp.array(sequence)


def sample_item_for_category(category, category_to_items, max_available_id, rng):
    """
    Given a chosen category, sample an actual item ID belonging to it,
    restricted to ids <= max_available_id.
    """
    num_items = category_to_items.shape[0]
    item_ids = jnp.arange(num_items)

    matches_category = category_to_items == category
    is_unlocked = item_ids <= max_available_id
    eligible = matches_category & is_unlocked

    eligible_ids = jnp.where(eligible, item_ids, -1)
    valid_ids = eligible_ids[eligible_ids >= 0]

    chosen = jax.random.choice(rng, valid_ids)
    return chosen


def generate_record(record_index, total_records, category_to_items, num_categories, seq_len, rng):
    """
    Generate one full 128-length record: pick 5 categories + H_c,
    sample alpha, run the Dirichlet process over positions, sample
    an item for each position.

    Returns: (seq_len,) int array of item ids
    """
    key, cat_select_key, prior_key, alpha_key, seq_key = jax.random.split(rng, 5)

    selected_categories = jax.random.choice(
        cat_select_key, num_categories, shape=(5,), replace=False
    )

    category_prior = jax.random.dirichlet(prior_key, alpha=jnp.ones(5))
    alpha = jax.random.uniform(alpha_key, minval=1.0, maxval=500.0)
    category_indices = sample_category_sequence(alpha, category_prior, seq_len, seq_key)
    category_ids = selected_categories[category_indices]
    max_available_id = available_vocab_size(record_index, total_records, category_to_items.shape[0])

    item_ids = []
    for i in range(seq_len):
        key, item_key = jax.random.split(key)
        item_id = sample_item_for_category(category_ids[i], category_to_items, max_available_id, item_key)
        item_ids.append(item_id)

    return jnp.array(item_ids)


def generate_dataset(num_records, num_items, num_categories, seq_len, rng):
    """
    Top-level: generate the full synthetic dataset.
    """
    key, assign_key = jax.random.split(rng)
    category_to_items = assign_items_to_categories(num_items, num_categories, assign_key)

    records = []
    for record_index in range(num_records):
        key, record_key = jax.random.split(key)
        record = generate_record(
            record_index, num_records, category_to_items, num_categories, seq_len, record_key
        )
        records.append(record)

    return jnp.stack(records)
