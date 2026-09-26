import jax
import jax.numpy as jnp


def interleave_items_and_actions(item_tokens, action_tokens):
    """
    Merge parallel item and action sequences into one alternating sequence:
    Φ0, a0, Φ1, a1, ...

    item_tokens:   (N, ...) — one entry per event
    action_tokens: (N, ...) — one entry per event, same N and leading structure as item_tokens

    Returns: (2N, ...) interleaved sequence
    """
    stacked = jnp.stack([item_tokens, action_tokens], axis=1)
    interleaved = stacked.reshape(-1, *item_tokens.shape[1:])
    return interleaved


def build_positions(num_events):
    """
    Positions for the interleaved sequence: 0, 1, 2, ..., 2N-1
    """
    return jnp.arange(2 * num_events)


def build_timestamps(event_time):
    """
    Timestamps for the interleaved sequence: each raw event_time[i]
    duplicated for both the item slot and its paired action slot
    (Option A — shared timestamp within a pair).

    event_time: (N,) raw per-event timestamps

    Returns: (2N,) timestamps, e.g. [t0, t0, t1, t1, ...]
    """
    return jnp.repeat(event_time, 2)


def build_ranking_target_mask(num_events):
    """
    Boolean mask over the interleaved sequence marking which positions
    are ranking prediction targets (item positions, not action positions).

    Returns: (2N,) bool array, e.g. [True, False, True, False, ...]
    """
    indices = jnp.arange(2 * num_events)
    return indices % 2 == 0


def build_retrieval_pairs(item_tokens, action_tokens):
    """
    combines each (Φᵢ, aᵢ) into one per-step representation (unlike ranking's alternating format)
    """
    concat = jnp.concatenate([item_tokens, action_tokens], axis=-1)
    return concat

def build_retrieval_positions(num_events):
    return jnp.arange(num_events)

def build_retrieval_timestamps(event_time):
    return event_time
