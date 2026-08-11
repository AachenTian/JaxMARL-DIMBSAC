from typing import NamedTuple

import jax.numpy as jnp


class NormalizationStats(NamedTuple):
    """Mean and standard deviation used for dynamics normalization."""

    mean: jnp.ndarray
    std: jnp.ndarray


def compute_normalization_stats(
    data,
    eps: float = 1e-6,
):
    """Compute normalization statistics along the batch dimension."""

    mean = jnp.mean(
        data,
        axis=0,
    )

    std = jnp.std(
        data,
        axis=0,
    )

    std = jnp.maximum(
        std,
        eps,
    )

    return NormalizationStats(
        mean=mean,
        std=std,
    )


def normalize(
    data,
    stats: NormalizationStats,
):
    """Normalize data using precomputed statistics."""

    return (
        data - stats.mean
    ) / stats.std


def denormalize(
    data,
    stats: NormalizationStats,
):
    """Transform normalized data back to the original scale."""

    return (
        data * stats.std
        + stats.mean
    )


def denormalize_variance(
    normalized_variance,
    target_stats: NormalizationStats,
):
    """
    Convert normalized predictive variance back to physical units.

    If:
        y = y_norm * std + mean

    then:
        Var[y] = Var[y_norm] * std^2
    """

    return (
        normalized_variance
        * jnp.square(target_stats.std)
    )


def build_agent_dynamics_data(
    batch,
    agent_idx: int,
):
    """
    Extract the local dynamics training data for one agent.

    Input:
        [x_i, a_i]

    Target:
        delta_x_i = next_x_i - x_i
    """

    x_i = batch.x[
        :, agent_idx
    ]

    action_i = batch.actions[
        :, agent_idx
    ]

    next_x_i = batch.next_x[
        :, agent_idx
    ]

    dynamics_input = jnp.concatenate(
        [
            x_i,
            action_i,
        ],
        axis=-1,
    )

    delta_x = (
        next_x_i - x_i
    )

    return (
        dynamics_input,
        delta_x,
    )