from typing import NamedTuple

import jax.numpy as jnp


class NormalizationStats(NamedTuple):
    mean: jnp.ndarray
    std: jnp.ndarray


def compute_normalization_stats(data, eps: float = 1e-6):
    mean = jnp.mean(data, axis=0)
    std = jnp.maximum(jnp.std(data, axis=0), eps)
    return NormalizationStats(mean=mean, std=std)


def normalize(data, stats: NormalizationStats):
    return (data - stats.mean) / stats.std


def denormalize(data, stats: NormalizationStats):
    return data * stats.std + stats.mean


def denormalize_variance(normalized_variance, target_stats: NormalizationStats):
    return normalized_variance * jnp.square(target_stats.std)


def build_local_dynamics_data(batch, agent_idx: int):
    """
    Baseline input: [x_i, a_i] = 4D + 5D = 9D.
    Target: delta_x_i = next_x_i - x_i = 4D.
    """
    x_i = batch.x[:, agent_idx]
    action_i = batch.actions[:, agent_idx]
    next_x_i = batch.next_x[:, agent_idx]

    dynamics_input = jnp.concatenate([x_i, action_i], axis=-1)
    delta_x = next_x_i - x_i
    return dynamics_input, delta_x


def build_oracle_pos_dynamics_data(
    batch,
    agent_idx: int,
):
    """
    Oracle-position dynamics data for one agent.

    Input:
        [x_i, a_i, true_relative_positions_of_other_agents]

    For 3-agent SimpleSpread:
        x_i                                  4D
        a_i                                  5D
        (p_j - p_i), (p_k - p_i)             4D
                                              ---
                                             13D

    Target:
        delta_x_i = next_x_i - x_i           4D

    Other agents are ordered by ascending agent index, excluding the ego
    agent. The relative positions are privileged training/evaluation inputs
    for this oracle diagnostic only.
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

    own_position = x_i[
        :, :2
    ]

    num_agents = batch.x.shape[1]
    other_relative_positions = []

    for other_idx in range(
        num_agents
    ):
        if other_idx == agent_idx:
            continue

        other_position = batch.x[
            :, other_idx, :2
        ]

        other_relative_positions.append(
            other_position - own_position
        )

    other_relative_positions = (
        jnp.concatenate(
            other_relative_positions,
            axis=-1,
        )
    )

    dynamics_input = jnp.concatenate(
        [
            x_i,
            action_i,
            other_relative_positions,
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


def build_oracle_pos_abs_vel_dynamics_data(
    batch,
    agent_idx: int,
):
    """
    Oracle position + absolute velocity dynamics data for one agent.

    Input:
        [x_i, a_i,
         true relative positions of other agents,
         true absolute velocities of other agents]

    For 3-agent continuous SimpleSpread:
        x_i                                      4D
        a_i                                      5D
        (p_j - p_i), (p_k - p_i)                 4D
        v_j, v_k                                 4D
                                                  ---
                                                 17D

    Target:
        delta_x_i = next_x_i - x_i               4D

    Other agents are ordered by ascending agent index, excluding ego.
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

    own_position = x_i[
        :, :2
    ]

    num_agents = batch.x.shape[1]
    other_relative_positions = []
    other_absolute_velocities = []

    for other_idx in range(
        num_agents
    ):
        if other_idx == agent_idx:
            continue

        other_position = batch.x[
            :, other_idx, :2
        ]
        other_velocity = batch.x[
            :, other_idx, 2:4
        ]

        other_relative_positions.append(
            other_position - own_position
        )
        other_absolute_velocities.append(
            other_velocity
        )

    other_relative_positions = jnp.concatenate(
        other_relative_positions,
        axis=-1,
    )
    other_absolute_velocities = jnp.concatenate(
        other_absolute_velocities,
        axis=-1,
    )

    dynamics_input = jnp.concatenate(
        [
            x_i,
            action_i,
            other_relative_positions,
            other_absolute_velocities,
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

