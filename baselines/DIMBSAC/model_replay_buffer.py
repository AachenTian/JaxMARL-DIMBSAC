from typing import NamedTuple

import jax
import jax.numpy as jnp


class ModelTransitionBatch(NamedTuple):
    """Learner-specific synthetic SAC transitions."""

    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    dones: jax.Array


class ModelReplayBufferState(NamedTuple):
    """Circular synthetic replay buffer for one learner."""

    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    dones: jax.Array
    position: jax.Array
    size: jax.Array


class IndependentModelReplayBuffers(NamedTuple):
    """One independent model replay buffer per learner."""

    agents: tuple


def init_model_replay_buffer(
    capacity: int,
    num_agents: int,
    obs_dim: int,
    action_dim: int,
):
    """Initialize one learner-specific model replay buffer."""

    return ModelReplayBufferState(
        obs=jnp.zeros(
            (
                capacity,
                num_agents,
                obs_dim,
            ),
            dtype=jnp.float32,
        ),
        actions=jnp.zeros(
            (
                capacity,
                num_agents,
                action_dim,
            ),
            dtype=jnp.float32,
        ),
        rewards=jnp.zeros(
            (capacity,),
            dtype=jnp.float32,
        ),
        next_obs=jnp.zeros(
            (
                capacity,
                num_agents,
                obs_dim,
            ),
            dtype=jnp.float32,
        ),
        dones=jnp.zeros(
            (capacity,),
            dtype=jnp.bool_,
        ),
        position=jnp.array(
            0,
            dtype=jnp.int32,
        ),
        size=jnp.array(
            0,
            dtype=jnp.int32,
        ),
    )


def init_independent_model_replay_buffers(
    num_agents: int,
    capacity: int,
    obs_dim: int,
    action_dim: int,
):
    """Initialize one model buffer for every learner."""

    buffers = tuple(
        init_model_replay_buffer(
            capacity=capacity,
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        for _ in range(num_agents)
    )

    return IndependentModelReplayBuffers(
        agents=buffers
    )


def add_model_batch(
    buffer,
    batch,
):
    """Add a batch of synthetic transitions to a circular buffer."""

    batch_size = batch.rewards.shape[0]
    capacity = buffer.rewards.shape[0]

    if batch_size == 0:
        return buffer

    # Keep only the most recent transitions if the incoming
    # batch itself is larger than the whole buffer.
    if batch_size >= capacity:
        return ModelReplayBufferState(
            obs=batch.obs[-capacity:],
            actions=batch.actions[-capacity:],
            rewards=batch.rewards[-capacity:],
            next_obs=batch.next_obs[-capacity:],
            dones=batch.dones[-capacity:],
            position=jnp.array(
                0,
                dtype=jnp.int32,
            ),
            size=jnp.array(
                capacity,
                dtype=jnp.int32,
            ),
        )

    indices = (
        jnp.arange(
            batch_size
        )
        + buffer.position
    ) % capacity

    obs = buffer.obs.at[
        indices
    ].set(
        batch.obs
    )

    actions = buffer.actions.at[
        indices
    ].set(
        batch.actions
    )

    rewards = buffer.rewards.at[
        indices
    ].set(
        batch.rewards
    )

    next_obs = buffer.next_obs.at[
        indices
    ].set(
        batch.next_obs
    )

    dones = buffer.dones.at[
        indices
    ].set(
        batch.dones
    )

    new_position = (
        buffer.position
        + batch_size
    ) % capacity

    new_size = jnp.minimum(
        buffer.size + batch_size,
        capacity,
    )

    return ModelReplayBufferState(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        position=new_position,
        size=new_size,
    )


def sample_model_batch(
    buffer,
    rng,
    batch_size: int,
):
    """Sample synthetic SAC transitions uniformly."""

    indices = jax.random.randint(
        rng,
        shape=(batch_size,),
        minval=0,
        maxval=buffer.size,
    )

    return ModelTransitionBatch(
        obs=buffer.obs[
            indices
        ],
        actions=buffer.actions[
            indices
        ],
        rewards=buffer.rewards[
            indices
        ],
        next_obs=buffer.next_obs[
            indices
        ],
        dones=buffer.dones[
            indices
        ],
    )


def trajectory_to_agent_model_batch(
    trajectory,
    agent_idx: int,
):
    """
    Convert valid synthetic trajectory transitions into
    learner-specific SAC transitions.
    """

    num_agents = (
        trajectory.obs.shape[2]
    )

    obs_dim = (
        trajectory.obs.shape[3]
    )

    action_dim = (
        trajectory.actions.shape[3]
    )

    flat_valid = (
        trajectory.valid.reshape(-1)
    )

    valid_indices = jnp.where(
        flat_valid
    )[0]

    flat_obs = trajectory.obs.reshape(
        -1,
        num_agents,
        obs_dim,
    )

    flat_actions = (
        trajectory.actions.reshape(
            -1,
            num_agents,
            action_dim,
        )
    )

    flat_rewards = (
        trajectory.rewards.reshape(
            -1,
            num_agents,
        )
    )

    flat_next_obs = (
        trajectory.next_obs.reshape(
            -1,
            num_agents,
            obs_dim,
        )
    )

    flat_dones = (
        trajectory.dones.reshape(-1)
    )

    return ModelTransitionBatch(
        obs=flat_obs[
            valid_indices
        ],
        actions=flat_actions[
            valid_indices
        ],
        rewards=flat_rewards[
            valid_indices,
            agent_idx,
        ],
        next_obs=flat_next_obs[
            valid_indices
        ],
        dones=flat_dones[
            valid_indices
        ],
    )


def add_trajectory_to_model_replay_buffers(
    model_buffers,
    trajectory,
):
    """
    Add one synthetic joint trajectory batch to all learners.

    Each learner receives the same joint observations/actions,
    but only its own scalar reward.
    """

    updated_buffers = []

    for agent_idx in range(
        len(model_buffers.agents)
    ):
        batch = (
            trajectory_to_agent_model_batch(
                trajectory=trajectory,
                agent_idx=agent_idx,
            )
        )

        updated_buffer = add_model_batch(
            buffer=(
                model_buffers.agents[
                    agent_idx
                ]
            ),
            batch=batch,
        )

        updated_buffers.append(
            updated_buffer
        )

    return IndependentModelReplayBuffers(
        agents=tuple(
            updated_buffers
        )
    )

def add_agent_trajectory_to_model_buffer(
    buffer,
    trajectory,
    agent_idx: int,
):
    """
    Add Learner_i's independently generated trajectory
    only to its own model replay buffer.
    """

    batch = (
        trajectory_to_agent_model_batch(
            trajectory=trajectory,
            agent_idx=agent_idx,
        )
    )

    return add_model_batch(
        buffer=buffer,
        batch=batch,
    )

def add_independent_rollouts_to_model_buffers(
    model_buffers,
    independent_rollouts,
):
    """
    Add each learner's own synthetic rollout only
    to its corresponding model replay buffer.
    """

    updated_buffers = []

    num_agents = len(
        model_buffers.agents
    )

    assert (
        len(
            independent_rollouts.agents
        )
        == num_agents
    )

    for agent_idx in range(
        num_agents
    ):
        updated_buffer = (
            add_agent_trajectory_to_model_buffer(
                buffer=(
                    model_buffers.agents[
                        agent_idx
                    ]
                ),
                trajectory=(
                    independent_rollouts.agents[
                        agent_idx
                    ]
                ),
                agent_idx=agent_idx,
            )
        )

        updated_buffers.append(
            updated_buffer
        )

    return model_buffers._replace(
        agents=tuple(
            updated_buffers
        )
    )