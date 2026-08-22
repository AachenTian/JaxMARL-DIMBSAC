from typing import NamedTuple

import jax
import jax.numpy as jnp


class RealTransitionBatch(NamedTuple):
    """Physical transitions shared by all independent dynamics learners."""

    x: jax.Array
    actions: jax.Array
    next_x: jax.Array
    episode_steps: jax.Array
    landmarks: jax.Array


def _stack_agent_state(env_state, num_agents: int):
    """Build x_i = [p_x, p_y, v_x, v_y] for every agent."""
    agent_pos = env_state.p_pos[:, :num_agents]
    agent_vel = env_state.p_vel[:, :num_agents]
    return jnp.concatenate([agent_pos, agent_vel], axis=-1)


def joint_action_to_dict(joint_action, agent_list):
    """Convert (N, A, action_dim) into the action dictionary expected by MPE."""
    return {
        agent: joint_action[:, agent_idx]
        for agent_idx, agent in enumerate(agent_list)
    }


def collect_random_transition_batch(
    env,
    rng,
    num_envs: int,
    num_collect_steps: int,
    episode_horizon: int,
):
    """
    Collect a fixed dataset with uniformly random continuous actions.

    We record exactly `episode_horizon` physical transitions and then reset.
    Therefore a reset jump is never stored as a dynamics target.
    """
    reset_keys = jax.random.split(rng, num_envs + 1)
    rng = reset_keys[0]
    _, env_state = jax.vmap(env.reset)(reset_keys[1:])

    num_agents = env.num_agents
    num_landmarks = env.num_landmarks
    action_space = env.action_space(env.agents[0])
    action_dim = action_space.shape[0]

    x_batches = []
    action_batches = []
    next_x_batches = []
    episode_step_batches = []
    landmark_batches = []

    episode_step = 0

    for _ in range(num_collect_steps):
        x = _stack_agent_state(env_state, num_agents)
        landmarks = env_state.p_pos[:, num_agents : num_agents + num_landmarks]

        rng, action_rng, step_rng = jax.random.split(rng, 3)

        joint_action = jax.random.uniform(
            action_rng,
            shape=(num_envs, num_agents, action_dim),
            minval=action_space.low,
            maxval=action_space.high,
        )

        action_dict = joint_action_to_dict(joint_action, env.agents)
        step_keys = jax.random.split(step_rng, num_envs)

        _, next_env_state, _, _, _ = jax.vmap(
            env.step_env,
            in_axes=(0, 0, 0),
        )(
            step_keys,
            env_state,
            action_dict,
        )

        next_x = _stack_agent_state(next_env_state, num_agents)

        x_batches.append(x)
        action_batches.append(joint_action)
        next_x_batches.append(next_x)
        episode_step_batches.append(
            jnp.full((num_envs,), episode_step, dtype=jnp.int32)
        )
        landmark_batches.append(landmarks)

        env_state = next_env_state
        episode_step += 1

        if episode_step == episode_horizon:
            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            _, env_state = jax.vmap(env.reset)(reset_keys)
            episode_step = 0

    batch = RealTransitionBatch(
        x=jnp.concatenate(x_batches, axis=0),
        actions=jnp.concatenate(action_batches, axis=0),
        next_x=jnp.concatenate(next_x_batches, axis=0),
        episode_steps=jnp.concatenate(episode_step_batches, axis=0),
        landmarks=jnp.concatenate(landmark_batches, axis=0),
    )

    return rng, batch
