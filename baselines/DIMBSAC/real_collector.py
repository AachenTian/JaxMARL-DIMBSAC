from typing import NamedTuple

import jax
import jax.numpy as jnp

from observation import (
    project_local_obs,
    local_obs_to_x,
    extract_landmarks,
)
from replay_buffer import RealTransitionBatch


class RealCollectorState(NamedTuple):
    """State carried by the parallel real-environment collector."""

    obs_dict: dict
    env_state: object
    episode_steps: jax.Array
    rng: jax.Array


def stack_rewards(rewards, agents):
    """
    Convert the JaxMARL reward dictionary into a joint reward tensor.

    Output shape:
        (num_envs, num_agents)
    """
    return jnp.stack(
        [rewards[agent] for agent in agents],
        axis=-1,
    )


def joint_action_to_dict(joint_action, agents):
    """
    Convert joint actions into the dictionary format expected by JaxMARL.

    Input:
        joint_action: (num_envs, num_agents, action_dim)
    """
    return {
        agent: joint_action[:, i]
        for i, agent in enumerate(agents)
    }


def init_real_collector(
    env,
    num_envs: int,
    rng: jax.Array,
) -> RealCollectorState:
    """Reset all parallel real environments."""

    rng, reset_rng = jax.random.split(rng)

    reset_keys = jax.random.split(
        reset_rng,
        num_envs,
    )

    obs_dict, env_state = jax.vmap(
        env.reset
    )(reset_keys)

    episode_steps = jnp.zeros(
        (num_envs,),
        dtype=jnp.int32,
    )

    return RealCollectorState(
        obs_dict=obs_dict,
        env_state=env_state,
        episode_steps=episode_steps,
        rng=rng,
    )


def _maybe_reset_single_env(
    env,
    reset_key,
    should_reset,
    stepped_obs,
    stepped_state,
):
    """
    Reset one environment only if the current episode has terminated.
    """

    def reset_branch(_):
        return env.reset(reset_key)

    def keep_branch(_):
        return stepped_obs, stepped_state

    return jax.lax.cond(
        should_reset,
        reset_branch,
        keep_branch,
        operand=None,
    )


def collect_real_step(
    env,
    collector_state: RealCollectorState,
    joint_action: jax.Array,
    num_landmarks: int,
    episode_horizon: int,
):
    """
    Collect one step from all parallel real environments.

    We use env.step_env instead of env.step so that terminal transitions
    are stored before resetting the environment.

    Parameters
    ----------
    joint_action:
        Shape (num_envs, num_agents, action_dim).

    episode_horizon:
        Maximum real episode length, e.g. 25.

    Returns
    -------
    next_collector_state
    transition
    info
    """

    agents = env.agents
    num_envs = joint_action.shape[0]

    obs_dict = collector_state.obs_dict
    env_state = collector_state.env_state
    episode_steps = collector_state.episode_steps
    rng = collector_state.rng

    # Current local observations and physical states.
    local_obs = project_local_obs(
        obs_dict,
        agents,
        num_landmarks,
    )

    joint_x = local_obs_to_x(
        local_obs
    )

    landmarks, _ = extract_landmarks(
        local_obs,
        num_landmarks,
    )

    action_dict = joint_action_to_dict(
        joint_action,
        agents,
    )

    # Generate independent PRNG keys for environment transitions and resets.
    rng, step_rng, reset_rng = jax.random.split(
        rng,
        3,
    )

    step_keys = jax.random.split(
        step_rng,
        num_envs,
    )

    # step_env does not automatically replace terminal states with reset states.
    (
        stepped_obs,
        stepped_state,
        rewards,
        _jaxmarl_done,
        info,
    ) = jax.vmap(
        env.step_env,
        in_axes=(0, 0, 0),
    )(
        step_keys,
        env_state,
        action_dict,
    )

    # Extract the true next physical state before any reset occurs.
    next_local_obs = project_local_obs(
        stepped_obs,
        agents,
        num_landmarks,
    )

    next_joint_x = local_obs_to_x(
        next_local_obs
    )

    reward_joint = stack_rewards(
        rewards,
        agents,
    )

    # episode_steps stores the starting step of each transition.
    # Therefore step 24 -> 25 is terminal when H = 25.
    env_done = (
        episode_steps + 1
        >= episode_horizon
    )

    transition = RealTransitionBatch(
        obs=local_obs,
        x=joint_x,
        actions=joint_action,
        rewards=reward_joint,
        next_obs=next_local_obs,
        next_x=next_joint_x,
        dones=env_done,
        episode_steps=episode_steps,
        landmarks=landmarks,
    )

    # Reset only after the terminal transition has been stored.
    reset_keys = jax.random.split(
        reset_rng,
        num_envs,
    )

    def reset_one(
        reset_key,
        done,
        obs,
        state,
    ):
        return _maybe_reset_single_env(
            env,
            reset_key,
            done,
            obs,
            state,
        )

    next_obs_dict, next_env_state = jax.vmap(
        reset_one,
        in_axes=(0, 0, 0, 0),
    )(
        reset_keys,
        env_done,
        stepped_obs,
        stepped_state,
    )

    next_episode_steps = jnp.where(
        env_done,
        0,
        episode_steps + 1,
    )

    next_collector_state = RealCollectorState(
        obs_dict=next_obs_dict,
        env_state=next_env_state,
        episode_steps=next_episode_steps,
        rng=rng,
    )

    return (
        next_collector_state,
        transition,
        info,
    )