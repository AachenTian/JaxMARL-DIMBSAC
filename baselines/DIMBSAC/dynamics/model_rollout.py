from typing import NamedTuple

import jax
import jax.numpy as jnp

from dynamics.reward import compute_model_rewards
from dynamics.rollout import (
    propagate_joint_x,
    sample_ensemble_members,
)
from observation import reconstruct_local_obs


class ModelRolloutState(NamedTuple):
    """Current state of a batch of synthetic trajectories."""

    x: jax.Array
    obs: jax.Array
    landmarks: jax.Array
    episode_steps: jax.Array
    member_indices: jax.Array
    active: jax.Array
    rng: jax.Array


class ModelRolloutTransition(NamedTuple):
    """One synthetic model transition."""

    obs: jax.Array
    x: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    rollout_stops: jax.Array
    valid: jax.Array
    episode_steps: jax.Array
    next_episode_steps: jax.Array
    delta_x: jax.Array
    next_obs: jax.Array
    next_x: jax.Array


class ModelRolloutTrajectory(NamedTuple):
    """K-step synthetic trajectory batch."""

    obs: jax.Array
    x: jax.Array
    actions: jax.Array
    rewards: jax.Array
    episode_steps: jax.Array
    next_episode_steps: jax.Array
    delta_x: jax.Array
    next_obs: jax.Array
    next_x: jax.Array
    member_indices: jax.Array
    dones: jax.Array
    rollout_stops: jax.Array
    valid: jax.Array


def init_model_rollout(
    rng,
    start_x,
    landmarks,
    start_episode_steps,
    num_agents: int,
    ensemble_size: int,
):
    """
    Initialize synthetic trajectories from real replay states.

    One dynamics member is sampled per trajectory and agent.
    Member identities remain fixed for the whole rollout.
    """

    batch_size = start_x.shape[0]

    rng, member_rng = jax.random.split(
        rng
    )

    member_indices = (
        sample_ensemble_members(
            rng=member_rng,
            batch_size=batch_size,
            num_agents=num_agents,
            ensemble_size=ensemble_size,
        )
    )

    start_obs = reconstruct_local_obs(
        start_x,
        landmarks,
    )

    active = jnp.ones(
        (batch_size,),
        dtype=jnp.bool_,
    )

    return ModelRolloutState(
        x=start_x,
        obs=start_obs,
        landmarks=landmarks,
        episode_steps=start_episode_steps,
        member_indices=member_indices,
        active=active,
        rng=rng,
    )


def model_rollout_step(
    rollout_state,
    dynamics_ensembles,
    action_sampler,
    env,
    rollout_step: int,
    rollout_horizon: int,
    episode_horizon: int,
):
    """Generate one synthetic transition."""

    rng, action_rng = jax.random.split(
        rollout_state.rng
    )

    # Frozen actor snapshots generate decentralized actions.
    joint_actions = action_sampler(
        action_rng,
        rollout_state.obs,
    )

    # Learned dynamics predict physical next states.
    next_x, delta_x = propagate_joint_x(
        dynamics_ensembles=(
            dynamics_ensembles
        ),
        joint_x=rollout_state.x,
        joint_actions=joint_actions,
        member_indices=(
            rollout_state.member_indices
        ),
    )

    next_obs = reconstruct_local_obs(
        next_x,
        rollout_state.landmarks,
    )

    next_episode_steps = (
        rollout_state.episode_steps
        + 1
    )

    # Environment termination is defined only by
    # the true task horizon.
    dones = (
        next_episode_steps
        >= episode_horizon
    )

    # Artificial model truncation is separate from env_done.
    reached_rollout_horizon = (
        rollout_step + 1
        >= rollout_horizon
    )

    rollout_stops = jnp.full(
        dones.shape,
        reached_rollout_horizon,
        dtype=jnp.bool_,
    )

    # If the environment terminates on this transition,
    # this is not an artificial truncation.
    rollout_stops = (
        rollout_stops
        & (~dones)
    )

    valid = rollout_state.active

    # Reward is computed from the predicted next physical state.
    rewards = compute_model_rewards(
        env=env,
        predicted_next_x=next_x,
        landmarks=rollout_state.landmarks,
        next_episode_steps=(
            next_episode_steps
        ),
        episode_horizon=(
            episode_horizon
        ),
    )

    transition = ModelRolloutTransition(
        obs=rollout_state.obs,
        x=rollout_state.x,
        actions=joint_actions,
        rewards=rewards,
        dones=dones,
        rollout_stops=rollout_stops,
        valid=valid,
        episode_steps=(
            rollout_state.episode_steps
        ),
        next_episode_steps=(
            next_episode_steps
        ),
        delta_x=delta_x,
        next_obs=next_obs,
        next_x=next_x,
    )

    next_active = (
        rollout_state.active
        & (~dones)
        & (~rollout_stops)
    )

    next_state = rollout_state._replace(
        x=next_x,
        obs=next_obs,
        episode_steps=(
            next_episode_steps
        ),
        active=next_active,
        rng=rng,
    )

    return (
        next_state,
        transition,
    )


def generate_model_rollout(
    rng,
    start_x,
    landmarks,
    start_episode_steps,
    dynamics_ensembles,
    action_sampler,
    env,
    ensemble_size: int,
    horizon: int,
    episode_horizon: int,
):
    """
    Generate a fixed-horizon synthetic rollout.

    Ensemble members are sampled once at trajectory start and
    remain unchanged throughout the trajectory.
    """

    num_agents = start_x.shape[1]

    rollout_state = init_model_rollout(
        rng=rng,
        start_x=start_x,
        landmarks=landmarks,
        start_episode_steps=(
            start_episode_steps
        ),
        num_agents=num_agents,
        ensemble_size=ensemble_size,
    )

    initial_member_indices = (
        rollout_state.member_indices
    )

    transitions = []

    for rollout_step in range(
            horizon
    ):
        (
            rollout_state,
            transition,
        ) = model_rollout_step(
            rollout_state=rollout_state,
            dynamics_ensembles=(
                dynamics_ensembles
            ),
            action_sampler=action_sampler,
            env=env,
            rollout_step=rollout_step,
            rollout_horizon=horizon,
            episode_horizon=(
                episode_horizon
            ),
        )

        transitions.append(
            transition
        )

    assert jnp.array_equal(
        rollout_state.member_indices,
        initial_member_indices,
    )

    trajectory = ModelRolloutTrajectory(
        obs=jnp.stack(
            [t.obs for t in transitions],
            axis=0,
        ),
        x=jnp.stack(
            [t.x for t in transitions],
            axis=0,
        ),
        actions=jnp.stack(
            [t.actions for t in transitions],
            axis=0,
        ),
        rewards=jnp.stack(
            [t.rewards for t in transitions],
            axis=0,
        ),
        episode_steps=jnp.stack(
            [
                t.episode_steps
                for t in transitions
            ],
            axis=0,
        ),
        next_episode_steps=jnp.stack(
            [
                t.next_episode_steps
                for t in transitions
            ],
            axis=0,
        ),
        delta_x=jnp.stack(
            [t.delta_x for t in transitions],
            axis=0,
        ),
        next_obs=jnp.stack(
            [t.next_obs for t in transitions],
            axis=0,
        ),
        next_x=jnp.stack(
            [t.next_x for t in transitions],
            axis=0,
        ),
        member_indices=(
            initial_member_indices
        ),
        dones=jnp.stack(
            [t.dones for t in transitions],
            axis=0,
        ),
        rollout_stops=jnp.stack(
            [
                t.rollout_stops
                for t in transitions
            ],
            axis=0,
        ),
        valid=jnp.stack(
            [t.valid for t in transitions],
            axis=0,
        ),
    )

    return trajectory