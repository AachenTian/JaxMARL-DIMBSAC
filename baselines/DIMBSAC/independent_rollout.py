from typing import NamedTuple

import jax

from dynamics.model_rollout import (
    generate_model_rollout,
)
from model_action_sampler import (
    build_actor_action_sampler,
)
from dynamics.ensemble import (
    IndependentDynamicsEnsembles,
)


class IndependentModelRollouts(NamedTuple):
    """One independently generated synthetic rollout per learner."""

    agents: tuple


def build_learner_actor_params(
    agent_states,
    actor_snapshots,
    learner_idx: int,
):
    """
    Build the actor parameter view used by Learner_i.

    Learner_i uses its current actor parameters.
    Other agents use frozen actor snapshots.
    """

    params = []

    for agent_idx in range(
        len(agent_states)
    ):
        if agent_idx == learner_idx:
            params.append(
                agent_states[
                    agent_idx
                ].actor.params
            )
        else:
            params.append(
                actor_snapshots[
                    agent_idx
                ]
            )

    return tuple(params)


def generate_independent_model_rollouts(
    rng,
    start_x,
    landmarks,
    start_episode_steps,
    agent_states,
    actor,
    actor_snapshots,
    current_dynamics,
    dynamics_snapshots,
    env,
    action_low,
    action_high,
    ensemble_size: int,
    horizon: int,
    episode_horizon: int,
):
    """
    Generate one independent synthetic rollout for each learner.

    Each learner receives an independent RNG stream, therefore
    actions and ensemble-member selections are sampled separately.
    """

    num_agents = len(
        agent_states
    )

    learner_keys = jax.random.split(
        rng,
        num_agents + 1,
    )

    next_rng = learner_keys[0]

    trajectories = []

    for learner_idx in range(
        num_agents
    ):
        learner_actor_params = (
            build_learner_actor_params(
                agent_states=agent_states,
                actor_snapshots=(
                    actor_snapshots
                ),
                learner_idx=learner_idx,
            )
        )

        action_sampler = (
            build_actor_action_sampler(
                actor=actor,
                actor_params=(
                    learner_actor_params
                ),
                action_low=action_low,
                action_high=action_high,
            )
        )

        learner_dynamics = (
            build_learner_dynamics_view(
                current_dynamics=(
                    current_dynamics
                ),
                dynamics_snapshots=(
                    dynamics_snapshots
                ),
                learner_idx=learner_idx,
            )
        )

        trajectory = (
            generate_model_rollout(
                rng=learner_keys[
                    learner_idx + 1
                ],
                start_x=start_x,
                landmarks=landmarks,
                start_episode_steps=(
                    start_episode_steps
                ),
                dynamics_ensembles=(
                    learner_dynamics
                ),
                action_sampler=(
                    action_sampler
                ),
                env=env,
                ensemble_size=(
                    ensemble_size
                ),
                horizon=horizon,
                episode_horizon=(
                    episode_horizon
                ),
            )
        )

        trajectories.append(
            trajectory
        )

    return (
        next_rng,
        IndependentModelRollouts(
            agents=tuple(
                trajectories
            )
        ),
    )

def build_learner_dynamics_view(
    current_dynamics,
    dynamics_snapshots,
    learner_idx: int,
):
    """
    Build the dynamics view used by Learner_i.

    Learner_i uses its current dynamics ensemble.
    Other agents are represented by frozen snapshots.
    """

    agent_ensembles = []

    num_agents = len(
        current_dynamics.agents
    )

    for agent_idx in range(
        num_agents
    ):
        if agent_idx == learner_idx:
            ensemble = (
                current_dynamics.agents[
                    agent_idx
                ]
            )
        else:
            ensemble = (
                dynamics_snapshots.agents[
                    agent_idx
                ]
            )

        agent_ensembles.append(
            ensemble
        )

    return IndependentDynamicsEnsembles(
        agents=tuple(
            agent_ensembles
        )
    )
