from typing import NamedTuple

import jax
import jax.numpy as jnp

from dynamics.trainer import (
    DynamicsTrainState,
    init_dynamics_train_state,
)

from dynamics.trainer import predict_dynamics

class DynamicsEnsembleState(NamedTuple):
    """State of one agent's probabilistic dynamics ensemble."""

    members: tuple

class IndependentDynamicsEnsembles(NamedTuple):
    """Independent dynamics ensemble owned by each agent."""

    agents: tuple


def init_dynamics_ensemble(
    rng,
    model,
    ensemble_size: int,
    input_dim: int,
    input_stats,
    target_stats,
    learning_rate: float,
):
    """
    Initialize independent ensemble members.

    All members share normalization statistics but have
    independently initialized model parameters and optimizers.
    """

    member_keys = jax.random.split(
        rng,
        ensemble_size,
    )

    members = tuple(
        init_dynamics_train_state(
            rng=member_keys[member_idx],
            model=model,
            input_dim=input_dim,
            input_stats=input_stats,
            target_stats=target_stats,
            learning_rate=learning_rate,
        )
        for member_idx in range(
            ensemble_size
        )
    )

    return DynamicsEnsembleState(
        members=members
    )


def create_bootstrap_indices(
    rng,
    ensemble_size: int,
    num_samples: int,
):
    """
    Generate one bootstrap dataset for each ensemble member.

    Sampling is performed with replacement.

    Output:
        (ensemble_size, num_samples)
    """

    bootstrap_keys = jax.random.split(
        rng,
        ensemble_size,
    )

    indices = jnp.stack(
        [
            jax.random.randint(
                bootstrap_keys[member_idx],
                shape=(num_samples,),
                minval=0,
                maxval=num_samples,
            )
            for member_idx in range(
                ensemble_size
            )
        ],
        axis=0,
    )

    return indices


def replace_ensemble_member(
    ensemble_state,
    member_idx: int,
    new_member_state,
):
    """Replace one member of an ensemble."""

    members = list(
        ensemble_state.members
    )

    members[member_idx] = (
        new_member_state
    )

    return DynamicsEnsembleState(
        members=tuple(members)
    )


def predict_dynamics_ensemble(
    ensemble_state: DynamicsEnsembleState,
    dynamics_inputs,
):
    """
    Predict with every member of one dynamics ensemble.

    Input:
        dynamics_inputs: (B, input_dim)

    Returns:
        member_means:     (M, B, 4)
        member_variances: (M, B, 4)

    The variance outputs are preserved for later uncertainty analysis,
    but are not used for rollout decisions at this stage.
    """

    member_means = []
    member_variances = []

    for member_state in ensemble_state.members:

        mean_delta_x, variance_delta_x = (
            predict_dynamics(
                state=member_state,
                dynamics_inputs=dynamics_inputs,
            )
        )

        member_means.append(
            mean_delta_x
        )

        member_variances.append(
            variance_delta_x
        )

    member_means = jnp.stack(
        member_means,
        axis=0,
    )

    member_variances = jnp.stack(
        member_variances,
        axis=0,
    )

    return (
        member_means,
        member_variances,
    )

def build_independent_dynamics_ensembles(
    agent_ensembles,
):
    """Pack one independent dynamics ensemble per agent."""

    return IndependentDynamicsEnsembles(
        agents=tuple(agent_ensembles)
    )


def replace_agent_ensemble(
    dynamics_ensembles: IndependentDynamicsEnsembles,
    agent_idx: int,
    new_ensemble_state: DynamicsEnsembleState,
):
    """Replace the full dynamics ensemble of one agent."""

    agents = list(
        dynamics_ensembles.agents
    )

    agents[agent_idx] = (
        new_ensemble_state
    )

    return IndependentDynamicsEnsembles(
        agents=tuple(agents)
    )


def predict_agent_dynamics_ensemble(
    dynamics_ensembles: IndependentDynamicsEnsembles,
    agent_idx: int,
    dynamics_inputs,
):
    """Predict with all dynamics members owned by one agent."""

    return predict_dynamics_ensemble(
        ensemble_state=(
            dynamics_ensembles.agents[
                agent_idx
            ]
        ),
        dynamics_inputs=dynamics_inputs,
    )
