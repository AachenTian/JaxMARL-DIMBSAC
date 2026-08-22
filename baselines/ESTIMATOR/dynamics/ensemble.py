from typing import NamedTuple

import jax
import jax.numpy as jnp

from dynamics.trainer import init_dynamics_train_state, predict_dynamics


class DynamicsEnsembleState(NamedTuple):
    members: tuple


class IndependentDynamicsEnsembles(NamedTuple):
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
    member_keys = jax.random.split(rng, ensemble_size)
    members = tuple(
        init_dynamics_train_state(
            rng=member_keys[idx],
            model=model,
            input_dim=input_dim,
            input_stats=input_stats,
            target_stats=target_stats,
            learning_rate=learning_rate,
        )
        for idx in range(ensemble_size)
    )
    return DynamicsEnsembleState(members=members)


def create_bootstrap_indices(rng, ensemble_size: int, num_samples: int):
    member_keys = jax.random.split(rng, ensemble_size)
    return jnp.stack(
        [
            jax.random.randint(
                member_keys[idx],
                shape=(num_samples,),
                minval=0,
                maxval=num_samples,
            )
            for idx in range(ensemble_size)
        ],
        axis=0,
    )


def replace_ensemble_member(ensemble_state, member_idx: int, new_member_state):
    members = list(ensemble_state.members)
    members[member_idx] = new_member_state
    return DynamicsEnsembleState(members=tuple(members))


def predict_dynamics_ensemble(ensemble_state, dynamics_inputs):
    means = []
    variances = []
    for member_state in ensemble_state.members:
        mean, variance = predict_dynamics(member_state, dynamics_inputs)
        means.append(mean)
        variances.append(variance)
    return jnp.stack(means, axis=0), jnp.stack(variances, axis=0)


def build_independent_dynamics_ensembles(agent_ensembles):
    return IndependentDynamicsEnsembles(agents=tuple(agent_ensembles))
