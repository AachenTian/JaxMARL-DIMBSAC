from typing import NamedTuple

import jax


class SACBatch(NamedTuple):
    """Learner-specific batch used by one SAC agent."""

    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    dones: jax.Array


def real_batch_to_sac_batch(
    real_batch,
    agent_idx: int,
):
    """
    Convert a shared real replay batch into the
    learner-specific view required by Agent_i.
    """

    return SACBatch(
        obs=real_batch.obs,
        actions=real_batch.actions,
        rewards=real_batch.rewards[
            :,
            agent_idx,
        ],
        next_obs=real_batch.next_obs,
        dones=real_batch.dones,
    )


def model_batch_to_sac_batch(
    model_batch,
):
    """Convert a learner-specific model batch to SACBatch."""

    return SACBatch(
        obs=model_batch.obs,
        actions=model_batch.actions,
        rewards=model_batch.rewards,
        next_obs=model_batch.next_obs,
        dones=model_batch.dones,
    )