from typing import NamedTuple

import jax
import jax.numpy as jnp

from model_replay_buffer import (
    sample_model_batch,
)
from sac_batch import (
    SACBatch,
    model_batch_to_sac_batch,
)


class MixedBatchInfo(NamedTuple):
    """Composition of one mixed critic batch."""

    num_real: int
    num_model: int


def compute_mixed_batch_sizes(
    batch_size: int,
    model_ratio: float,
):
    """
    Compute real/model sample counts for a mixed batch.

    model_ratio = 0.0 -> real only
    model_ratio = 1.0 -> model only
    """

    if not 0.0 <= model_ratio <= 1.0:
        raise ValueError(
            "model_ratio must be in [0, 1], "
            f"got {model_ratio}"
        )

    num_model = int(
        round(
            batch_size
            * model_ratio
        )
    )

    num_model = min(
        max(
            num_model,
            0,
        ),
        batch_size,
    )

    num_real = (
        batch_size
        - num_model
    )

    return (
        num_real,
        num_model,
    )


def sample_real_sac_batch(
    real_buffer,
    rng,
    batch_size: int,
    agent_idx: int,
):
    """
    Sample a learner-specific SAC batch directly from D_real.

    D_real stores joint rewards, so only reward_i is selected.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if int(real_buffer.size) <= 0:
        raise ValueError(
            "Cannot sample from an empty real replay buffer."
        )

    indices = jax.random.randint(
        rng,
        shape=(batch_size,),
        minval=0,
        maxval=real_buffer.size,
    )

    return SACBatch(
        obs=real_buffer.obs[
            indices
        ],
        actions=real_buffer.actions[
            indices
        ],
        rewards=real_buffer.rewards[
            indices,
            agent_idx,
        ],
        next_obs=real_buffer.next_obs[
            indices
        ],
        dones=real_buffer.dones[
            indices
        ],
    )


def concatenate_sac_batches(
    first_batch,
    second_batch,
):
    """Concatenate two learner-specific SAC batches."""

    return SACBatch(
        obs=jnp.concatenate(
            [
                first_batch.obs,
                second_batch.obs,
            ],
            axis=0,
        ),
        actions=jnp.concatenate(
            [
                first_batch.actions,
                second_batch.actions,
            ],
            axis=0,
        ),
        rewards=jnp.concatenate(
            [
                first_batch.rewards,
                second_batch.rewards,
            ],
            axis=0,
        ),
        next_obs=jnp.concatenate(
            [
                first_batch.next_obs,
                second_batch.next_obs,
            ],
            axis=0,
        ),
        dones=jnp.concatenate(
            [
                first_batch.dones,
                second_batch.dones,
            ],
            axis=0,
        ),
    )


def shuffle_sac_batch(
    batch,
    rng,
):
    """Randomly shuffle transitions inside one SAC batch."""

    batch_size = (
        batch.rewards.shape[0]
    )

    permutation = (
        jax.random.permutation(
            rng,
            batch_size,
        )
    )

    return SACBatch(
        obs=batch.obs[
            permutation
        ],
        actions=batch.actions[
            permutation
        ],
        rewards=batch.rewards[
            permutation
        ],
        next_obs=batch.next_obs[
            permutation
        ],
        dones=batch.dones[
            permutation
        ],
    )


def sample_mixed_critic_batch(
    real_buffer,
    model_buffer,
    agent_idx: int,
    rng,
    batch_size: int,
    model_ratio: float,
):
    """
    Sample a learner-specific critic batch from real and model replay.

    The returned reward is always scalar reward_i.
    """

    (
        num_real,
        num_model,
    ) = compute_mixed_batch_sizes(
        batch_size=batch_size,
        model_ratio=model_ratio,
    )

    rng, real_rng, model_rng, shuffle_rng = (
        jax.random.split(
            rng,
            4,
        )
    )

    real_batch = None
    model_batch = None

    if num_real > 0:
        real_batch = sample_real_sac_batch(
            real_buffer=real_buffer,
            rng=real_rng,
            batch_size=num_real,
            agent_idx=agent_idx,
        )

    if num_model > 0:
        if int(model_buffer.size) <= 0:
            raise ValueError(
                "model_ratio > 0 but model replay buffer is empty."
            )

        sampled_model_batch = (
            sample_model_batch(
                buffer=model_buffer,
                rng=model_rng,
                batch_size=num_model,
            )
        )

        model_batch = (
            model_batch_to_sac_batch(
                sampled_model_batch
            )
        )

    if real_batch is None:
        mixed_batch = model_batch

    elif model_batch is None:
        mixed_batch = real_batch

    else:
        mixed_batch = (
            concatenate_sac_batches(
                real_batch,
                model_batch,
            )
        )

        mixed_batch = shuffle_sac_batch(
            batch=mixed_batch,
            rng=shuffle_rng,
        )

    info = MixedBatchInfo(
        num_real=num_real,
        num_model=num_model,
    )

    return (
        mixed_batch,
        info,
    )