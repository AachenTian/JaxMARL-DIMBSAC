from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class SACActor(nn.Module):
    """
    Feed-forward Gaussian actor for continuous SAC.

    Input:
        local observation o_i

    Output:
        Gaussian mean and log standard deviation
    """

    action_dim: int
    hidden_dims: Sequence[int]
    log_std_min: float
    log_std_max: float

    @nn.compact
    def __call__(self, obs):
        x = obs

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)

        log_std = jnp.clip(
            log_std,
            self.log_std_min,
            self.log_std_max,
        )

        return mean, log_std


def sample_squashed_gaussian(
    rng,
    mean,
    log_std,
    action_low,
    action_high,
):
    """Sample a reparameterized and squashed Gaussian action."""

    std = jnp.exp(log_std)

    noise = jax.random.normal(
        rng,
        shape=mean.shape,
    )

    pre_tanh_action = (
        mean + std * noise
    )

    squashed_action = jnp.tanh(
        pre_tanh_action
    )

    action_scale, action_bias = (
        _action_scale_and_bias(
            action_low=action_low,
            action_high=action_high,
            action_dim=mean.shape[-1],
            dtype=mean.dtype,
        )
    )

    action = (
        action_bias
        + action_scale * squashed_action
    )

    gaussian_log_prob = (
        -0.5
        * (
            noise ** 2
            + 2.0 * log_std
            + jnp.log(2.0 * jnp.pi)
        )
    )

    gaussian_log_prob = jnp.sum(
        gaussian_log_prob,
        axis=-1,
    )

    tanh_log_det = jnp.sum(
        jnp.log(
            1.0
            - squashed_action ** 2
            + 1e-6
        ),
        axis=-1,
    )

    # The affine correction must include every action dimension.
    scale_log_det = jnp.sum(
        jnp.log(
            jnp.abs(action_scale)
            + 1e-8
        )
    )

    log_prob = (
        gaussian_log_prob
        - tanh_log_det
        - scale_log_det
    )

    return (
        action,
        log_prob,
        pre_tanh_action,
    )


def deterministic_action(
    mean,
    action_low,
    action_high,
):
    """Compute the deterministic mean action used for evaluation."""

    squashed_action = jnp.tanh(
        mean
    )

    action_scale, action_bias = (
        _action_scale_and_bias(
            action_low=action_low,
            action_high=action_high,
            action_dim=mean.shape[-1],
            dtype=mean.dtype,
        )
    )

    return (
        action_bias
        + action_scale * squashed_action
    )

def _action_scale_and_bias(
    action_low,
    action_high,
    action_dim,
    dtype,
):
    """Build per-dimension affine parameters for the action space."""

    action_low = jnp.asarray(
        action_low,
        dtype=dtype,
    )

    action_high = jnp.asarray(
        action_high,
        dtype=dtype,
    )

    action_scale = (
        action_high - action_low
    ) / 2.0

    action_bias = (
        action_high + action_low
    ) / 2.0

    # JaxMARL may expose scalar bounds even for multi-dimensional actions.
    action_scale = jnp.broadcast_to(
        action_scale,
        (action_dim,),
    )

    action_bias = jnp.broadcast_to(
        action_bias,
        (action_dim,),
    )

    return action_scale, action_bias