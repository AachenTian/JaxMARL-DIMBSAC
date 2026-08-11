from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class ProbabilisticDynamicsModel(nn.Module):
    """
    One probabilistic dynamics model.

    Input:
        normalized [x_i, a_i]

    Output:
        mean and log-variance of normalized delta_x_i
    """

    output_dim: int
    hidden_dims: Sequence[int]
    log_var_min: float
    log_var_max: float

    @nn.compact
    def __call__(self, inputs):

        x = inputs

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.silu(x)

        mean = nn.Dense(
            self.output_dim
        )(x)

        log_var = nn.Dense(
            self.output_dim
        )(x)

        log_var = jnp.clip(
            log_var,
            self.log_var_min,
            self.log_var_max,
        )

        return mean, log_var