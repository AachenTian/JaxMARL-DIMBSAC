from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class SACCritic(nn.Module):
    """
    Q-network for one SAC critic.

    Input:
        joint observation: (..., num_agents, obs_dim)
        joint action:      (..., num_agents, action_dim)

    Output:
        Q-value: (...)
    """

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self,
        joint_obs,
        joint_action,
    ):
        # Flatten the agent dimension while preserving batch dimensions.
        obs_flat = joint_obs.reshape(
            joint_obs.shape[:-2] + (-1,)
        )

        action_flat = joint_action.reshape(
            joint_action.shape[:-2] + (-1,)
        )

        x = jnp.concatenate(
            [
                obs_flat,
                action_flat,
            ],
            axis=-1,
        )

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        q_value = nn.Dense(1)(x)

        return jnp.squeeze(
            q_value,
            axis=-1,
        )