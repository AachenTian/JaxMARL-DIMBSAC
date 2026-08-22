import json
from pathlib import Path
from typing import Dict

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.linen.initializers import constant, orthogonal


class ActorCriticFFMatchedBeta(nn.Module):
    """
    Exact inference-side architecture used by the continuous Beta IPPO actor.

    The critic is retained because the saved checkpoint contains the complete
    shared ActorCritic parameter tree. Estimator rollout only uses the policy.
    """

    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, x):
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        embedding = nn.relu(
            embedding
        )

        latent = nn.Dense(
            self.config["FF_LATENT_DIM"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        latent = nn.relu(
            latent
        )

        actor = nn.Dense(
            self.config["FF_LATENT_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(latent)
        actor = nn.relu(
            actor
        )

        alpha_raw = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_alpha",
        )(actor)
        beta_raw = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_beta",
        )(actor)

        min_concentration = float(
            self.config[
                "BETA_MIN_CONCENTRATION"
            ]
        )
        max_concentration = float(
            self.config[
                "BETA_MAX_CONCENTRATION"
            ]
        )

        alpha = jnp.clip(
            nn.softplus(alpha_raw)
            + min_concentration,
            min_concentration,
            max_concentration,
        )
        beta = jnp.clip(
            nn.softplus(beta_raw)
            + min_concentration,
            min_concentration,
            max_concentration,
        )

        pi = distrax.Independent(
            distrax.Beta(
                alpha,
                beta,
            ),
            reinterpreted_batch_ndims=1,
        )

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(latent)
        critic = nn.relu(
            critic
        )
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return (
            pi,
            jnp.squeeze(
                critic,
                axis=-1,
            ),
        )


def load_beta_policy(
    checkpoint_path,
    policy_cfg,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Policy checkpoint not found: "
            f"{checkpoint_path}"
        )

    input_dim = int(
        policy_cfg.INPUT_DIM
    )
    action_dim = int(
        policy_cfg.ACTION_DIM
    )

    network_config = {
        "FC_DIM_SIZE": int(
            policy_cfg.FC_DIM_SIZE
        ),
        "FF_LATENT_DIM": int(
            policy_cfg.FF_LATENT_DIM
        ),
        "BETA_MIN_CONCENTRATION": float(
            policy_cfg.BETA_MIN_CONCENTRATION
        ),
        "BETA_MAX_CONCENTRATION": float(
            policy_cfg.BETA_MAX_CONCENTRATION
        ),
    }

    network = (
        ActorCriticFFMatchedBeta(
            action_dim=action_dim,
            config=network_config,
        )
    )

    template_variables = (
        network.init(
            jax.random.PRNGKey(0),
            jnp.zeros(
                (input_dim,),
                dtype=jnp.float32,
            ),
        )
    )

    checkpoint_bytes = (
        checkpoint_path.read_bytes()
    )

    # The continuous-IPPO training script stores TrainState.params exactly as
    # created by network.init(...). In that script TrainState.params therefore
    # contains the top-level "params" collection:
    #
    #     {"params": {"Dense_0": ..., "actor_alpha": ..., ...}}
    #
    # Restore against the full variables template so the checkpoint structure
    # matches exactly. A fallback is included for any older checkpoint that
    # stored only the inner parameter collection.
    try:
        policy_variables = (
            serialization.from_bytes(
                template_variables,
                checkpoint_bytes,
            )
        )
    except ValueError:
        inner_params = (
            serialization.from_bytes(
                template_variables["params"],
                checkpoint_bytes,
            )
        )
        policy_variables = {
            "params": inner_params
        }

    params = policy_variables

    finite_leaves = [
        jnp.all(
            jnp.isfinite(leaf)
        )
        for leaf in jax.tree.leaves(
            params
        )
    ]
    all_finite = bool(
        np.all(
            np.asarray(
                jax.device_get(
                    jnp.stack(
                        finite_leaves
                    )
                )
            )
        )
    )

    if not all_finite:
        raise ValueError(
            "Loaded policy checkpoint contains NaN/Inf parameters. "
            "Use a stable checkpoint from the clipped Beta run."
        )

    return (
        network,
        params,
    )


def deterministic_beta_action(
    network,
    params,
    policy_inputs,
    action_eps: float,
):
    """
    Deterministic execution action = Beta distribution mean.

    Using the mean avoids injecting policy sampling noise into estimator
    diagnostics.
    """
    pi, _ = network.apply(
        params,
        policy_inputs,
    )
    action = pi.mean()
    return jnp.clip(
        action,
        action_eps,
        1.0 - action_eps,
    )
