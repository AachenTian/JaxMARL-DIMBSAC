"""
Continuous-action agent-centric feed-forward IPPO for JaxMARL SimpleSpread.

Policy input (19D for 3 agents / 3 landmarks):
  own velocity                      2
  own position                      2
  landmark relative positions       6
  other-agent relative positions    4
  other-agent absolute velocities   4
  normalized episode step           1
  -----------------------------------
  total                             19

Action:
  Independent 5D Beta policy on (0, 1)^5.

Environment:
  MPE_simple_spread_v3
  continuous actions
  contact_force = 0.0

The PPO network width, trajectory-wise minibatching, and all inherited PPO
hyperparameters remain controlled by the original matched-FF Hydra config.
"""

import json
import os
from pathlib import Path
from typing import Dict, NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
import wandb
from jaxmarl.wrappers.baselines import MPELogWrapper


class ActorCriticFFMatchedBeta(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, x):
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        embedding = nn.relu(embedding)

        # Feed-forward replacement for the RNN recurrent block.
        latent = nn.Dense(
            self.config["FF_LATENT_DIM"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        latent = nn.relu(latent)

        actor = nn.Dense(
            self.config["FF_LATENT_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(latent)
        actor = nn.relu(actor)

        # Two heads parameterize one Beta distribution per action dimension.
        # + BETA_MIN_CONCENTRATION keeps the concentrations safely positive.
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
            self.config["BETA_MIN_CONCENTRATION"]
        )
        max_concentration = float(
            self.config["BETA_MAX_CONCENTRATION"]
        )

        # Numerical stabilization for the Beta actor.
        alpha = jnp.clip(
            nn.softplus(alpha_raw) + min_concentration,
            min_concentration,
            max_concentration,
        )
        beta = jnp.clip(
            nn.softplus(beta_raw) + min_concentration,
            min_concentration,
            max_concentration,
        )

        # Reinterpret the 5 scalar Betas as one joint 5D action.
        # Therefore log_prob() and entropy() are summed over action dimensions,
        # which is what PPO needs for the joint action likelihood ratio.
        pi = distrax.Independent(
            distrax.Beta(alpha, beta),
            reinterpreted_batch_ndims=1,
        )

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(latent)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    policy_input: jnp.ndarray
    info: dict


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {
        agent: x[idx]
        for idx, agent in enumerate(agent_list)
    }


def agent_centric_state_input_dim(env):
    """Dimension of the ego-centric physical-state policy input."""
    return (
        2 * env.dim_p
        + env.num_landmarks * env.dim_p
        + 2 * (env.num_agents - 1) * env.dim_p
        + 1
    )


def build_agent_centric_state_batch(log_env_state, env, num_envs):
    """Build the 19D agent-centric physical-state input for every actor."""
    state = log_env_state.env_state
    per_agent_inputs = []

    for agent_idx in range(env.num_agents):
        own_pos = state.p_pos[:, agent_idx, :]
        own_vel = state.p_vel[:, agent_idx, :]

        landmark_rel_pos = (
            state.p_pos[:, env.num_agents :, :]
            - own_pos[:, None, :]
        )

        other_indices = [
            idx
            for idx in range(env.num_agents)
            if idx != agent_idx
        ]
        other_rel_pos = (
            state.p_pos[:, other_indices, :]
            - own_pos[:, None, :]
        )
        other_abs_vel = state.p_vel[
            :, other_indices, :
        ]

        normalized_step = (
            state.step.astype(jnp.float32)
            / float(env.max_steps)
        ).reshape((num_envs, 1))

        agent_input = jnp.concatenate(
            [
                own_vel,
                own_pos,
                landmark_rel_pos.reshape((num_envs, -1)),
                other_rel_pos.reshape((num_envs, -1)),
                other_abs_vel.reshape((num_envs, -1)),
                normalized_step,
            ],
            axis=-1,
        )
        per_agent_inputs.append(agent_input)

    return jnp.stack(
        per_agent_inputs,
        axis=0,
    ).reshape((env.num_agents * num_envs, -1))


def build_policy_input(mode, obs, env_state, env, config):
    del obs

    if mode == "agent_centric_state_abs_vel_step":
        return build_agent_centric_state_batch(
            env_state,
            env,
            config["NUM_ENVS"],
        )

    raise ValueError(
        f"Unknown INPUT_MODE: {mode}"
    )


def make_train(config):
    base_env = jaxmarl.make(
        config["ENV_NAME"],
        **config.get("ENV_KWARGS", {}),
    )

    action_space = base_env.action_space(
        base_env.agents[0]
    )

    if len(action_space.shape) != 1:
        raise ValueError(
            "Continuous Beta IPPO expects a vector Box action space, "
            f"got shape {action_space.shape}."
        )

    action_dim = action_space.shape[0]

    if action_dim != 5:
        raise ValueError(
            "SimpleSpread continuous action is expected to be 5D, "
            f"got {action_dim}D."
        )

    if float(base_env.contact_force) != 0.0:
        raise ValueError(
            "This experiment requires contact_force=0.0, "
            f"got {base_env.contact_force}."
        )

    config["NUM_ACTORS"] = (
        base_env.num_agents
        * config["NUM_ENVS"]
    )
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"])
        // config["NUM_STEPS"]
        // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"]
        * config["NUM_STEPS"]
        // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"]
        / base_env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    input_mode = config["INPUT_MODE"]

    if input_mode != "agent_centric_state_abs_vel_step":
        raise ValueError(
            "This script expects "
            "INPUT_MODE='agent_centric_state_abs_vel_step', "
            f"got {input_mode}."
        )

    policy_input_dim = (
        agent_centric_state_input_dim(
            base_env
        )
    )
    standard_obs_dim = (
        base_env.observation_space(
            base_env.agents[0]
        ).shape[0]
    )
    expected_policy_input_dim = (
        standard_obs_dim + 1
    )

    if (
        policy_input_dim
        != expected_policy_input_dim
    ):
        raise ValueError(
            "Unexpected policy input dimension: "
            f"{policy_input_dim}; expected "
            f"{expected_policy_input_dim}."
        )

    env = MPELogWrapper(base_env)

    def linear_schedule(count):
        frac = (
            1.0
            - (
                count
                // (
                    config[
                        "NUM_MINIBATCHES"
                    ]
                    * config[
                        "UPDATE_EPOCHS"
                    ]
                )
            )
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        network = (
            ActorCriticFFMatchedBeta(
                action_dim=action_dim,
                config=config,
            )
        )

        rng, init_rng = (
            jax.random.split(rng)
        )
        init_x = jnp.zeros(
            (policy_input_dim,),
            dtype=jnp.float32,
        )
        network_params = network.init(
            init_rng,
            init_x,
        )

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(
                    config[
                        "MAX_GRAD_NORM"
                    ]
                ),
                optax.adam(
                    learning_rate=(
                        linear_schedule
                    ),
                    eps=1e-5,
                ),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(
                    config[
                        "MAX_GRAD_NORM"
                    ]
                ),
                optax.adam(
                    config["LR"],
                    eps=1e-5,
                ),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, reset_rng = (
            jax.random.split(rng)
        )
        reset_rng = (
            jax.random.split(
                reset_rng,
                config["NUM_ENVS"],
            )
        )
        obs, env_state = jax.vmap(
            env.reset,
            in_axes=(0,),
        )(reset_rng)

        def _update_step(
            update_runner_state,
            unused,
        ):
            del unused

            (
                runner_state,
                update_steps,
            ) = update_runner_state

            def _env_step(
                runner_state,
                unused,
            ):
                del unused

                (
                    train_state,
                    env_state,
                    last_obs,
                    rng,
                ) = runner_state

                policy_input = (
                    build_policy_input(
                        input_mode,
                        last_obs,
                        env_state,
                        env,
                        config,
                    )
                )

                rng, action_rng = (
                    jax.random.split(rng)
                )
                pi, value = (
                    network.apply(
                        train_state.params,
                        policy_input,
                    )
                )

                # Beta samples lie in [0, 1], but values numerically equal
                # to 0 or 1 can make Beta log_prob unstable. Keep the action
                # inside the open interval before both PPO log_prob and env.step.
                action = pi.sample(
                    seed=action_rng
                )
                action_eps = float(
                    config["ACTION_EPS"]
                )
                action = jnp.clip(
                    action,
                    action_eps,
                    1.0 - action_eps,
                )
                log_prob = pi.log_prob(
                    action
                )

                env_action = unbatchify(
                    action,
                    env.agents,
                    config["NUM_ENVS"],
                    env.num_agents,
                )

                rng, step_rng = (
                    jax.random.split(rng)
                )
                step_keys = (
                    jax.random.split(
                        step_rng,
                        config["NUM_ENVS"],
                    )
                )

                (
                    next_obs,
                    next_env_state,
                    reward,
                    done,
                    info,
                ) = jax.vmap(
                    env.step,
                    in_axes=(0, 0, 0),
                )(
                    step_keys,
                    env_state,
                    env_action,
                )

                info = jax.tree.map(
                    lambda x: x.reshape(
                        (
                            config[
                                "NUM_ACTORS"
                            ],
                        )
                    ),
                    info,
                )

                transition = Transition(
                    global_done=jnp.tile(
                        done["__all__"],
                        env.num_agents,
                    ),
                    action=action,
                    value=value.squeeze(),
                    reward=batchify(
                        reward,
                        env.agents,
                        config["NUM_ACTORS"],
                    ).squeeze(),
                    log_prob=log_prob,
                    policy_input=(
                        policy_input
                    ),
                    info=info,
                )

                runner_state = (
                    train_state,
                    next_env_state,
                    next_obs,
                    rng,
                )
                return (
                    runner_state,
                    transition,
                )

            (
                runner_state,
                traj_batch,
            ) = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                config["NUM_STEPS"],
            )

            (
                train_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state

            last_input = (
                build_policy_input(
                    input_mode,
                    last_obs,
                    env_state,
                    env,
                    config,
                )
            )
            _, last_value = network.apply(
                train_state.params,
                last_input,
            )

            def _calculate_gae(
                traj_batch,
                last_value,
            ):
                def _get_advantages(
                    gae_and_next_value,
                    transition,
                ):
                    (
                        gae,
                        next_value,
                    ) = gae_and_next_value

                    delta = (
                        transition.reward
                        + config["GAMMA"]
                        * next_value
                        * (
                            1
                            - transition
                            .global_done
                        )
                        - transition.value
                    )
                    gae = (
                        delta
                        + config[
                            "GAMMA"
                        ]
                        * config[
                            "GAE_LAMBDA"
                        ]
                        * (
                            1
                            - transition
                            .global_done
                        )
                        * gae
                    )

                    return (
                        (
                            gae,
                            transition.value,
                        ),
                        gae,
                    )

                _, advantages = (
                    jax.lax.scan(
                        _get_advantages,
                        (
                            jnp.zeros_like(
                                last_value
                            ),
                            last_value,
                        ),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )
                )

                return (
                    advantages,
                    (
                        advantages
                        + traj_batch.value
                    ),
                )

            (
                advantages,
                targets,
            ) = _calculate_gae(
                traj_batch,
                last_value,
            )

            def _update_epoch(
                update_state,
                unused,
            ):
                del unused

                def _update_minibatch(
                    train_state,
                    batch_info,
                ):
                    (
                        traj_mb,
                        advantages_mb,
                        targets_mb,
                    ) = batch_info

                    def _loss_fn(
                        params,
                        traj_mb,
                        gae,
                        targets,
                    ):
                        (
                            pi,
                            value,
                        ) = network.apply(
                            params,
                            traj_mb.policy_input,
                        )

                        # Independent(Beta) returns one joint log-probability
                        # per actor after summing the 5 action dimensions.
                        log_prob = pi.log_prob(
                            traj_mb.action
                        )

                        value_pred_clipped = (
                            traj_mb.value
                            + (
                                value
                                - traj_mb.value
                            ).clip(
                                -config[
                                    "CLIP_EPS"
                                ],
                                config[
                                    "CLIP_EPS"
                                ],
                            )
                        )

                        value_losses = (
                            jnp.square(
                                value
                                - targets
                            )
                        )
                        (
                            value_losses_clipped
                        ) = jnp.square(
                            value_pred_clipped
                            - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                value_losses,
                                value_losses_clipped,
                            ).mean()
                        )

                        logratio = (
                            log_prob
                            - traj_mb.log_prob
                        )
                        ratio = jnp.exp(
                            logratio
                        )

                        gae = (
                            gae - gae.mean()
                        ) / (
                            gae.std() + 1e-8
                        )

                        loss_actor1 = (
                            ratio * gae
                        )
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0
                                - config[
                                    "CLIP_EPS"
                                ],
                                1.0
                                + config[
                                    "CLIP_EPS"
                                ],
                            )
                            * gae
                        )
                        loss_actor = (
                            -jnp.minimum(
                                loss_actor1,
                                loss_actor2,
                            ).mean()
                        )

                        # This is differential entropy of the joint 5D
                        # continuous action distribution and can be negative.
                        entropy = (
                            pi.entropy().mean()
                        )

                        approx_kl = (
                            (
                                ratio - 1
                            )
                            - logratio
                        ).mean()
                        clip_frac = jnp.mean(
                            jnp.abs(
                                ratio - 1
                            )
                            > config[
                                "CLIP_EPS"
                            ]
                        )

                        total_loss = (
                            loss_actor
                            + config[
                                "VF_COEF"
                            ]
                            * value_loss
                            - config[
                                "ENT_COEF"
                            ]
                            * entropy
                        )

                        return (
                            total_loss,
                            (
                                value_loss,
                                loss_actor,
                                entropy,
                                ratio,
                                approx_kl,
                                clip_frac,
                            ),
                        )

                    grad_fn = (
                        jax.value_and_grad(
                            _loss_fn,
                            has_aux=True,
                        )
                    )
                    total_loss, grads = (
                        grad_fn(
                            train_state.params,
                            traj_mb,
                            advantages_mb,
                            targets_mb,
                        )
                    )
                    train_state = (
                        train_state
                        .apply_gradients(
                            grads=grads
                        )
                    )

                    return (
                        train_state,
                        total_loss,
                    )

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, shuffle_rng = (
                    jax.random.split(rng)
                )

                # Preserve full actor trajectories exactly as in the matched
                # FF diagnostic used against the official RNN baseline.
                permutation = (
                    jax.random.permutation(
                        shuffle_rng,
                        config[
                            "NUM_ACTORS"
                        ],
                    )
                )

                batch = (
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                shuffled_batch = (
                    jax.tree.map(
                        lambda x: jnp.take(
                            x,
                            permutation,
                            axis=1,
                        ),
                        batch,
                    )
                )
                minibatches = (
                    jax.tree.map(
                        lambda x: (
                            jnp.swapaxes(
                                jnp.reshape(
                                    x,
                                    [
                                        x.shape[
                                            0
                                        ],
                                        config[
                                            "NUM_MINIBATCHES"
                                        ],
                                        -1,
                                    ]
                                    + list(
                                        x.shape[
                                            2:
                                        ]
                                    ),
                                ),
                                1,
                                0,
                            )
                        ),
                        shuffled_batch,
                    )
                )

                (
                    train_state,
                    loss_info,
                ) = jax.lax.scan(
                    _update_minibatch,
                    train_state,
                    minibatches,
                )

                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )

                return (
                    update_state,
                    loss_info,
                )

            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )

            (
                update_state,
                loss_info,
            ) = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                config[
                    "UPDATE_EPOCHS"
                ],
            )

            train_state = (
                update_state[0]
            )
            rng = update_state[-1]

            metric = jax.tree.map(
                lambda x: x.reshape(
                    (
                        config[
                            "NUM_STEPS"
                        ],
                        config[
                            "NUM_ENVS"
                        ],
                        env.num_agents,
                    )
                ),
                traj_batch.info,
            )

            ratio_0 = (
                loss_info[1][3]
                .at[0, 0]
                .get()
                .mean()
            )
            loss_info = jax.tree.map(
                lambda x: x.mean(),
                loss_info,
            )

            metric["loss"] = {
                "total_loss": (
                    loss_info[0]
                ),
                "value_loss": (
                    loss_info[1][0]
                ),
                "actor_loss": (
                    loss_info[1][1]
                ),
                "entropy": (
                    loss_info[1][2]
                ),
                "ratio": (
                    loss_info[1][3]
                ),
                "ratio_0": ratio_0,
                "approx_kl": (
                    loss_info[1][4]
                ),
                "clip_frac": (
                    loss_info[1][5]
                ),
            }

            # Action diagnostics are important for checking that the Beta actor
            # is using the continuous [0, 1]^5 action space sensibly.
            metric["action_stats"] = {
                "mean": (
                    traj_batch.action.mean()
                ),
                "std": (
                    traj_batch.action.std()
                ),
                "min": (
                    traj_batch.action.min()
                ),
                "max": (
                    traj_batch.action.max()
                ),
            }

            for action_idx in range(
                action_dim
            ):
                metric[
                    "action_stats"
                ][
                    f"mean_a{action_idx}"
                ] = (
                    traj_batch.action[
                        ..., action_idx
                    ].mean()
                )
                metric[
                    "action_stats"
                ][
                    f"std_a{action_idx}"
                ] = (
                    traj_batch.action[
                        ..., action_idx
                    ].std()
                )

            metric[
                "update_steps"
            ] = update_steps

            def callback(metric):
                completed = (
                    metric[
                        "returned_episode"
                    ][:, :, 0]
                )
                returns = (
                    metric[
                        "returned_episode_returns"
                    ][:, :, 0]
                )

                if np.any(completed):
                    mean_return = float(
                        np.mean(
                            returns[
                                completed
                            ]
                        )
                    )
                else:
                    mean_return = (
                        float("nan")
                    )

                env_step = int(
                    metric[
                        "update_steps"
                    ]
                    * config[
                        "NUM_ENVS"
                    ]
                    * config[
                        "NUM_STEPS"
                    ]
                )

                log_data = {
                    "env_step": env_step,

                    # Same metric name as the previous discrete IPPO runs.
                    # Set the W&B panel X-axis to env_step for direct overlay.
                    "returns": mean_return,

                    # Keep the hierarchical copy as well.
                    "rollout/returns": mean_return,
                    (
                        "train/total_loss"
                    ): float(
                        metric[
                            "loss"
                        ][
                            "total_loss"
                        ]
                    ),
                    (
                        "train/value_loss"
                    ): float(
                        metric[
                            "loss"
                        ][
                            "value_loss"
                        ]
                    ),
                    (
                        "train/actor_loss"
                    ): float(
                        metric[
                            "loss"
                        ][
                            "actor_loss"
                        ]
                    ),
                    (
                        "train/entropy"
                    ): float(
                        metric[
                            "loss"
                        ][
                            "entropy"
                        ]
                    ),
                    (
                        "train/ratio"
                    ): float(
                        metric[
                            "loss"
                        ][
                            "ratio"
                        ]
                    ),
                    (
                        "train/ratio_0"
                    ): float(
                        metric[
                            "loss"
                        ][
                            "ratio_0"
                        ]
                    ),
                    (
                        "train/approx_kl"
                    ): float(
                        metric[
                            "loss"
                        ][
                            "approx_kl"
                        ]
                    ),
                    (
                        "train/clip_frac"
                    ): float(
                        metric[
                            "loss"
                        ][
                            "clip_frac"
                        ]
                    ),
                    (
                        "policy/action_mean"
                    ): float(
                        metric[
                            "action_stats"
                        ][
                            "mean"
                        ]
                    ),
                    (
                        "policy/action_std"
                    ): float(
                        metric[
                            "action_stats"
                        ][
                            "std"
                        ]
                    ),
                    (
                        "policy/action_min"
                    ): float(
                        metric[
                            "action_stats"
                        ][
                            "min"
                        ]
                    ),
                    (
                        "policy/action_max"
                    ): float(
                        metric[
                            "action_stats"
                        ][
                            "max"
                        ]
                    ),
                }

                for action_idx in range(
                    action_dim
                ):
                    log_data[
                        (
                            "policy/"
                            f"action_mean_a{action_idx}"
                        )
                    ] = float(
                        metric[
                            "action_stats"
                        ][
                            f"mean_a{action_idx}"
                        ]
                    )
                    log_data[
                        (
                            "policy/"
                            f"action_std_a{action_idx}"
                        )
                    ] = float(
                        metric[
                            "action_stats"
                        ][
                            f"std_a{action_idx}"
                        ]
                    )

                wandb.log(
                    log_data
                )

            jax.experimental.io_callback(
                callback,
                None,
                metric,
            )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                rng,
            )

            return (
                (
                    runner_state,
                    update_steps + 1,
                ),
                metric,
            )

        runner_state = (
            train_state,
            env_state,
            obs,
            rng,
        )

        (
            runner_state,
            metric,
        ) = jax.lax.scan(
            _update_step,
            (
                runner_state,
                0,
            ),
            None,
            config["NUM_UPDATES"],
        )

        return {
            "runner_state": (
                runner_state
            ),
            "metrics": metric,
            "policy_input_dim": (
                jnp.asarray(
                    policy_input_dim
                )
            ),
            "action_dim": (
                jnp.asarray(
                    action_dim
                )
            ),
        }

    return train


def save_policy_checkpoint(
    params,
    config,
    policy_input_dim,
    action_dim,
    wandb_run,
):
    checkpoint_cfg = (
        config.get(
            "CHECKPOINT",
            {},
        )
    )

    if not checkpoint_cfg.get(
        "ENABLED",
        True,
    ):
        return None

    checkpoint_dir = Path(
        checkpoint_cfg.get(
            "DIR",
            (
                "checkpoints/"
                "ippo_ff_mpe_agent_state_"
                "abs_vel_step_continuous"
            ),
        )
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = checkpoint_cfg.get(
        "FILE_PREFIX",
        (
            "ippo_ff_abs_vel_step_"
            "beta_no_collision"
        ),
    )
    seed = int(config["SEED"])

    params_path = (
        checkpoint_dir
        / f"{prefix}_seed{seed}.msgpack"
    )
    metadata_path = (
        checkpoint_dir
        / f"{prefix}_seed{seed}.json"
    )

    params_path.write_bytes(
        serialization.to_bytes(
            params
        )
    )

    metadata = {
        "format": (
            "flax.serialization.to_bytes"
        ),
        "network_class": (
            "ActorCriticFFMatchedBeta"
        ),
        "action_distribution": (
            "Independent(Beta)"
        ),
        "policy_input_mode": (
            config["INPUT_MODE"]
        ),
        "policy_input_dim": int(
            policy_input_dim
        ),
        "action_dim": int(
            action_dim
        ),
        "action_range": [
            0.0,
            1.0,
        ],
        "seed": seed,
        "env_name": (
            config["ENV_NAME"]
        ),
        "env_kwargs": (
            config.get(
                "ENV_KWARGS",
                {},
            )
        ),
        "beta_min_concentration": float(
            config[
                "BETA_MIN_CONCENTRATION"
            ]
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    if wandb_run is not None:
        wandb_run.summary[
            "checkpoint/params_path"
        ] = str(params_path)
        wandb_run.summary[
            "checkpoint/metadata_path"
        ] = str(metadata_path)
        wandb_run.summary[
            "policy/action_dim"
        ] = int(action_dim)
        wandb_run.summary[
            "policy/input_dim"
        ] = int(policy_input_dim)
        wandb_run.summary[
            "environment/contact_force"
        ] = float(
            config[
                "ENV_KWARGS"
            ][
                "contact_force"
            ]
        )

        if checkpoint_cfg.get(
            "LOG_WANDB_ARTIFACT",
            True,
        ):
            artifact = wandb.Artifact(
                name=(
                    f"{prefix}-seed{seed}"
                ),
                type="model",
                metadata=metadata,
            )
            artifact.add_file(
                str(params_path)
            )
            artifact.add_file(
                str(metadata_path)
            )
            wandb_run.log_artifact(
                artifact
            )

    return params_path


@hydra.main(
    version_base=None,
    config_path="config",
    config_name=(
        "ippo_ff_mpe_agent_state_"
        "abs_vel_step_continuous"
    ),
)
def main(config):
    config = OmegaConf.to_container(
        config,
        resolve=True,
    )

    config_tags = list(
        config.get(
            "TAGS",
            [],
        )
    )
    env_tags = [
        tag
        for tag in os.environ.get(
            "WANDB_TAGS",
            "",
        ).split(",")
        if tag
    ]

    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=(
            config_tags
            + env_tags
        ),
        group=(
            os.environ.get(
                "WANDB_RUN_GROUP"
            )
            or config.get(
                "WANDB_GROUP"
            )
            or None
        ),
        name=(
            os.environ.get(
                "WANDB_NAME"
            )
            or config.get(
                "WANDB_RUN_NAME"
            )
            or None
        ),
        config=config,
        mode=config["WANDB_MODE"],
    )

    # Use env_step as the x-axis for all training/policy curves.
    wandb.define_metric(
        "env_step"
    )
    wandb.define_metric(
        "returns",
        step_metric="env_step",
    )
    wandb.define_metric(
        "rollout/*",
        step_metric="env_step",
    )
    wandb.define_metric(
        "train/*",
        step_metric="env_step",
    )
    wandb.define_metric(
        "policy/*",
        step_metric="env_step",
    )

    print(
        "IPPO FF continuous Beta | "
        "agent-centric abs velocity + normalized step"
    )
    print(
        "env:",
        config["ENV_NAME"],
    )
    print(
        "env kwargs:",
        config["ENV_KWARGS"],
    )
    print(
        "input mode:",
        config["INPUT_MODE"],
    )
    print(
        "action distribution: Independent Beta"
    )

    rng = jax.random.PRNGKey(
        config["SEED"]
    )

    train_jit = jax.jit(
        make_train(config),
        device=jax.devices()[0],
    )

    out = train_jit(rng)

    # runner_state structure after the outer scan:
    # ((train_state, env_state, obs, rng), update_steps)
    final_train_state = (
        out[
            "runner_state"
        ][0][0]
    )
    trained_params = jax.device_get(
        final_train_state.params
    )

    policy_input_dim = int(
        jax.device_get(
            out[
                "policy_input_dim"
            ]
        )
    )
    action_dim = int(
        jax.device_get(
            out[
                "action_dim"
            ]
        )
    )

    checkpoint_path = (
        save_policy_checkpoint(
            params=trained_params,
            config=config,
            policy_input_dim=(
                policy_input_dim
            ),
            action_dim=action_dim,
            wandb_run=run,
        )
    )

    if checkpoint_path is not None:
        print(
            "Saved policy checkpoint:",
            checkpoint_path,
        )

    run.finish()


if __name__ == "__main__":
    main()
