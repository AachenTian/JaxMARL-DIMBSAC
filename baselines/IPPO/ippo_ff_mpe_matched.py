"""
Controlled feed-forward PPO ablations for JaxMARL MPE SimpleSpread.

Modes:
  local:
      Parameter-sharing PPO with each agent's ordinary local observation.
      Architecture and PPO settings are designed to match the official
      IPPO-RNN baseline as closely as possible without recurrence.

  full_state:
      Diagnostic centralized-execution variant. Each policy input contains
      the full MPE environment state plus a one-hot agent ID. This is NOT
      decentralized IPPO and should be treated as an observability upper bound.

The feed-forward latent block replaces the GRU while preserving trajectory-wise
minibatching used by the RNN baseline.
"""

import os
from typing import Dict, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
import wandb
from jaxmarl.wrappers.baselines import MPELogWrapper


class ActorCriticFFMatched(nn.Module):
    action_dim: Sequence[int]
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
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        pi = distrax.Categorical(logits=logits)

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
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def full_state_input_dim(env):
    # State fields used by SimpleSpread:
    # p_pos, p_vel, c, step, done, plus controlled-agent identity.
    physical_dim = (
        env.num_entities * env.dim_p
        + env.num_entities * env.dim_p
        + env.num_agents * env.dim_c
        + 1
        + 1
    )
    return physical_dim + env.num_agents


def build_full_state_batch(log_env_state, env, num_envs):
    """Return agent-major full-state policy inputs.

    For 3-agent / 3-landmark SimpleSpread this is:
      p_pos: 12
      p_vel: 12
      c: 6
      normalized step: 1
      done: 1
      agent ID: 3
      total: 35 dimensions per actor.
    """
    state = log_env_state.env_state

    global_state = jnp.concatenate(
        [
            state.p_pos.reshape((num_envs, -1)),
            state.p_vel.reshape((num_envs, -1)),
            state.c.reshape((num_envs, -1)),
            (
                state.step.astype(jnp.float32)
                / float(env.max_steps)
            ).reshape((num_envs, 1)),
            state.done.astype(jnp.float32).reshape((num_envs, 1)),
        ],
        axis=-1,
    )

    agent_ids = jnp.eye(env.num_agents, dtype=jnp.float32)
    tiled_state = jnp.broadcast_to(
        global_state[None, :, :],
        (env.num_agents, num_envs, global_state.shape[-1]),
    )
    tiled_ids = jnp.broadcast_to(
        agent_ids[:, None, :],
        (env.num_agents, num_envs, env.num_agents),
    )

    return jnp.concatenate(
        [tiled_state, tiled_ids],
        axis=-1,
    ).reshape((env.num_agents * num_envs, -1))


def build_policy_input(mode, obs, env_state, env, config):
    if mode == "local":
        return batchify(
            obs,
            env.agents,
            config["NUM_ACTORS"],
        )
    if mode == "full_state":
        return build_full_state_batch(
            env_state,
            env,
            config["NUM_ENVS"],
        )
    raise ValueError(f"Unknown INPUT_MODE: {mode}")


def make_train(config):
    base_env = jaxmarl.make(
        config["ENV_NAME"],
        **config.get("ENV_KWARGS", {}),
    )
    config["NUM_ACTORS"] = (
        base_env.num_agents * config["NUM_ENVS"]
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
        config["CLIP_EPS"] / base_env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    input_mode = config["INPUT_MODE"]
    if input_mode == "local":
        policy_input_dim = base_env.observation_space(
            base_env.agents[0]
        ).shape[0]
    elif input_mode == "full_state":
        policy_input_dim = full_state_input_dim(base_env)
    else:
        raise ValueError(
            f"INPUT_MODE must be 'local' or 'full_state', got {input_mode}"
        )

    env = MPELogWrapper(base_env)

    def linear_schedule(count):
        frac = (
            1.0
            - (
                count
                // (
                    config["NUM_MINIBATCHES"]
                    * config["UPDATE_EPOCHS"]
                )
            )
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        network = ActorCriticFFMatched(
            env.action_space(env.agents[0]).n,
            config=config,
        )

        rng, init_rng = jax.random.split(rng)
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
                    config["MAX_GRAD_NORM"]
                ),
                optax.adam(
                    learning_rate=linear_schedule,
                    eps=1e-5,
                ),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(
                    config["MAX_GRAD_NORM"]
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

        rng, reset_rng = jax.random.split(rng)
        reset_rng = jax.random.split(
            reset_rng,
            config["NUM_ENVS"],
        )
        obs, env_state = jax.vmap(
            env.reset,
            in_axes=(0,),
        )(reset_rng)

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    rng,
                ) = runner_state

                policy_input = build_policy_input(
                    input_mode,
                    last_obs,
                    env_state,
                    env,
                    config,
                )

                rng, action_rng = jax.random.split(rng)
                pi, value = network.apply(
                    train_state.params,
                    policy_input,
                )
                action = pi.sample(seed=action_rng)
                log_prob = pi.log_prob(action)

                env_action = unbatchify(
                    action,
                    env.agents,
                    config["NUM_ENVS"],
                    env.num_agents,
                )
                env_action = {
                    k: v.squeeze()
                    for k, v in env_action.items()
                }

                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(
                    step_rng,
                    config["NUM_ENVS"],
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
                        (config["NUM_ACTORS"],)
                    ),
                    info,
                )

                transition = Transition(
                    global_done=jnp.tile(
                        done["__all__"],
                        env.num_agents,
                    ),
                    action=action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(
                        reward,
                        env.agents,
                        config["NUM_ACTORS"],
                    ).squeeze(),
                    log_prob=log_prob.squeeze(),
                    policy_input=policy_input,
                    info=info,
                )

                runner_state = (
                    train_state,
                    next_env_state,
                    next_obs,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
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

            last_input = build_policy_input(
                input_mode,
                last_obs,
                env_state,
                env,
                config,
            )
            _, last_value = network.apply(
                train_state.params,
                last_input,
            )

            def _calculate_gae(traj_batch, last_value):
                def _get_advantages(
                    gae_and_next_value,
                    transition,
                ):
                    gae, next_value = gae_and_next_value

                    delta = (
                        transition.reward
                        + config["GAMMA"]
                        * next_value
                        * (1 - transition.global_done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1 - transition.global_done)
                        * gae
                    )
                    return (
                        (gae, transition.value),
                        gae,
                    )

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (
                        jnp.zeros_like(last_value),
                        last_value,
                    ),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return (
                    advantages,
                    advantages + traj_batch.value,
                )

            advantages, targets = _calculate_gae(
                traj_batch,
                last_value,
            )


            def _update_epoch(update_state, unused):
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
                        pi, value = network.apply(
                            params,
                            traj_mb.policy_input,
                        )
                        log_prob = pi.log_prob(
                            traj_mb.action
                        )

                        value_pred_clipped = (
                            traj_mb.value
                            + (
                                value
                                - traj_mb.value
                            ).clip(
                                -config["CLIP_EPS"],
                                config["CLIP_EPS"],
                            )
                        )
                        value_losses = jnp.square(
                            value - targets
                        )
                        value_losses_clipped = (
                            jnp.square(
                                value_pred_clipped
                                - targets
                            )
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
                        ratio = jnp.exp(logratio)
                        gae = (
                            gae - gae.mean()
                        ) / (
                            gae.std() + 1e-8
                        )

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0
                                - config["CLIP_EPS"],
                                1.0
                                + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = (
                            -jnp.minimum(
                                loss_actor1,
                                loss_actor2,
                            ).mean()
                        )

                        entropy = pi.entropy().mean()
                        approx_kl = (
                            (ratio - 1)
                            - logratio
                        ).mean()
                        clip_frac = jnp.mean(
                            jnp.abs(ratio - 1)
                            > config["CLIP_EPS"]
                        )

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"]
                            * value_loss
                            - config["ENT_COEF"]
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

                    grad_fn = jax.value_and_grad(
                        _loss_fn,
                        has_aux=True,
                    )
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_mb,
                        advantages_mb,
                        targets_mb,
                    )
                    train_state = (
                        train_state.apply_gradients(
                            grads=grads
                        )
                    )
                    return train_state, total_loss

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

                # Match the official RNN baseline:
                # shuffle complete actor trajectories,
                # preserving the full rollout time axis.
                permutation = (
                    jax.random.permutation(
                        shuffle_rng,
                        config["NUM_ACTORS"],
                    )
                )
                batch = (
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(
                        x,
                        permutation,
                        axis=1,
                    ),
                    batch,
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [
                                x.shape[0],
                                config[
                                    "NUM_MINIBATCHES"
                                ],
                                -1,
                            ]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, loss_info = (
                    jax.lax.scan(
                        _update_minibatch,
                        train_state,
                        minibatches,
                    )
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
            update_state, loss_info = (
                jax.lax.scan(
                    _update_epoch,
                    update_state,
                    None,
                    config["UPDATE_EPOCHS"],
                )
            )

            train_state = update_state[0]
            rng = update_state[-1]

            metric = jax.tree.map(
                lambda x: x.reshape(
                    (
                        config["NUM_STEPS"],
                        config["NUM_ENVS"],
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
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            metric["update_steps"] = update_steps

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
                wandb.log(
                    {
                        "returns": (
                            returns[completed].mean()
                        ),
                        "env_step": (
                            (metric["update_steps"])
                            * config["NUM_ENVS"]
                            * config["NUM_STEPS"]
                        ),
                        "input_mode": (
                            0
                            if input_mode == "local"
                            else 1
                        ),
                        **metric["loss"],
                    }
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
        runner_state, metric = jax.lax.scan(
            _update_step,
            (runner_state, 0),
            None,
            config["NUM_UPDATES"],
        )

        return {
            "runner_state": runner_state,
            "metrics": metric,
        }

    return train


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="ippo_ff_mpe_matched",
)
def main(config):
    config = OmegaConf.to_container(
        config,
        resolve=True,
    )

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            t
            for t in os.environ.get(
                "WANDB_TAGS",
                "",
            ).split(",")
            if t
        ],
        group=(
            os.environ.get(
                "WANDB_RUN_GROUP"
            )
            or None
        ),
        name=(
            os.environ.get("WANDB_NAME")
            or None
        ),
        config=config,
        mode=config["WANDB_MODE"],
    )

    print(
        "IPPO FF ablation | "
        f"input_mode={config['INPUT_MODE']} | "
        f"env={config['ENV_NAME']}"
    )

    rng = jax.random.PRNGKey(
        config["SEED"]
    )
    train_jit = jax.jit(
        make_train(config),
        device=jax.devices()[0],
    )
    train_jit(rng)


if __name__ == "__main__":
    main()
