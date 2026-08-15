"""
Agent-centric physical-state feed-forward PPO for JaxMARL SimpleSpread.

Each shared policy receives a different ego-centric physical state:
  - ego velocity
  - ego position
  - landmark positions relative to ego
  - other-agent positions relative to ego
  - other-agent absolute velocities

For the default 3-agent / 3-landmark task this is 18 dimensions, the same
input dimensionality as the standard SimpleSpread observation. The standard
observation spends its final 4 dimensions on communication from the two other
agents; SimpleSpread agents are silent by default, so those entries are zero.
This diagnostic replaces those zero communication features with the missing
absolute velocities while keeping the network and PPO optimization regime
matched to the IPPO-RNN baseline.

This remains decentralized with respect to the constructed per-agent input:
there is no shared global-state vector and no agent-ID feature. Parameters are
shared, but each agent receives its own ego-centric state representation.
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


def agent_centric_state_input_dim(env):
    """Dimension of the ego-centric physical state.

    Layout:
      own velocity:                    2
      own position:                    2
      landmark relative positions:    2 * num_landmarks
      other-agent relative positions: 2 * (num_agents - 1)
      other-agent absolute velocities:2 * (num_agents - 1)
      agent one-hot ID:                 num_agents
    """
    return (
        2 * env.dim_p
        + env.num_landmarks * env.dim_p
        + 2 * (env.num_agents - 1) * env.dim_p
        + env.num_agents
    )


def build_agent_centric_state_batch(log_env_state, env, num_envs):
    """Build one ego-centric physical-state vector per actor.

    The entity order in JaxMARL MPE is agents first, then landmarks. For each
    controlled agent i, other agents are kept in ascending agent-index order
    with i removed. Other-agent velocity uses the absolute physical velocity v_j.
    """
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

        agent_id = jnp.broadcast_to(
            jax.nn.one_hot(
                agent_idx,
                env.num_agents,
                dtype=jnp.float32,
            ),
            (num_envs, env.num_agents),
        )

        agent_input = jnp.concatenate(
            [
                own_vel,
                own_pos,
                landmark_rel_pos.reshape((num_envs, -1)),
                other_rel_pos.reshape((num_envs, -1)),
                other_abs_vel.reshape((num_envs, -1)),
                agent_id,
            ],
            axis=-1,
        )
        per_agent_inputs.append(agent_input)

    # Agent-major ordering matches batchify/unbatchify in the IPPO baselines.
    return jnp.stack(
        per_agent_inputs,
        axis=0,
    ).reshape((env.num_agents * num_envs, -1))


def build_policy_input(mode, obs, env_state, env, config):
    if mode == "agent_centric_state_abs_vel_onehot":
        return build_agent_centric_state_batch(
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
    if input_mode != "agent_centric_state_abs_vel_onehot":
        raise ValueError(
            "This script expects INPUT_MODE='agent_centric_state_abs_vel_onehot', "
            f"got {input_mode}"
        )

    policy_input_dim = agent_centric_state_input_dim(base_env)
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
                            metric["update_steps"]
                            * config["NUM_ENVS"]
                            * config["NUM_STEPS"]
                        ),
                        "input_mode": 2,
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
    config_name="ippo_ff_mpe_agent_state_abs_vel_onehot",
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
        "IPPO FF agent-centric state (absolute other velocity + one-hot ID) | "
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
