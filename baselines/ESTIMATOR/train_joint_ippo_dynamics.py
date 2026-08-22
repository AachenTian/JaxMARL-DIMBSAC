"""
Model-free shared Beta IPPO + auxiliary local-dynamics training.

Important separation:

  POLICY TRAINING
  ---------------
  Three real agents interact with the real JaxMARL environment.
  Every agent receives the TRUE 19D training input and samples its real action
  from the shared Beta policy. PPO is trained only from these real on-policy
  transitions. No estimator/model rollout is used to generate PPO data.

  DYNAMICS TRAINING
  -----------------
  The exact same real transitions also supervise three independent local
  probabilistic dynamics ensembles:

      [x_i(real, 4D), a_i(real, 5D)] -> delta x_i(real, 4D)

  where x_i = [px, py, vx, vy]. The dynamics models are auxiliary models for
  later estimator execution; they never choose training actions here.

This therefore remains model-free RL.
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
import jaxmarl
import numpy as np
import optax
import wandb
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import MPELogWrapper
from omegaconf import DictConfig, OmegaConf

from dynamics.checkpoint import save_local_dynamics_checkpoint
from dynamics.ensemble import (
    build_independent_dynamics_ensembles,
    init_dynamics_ensemble,
    predict_dynamics_ensemble,
    replace_ensemble_member,
)
from dynamics.model import ProbabilisticDynamicsModel
from dynamics.normalization import compute_normalization_stats, normalize
from dynamics.policy_replay_buffer import PolicyDynamicsReplayBuffer
from dynamics.trainer import update_dynamics_model


DIM_NAMES = ("px", "py", "vx", "vy")


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

        min_concentration = float(self.config["BETA_MIN_CONCENTRATION"])
        max_concentration = float(self.config["BETA_MAX_CONCENTRATION"])

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


class PPOTransition(NamedTuple):
    global_done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    policy_input: jnp.ndarray
    info: dict


class DynamicsTransition(NamedTuple):
    x: jnp.ndarray
    action: jnp.ndarray
    next_x: jnp.ndarray
    valid: jnp.ndarray


class PPORunnerState(NamedTuple):
    train_state: TrainState
    env_state: object
    obs: object
    rng: jax.Array


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {agent: x[idx] for idx, agent in enumerate(agent_list)}


def agent_centric_state_input_dim(env):
    return (
        2 * env.dim_p
        + env.num_landmarks * env.dim_p
        + 2 * (env.num_agents - 1) * env.dim_p
        + 1
    )


def build_agent_centric_state_batch(log_env_state, env, num_envs):
    """TRUE 19D training input for every real agent."""
    state = log_env_state.env_state
    per_agent_inputs = []

    for agent_idx in range(env.num_agents):
        own_pos = state.p_pos[:, agent_idx, :]
        own_vel = state.p_vel[:, agent_idx, :]

        landmark_rel_pos = (
            state.p_pos[:, env.num_agents:, :]
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
        other_abs_vel = state.p_vel[:, other_indices, :]

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

    return jnp.stack(per_agent_inputs, axis=0).reshape(
        (env.num_agents * num_envs, -1)
    )


def stack_real_agent_state(log_env_state, env):
    """Return [N,A,4] physical x_i=[px,py,vx,vy]."""
    state = log_env_state.env_state
    pos = state.p_pos[:, :env.num_agents, :]
    vel = state.p_vel[:, :env.num_agents, :]
    return jnp.concatenate([pos, vel], axis=-1)


def make_policy_components(cfg: DictConfig):
    ppo = OmegaConf.to_container(cfg.PPO, resolve=True)
    env_kwargs = OmegaConf.to_container(cfg.ENV.KWARGS, resolve=True)

    base_env = jaxmarl.make(
        str(cfg.ENV.NAME),
        **env_kwargs,
    )

    if float(base_env.contact_force) != 0.0:
        raise ValueError(
            "Joint trainer currently expects contact_force=0.0, "
            f"got {base_env.contact_force}."
        )

    action_space = base_env.action_space(base_env.agents[0])
    action_dim = int(action_space.shape[0])
    if action_dim != 5:
        raise ValueError(f"Expected 5D continuous MPE action, got {action_dim}D.")

    num_envs = int(ppo["NUM_ENVS"])
    num_steps = int(ppo["NUM_STEPS"])
    num_actors = int(base_env.num_agents * num_envs)
    num_updates = int(float(ppo["TOTAL_TIMESTEPS"])) // num_steps // num_envs

    ppo["NUM_ACTORS"] = num_actors
    ppo["NUM_UPDATES"] = num_updates
    ppo["MINIBATCH_SIZE"] = (
        num_actors * num_steps // int(ppo["NUM_MINIBATCHES"])
    )
    if bool(ppo["SCALE_CLIP_EPS"]):
        ppo["CLIP_EPS"] = float(ppo["CLIP_EPS"]) / base_env.num_agents

    policy_input_dim = agent_centric_state_input_dim(base_env)
    if policy_input_dim != 19:
        raise ValueError(
            f"Expected 19D SimpleSpread training input, got {policy_input_dim}D."
        )

    env = MPELogWrapper(base_env)
    network = ActorCriticFFMatchedBeta(action_dim=action_dim, config=ppo)

    def linear_schedule(count):
        frac = 1.0 - (
            count
            // (int(ppo["NUM_MINIBATCHES"]) * int(ppo["UPDATE_EPOCHS"]))
        ) / num_updates
        return float(ppo["LR"]) * frac

    if bool(ppo["ANNEAL_LR"]):
        learning_rate = linear_schedule
    else:
        learning_rate = float(ppo["LR"])

    tx = optax.chain(
        optax.clip_by_global_norm(float(ppo["MAX_GRAD_NORM"])),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )

    return (
        ppo,
        base_env,
        env,
        network,
        tx,
        policy_input_dim,
        action_dim,
        num_updates,
    )


def initialize_ppo_runner(rng, ppo, env, network, tx, policy_input_dim):
    rng, init_rng = jax.random.split(rng)
    variables = network.init(
        init_rng,
        jnp.zeros((policy_input_dim,), dtype=jnp.float32),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=variables,
        tx=tx,
    )

    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, int(ppo["NUM_ENVS"]))
    obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

    return PPORunnerState(
        train_state=train_state,
        env_state=env_state,
        obs=obs,
        rng=rng,
    )


def make_one_ppo_update(ppo, env, network, action_dim):
    num_envs = int(ppo["NUM_ENVS"])
    num_steps = int(ppo["NUM_STEPS"])
    num_actors = int(ppo["NUM_ACTORS"])
    num_agents = int(env.num_agents)
    action_eps = float(ppo["ACTION_EPS"])

    def one_update(runner_state, update_idx):
        def _env_step(runner_state, unused):
            del unused
            train_state, env_state, last_obs, rng = runner_state

            policy_input = build_agent_centric_state_batch(
                env_state,
                env,
                num_envs,
            )
            current_x = stack_real_agent_state(env_state, env)

            rng, action_rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, policy_input)
            action = pi.sample(seed=action_rng)
            action = jnp.clip(action, action_eps, 1.0 - action_eps)
            log_prob = pi.log_prob(action)

            joint_action = action.reshape(
                (num_agents, num_envs, action_dim)
            ).transpose((1, 0, 2))
            env_action = unbatchify(
                action,
                env.agents,
                num_envs,
                num_agents,
            )

            rng, step_rng = jax.random.split(rng)
            step_keys = jax.random.split(step_rng, num_envs)
            next_obs, next_env_state, reward, done, info = jax.vmap(
                env.step,
                in_axes=(0, 0, 0),
            )(
                step_keys,
                env_state,
                env_action,
            )

            next_x = stack_real_agent_state(next_env_state, env)
            global_done = done["__all__"]

            info_actor = jax.tree.map(
                lambda x: x.reshape((num_actors,)),
                info,
            )

            ppo_transition = PPOTransition(
                global_done=jnp.tile(global_done, num_agents),
                action=action,
                value=value.squeeze(),
                reward=batchify(reward, env.agents, num_actors).squeeze(),
                log_prob=log_prob,
                policy_input=policy_input,
                info=info_actor,
            )

            # MPELogWrapper auto-resets completed episodes. Therefore any
            # transition whose global_done is true may contain a reset state in
            # next_env_state and must NOT be a dynamics target.
            dynamics_transition = DynamicsTransition(
                x=current_x,
                action=joint_action,
                next_x=next_x,
                valid=jnp.logical_not(global_done),
            )

            return (
                PPORunnerState(
                    train_state=train_state,
                    env_state=next_env_state,
                    obs=next_obs,
                    rng=rng,
                ),
                (ppo_transition, dynamics_transition),
            )

        runner_state, scan_out = jax.lax.scan(
            _env_step,
            runner_state,
            None,
            num_steps,
        )
        traj_batch, dynamics_batch = scan_out

        train_state = runner_state.train_state
        env_state = runner_state.env_state
        last_obs = runner_state.obs
        rng = runner_state.rng

        last_input = build_agent_centric_state_batch(
            env_state,
            env,
            num_envs,
        )
        _, last_value = network.apply(train_state.params, last_input)

        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            delta = (
                transition.reward
                + float(ppo["GAMMA"])
                * next_value
                * (1 - transition.global_done)
                - transition.value
            )
            gae = (
                delta
                + float(ppo["GAMMA"])
                * float(ppo["GAE_LAMBDA"])
                * (1 - transition.global_done)
                * gae
            )
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_value), last_value),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        targets = advantages + traj_batch.value

        def _update_epoch(update_state, unused):
            del unused
            train_state, rng = update_state
            rng, shuffle_rng = jax.random.split(rng)

            permutation = jax.random.permutation(
                shuffle_rng,
                num_actors,
            )
            batch = (
                traj_batch,
                advantages.squeeze(),
                targets.squeeze(),
            )
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=1),
                batch,
            )
            minibatches = jax.tree.map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [
                            x.shape[0],
                            int(ppo["NUM_MINIBATCHES"]),
                            -1,
                        ]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            def _update_minibatch(train_state, batch_info):
                traj_mb, advantages_mb, targets_mb = batch_info

                def _loss_fn(params):
                    pi, value = network.apply(params, traj_mb.policy_input)
                    log_prob = pi.log_prob(traj_mb.action)

                    value_pred_clipped = traj_mb.value + (
                        value - traj_mb.value
                    ).clip(
                        -float(ppo["CLIP_EPS"]),
                        float(ppo["CLIP_EPS"]),
                    )
                    value_losses = jnp.square(value - targets_mb)
                    value_losses_clipped = jnp.square(
                        value_pred_clipped - targets_mb
                    )
                    value_loss = 0.5 * jnp.maximum(
                        value_losses,
                        value_losses_clipped,
                    ).mean()

                    logratio = log_prob - traj_mb.log_prob
                    ratio = jnp.exp(logratio)
                    normalized_advantage = (
                        advantages_mb - advantages_mb.mean()
                    ) / (advantages_mb.std() + 1e-8)

                    loss_actor1 = ratio * normalized_advantage
                    loss_actor2 = jnp.clip(
                        ratio,
                        1.0 - float(ppo["CLIP_EPS"]),
                        1.0 + float(ppo["CLIP_EPS"]),
                    ) * normalized_advantage
                    actor_loss = -jnp.minimum(
                        loss_actor1,
                        loss_actor2,
                    ).mean()

                    entropy = pi.entropy().mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = jnp.mean(
                        jnp.abs(ratio - 1) > float(ppo["CLIP_EPS"])
                    )
                    total_loss = (
                        actor_loss
                        + float(ppo["VF_COEF"]) * value_loss
                        - float(ppo["ENT_COEF"]) * entropy
                    )
                    aux = (
                        value_loss,
                        actor_loss,
                        entropy,
                        ratio.mean(),
                        approx_kl,
                        clip_frac,
                    )
                    return total_loss, aux

                (loss_and_aux, grads) = jax.value_and_grad(
                    _loss_fn,
                    has_aux=True,
                )(train_state.params)
                total_loss, aux = loss_and_aux
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (total_loss, aux)

            train_state, loss_info = jax.lax.scan(
                _update_minibatch,
                train_state,
                minibatches,
            )
            return (train_state, rng), loss_info

        (train_state, rng), loss_info = jax.lax.scan(
            _update_epoch,
            (train_state, rng),
            None,
            int(ppo["UPDATE_EPOCHS"]),
        )

        # Scalar PPO diagnostics.
        loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
        total_loss = loss_info[0]
        value_loss = loss_info[1][0]
        actor_loss = loss_info[1][1]
        entropy = loss_info[1][2]
        ratio = loss_info[1][3]
        approx_kl = loss_info[1][4]
        clip_frac = loss_info[1][5]

        info_metric = jax.tree.map(
            lambda x: x.reshape((num_steps, num_envs, num_agents)),
            traj_batch.info,
        )
        completed = info_metric["returned_episode"][:, :, 0]
        episode_returns = info_metric["returned_episode_returns"][:, :, 0]
        completed_count = jnp.sum(completed)
        mean_return = jnp.where(
            completed_count > 0,
            jnp.sum(jnp.where(completed, episode_returns, 0.0))
            / completed_count,
            jnp.nan,
        )

        action_means = jnp.mean(traj_batch.action, axis=(0, 1))
        action_stds = jnp.std(traj_batch.action, axis=(0, 1))

        metrics = {
            "mean_return": mean_return,
            "total_loss": total_loss,
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "entropy": entropy,
            "ratio": ratio,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "action_mean": traj_batch.action.mean(),
            "action_std": traj_batch.action.std(),
            "action_min": traj_batch.action.min(),
            "action_max": traj_batch.action.max(),
            "action_means": action_means,
            "action_stds": action_stds,
            "update_idx": update_idx,
        }

        next_runner_state = PPORunnerState(
            train_state=train_state,
            env_state=env_state,
            obs=last_obs,
            rng=rng,
        )
        return next_runner_state, metrics, dynamics_batch

    return one_update


def extract_valid_dynamics_batch(dynamics_batch):
    """Device -> CPU; remove terminal/reset transitions."""
    batch = jax.device_get(dynamics_batch)
    x = np.asarray(batch.x, dtype=np.float32)
    actions = np.asarray(batch.action, dtype=np.float32)
    next_x = np.asarray(batch.next_x, dtype=np.float32)
    valid = np.asarray(batch.valid, dtype=bool)

    # [T,N,A,*] -> [T*N,A,*]
    x = x.reshape((-1,) + x.shape[2:])
    actions = actions.reshape((-1,) + actions.shape[2:])
    next_x = next_x.reshape((-1,) + next_x.shape[2:])
    valid = valid.reshape(-1)

    return x[valid], actions[valid], next_x[valid]


def initialize_dynamics_from_replay(replay, model, cfg, rng):
    """Freeze normalization stats once, then initialize persistent ensembles."""
    sample_size = min(
        int(cfg.DYNAMICS.NORMALIZATION_SAMPLES),
        len(replay),
    )
    x, actions, next_x = replay.sample_without_replacement(sample_size)

    agent_ensembles = []
    for agent_idx in range(x.shape[1]):
        dynamics_inputs = np.concatenate(
            [x[:, agent_idx], actions[:, agent_idx]],
            axis=-1,
        )
        targets = next_x[:, agent_idx] - x[:, agent_idx]

        input_stats = compute_normalization_stats(jnp.asarray(dynamics_inputs))
        target_stats = compute_normalization_stats(jnp.asarray(targets))

        rng, ensemble_rng = jax.random.split(rng)
        ensemble_state = init_dynamics_ensemble(
            rng=ensemble_rng,
            model=model,
            ensemble_size=int(cfg.DYNAMICS.ENSEMBLE_SIZE),
            input_dim=9,
            input_stats=input_stats,
            target_stats=target_stats,
            learning_rate=float(cfg.DYNAMICS.LR),
        )
        agent_ensembles.append(ensemble_state)

    return rng, build_independent_dynamics_ensembles(agent_ensembles)


def train_dynamics_from_replay(independent_ensembles, replay, cfg, rng):
    """Continue training existing ensemble weights; never reinitialize them."""
    update_fn = jax.jit(update_dynamics_model)

    for agent_idx in range(len(independent_ensembles.agents)):
        ensemble_state = independent_ensembles.agents[agent_idx]

        for member_idx in range(int(cfg.DYNAMICS.ENSEMBLE_SIZE)):
            member_state = ensemble_state.members[member_idx]

            for _ in range(int(cfg.DYNAMICS.GRAD_STEPS_PER_UPDATE)):
                x, actions, next_x = replay.sample(int(cfg.DYNAMICS.BATCH_SIZE))
                dynamics_inputs = np.concatenate(
                    [x[:, agent_idx], actions[:, agent_idx]],
                    axis=-1,
                )
                targets = next_x[:, agent_idx] - x[:, agent_idx]

                normalized_inputs = normalize(
                    jnp.asarray(dynamics_inputs),
                    member_state.input_stats,
                )
                normalized_targets = normalize(
                    jnp.asarray(targets),
                    member_state.target_stats,
                )

                member_state, _ = update_fn(
                    member_state,
                    normalized_inputs,
                    normalized_targets,
                )

            ensemble_state = replace_ensemble_member(
                ensemble_state=ensemble_state,
                member_idx=member_idx,
                new_member_state=member_state,
            )

        agents = list(independent_ensembles.agents)
        agents[agent_idx] = ensemble_state
        independent_ensembles = build_independent_dynamics_ensembles(agents)

    return rng, independent_ensembles


def evaluate_dynamics_on_batch(independent_ensembles, x, actions, next_x):
    """Evaluate ensemble-mean physical RMSE on a fresh real policy batch."""
    metrics = {}
    agent_per_dim = []

    for agent_idx in range(x.shape[1]):
        dynamics_inputs = jnp.asarray(
            np.concatenate(
                [x[:, agent_idx], actions[:, agent_idx]],
                axis=-1,
            )
        )
        targets = jnp.asarray(next_x[:, agent_idx] - x[:, agent_idx])

        member_means, member_variances = predict_dynamics_ensemble(
            independent_ensembles.agents[agent_idx],
            dynamics_inputs,
        )
        ensemble_mean = jnp.mean(member_means, axis=0)
        error = ensemble_mean - targets
        per_dim_rmse = jnp.sqrt(jnp.mean(jnp.square(error), axis=0))
        physical_rmse = jnp.sqrt(jnp.mean(jnp.square(error)))

        # Useful later for the uncertainty work: separate mean aleatoric
        # variance from ensemble disagreement, but do not use either in PPO.
        mean_aleatoric_variance = jnp.mean(member_variances, axis=(0, 1))
        epistemic_variance = jnp.mean(
            jnp.var(member_means, axis=0),
            axis=0,
        )

        metrics[f"dynamics/agent_{agent_idx}/ensemble_rmse"] = float(physical_rmse)
        for dim_idx, dim_name in enumerate(DIM_NAMES):
            metrics[
                f"dynamics/agent_{agent_idx}/rmse_{dim_name}"
            ] = float(per_dim_rmse[dim_idx])
            metrics[
                f"dynamics/agent_{agent_idx}/aleatoric_var_{dim_name}"
            ] = float(mean_aleatoric_variance[dim_idx])
            metrics[
                f"dynamics/agent_{agent_idx}/epistemic_var_{dim_name}"
            ] = float(epistemic_variance[dim_idx])

        agent_per_dim.append(per_dim_rmse)

    mean_per_dim = jnp.mean(jnp.stack(agent_per_dim, axis=0), axis=0)
    metrics["dynamics/mean/position_rmse"] = float(jnp.mean(mean_per_dim[:2]))
    metrics["dynamics/mean/velocity_rmse"] = float(jnp.mean(mean_per_dim[2:]))
    return metrics


def policy_params_are_finite(params):
    leaves = jax.tree.leaves(params)
    checks = [jnp.all(jnp.isfinite(x)) for x in leaves]
    return bool(np.all(np.asarray(jax.device_get(jnp.stack(checks)))))


def save_policy_checkpoint(params, cfg, policy_input_dim, action_dim, run, suffix="final"):
    checkpoint_dir = Path(str(cfg.CHECKPOINT.POLICY_DIR))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(cfg.CHECKPOINT.POLICY_PREFIX)
    seed = int(cfg.PPO.SEED)
    params_path = checkpoint_dir / f"{prefix}_seed{seed}_{suffix}.msgpack"
    metadata_path = checkpoint_dir / f"{prefix}_seed{seed}_{suffix}.json"

    host_params = jax.device_get(params)
    if not policy_params_are_finite(host_params):
        raise ValueError("Refusing to save policy checkpoint containing NaN/Inf.")

    params_path.write_bytes(serialization.to_bytes(host_params))
    metadata = {
        "format": "flax.serialization.to_bytes",
        "network_class": "ActorCriticFFMatchedBeta",
        "action_distribution": "Independent(Beta)",
        "policy_training_input": "true_agent_centric_state_abs_vel_step",
        "policy_input_dim": int(policy_input_dim),
        "action_dim": int(action_dim),
        "seed": seed,
        "env_name": str(cfg.ENV.NAME),
        "env_kwargs": OmegaConf.to_container(cfg.ENV.KWARGS, resolve=True),
        "beta_min_concentration": float(cfg.PPO.BETA_MIN_CONCENTRATION),
        "beta_max_concentration": float(cfg.PPO.BETA_MAX_CONCENTRATION),
        "action_eps": float(cfg.PPO.ACTION_EPS),
        "joint_training": True,
        "estimator_used_for_ppo_data": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if run is not None:
        run.summary[f"checkpoint/policy_{suffix}"] = str(params_path)

    return params_path


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="joint_ippo_dynamics",
)
def main(cfg: DictConfig):
    (
        ppo,
        base_env,
        env,
        network,
        tx,
        policy_input_dim,
        action_dim,
        num_updates,
    ) = make_policy_components(cfg)

    config_container = OmegaConf.to_container(cfg, resolve=True)
    env_tags = [
        tag
        for tag in os.environ.get("WANDB_TAGS", "").split(",")
        if tag
    ]
    run = wandb.init(
        entity=str(cfg.WANDB.ENTITY),
        project=str(cfg.WANDB.PROJECT),
        name=(os.environ.get("WANDB_NAME") or cfg.WANDB.RUN_NAME or None),
        group=(os.environ.get("WANDB_RUN_GROUP") or cfg.WANDB.GROUP or None),
        tags=list(cfg.WANDB.TAGS) + env_tags,
        mode=str(cfg.WANDB.MODE),
        config=config_container,
    )

    wandb.define_metric("env_step")
    wandb.define_metric("returns", step_metric="env_step")
    wandb.define_metric("rollout/*", step_metric="env_step")
    wandb.define_metric("train/*", step_metric="env_step")
    wandb.define_metric("policy/*", step_metric="env_step")
    wandb.define_metric("dynamics/*", step_metric="env_step")

    print("Joint model-free training: shared Beta IPPO + auxiliary dynamics")
    print("=" * 78)
    print("environment:", cfg.ENV.NAME)
    print("env kwargs:", OmegaConf.to_container(cfg.ENV.KWARGS, resolve=True))
    print("num agents:", base_env.num_agents)
    print("num envs:", ppo["NUM_ENVS"])
    print("ppo rollout steps:", ppo["NUM_STEPS"])
    print("total PPO updates:", num_updates)
    print("policy training input: TRUE 19D state input")
    print("estimator/model rollout used for PPO data: NO")
    print("dynamics target: [real x_i, real a_i] -> real delta x_i")
    print("dynamics replay capacity:", cfg.DYNAMICS.REPLAY_CAPACITY)
    print("dynamics warmup transitions:", cfg.DYNAMICS.WARMUP_TRANSITIONS)
    print("dynamics update interval (PPO updates):", cfg.DYNAMICS.UPDATE_INTERVAL)

    rng = jax.random.PRNGKey(int(cfg.PPO.SEED))
    runner_state = initialize_ppo_runner(
        rng=rng,
        ppo=ppo,
        env=env,
        network=network,
        tx=tx,
        policy_input_dim=policy_input_dim,
    )

    one_update_jit = jax.jit(
        make_one_ppo_update(
            ppo=ppo,
            env=env,
            network=network,
            action_dim=action_dim,
        ),
        device=jax.devices()[0],
    )

    replay = PolicyDynamicsReplayBuffer(
        capacity=int(cfg.DYNAMICS.REPLAY_CAPACITY),
        num_agents=base_env.num_agents,
        state_dim=4,
        action_dim=action_dim,
        seed=int(cfg.PPO.SEED) + 1000,
    )

    dynamics_model = ProbabilisticDynamicsModel(
        output_dim=4,
        hidden_dims=tuple(int(x) for x in cfg.DYNAMICS.HIDDEN_DIMS),
        log_var_min=float(cfg.DYNAMICS.LOG_VAR_MIN),
        log_var_max=float(cfg.DYNAMICS.LOG_VAR_MAX),
    )
    dynamics_ensembles = None
    dynamics_rng = jax.random.PRNGKey(int(cfg.PPO.SEED) + 2000)

    try:
        for update_idx in range(num_updates):
            runner_state, ppo_metrics, dynamics_batch = one_update_jit(
                runner_state,
                jnp.asarray(update_idx, dtype=jnp.int32),
            )

            ppo_metrics = jax.device_get(ppo_metrics)
            x, actions, next_x = extract_valid_dynamics_batch(dynamics_batch)

            # This exact real on-policy batch feeds the auxiliary dynamics
            # replay buffer. No estimator-generated/synthetic transitions.
            replay.add(x, actions, next_x)

            env_step = int(
                (update_idx + 1)
                * int(ppo["NUM_ENVS"])
                * int(ppo["NUM_STEPS"])
            )

            log_data = {
                "env_step": env_step,
                "returns": float(ppo_metrics["mean_return"]),
                "rollout/returns": float(ppo_metrics["mean_return"]),
                "train/total_loss": float(ppo_metrics["total_loss"]),
                "train/value_loss": float(ppo_metrics["value_loss"]),
                "train/actor_loss": float(ppo_metrics["actor_loss"]),
                "train/entropy": float(ppo_metrics["entropy"]),
                "train/ratio": float(ppo_metrics["ratio"]),
                "train/approx_kl": float(ppo_metrics["approx_kl"]),
                "train/clip_frac": float(ppo_metrics["clip_frac"]),
                "policy/action_mean": float(ppo_metrics["action_mean"]),
                "policy/action_std": float(ppo_metrics["action_std"]),
                "policy/action_min": float(ppo_metrics["action_min"]),
                "policy/action_max": float(ppo_metrics["action_max"]),
                "dynamics/replay_size": len(replay),
                "dynamics/total_real_transitions_seen": replay.total_added,
                "dynamics/initialized": int(dynamics_ensembles is not None),
            }
            for action_idx in range(action_dim):
                log_data[f"policy/action_mean_a{action_idx}"] = float(
                    ppo_metrics["action_means"][action_idx]
                )
                log_data[f"policy/action_std_a{action_idx}"] = float(
                    ppo_metrics["action_stds"][action_idx]
                )

            if (
                dynamics_ensembles is None
                and len(replay) >= int(cfg.DYNAMICS.WARMUP_TRANSITIONS)
            ):
                dynamics_rng, dynamics_ensembles = initialize_dynamics_from_replay(
                    replay=replay,
                    model=dynamics_model,
                    cfg=cfg,
                    rng=dynamics_rng,
                )
                print(
                    f"[env_step={env_step}] initialized dynamics ensembles "
                    f"and froze normalization stats from {len(replay)} real transitions"
                )
                log_data["dynamics/initialized"] = 1

            should_update_dynamics = (
                dynamics_ensembles is not None
                and (update_idx + 1) % int(cfg.DYNAMICS.UPDATE_INTERVAL) == 0
            )

            if should_update_dynamics:
                # PRE update = generalization to the just-collected current-policy
                # rollout before fitting on the newly enlarged replay.
                pre_metrics = evaluate_dynamics_on_batch(
                    dynamics_ensembles,
                    x,
                    actions,
                    next_x,
                )
                for key, value in pre_metrics.items():
                    log_data[key.replace("dynamics/", "dynamics/pre_update/", 1)] = value

                dynamics_rng, dynamics_ensembles = train_dynamics_from_replay(
                    independent_ensembles=dynamics_ensembles,
                    replay=replay,
                    cfg=cfg,
                    rng=dynamics_rng,
                )

                post_metrics = evaluate_dynamics_on_batch(
                    dynamics_ensembles,
                    x,
                    actions,
                    next_x,
                )
                for key, value in post_metrics.items():
                    log_data[key.replace("dynamics/", "dynamics/post_update/", 1)] = value

            wandb.log(log_data)

            print_interval = int(cfg.LOGGING.PRINT_INTERVAL_UPDATES)
            if (
                update_idx == 0
                or (update_idx + 1) % print_interval == 0
                or update_idx + 1 == num_updates
            ):
                print(
                    f"update {update_idx + 1:5d}/{num_updates} | "
                    f"env_step={env_step:9d} | "
                    f"return={float(ppo_metrics['mean_return']):8.3f} | "
                    f"replay={len(replay):6d} | "
                    f"dyn={'on' if dynamics_ensembles is not None else 'warmup'}"
                )

            save_interval = int(cfg.CHECKPOINT.SAVE_INTERVAL_UPDATES)
            if save_interval > 0 and (update_idx + 1) % save_interval == 0:
                save_policy_checkpoint(
                    params=runner_state.train_state.params,
                    cfg=cfg,
                    policy_input_dim=policy_input_dim,
                    action_dim=action_dim,
                    run=run,
                    suffix=f"update{update_idx + 1}",
                )
                if dynamics_ensembles is not None:
                    checkpoint_dir = Path(str(cfg.DYNAMICS.CHECKPOINT_DIR))
                    # Periodic snapshots go into sibling directories so final
                    # checkpoint remains a clean stable path for evaluators.
                    periodic_cfg = OmegaConf.create(
                        OmegaConf.to_container(cfg, resolve=True)
                    )
                    periodic_cfg.DYNAMICS.CHECKPOINT_DIR = str(
                        checkpoint_dir.parent
                        / f"{checkpoint_dir.name}_update{update_idx + 1}"
                    )
                    save_local_dynamics_checkpoint(
                        dynamics_ensembles,
                        periodic_cfg,
                    )

        final_policy_path = save_policy_checkpoint(
            params=runner_state.train_state.params,
            cfg=cfg,
            policy_input_dim=policy_input_dim,
            action_dim=action_dim,
            run=run,
            suffix="final",
        )
        print("Saved final policy checkpoint:", final_policy_path)

        if dynamics_ensembles is None:
            raise RuntimeError(
                "Training ended before dynamics warmup completed. "
                "Reduce DYNAMICS.WARMUP_TRANSITIONS or train longer."
            )

        final_dynamics_dir = save_local_dynamics_checkpoint(
            dynamics_ensembles,
            cfg,
        )
        print("Saved final dynamics checkpoint:", final_dynamics_dir)

        run.summary["final/policy_checkpoint"] = str(final_policy_path)
        run.summary["final/dynamics_checkpoint"] = str(final_dynamics_dir)
        run.summary["final/dynamics_replay_size"] = len(replay)
        run.summary["training/model_based"] = False
        run.summary["training/estimator_used_for_policy_data"] = False

    finally:
        run.finish()


if __name__ == "__main__":
    main()
