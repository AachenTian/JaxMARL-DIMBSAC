import hydra
import jax
import jax.numpy as jnp
import jaxmarl
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
except ImportError:
    wandb = None

from dynamics.ensemble import (
    build_independent_dynamics_ensembles,
    init_dynamics_ensemble,
)
from dynamics.model import ProbabilisticDynamicsModel
from dynamics.normalization import (
    build_agent_dynamics_data,
    compute_normalization_stats,
)
from dynamics.online_trainer import (
    retrain_agent_ensemble,
    retrain_independent_dynamics,
)
from dynamics.trainer import evaluate_dynamics_model
from independent_rollout import (
    generate_independent_model_rollouts,
)
from mixed_replay import (
    sample_mixed_critic_batch,
)
from model_replay_buffer import (
    add_independent_rollouts_to_model_buffers,
    init_independent_model_replay_buffers,
)
from networks.actor import (
    SACActor,
    deterministic_action,
    sample_squashed_gaussian,
)
from networks.critic import SACCritic
from observation import project_local_obs
from real_collector import (
    collect_real_step,
    init_real_collector,
    joint_action_to_dict,
    stack_rewards,
)
from replay_buffer import (
    RealTransitionBatch,
    add_real_batch,
    init_real_replay_buffer,
)
from sac_agent import (
    init_sac_agent,
    update_sac_agent_with_batches,
)
from training_schedule import (
    advance_environment_steps,
    init_dimbsac_schedule,
    mark_dynamics_initialized,
    mark_dynamics_retrained,
    should_initialize_dynamics,
    should_retrain_dynamics,
)


# ============================================================
# Phase 6E.2 adds paired pre/post current-replay diagnostics around each dynamics refresh.
# ============================================================


# ============================================================
# Actor / evaluation helpers
# ============================================================


def sample_policy_actions(
    actor,
    agent_states,
    local_obs,
    rng,
    action_low,
    action_high,
):
    """Sample decentralized environment actions from current actors."""

    num_agents = local_obs.shape[1]

    action_keys = jax.random.split(
        rng,
        num_agents,
    )

    actions = []

    for agent_idx in range(num_agents):
        mean, log_std = actor.apply(
            {
                "params":
                    agent_states[
                        agent_idx
                    ].actor.params
            },
            local_obs[:, agent_idx],
        )

        action, _, _ = sample_squashed_gaussian(
            rng=action_keys[agent_idx],
            mean=mean,
            log_std=log_std,
            action_low=action_low,
            action_high=action_high,
        )

        actions.append(action)

    return jnp.stack(
        actions,
        axis=1,
    )


def evaluate_policy(
    env,
    actor,
    agent_states,
    rng,
    num_eval_envs,
    num_landmarks,
    episode_horizon,
    action_low,
    action_high,
):
    """Evaluate current actors using deterministic actions."""

    reset_keys = jax.random.split(
        rng,
        num_eval_envs,
    )

    obs_dict, env_state = jax.vmap(
        env.reset
    )(reset_keys)

    num_agents = env.num_agents

    episode_returns = jnp.zeros(
        (
            num_eval_envs,
            num_agents,
        ),
        dtype=jnp.float32,
    )

    for _ in range(episode_horizon):
        local_obs = project_local_obs(
            obs_dict,
            env.agents,
            num_landmarks,
        )

        actions = []

        for agent_idx in range(num_agents):
            mean, _ = actor.apply(
                {
                    "params":
                        agent_states[
                            agent_idx
                        ].actor.params
                },
                local_obs[:, agent_idx],
            )

            action = deterministic_action(
                mean=mean,
                action_low=action_low,
                action_high=action_high,
            )

            actions.append(action)

        joint_action = jnp.stack(
            actions,
            axis=1,
        )

        action_dict = joint_action_to_dict(
            joint_action,
            env.agents,
        )

        rng, step_rng = jax.random.split(rng)

        step_keys = jax.random.split(
            step_rng,
            num_eval_envs,
        )

        (
            obs_dict,
            env_state,
            rewards,
            _,
            _,
        ) = jax.vmap(
            env.step_env,
            in_axes=(0, 0, 0),
        )(
            step_keys,
            env_state,
            action_dict,
        )

        episode_returns += stack_rewards(
            rewards,
            env.agents,
        )

    mean_agent_returns = jnp.mean(
        episode_returns,
        axis=0,
    )

    mean_team_return = jnp.mean(
        mean_agent_returns
    )

    return (
        mean_team_return,
        mean_agent_returns,
    )


# ============================================================
# Real replay / dynamics helpers
# ============================================================


def get_valid_replay_batch(
    replay_buffer,
):
    """Extract every currently valid transition from real replay."""

    valid_size = int(
        replay_buffer.size
    )

    return RealTransitionBatch(
        obs=replay_buffer.obs[:valid_size],
        x=replay_buffer.x[:valid_size],
        actions=replay_buffer.actions[:valid_size],
        rewards=replay_buffer.rewards[:valid_size],
        next_obs=replay_buffer.next_obs[:valid_size],
        next_x=replay_buffer.next_x[:valid_size],
        dones=replay_buffer.dones[:valid_size],
        episode_steps=(
            replay_buffer.episode_steps[:valid_size]
        ),
        landmarks=(
            replay_buffer.landmarks[:valid_size]
        ),
    )


def sample_replay_diagnostic_batch(
    replay_buffer,
    rng,
    batch_size: int,
):
    """Sample a logging-only diagnostic batch from current D_real."""

    replay_size = int(replay_buffer.size)

    if replay_size <= 0:
        raise ValueError(
            "Cannot evaluate dynamics on an empty real replay buffer."
        )

    sample_size = min(
        int(batch_size),
        replay_size,
    )

    indices = jax.random.choice(
        rng,
        replay_size,
        shape=(sample_size,),
        replace=False,
    )

    return RealTransitionBatch(
        obs=replay_buffer.obs[indices],
        x=replay_buffer.x[indices],
        actions=replay_buffer.actions[indices],
        rewards=replay_buffer.rewards[indices],
        next_obs=replay_buffer.next_obs[indices],
        next_x=replay_buffer.next_x[indices],
        dones=replay_buffer.dones[indices],
        episode_steps=replay_buffer.episode_steps[indices],
        landmarks=replay_buffer.landmarks[indices],
    )


def evaluate_agent_dynamics_diagnostics(
    ensemble_state,
    diagnostic_batch,
    agent_idx: int,
):
    """Evaluate one agent ensemble on a fixed logging-only real batch.

    The same batch can be evaluated before and after a dynamics refresh so
    changes measure model improvement rather than a change of sampled data.
    """

    dynamics_inputs, delta_x_targets = build_agent_dynamics_data(
        batch=diagnostic_batch,
        agent_idx=agent_idx,
    )

    member_rmses = []
    member_nlls = []
    member_per_dim_rmses = []
    member_per_dim_nlls = []

    for member_state in ensemble_state.members:
        metrics = evaluate_dynamics_model(
            state=member_state,
            dynamics_inputs=dynamics_inputs,
            delta_x_targets=delta_x_targets,
        )

        member_rmses.append(metrics["physical_rmse"])
        member_nlls.append(metrics["nll"])
        member_per_dim_rmses.append(metrics["per_dim_rmse"])
        member_per_dim_nlls.append(metrics["per_dim_nll"])

    mean_rmse = jnp.mean(jnp.stack(member_rmses))
    mean_nll = jnp.mean(jnp.stack(member_nlls))
    mean_per_dim_rmse = jnp.mean(
        jnp.stack(member_per_dim_rmses),
        axis=0,
    )
    mean_per_dim_nll = jnp.mean(
        jnp.stack(member_per_dim_nlls),
        axis=0,
    )

    zero_delta_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(delta_x_targets)
        )
    )

    return {
        "rmse": mean_rmse,
        "nll": mean_nll,
        "per_dim_rmse": mean_per_dim_rmse,
        "per_dim_nll": mean_per_dim_nll,
        "zero_delta_rmse": zero_delta_rmse,
        "relative_rmse": (
            mean_rmse / (zero_delta_rmse + 1e-8)
        ),
    }


def evaluate_dynamics_batch_diagnostics(
    dynamics_ensembles,
    diagnostic_batch,
    num_agents: int,
):
    """Evaluate all independent dynamics ensembles on one fixed batch."""

    return tuple(
        evaluate_agent_dynamics_diagnostics(
            ensemble_state=dynamics_ensembles.agents[agent_idx],
            diagnostic_batch=diagnostic_batch,
            agent_idx=agent_idx,
        )
        for agent_idx in range(num_agents)
    )


def sample_dynamics_diagnostic_batch(
    rng,
    replay_buffer,
    batch_size: int,
):
    """Sample one current-replay batch without touching the training RNG."""

    rng, sample_rng = jax.random.split(rng)

    diagnostic_batch = sample_replay_diagnostic_batch(
        replay_buffer=replay_buffer,
        rng=sample_rng,
        batch_size=batch_size,
    )

    return rng, diagnostic_batch


def init_wandb_run(cfg, seed: int, num_agents: int):
    """Initialize W&B and define separate axes for model-training traces."""

    if not bool(cfg.WANDB.ENABLED):
        return None

    if wandb is None:
        raise ImportError(
            "W&B logging is enabled but the 'wandb' package is not installed. "
            "Install it with: pip install wandb"
        )

    run_name = (
        f"{cfg.WANDB.RUN_NAME_PREFIX}-seed{seed}"
    )

    run = wandb.init(
        entity=str(cfg.WANDB.ENTITY),
        project=str(cfg.WANDB.PROJECT),
        group=str(cfg.WANDB.GROUP),
        mode=str(cfg.WANDB.MODE),
        name=run_name,
        config=OmegaConf.to_container(
            cfg,
            resolve=True,
        ),
    )

    # Main learning curves use real environment interactions as x-axis.
    run.define_metric("env_steps")
    run.define_metric("*", step_metric="env_steps")

    # A retraining round contains several epochs at the same env_steps. Use a
    # separate monotonically increasing axis for those inner optimization
    # traces, while retaining env_steps and retrain_index as context fields.
    run.define_metric("dynamics_epoch_step")
    run.define_metric(
        "dynamics/retrain/epoch",
        step_metric="dynamics_epoch_step",
    )
    run.define_metric(
        "dynamics/retrain/refresh_index",
        step_metric="dynamics_epoch_step",
    )
    run.define_metric(
        "dynamics/retrain/refresh_kind",
        step_metric="dynamics_epoch_step",
    )

    for agent_idx in range(num_agents):
        run.define_metric(
            f"dynamics/retrain/agent_{agent_idx}/rmse",
            step_metric="dynamics_epoch_step",
        )
        run.define_metric(
            f"dynamics/retrain/agent_{agent_idx}/nll",
            step_metric="dynamics_epoch_step",
        )

    run.define_metric("eval/team_return", summary="max")

    return run


def log_dynamics_retrain_histories(
    run,
    metrics_by_agent,
    env_steps: int,
    refresh_index: int,
    refresh_kind: str,
    dynamics_epoch_step: int,
):
    """Log per-epoch held-out RMSE/NLL without coupling W&B to the trainer."""

    if run is None:
        return dynamics_epoch_step

    max_epochs = max(
        metric.epochs_trained
        for metric in metrics_by_agent
    )

    refresh_kind_id = (
        0 if refresh_kind == "initial" else 1
    )

    for epoch_idx in range(max_epochs):
        payload = {
            "dynamics_epoch_step": dynamics_epoch_step,
            "env_steps": int(env_steps),
            "dynamics/retrain/epoch": epoch_idx + 1,
            "dynamics/retrain/refresh_index": int(refresh_index),
            "dynamics/retrain/refresh_kind": refresh_kind_id,
        }

        for agent_idx, metrics in enumerate(metrics_by_agent):
            if epoch_idx >= metrics.epochs_trained:
                continue

            payload[
                f"dynamics/retrain/agent_{agent_idx}/rmse"
            ] = float(metrics.epoch_mean_rmse[epoch_idx])
            payload[
                f"dynamics/retrain/agent_{agent_idx}/nll"
            ] = float(metrics.epoch_mean_nll[epoch_idx])

        run.log(payload)
        dynamics_epoch_step += 1

    return dynamics_epoch_step


def mean_sac_metric(metric_history, key: str):
    """Average one SAC scalar over the update rounds of a collector step."""

    return float(
        jnp.mean(
            jnp.stack(
                [metrics[key] for metrics in metric_history]
            )
        )
    )


def initialize_and_train_dynamics(
    rng,
    real_batch,
    diagnostic_batch,
    model,
    num_agents: int,
    ensemble_size: int,
    dynamics_config,
):
    """Initialize all dynamics ensembles once and train the first round.

    Normalization statistics are computed from the initial training split and
    remain frozen afterward. Before the first gradient update, each randomly
    initialized ensemble is evaluated on the same current-replay diagnostic
    batch that will also be used for the post-training measurement.
    """

    agent_ensembles = []
    all_metrics = []
    pre_diagnostic_metrics = []

    train_fraction = float(
        dynamics_config.TRAIN_FRACTION
    )

    for agent_idx in range(num_agents):
        print(
            f"\nInitializing dynamics Agent {agent_idx}"
        )
        print("================================")

        (
            dynamics_inputs,
            delta_x_targets,
        ) = build_agent_dynamics_data(
            batch=real_batch,
            agent_idx=agent_idx,
        )

        num_samples = dynamics_inputs.shape[0]

        train_size = int(
            num_samples * train_fraction
        )
        train_size = min(
            max(train_size, 1),
            num_samples - 1,
        )

        # Reproduce the first retraining split so normalization is computed
        # only from the same train subset used in the first training round.
        rng, ensemble_rng = jax.random.split(rng)
        rng, retrain_rng = jax.random.split(rng)

        _, normalization_split_rng = jax.random.split(
            retrain_rng
        )

        permutation = jax.random.permutation(
            normalization_split_rng,
            num_samples,
        )

        train_indices = permutation[
            :train_size
        ]

        train_inputs = dynamics_inputs[
            train_indices
        ]

        train_targets = delta_x_targets[
            train_indices
        ]

        input_stats = compute_normalization_stats(
            train_inputs
        )

        target_stats = compute_normalization_stats(
            train_targets
        )

        ensemble_state = init_dynamics_ensemble(
            rng=ensemble_rng,
            model=model,
            ensemble_size=ensemble_size,
            input_dim=dynamics_inputs.shape[-1],
            input_stats=input_stats,
            target_stats=target_stats,
            learning_rate=(
                dynamics_config.LR
            ),
        )

        # This is the true pre-training point for the initial refresh:
        # random parameters, fixed normalization, zero gradient updates.
        pre_diagnostic_metrics.append(
            evaluate_agent_dynamics_diagnostics(
                ensemble_state=ensemble_state,
                diagnostic_batch=diagnostic_batch,
                agent_idx=agent_idx,
            )
        )

        (
            retrain_rng,
            ensemble_state,
            metrics,
        ) = retrain_agent_ensemble(
            rng=retrain_rng,
            ensemble_state=ensemble_state,
            real_batch=real_batch,
            agent_idx=agent_idx,
            train_fraction=train_fraction,
            batch_size=int(
                dynamics_config.BATCH_SIZE
            ),
            max_epochs=int(
                dynamics_config.MAX_RETRAIN_EPOCHS
            ),
            patience=int(
                dynamics_config.EARLY_STOPPING_PATIENCE
            ),
            min_delta=float(
                dynamics_config.EARLY_STOPPING_MIN_DELTA
            ),
        )

        rng = retrain_rng

        agent_ensembles.append(
            ensemble_state
        )
        all_metrics.append(
            metrics
        )

        print(
            "initial round epochs:",
            metrics.epochs_trained,
        )
        print(
            "initial round best mean RMSE:",
            metrics.best_mean_rmse,
        )

    dynamics_ensembles = (
        build_independent_dynamics_ensembles(
            agent_ensembles
        )
    )

    return (
        rng,
        dynamics_ensembles,
        tuple(all_metrics),
        tuple(pre_diagnostic_metrics),
    )


# ============================================================
# Model-rollout helpers
# ============================================================


def sample_model_rollout_starts(
    real_buffer,
    rng,
    batch_size: int,
):
    """Sample real replay states used as synthetic rollout starts."""

    replay_size = int(
        real_buffer.size
    )

    if replay_size <= 0:
        raise ValueError(
            "Cannot start model rollouts from an empty real replay buffer."
        )

    indices = jax.random.randint(
        rng,
        shape=(batch_size,),
        minval=0,
        maxval=real_buffer.size,
    )

    return (
        real_buffer.x[indices],
        real_buffer.landmarks[indices],
        real_buffer.episode_steps[indices],
    )


def synchronize_rollout_snapshots(
    agent_states,
    dynamics_ensembles,
):
    """Synchronize actor and dynamics snapshots at one communication point."""

    actor_snapshots = tuple(
        state.actor.params
        for state in agent_states
    )

    dynamics_snapshots = (
        dynamics_ensembles
    )

    return (
        actor_snapshots,
        dynamics_snapshots,
    )


def build_sac_actor_snapshot_view(
    agent_states,
    communication_actor_snapshots,
    dynamics_initialized: bool,
):
    """Build the opponent-policy snapshot view used by one SAC update round.

    Before learned dynamics are available, preserve the verified real-only SAC
    baseline semantics: all agents use the same current pre-update actor view.

    After the first dynamics initialization, use only the actor snapshots from
    the most recent model-refresh communication point.
    """

    if not dynamics_initialized:
        return tuple(
            state.actor.params
            for state in agent_states
        )

    if communication_actor_snapshots is None:
        raise RuntimeError(
            "Dynamics are initialized but actor communication snapshots "
            "have not been synchronized."
        )

    return communication_actor_snapshots


def generate_and_store_model_rollouts(
    rng,
    real_buffer,
    model_buffers,
    agent_states,
    actor,
    actor_snapshots,
    dynamics_ensembles,
    dynamics_snapshots,
    env,
    action_low,
    action_high,
    ensemble_size: int,
    rollout_batch_size: int,
    rollout_horizon: int,
    episode_horizon: int,
):
    """Generate learner-specific rollouts and append them to persistent buffers."""

    rng, start_rng = jax.random.split(
        rng
    )

    (
        start_x,
        landmarks,
        start_episode_steps,
    ) = sample_model_rollout_starts(
        real_buffer=real_buffer,
        rng=start_rng,
        batch_size=rollout_batch_size,
    )

    (
        rng,
        independent_rollouts,
    ) = generate_independent_model_rollouts(
        rng=rng,
        start_x=start_x,
        landmarks=landmarks,
        start_episode_steps=start_episode_steps,
        agent_states=agent_states,
        actor=actor,
        actor_snapshots=actor_snapshots,
        current_dynamics=dynamics_ensembles,
        dynamics_snapshots=dynamics_snapshots,
        env=env,
        action_low=action_low,
        action_high=action_high,
        ensemble_size=ensemble_size,
        horizon=rollout_horizon,
        episode_horizon=episode_horizon,
    )

    valid_counts = tuple(
        int(
            jnp.sum(
                trajectory.valid
            )
        )
        for trajectory in independent_rollouts.agents
    )

    model_buffers = (
        add_independent_rollouts_to_model_buffers(
            model_buffers=model_buffers,
            independent_rollouts=independent_rollouts,
        )
    )

    model_buffer_sizes = tuple(
        int(buffer.size)
        for buffer in model_buffers.agents
    )

    print("\nModel rollout generation")
    print("--------------------------------")
    print(
        "valid synthetic transitions:",
        valid_counts,
    )
    print(
        "persistent model replay sizes:",
        model_buffer_sizes,
    )

    return (
        rng,
        model_buffers,
        valid_counts,
    )


# ============================================================
# Main training loop: Phase 6D
# Real collection + periodic model retraining + independent model
# rollouts + persistent D_model_i + mixed SAC updates.
# ============================================================


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="simple_spread",
)
def main(
    cfg: DictConfig,
):
    # ========================================================
    # Environment
    # ========================================================

    env = jaxmarl.make(
        cfg.ENV.NAME,
        action_type=cfg.ENV.ACTION_TYPE,
    )

    num_agents = env.num_agents
    num_landmarks = env.num_landmarks

    num_envs = int(
        cfg.ENV.NUM_ENVS
    )
    episode_horizon = int(
        cfg.ENV.EPISODE_HORIZON
    )

    obs_dim = (
        4 + 2 * num_landmarks
    )

    action_space = env.action_space(
        env.agents[0]
    )
    action_dim = action_space.shape[0]

    ensemble_size = int(
        cfg.DYNAMICS.ENSEMBLE_SIZE
    )

    rollout_batch_size = int(
        cfg.MODEL_ROLLOUT.BATCH_SIZE
    )
    rollout_horizon = int(
        cfg.MODEL_ROLLOUT.HORIZON
    )

    model_buffer_capacity = int(
        cfg.MODEL_REPLAY.CAPACITY
    )

    model_ratio = float(
        cfg.SAC.MODEL_RATIO
    )

    num_eval_envs = int(
        cfg.ENV.NUM_EVAL_ENVS
    )
    eval_seed = int(
        cfg.ENV.EVAL_SEED
    )
    train_seed = int(
        cfg.ENV.SEED
    )

    target_periodic_retrains = int(
        cfg.DYNAMICS.NUM_PERIODIC_RETRAINS
    )
    dynamics_diagnostic_batch_size = int(
        cfg.WANDB.DYNAMICS_DIAGNOSTIC_BATCH_SIZE
    )
    dynamics_diagnostic_seed = int(
        cfg.WANDB.DYNAMICS_DIAGNOSTIC_SEED
    )

    real_buffer_capacity = int(
        cfg.REPLAY.CAPACITY
    )
    sac_batch_size = int(
        cfg.REPLAY.BATCH_SIZE
    )

    learning_starts = int(
        cfg.TRAIN.LEARNING_STARTS
    )
    total_env_steps = int(
        cfg.TRAIN.TOTAL_ENV_STEPS
    )
    updates_per_collect_step = int(
        cfg.TRAIN.UPDATES_PER_COLLECT_STEP
    )
    eval_interval_env_steps = int(
        cfg.TRAIN.EVAL_INTERVAL_ENV_STEPS
    )

    actor_lr = float(
        cfg.ACTOR.LR
    )
    critic_lr = float(
        cfg.CRITIC.LR
    )

    gamma = float(
        cfg.SAC.GAMMA
    )
    alpha = float(
        cfg.SAC.ALPHA
    )
    tau = float(
        cfg.SAC.TAU
    )

    if num_envs <= 0:
        raise ValueError("ENV.NUM_ENVS must be positive.")

    if total_env_steps <= 0:
        raise ValueError("TRAIN.TOTAL_ENV_STEPS must be positive.")

    if total_env_steps % num_envs != 0:
        raise ValueError(
            "TRAIN.TOTAL_ENV_STEPS must be divisible by ENV.NUM_ENVS "
            "so the vectorized collector has an exact real-transition budget."
        )

    if eval_interval_env_steps <= 0:
        raise ValueError(
            "TRAIN.EVAL_INTERVAL_ENV_STEPS must be positive."
        )

    if real_buffer_capacity < max(learning_starts, sac_batch_size):
        raise ValueError(
            "REPLAY.CAPACITY must be at least max(TRAIN.LEARNING_STARTS, "
            "REPLAY.BATCH_SIZE)."
        )

    if target_periodic_retrains < 0:
        raise ValueError(
            "DYNAMICS.NUM_PERIODIC_RETRAINS must be non-negative."
        )

    if dynamics_diagnostic_batch_size <= 0:
        raise ValueError(
            "WANDB.DYNAMICS_DIAGNOSTIC_BATCH_SIZE must be positive."
        )

    init_trigger_env_steps = (
        (
            int(cfg.DYNAMICS.INIT_REAL_TRANSITIONS)
            + num_envs
            - 1
        )
        // num_envs
        * num_envs
    )

    retrain_stride_env_steps = (
        (
            int(cfg.DYNAMICS.RETRAIN_INTERVAL_ENV_STEPS)
            + num_envs
            - 1
        )
        // num_envs
        * num_envs
    )

    expected_total_env_steps = (
        init_trigger_env_steps
        + target_periodic_retrains
        * retrain_stride_env_steps
    )

    if total_env_steps != expected_total_env_steps:
        raise ValueError(
            "TRAIN.TOTAL_ENV_STEPS does not match the requested periodic "
            "dynamics-retrain budget. Expected "
            f"{expected_total_env_steps} env steps for "
            f"{target_periodic_retrains} retrains, got {total_env_steps}."
        )

    total_collect_steps = (
        total_env_steps // num_envs
    )

    print("DIMBSAC Phase 6E.2")
    print("================================")
    print("num_agents:", num_agents)
    print("num_envs:", num_envs)
    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)
    print("ensemble_size:", ensemble_size)
    print("rollout batch size:", rollout_batch_size)
    print("rollout horizon:", rollout_horizon)
    print("model replay capacity:", model_buffer_capacity)
    print("mixed model ratio:", model_ratio)
    print("total env steps:", total_env_steps)
    print("total collect steps:", total_collect_steps)
    print("learning starts:", learning_starts)
    print("SAC batch size:", sac_batch_size)
    print("updates per collect step:", updates_per_collect_step)
    print("eval interval env steps:", eval_interval_env_steps)
    print(
        "dynamics init transitions:",
        int(
            cfg.DYNAMICS.INIT_REAL_TRANSITIONS
        ),
    )
    print(
        "dynamics retrain interval:",
        int(
            cfg.DYNAMICS.RETRAIN_INTERVAL_ENV_STEPS
        ),
    )
    print(
        "actual vectorized retrain stride:",
        retrain_stride_env_steps,
    )
    print(
        "target periodic dynamics retrains:",
        target_periodic_retrains,
    )
    print(
        "dynamics max epochs / patience:",
        int(cfg.DYNAMICS.MAX_RETRAIN_EPOCHS),
        "/",
        int(cfg.DYNAMICS.EARLY_STOPPING_PATIENCE),
    )

    if int(cfg.DYNAMICS.MAX_RETRAIN_EPOCHS) <= int(
        cfg.DYNAMICS.EARLY_STOPPING_PATIENCE
    ):
        print(
            "WARNING: MAX_RETRAIN_EPOCHS <= EARLY_STOPPING_PATIENCE; "
            "patience-based early stopping cannot trigger in this run."
        )

    wandb_run = init_wandb_run(
        cfg=cfg,
        seed=train_seed,
        num_agents=num_agents,
    )

    rng = jax.random.PRNGKey(
        train_seed
    )

    # Logging-only RNG. It never feeds back into training or rollout RNG.
    dynamics_diagnostic_rng = jax.random.PRNGKey(
        dynamics_diagnostic_seed
    )

    # ========================================================
    # SAC networks
    # ========================================================

    actor = SACActor(
        action_dim=action_dim,
        hidden_dims=tuple(
            cfg.ACTOR.HIDDEN_DIMS
        ),
        log_std_min=float(
            cfg.ACTOR.LOG_STD_MIN
        ),
        log_std_max=float(
            cfg.ACTOR.LOG_STD_MAX
        ),
    )

    critic = SACCritic(
        hidden_dims=tuple(
            cfg.CRITIC.HIDDEN_DIMS
        ),
    )

    dummy_local_obs = jnp.zeros(
        (1, obs_dim),
        dtype=jnp.float32,
    )

    dummy_joint_obs = jnp.zeros(
        (
            1,
            num_agents,
            obs_dim,
        ),
        dtype=jnp.float32,
    )

    dummy_joint_action = jnp.zeros(
        (
            1,
            num_agents,
            action_dim,
        ),
        dtype=jnp.float32,
    )

    agent_states = []

    for _ in range(num_agents):
        rng, init_rng = jax.random.split(rng)

        state = init_sac_agent(
            rng=init_rng,
            actor=actor,
            critic=critic,
            dummy_local_obs=dummy_local_obs,
            dummy_joint_obs=dummy_joint_obs,
            dummy_joint_action=dummy_joint_action,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        )

        agent_states.append(state)

    # Communication snapshots do not exist until the first dynamics refresh.
    # Before that point, SAC uses current pre-update actors exactly like the
    # verified real-only baseline.
    communication_actor_snapshots = None

    # ========================================================
    # Dynamics architecture
    # ========================================================

    dynamics_model = ProbabilisticDynamicsModel(
        output_dim=4,
        hidden_dims=tuple(
            cfg.DYNAMICS.HIDDEN_DIMS
        ),
        log_var_min=(
            cfg.DYNAMICS.LOG_VAR_MIN
        ),
        log_var_max=(
            cfg.DYNAMICS.LOG_VAR_MAX
        ),
    )

    dynamics_ensembles = None
    dynamics_snapshots = None
    last_dynamics_metrics = None

    # ========================================================
    # Real environment and replay
    # ========================================================

    rng, collector_rng = jax.random.split(rng)

    collector_state = init_real_collector(
        env=env,
        num_envs=num_envs,
        rng=collector_rng,
    )

    replay_buffer = init_real_replay_buffer(
        capacity=real_buffer_capacity,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
    )

    replay_size = 0

    # ========================================================
    # Persistent learner-specific model replay buffers
    # ========================================================

    model_buffers = (
        init_independent_model_replay_buffers(
            num_agents=num_agents,
            capacity=model_buffer_capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    )

    # ========================================================
    # Scheduling / compiled update
    # ========================================================

    schedule_state = (
        init_dimbsac_schedule()
    )

    update_fn = jax.jit(
        update_sac_agent_with_batches,
        static_argnames=(
            "agent_idx",
            "actor",
            "critic",
            "gamma",
            "alpha",
            "tau",
        ),
    )

    # ========================================================
    # Initial evaluation
    # ========================================================

    eval_rng = jax.random.PRNGKey(
        eval_seed
    )

    (
        initial_return,
        initial_agent_returns,
    ) = evaluate_policy(
        env=env,
        actor=actor,
        agent_states=agent_states,
        rng=eval_rng,
        num_eval_envs=num_eval_envs,
        num_landmarks=num_landmarks,
        episode_horizon=episode_horizon,
        action_low=action_space.low,
        action_high=action_space.high,
    )

    print("\nInitial evaluation")
    print("------------------------------")
    print(
        "mean team return:",
        float(initial_return),
    )
    print(
        "agent returns:",
        initial_agent_returns,
    )

    if wandb_run is not None:
        initial_log = {
            "env_steps": 0,
            "eval/team_return": float(initial_return),
            "replay/real_size": 0,
            "train/periodic_retrain_count": 0,
        }

        for agent_idx in range(num_agents):
            initial_log[
                f"eval/agent_{agent_idx}_return"
            ] = float(initial_agent_returns[agent_idx])

        wandb_run.log(initial_log)

    # ========================================================
    # Training
    # ========================================================

    last_sac_metrics = None
    last_mixed_info = None
    last_rollout_valid_counts = None
    dynamics_epoch_step = 0
    last_sac_snapshot_mode = None
    last_eval_env_step = 0
    periodic_retrain_count = 0

    for collect_step in range(
        1,
        total_collect_steps + 1,
    ):
        # ----------------------------------------------------
        # Real environment interaction
        # ----------------------------------------------------

        local_obs = project_local_obs(
            collector_state.obs_dict,
            env.agents,
            num_landmarks,
        )

        rng, action_rng = jax.random.split(rng)

        if replay_size < learning_starts:
            joint_action = jax.random.uniform(
                action_rng,
                shape=(
                    num_envs,
                    num_agents,
                    action_dim,
                ),
                minval=action_space.low,
                maxval=action_space.high,
            )
        else:
            joint_action = sample_policy_actions(
                actor=actor,
                agent_states=agent_states,
                local_obs=local_obs,
                rng=action_rng,
                action_low=action_space.low,
                action_high=action_space.high,
            )

        (
            collector_state,
            transition,
            _,
        ) = collect_real_step(
            env=env,
            collector_state=collector_state,
            joint_action=joint_action,
            num_landmarks=num_landmarks,
            episode_horizon=episode_horizon,
        )

        replay_buffer = add_real_batch(
            replay_buffer,
            transition,
        )

        replay_size = min(
            replay_size + num_envs,
            real_buffer_capacity,
        )

        schedule_state = advance_environment_steps(
            schedule_state=schedule_state,
            num_transitions=num_envs,
        )

        wandb_step_metrics = {
            "env_steps": int(schedule_state.total_env_steps),
            "replay/real_size": replay_size,
            "train/periodic_retrain_count": periodic_retrain_count,
        }

        # ----------------------------------------------------
        # Dynamics initialization / periodic retraining
        # ----------------------------------------------------

        model_refresh_completed = False
        model_refresh_kind = None
        refresh_diagnostic_batch = None
        dynamics_pre_metrics = None

        if should_initialize_dynamics(
            schedule_state=schedule_state,
            init_real_transitions=int(
                cfg.DYNAMICS.INIT_REAL_TRANSITIONS
            ),
        ):
            print("\n\n================================")
            print(
                "INITIALIZING DYNAMICS AT ENV STEPS:",
                schedule_state.total_env_steps,
            )
            print("================================")

            real_batch = get_valid_replay_batch(
                replay_buffer
            )

            (
                dynamics_diagnostic_rng,
                refresh_diagnostic_batch,
            ) = sample_dynamics_diagnostic_batch(
                rng=dynamics_diagnostic_rng,
                replay_buffer=replay_buffer,
                batch_size=dynamics_diagnostic_batch_size,
            )

            (
                rng,
                dynamics_ensembles,
                last_dynamics_metrics,
                dynamics_pre_metrics,
            ) = initialize_and_train_dynamics(
                rng=rng,
                real_batch=real_batch,
                diagnostic_batch=refresh_diagnostic_batch,
                model=dynamics_model,
                num_agents=num_agents,
                ensemble_size=ensemble_size,
                dynamics_config=cfg.DYNAMICS,
            )

            schedule_state = mark_dynamics_initialized(
                schedule_state
            )

            model_refresh_completed = True
            model_refresh_kind = "initial"

            print(
                "Dynamics initialized. "
                "Normalization statistics are now frozen."
            )

        elif should_retrain_dynamics(
            schedule_state=schedule_state,
            retrain_interval_env_steps=int(
                cfg.DYNAMICS.RETRAIN_INTERVAL_ENV_STEPS
            ),
        ):
            print("\n\n================================")
            print(
                "RETRAINING DYNAMICS AT ENV STEPS:",
                schedule_state.total_env_steps,
            )
            print("================================")

            real_batch = get_valid_replay_batch(
                replay_buffer
            )

            (
                dynamics_diagnostic_rng,
                refresh_diagnostic_batch,
            ) = sample_dynamics_diagnostic_batch(
                rng=dynamics_diagnostic_rng,
                replay_buffer=replay_buffer,
                batch_size=dynamics_diagnostic_batch_size,
            )

            # Periodic pre metrics use the old model on exactly the same
            # diagnostic batch that will be reused after retraining.
            dynamics_pre_metrics = evaluate_dynamics_batch_diagnostics(
                dynamics_ensembles=dynamics_ensembles,
                diagnostic_batch=refresh_diagnostic_batch,
                num_agents=num_agents,
            )

            (
                rng,
                dynamics_ensembles,
                last_dynamics_metrics,
            ) = retrain_independent_dynamics(
                rng=rng,
                dynamics_ensembles=(
                    dynamics_ensembles
                ),
                real_batch=real_batch,
                train_fraction=float(
                    cfg.DYNAMICS.TRAIN_FRACTION
                ),
                batch_size=int(
                    cfg.DYNAMICS.BATCH_SIZE
                ),
                max_epochs=int(
                    cfg.DYNAMICS.MAX_RETRAIN_EPOCHS
                ),
                patience=int(
                    cfg.DYNAMICS.EARLY_STOPPING_PATIENCE
                ),
                min_delta=float(
                    cfg.DYNAMICS.EARLY_STOPPING_MIN_DELTA
                ),
            )

            schedule_state = mark_dynamics_retrained(
                schedule_state
            )

            periodic_retrain_count += 1
            model_refresh_completed = True
            model_refresh_kind = "retrain"

        # ----------------------------------------------------
        # Snapshot sync + model rollout after each model refresh
        # ----------------------------------------------------

        if model_refresh_completed:
            if refresh_diagnostic_batch is None or dynamics_pre_metrics is None:
                raise RuntimeError(
                    "Dynamics refresh completed without paired pre diagnostics."
                )

            refresh_index = (
                0
                if model_refresh_kind == "initial"
                else periodic_retrain_count
            )

            # 1) Per-epoch held-out validation traces from the retraining
            # round itself. These use a dedicated dynamics_epoch_step axis.
            dynamics_epoch_step = log_dynamics_retrain_histories(
                run=wandb_run,
                metrics_by_agent=last_dynamics_metrics,
                env_steps=schedule_state.total_env_steps,
                refresh_index=refresh_index,
                refresh_kind=model_refresh_kind,
                dynamics_epoch_step=dynamics_epoch_step,
            )

            # 2) Paired current-replay diagnostics. Pre and post use the
            # exact same sampled transitions, so their difference measures
            # the effect of this refresh instead of diagnostic sampling noise.
            dynamics_post_metrics = evaluate_dynamics_batch_diagnostics(
                dynamics_ensembles=dynamics_ensembles,
                diagnostic_batch=refresh_diagnostic_batch,
                num_agents=num_agents,
            )

            eval_rmses = []
            eval_nlls = []
            pre_rmses = []
            pre_nlls = []
            post_rmses = []
            post_nlls = []
            dim_names = ("px", "py", "vx", "vy")

            print("\nDynamics refresh diagnostics")
            print("--------------------------------")

            for agent_idx in range(num_agents):
                retrain_metrics = last_dynamics_metrics[agent_idx]
                pre_metrics = dynamics_pre_metrics[agent_idx]
                post_metrics = dynamics_post_metrics[agent_idx]

                eval_rmse = float(
                    retrain_metrics.final_eval_mean_rmse
                )
                eval_nll = float(
                    retrain_metrics.final_eval_mean_nll
                )
                pre_rmse = float(pre_metrics["rmse"])
                pre_nll = float(pre_metrics["nll"])
                post_rmse = float(post_metrics["rmse"])
                post_nll = float(post_metrics["nll"])

                eval_rmses.append(eval_rmse)
                eval_nlls.append(eval_nll)
                pre_rmses.append(pre_rmse)
                pre_nlls.append(pre_nll)
                post_rmses.append(post_rmse)
                post_nlls.append(post_nll)

                # Held-out metrics: this refresh round's validation split,
                # re-evaluated after restoring the selected checkpoints.
                wandb_step_metrics[
                    f"dynamics/eval/agent_{agent_idx}/rmse"
                ] = eval_rmse
                wandb_step_metrics[
                    f"dynamics/eval/agent_{agent_idx}/nll"
                ] = eval_nll
                wandb_step_metrics[
                    f"dynamics/eval/agent_{agent_idx}/zero_delta_rmse"
                ] = float(
                    retrain_metrics.final_eval_zero_delta_rmse
                )
                wandb_step_metrics[
                    f"dynamics/eval/agent_{agent_idx}/relative_rmse"
                ] = float(
                    retrain_metrics.final_eval_relative_rmse
                )
                wandb_step_metrics[
                    f"dynamics/eval/agent_{agent_idx}/epochs_trained"
                ] = int(retrain_metrics.epochs_trained)
                wandb_step_metrics[
                    f"dynamics/eval/agent_{agent_idx}/stopped_early"
                ] = int(retrain_metrics.stopped_early)

                # Current-replay PRE: before this refresh. For the initial
                # event this is the random ensemble before any gradient step.
                wandb_step_metrics[
                    f"dynamics/pre/agent_{agent_idx}/rmse"
                ] = pre_rmse
                wandb_step_metrics[
                    f"dynamics/pre/agent_{agent_idx}/nll"
                ] = pre_nll
                wandb_step_metrics[
                    f"dynamics/pre/agent_{agent_idx}/zero_delta_rmse"
                ] = float(pre_metrics["zero_delta_rmse"])
                wandb_step_metrics[
                    f"dynamics/pre/agent_{agent_idx}/relative_rmse"
                ] = float(pre_metrics["relative_rmse"])

                # Current-replay POST: restored model after this refresh,
                # evaluated on exactly the same diagnostic transitions.
                wandb_step_metrics[
                    f"dynamics/post/agent_{agent_idx}/rmse"
                ] = post_rmse
                wandb_step_metrics[
                    f"dynamics/post/agent_{agent_idx}/nll"
                ] = post_nll
                wandb_step_metrics[
                    f"dynamics/post/agent_{agent_idx}/zero_delta_rmse"
                ] = float(post_metrics["zero_delta_rmse"])
                wandb_step_metrics[
                    f"dynamics/post/agent_{agent_idx}/relative_rmse"
                ] = float(post_metrics["relative_rmse"])

                wandb_step_metrics[
                    f"dynamics/change/agent_{agent_idx}/rmse_improvement"
                ] = pre_rmse - post_rmse
                wandb_step_metrics[
                    f"dynamics/change/agent_{agent_idx}/nll_change"
                ] = post_nll - pre_nll

                for dim_idx, dim_name in enumerate(dim_names):
                    wandb_step_metrics[
                        f"dynamics/eval/agent_{agent_idx}/rmse_{dim_name}"
                    ] = float(
                        retrain_metrics.final_eval_per_dim_rmse[dim_idx]
                    )
                    wandb_step_metrics[
                        f"dynamics/eval/agent_{agent_idx}/nll_{dim_name}"
                    ] = float(
                        retrain_metrics.final_eval_per_dim_nll[dim_idx]
                    )
                    wandb_step_metrics[
                        f"dynamics/pre/agent_{agent_idx}/rmse_{dim_name}"
                    ] = float(
                        pre_metrics["per_dim_rmse"][dim_idx]
                    )
                    wandb_step_metrics[
                        f"dynamics/pre/agent_{agent_idx}/nll_{dim_name}"
                    ] = float(
                        pre_metrics["per_dim_nll"][dim_idx]
                    )
                    wandb_step_metrics[
                        f"dynamics/post/agent_{agent_idx}/rmse_{dim_name}"
                    ] = float(
                        post_metrics["per_dim_rmse"][dim_idx]
                    )
                    wandb_step_metrics[
                        f"dynamics/post/agent_{agent_idx}/nll_{dim_name}"
                    ] = float(
                        post_metrics["per_dim_nll"][dim_idx]
                    )

                print(
                    f"Agent {agent_idx}: "
                    f"pre RMSE/NLL={pre_rmse:.6f}/{pre_nll:.4f} -> "
                    f"post={post_rmse:.6f}/{post_nll:.4f}; "
                    f"held-out eval={eval_rmse:.6f}/{eval_nll:.4f}"
                )

            wandb_step_metrics["dynamics/eval/mean_rmse"] = (
                sum(eval_rmses) / len(eval_rmses)
            )
            wandb_step_metrics["dynamics/eval/mean_nll"] = (
                sum(eval_nlls) / len(eval_nlls)
            )
            wandb_step_metrics["dynamics/pre/mean_rmse"] = (
                sum(pre_rmses) / len(pre_rmses)
            )
            wandb_step_metrics["dynamics/pre/mean_nll"] = (
                sum(pre_nlls) / len(pre_nlls)
            )
            wandb_step_metrics["dynamics/post/mean_rmse"] = (
                sum(post_rmses) / len(post_rmses)
            )
            wandb_step_metrics["dynamics/post/mean_nll"] = (
                sum(post_nlls) / len(post_nlls)
            )
            wandb_step_metrics["dynamics/change/mean_rmse_improvement"] = (
                sum(pre_rmses) / len(pre_rmses)
                - sum(post_rmses) / len(post_rmses)
            )
            wandb_step_metrics["dynamics/refresh_kind"] = (
                0 if model_refresh_kind == "initial" else 1
            )
            wandb_step_metrics["train/periodic_retrain_count"] = (
                periodic_retrain_count
            )

            (
                communication_actor_snapshots,
                dynamics_snapshots,
            ) = synchronize_rollout_snapshots(
                agent_states=agent_states,
                dynamics_ensembles=dynamics_ensembles,
            )

            print(
                "Snapshot synchronization completed at env steps:",
                schedule_state.total_env_steps,
            )

            (
                rng,
                model_buffers,
                last_rollout_valid_counts,
            ) = generate_and_store_model_rollouts(
                rng=rng,
                real_buffer=replay_buffer,
                model_buffers=model_buffers,
                agent_states=agent_states,
                actor=actor,
                actor_snapshots=communication_actor_snapshots,
                dynamics_ensembles=dynamics_ensembles,
                dynamics_snapshots=dynamics_snapshots,
                env=env,
                action_low=action_space.low,
                action_high=action_space.high,
                ensemble_size=ensemble_size,
                rollout_batch_size=rollout_batch_size,
                rollout_horizon=rollout_horizon,
                episode_horizon=episode_horizon,
            )

            for agent_idx in range(num_agents):
                wandb_step_metrics[
                    f"rollout/agent_{agent_idx}_valid"
                ] = int(last_rollout_valid_counts[agent_idx])

        # ----------------------------------------------------
        # SAC updates
        # Same mixed batch is used for Actor_i and Critic_i.
        # ----------------------------------------------------

        if replay_size >= max(
            learning_starts,
            sac_batch_size,
        ):
            sac_snapshot_mode = (
                "periodic communication snapshots"
                if schedule_state.dynamics_initialized
                else "real-only current pre-update actors"
            )

            if sac_snapshot_mode != last_sac_snapshot_mode:
                print(
                    "SAC snapshot mode ->",
                    sac_snapshot_mode,
                    "at env steps:",
                    schedule_state.total_env_steps,
                )
                last_sac_snapshot_mode = sac_snapshot_mode

            model_data_available = all(
                int(buffer.size) > 0
                for buffer in model_buffers.agents
            )

            effective_model_ratio = (
                model_ratio
                if model_data_available
                else 0.0
            )

            sac_metric_histories = [
                [] for _ in range(num_agents)
            ]

            for _ in range(
                updates_per_collect_step
            ):
                # One shared snapshot view is frozen before sequentially
                # updating the agents in this SAC update round.
                sac_actor_snapshots = build_sac_actor_snapshot_view(
                    agent_states=agent_states,
                    communication_actor_snapshots=(
                        communication_actor_snapshots
                    ),
                    dynamics_initialized=(
                        schedule_state.dynamics_initialized
                    ),
                )

                current_metrics = []
                current_mixed_info = []

                for agent_idx in range(
                    num_agents
                ):
                    rng, sample_rng, update_rng = (
                        jax.random.split(
                            rng,
                            3,
                        )
                    )

                    (
                        mixed_batch,
                        mixed_info,
                    ) = sample_mixed_critic_batch(
                        real_buffer=replay_buffer,
                        model_buffer=(
                            model_buffers.agents[
                                agent_idx
                            ]
                        ),
                        agent_idx=agent_idx,
                        rng=sample_rng,
                        batch_size=sac_batch_size,
                        model_ratio=effective_model_ratio,
                    )

                    (
                        new_state,
                        metrics,
                    ) = update_fn(
                        agent_idx=agent_idx,
                        agent_state=agent_states[
                            agent_idx
                        ],
                        actor=actor,
                        critic=critic,
                        actor_param_snapshots=(
                            sac_actor_snapshots
                        ),
                        critic_batch=mixed_batch,
                        actor_batch=mixed_batch,
                        rng=update_rng,
                        action_low=action_space.low,
                        action_high=action_space.high,
                        gamma=gamma,
                        alpha=alpha,
                        tau=tau,
                    )

                    agent_states[
                        agent_idx
                    ] = new_state

                    current_metrics.append(
                        metrics
                    )
                    current_mixed_info.append(
                        mixed_info
                    )
                    sac_metric_histories[
                        agent_idx
                    ].append(metrics)

                last_sac_metrics = (
                    current_metrics
                )
                last_mixed_info = (
                    current_mixed_info
                )

            sac_metric_keys = (
                "critic_loss",
                "actor_loss",
                "q1_mean",
                "q2_mean",
                "target_q_mean",
                "log_prob_mean",
            )

            for agent_idx in range(num_agents):
                for metric_key in sac_metric_keys:
                    wandb_step_metrics[
                        f"sac/agent_{agent_idx}/{metric_key}"
                    ] = mean_sac_metric(
                        sac_metric_histories[agent_idx],
                        metric_key,
                    )

            wandb_step_metrics["train/model_ratio"] = (
                effective_model_ratio
            )

        # Replay sizes are cheap and useful for diagnosing buffer saturation.
        for agent_idx in range(num_agents):
            wandb_step_metrics[
                f"replay/model_{agent_idx}_size"
            ] = int(model_buffers.agents[agent_idx].size)

        # ----------------------------------------------------
        # Deterministic evaluation / integration logging
        # ----------------------------------------------------

        eval_due = (
            schedule_state.total_env_steps
            - last_eval_env_step
            >= eval_interval_env_steps
        )

        if (
            eval_due
            or collect_step == total_collect_steps
        ):
            (
                mean_return,
                agent_returns,
            ) = evaluate_policy(
                env=env,
                actor=actor,
                agent_states=agent_states,
                rng=eval_rng,
                num_eval_envs=num_eval_envs,
                num_landmarks=num_landmarks,
                episode_horizon=episode_horizon,
                action_low=action_space.low,
                action_high=action_space.high,
            )

            last_eval_env_step = (
                schedule_state.total_env_steps
            )

            print(
                f"\ncollect_step = {collect_step}"
            )
            print(
                "total env steps:",
                schedule_state.total_env_steps,
            )
            print(
                "real replay size:",
                replay_size,
            )
            print(
                "model replay sizes:",
                [
                    int(buffer.size)
                    for buffer in model_buffers.agents
                ],
            )
            print(
                "dynamics initialized:",
                schedule_state.dynamics_initialized,
            )
            print(
                "last dynamics retrain env step:",
                schedule_state.last_dynamics_retrain_env_step,
            )
            print(
                "mean team return:",
                float(mean_return),
            )
            print(
                "agent returns:",
                agent_returns,
            )

            wandb_step_metrics["eval/team_return"] = float(
                mean_return
            )
            for agent_idx in range(num_agents):
                wandb_step_metrics[
                    f"eval/agent_{agent_idx}_return"
                ] = float(agent_returns[agent_idx])

            if last_dynamics_metrics is not None:
                print(
                    "last dynamics restored eval RMSE:",
                    [
                        metric.final_eval_mean_rmse
                        for metric in last_dynamics_metrics
                    ],
                )

            if last_rollout_valid_counts is not None:
                print(
                    "last rollout valid counts:",
                    last_rollout_valid_counts,
                )

            if last_mixed_info is not None:
                print(
                    "last mixed SAC composition:",
                    [
                        (
                            info.num_real,
                            info.num_model,
                        )
                        for info in last_mixed_info
                    ],
                )

            if last_sac_metrics is not None:
                for agent_idx in range(
                    num_agents
                ):
                    print(
                        f"agent {agent_idx} critic loss:",
                        float(
                            last_sac_metrics[
                                agent_idx
                            ]["critic_loss"]
                        ),
                    )
                    print(
                        f"agent {agent_idx} actor loss:",
                        float(
                            last_sac_metrics[
                                agent_idx
                            ]["actor_loss"]
                        ),
                    )

        if wandb_run is not None:
            wandb_run.log(
                wandb_step_metrics
            )

    if periodic_retrain_count != target_periodic_retrains:
        raise RuntimeError(
            "Periodic dynamics retrain count mismatch: "
            f"expected {target_periodic_retrains}, "
            f"got {periodic_retrain_count}."
        )

    if wandb_run is not None:
        wandb_run.summary[
            "periodic_retrains_completed"
        ] = periodic_retrain_count
        wandb_run.finish()

    print("\n================================")
    print("DIMBSAC PHASE 6E.1 FINISHED")
    print("================================")


if __name__ == "__main__":
    main()
