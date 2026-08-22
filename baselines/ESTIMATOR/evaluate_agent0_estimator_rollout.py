"""
Single-ego shadow estimator diagnostic.

Reference trajectory:
    All agents use the same saved shared Beta policy.
    Every agent receives a TRUE 19D policy input reconstructed from the real
    environment state, and all actions are executed in the real environment.

Agent-0 shadow estimator:
    - At t=0, all agent states are initialized from the true environment state.
    - At every later step, only ego Agent 0 is refreshed from the real
      environment.
    - Agent 1 and Agent 2 states are NEVER corrected with true state.
    - Their 19D policy inputs are reconstructed from:
          real ego state + estimated other-agent states + true landmarks + step.
    - The shared Beta policy predicts their 5D actions.
    - Their own 9D local dynamics models propagate:
          [estimated px, py, vx, vy, predicted action(5)] -> delta state(4).

The true Agent 1 / Agent 2 states are used only for evaluation metrics.
"""

import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import wandb
from omegaconf import DictConfig

from data.collector import joint_action_to_dict
from dynamics.checkpoint import load_local_dynamics_checkpoint
from dynamics.ensemble import predict_dynamics_ensemble
from estimator.animation import create_agent0_shadow_animation
from estimator.policy_beta import (
    deterministic_beta_action,
    load_beta_policy,
)
from estimator.state_rollout import (
    DIM_NAMES,
    build_policy_inputs_from_joint_state,
    get_landmark_positions,
    stack_physical_agent_state,
)
from wandb_utils import finish_wandb, init_wandb


EXPERIMENT_NAME = "agent0_shadow_estimator_rollout"

# Current diagnostic is intentionally fixed to a single ego first.
DEFAULT_EGO_AGENT_IDX = 0
DEFAULT_DEBUG_ENV_INDEX = 0


def _cfg_int(section, key, default):
    """Read an optional Hydra key without forcing an immediate YAML migration."""
    try:
        return int(section[key])
    except Exception:
        return int(default)


def validate_checkpoint_environment(
    dynamics_metadata,
    cfg,
):
    expected_force = float(
        cfg.ENV.CONTACT_FORCE
    )
    saved_force = float(
        dynamics_metadata["contact_force"]
    )

    if not np.isclose(
        expected_force,
        saved_force,
    ):
        raise ValueError(
            "Dynamics/environment mismatch: "
            f"checkpoint contact_force={saved_force}, "
            f"current config contact_force={expected_force}."
        )

    if int(
        dynamics_metadata["input_dim"]
    ) != 9:
        raise ValueError(
            "This diagnostic expects the 9D local dynamics checkpoint."
        )


def deterministic_shared_policy_actions(
    network,
    params,
    joint_state,
    landmark_positions,
    normalized_step,
    action_eps,
):
    """
    Construct the same 19D input used during continuous-IPPO training and
    return the deterministic Beta mean action for every agent.

    joint_state: [N, A, 4]
    output:      [N, A, 5]
    """
    policy_inputs = (
        build_policy_inputs_from_joint_state(
            joint_state=joint_state,
            landmark_positions=landmark_positions,
            normalized_step=normalized_step,
        )
    )

    num_envs = policy_inputs.shape[0]
    num_agents = policy_inputs.shape[1]

    flat_inputs = policy_inputs.reshape(
        (
            num_envs * num_agents,
            -1,
        )
    )

    flat_actions = deterministic_beta_action(
        network=network,
        params=params,
        policy_inputs=flat_inputs,
        action_eps=action_eps,
    )

    return flat_actions.reshape(
        (
            num_envs,
            num_agents,
            -1,
        )
    )


def propagate_estimated_other_agents(
    independent_ensembles,
    estimated_joint_state,
    predicted_actions,
    ego_agent_idx,
):
    """
    Propagate only the non-ego states.

    The ego state is real and is refreshed from the environment every step.
    Other-agent states are recursively propagated without any truth correction.
    """
    num_agents = (
        estimated_joint_state.shape[1]
    )

    next_estimated = (
        estimated_joint_state
    )

    for agent_idx in range(
        num_agents
    ):
        if agent_idx == ego_agent_idx:
            continue

        estimated_agent_state = (
            estimated_joint_state[
                :, agent_idx, :
            ]
        )
        estimated_agent_action = (
            predicted_actions[
                :, agent_idx, :
            ]
        )

        dynamics_input = jnp.concatenate(
            [
                estimated_agent_state,
                estimated_agent_action,
            ],
            axis=-1,
        )

        if dynamics_input.shape[-1] != 9:
            raise ValueError(
                "Local dynamics input must be 9D, "
                f"got {dynamics_input.shape[-1]}D."
            )

        member_means, _ = (
            predict_dynamics_ensemble(
                ensemble_state=(
                    independent_ensembles
                    .agents[agent_idx]
                ),
                dynamics_inputs=(
                    dynamics_input
                ),
            )
        )

        mean_delta = jnp.mean(
            member_means,
            axis=0,
        )

        next_agent_state = (
            estimated_agent_state
            + mean_delta
        )

        next_estimated = (
            next_estimated.at[
                :, agent_idx, :
            ].set(
                next_agent_state
            )
        )

    return next_estimated


def state_error_metrics(
    estimated_state,
    true_state,
):
    """
    estimated_state / true_state: [N, 4] = [px, py, vx, vy]

    RMSE is computed across evaluation environments.
    """
    error = (
        estimated_state
        - true_state
    )

    per_dim_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(error),
            axis=0,
        )
    )

    position_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error[:, :2]
            )
        )
    )
    velocity_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error[:, 2:4]
            )
        )
    )

    return {
        "error": error,
        "per_dim_rmse": per_dim_rmse,
        "position_rmse": position_rmse,
        "velocity_rmse": velocity_rmse,
    }


def combined_other_agent_metrics(
    estimated_joint_state,
    true_joint_state,
    other_agent_indices,
):
    """
    Aggregate the estimator error over both hidden / modeled agents.
    """
    errors = []

    for agent_idx in other_agent_indices:
        errors.append(
            estimated_joint_state[
                :, agent_idx, :
            ]
            - true_joint_state[
                :, agent_idx, :
            ]
        )

    error = jnp.stack(
        errors,
        axis=1,
    )

    per_dim_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(error),
            axis=(0, 1),
        )
    )

    position_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error[..., :2]
            )
        )
    )
    velocity_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error[..., 2:4]
            )
        )
    )

    return {
        "per_dim_rmse": per_dim_rmse,
        "position_rmse": position_rmse,
        "velocity_rmse": velocity_rmse,
    }


def compute_true_range(
    true_trajectory,
    other_agent_indices,
):
    """
    Range denominator for optional percentage diagnostics.

    It is computed only from the TRUE trajectories of the agents that Agent 0
    is actually estimating, never from predicted trajectories.
    """
    stacked = jnp.stack(
        true_trajectory,
        axis=0,
    )
    selected = stacked[
        :, :, other_agent_indices, :
    ]
    flat = selected.reshape(
        (-1, 4)
    )

    return (
        jnp.max(
            flat,
            axis=0,
        )
        - jnp.min(
            flat,
            axis=0,
        )
    )


def add_range_percentages(
    metrics,
    true_range,
    eps=1e-8,
):
    safe_range = jnp.maximum(
        true_range,
        eps,
    )

    per_dim_pct = (
        100.0
        * metrics["per_dim_rmse"]
        / safe_range
    )

    return {
        **metrics,
        "per_dim_nrmse_range_pct": (
            per_dim_pct
        ),
        "position_nrmse_range_pct": (
            jnp.mean(
                per_dim_pct[:2]
            )
        ),
        "velocity_nrmse_range_pct": (
            jnp.mean(
                per_dim_pct[2:4]
            )
        ),
    }


def log_shadow_rollout(
    run,
    true_trajectory,
    estimated_trajectory,
    ego_agent_idx,
    debug_env_idx,
):
    """
    Log exactly the quantities needed for this diagnostic:

      1) Agent-0 estimator error for Agent 1.
      2) Agent-0 estimator error for Agent 2.
      3) Mean error over both estimated agents.
      4) One debug environment's true-vs-estimated px/py/vx/vy traces.

    One wandb.log call is used per episode step so W&B's internal Step is also
    aligned with episode_step.
    """
    num_steps = len(
        true_trajectory
    )
    num_agents = (
        true_trajectory[0].shape[1]
    )

    other_agent_indices = [
        idx
        for idx in range(num_agents)
        if idx != ego_agent_idx
    ]

    true_range = compute_true_range(
        true_trajectory=(
            true_trajectory
        ),
        other_agent_indices=(
            other_agent_indices
        ),
    )

    print(
        "\nTrue-state range for optional range-normalized diagnostics:"
    )
    print(
        {
            name: float(value)
            for name, value in zip(
                DIM_NAMES,
                true_range,
            )
        }
    )

    if run is not None:
        wandb.define_metric(
            "episode_step"
        )
        wandb.define_metric(
            "estimator/*",
            step_metric="episode_step",
        )
        wandb.define_metric(
            "trace/*",
            step_metric="episode_step",
        )

        for dim_idx, dim_name in enumerate(
            DIM_NAMES
        ):
            run.summary[
                f"true_range/{dim_name}"
            ] = float(
                true_range[
                    dim_idx
                ]
            )

    final_mean_metrics = None

    print(
        "\nAgent-0 shadow estimator rollout"
    )
    print(
        "=" * 84
    )

    for episode_step in range(
        num_steps
    ):
        true_joint_state = (
            true_trajectory[
                episode_step
            ]
        )
        estimated_joint_state = (
            estimated_trajectory[
                episode_step
            ]
        )

        log_data = {
            "episode_step": (
                episode_step
            ),
        }

        # Per-other-agent metrics.
        per_agent_metrics = {}

        for agent_idx in (
            other_agent_indices
        ):
            metrics = (
                state_error_metrics(
                    estimated_state=(
                        estimated_joint_state[
                            :, agent_idx, :
                        ]
                    ),
                    true_state=(
                        true_joint_state[
                            :, agent_idx, :
                        ]
                    ),
                )
            )
            metrics = (
                add_range_percentages(
                    metrics=metrics,
                    true_range=(
                        true_range
                    ),
                )
            )
            per_agent_metrics[
                agent_idx
            ] = metrics

            prefix = (
                f"estimator/agent_{ego_agent_idx}/"
                f"agent_{agent_idx}"
            )

            log_data[
                f"{prefix}/position_rmse"
            ] = float(
                metrics[
                    "position_rmse"
                ]
            )
            log_data[
                f"{prefix}/velocity_rmse"
            ] = float(
                metrics[
                    "velocity_rmse"
                ]
            )
            log_data[
                f"{prefix}/position_nrmse_range_pct"
            ] = float(
                metrics[
                    "position_nrmse_range_pct"
                ]
            )
            log_data[
                f"{prefix}/velocity_nrmse_range_pct"
            ] = float(
                metrics[
                    "velocity_nrmse_range_pct"
                ]
            )

            for dim_idx, dim_name in enumerate(
                DIM_NAMES
            ):
                log_data[
                    f"{prefix}/rmse_{dim_name}"
                ] = float(
                    metrics[
                        "per_dim_rmse"
                    ][
                        dim_idx
                    ]
                )

        # Mean error over both modeled agents.
        mean_metrics = (
            combined_other_agent_metrics(
                estimated_joint_state=(
                    estimated_joint_state
                ),
                true_joint_state=(
                    true_joint_state
                ),
                other_agent_indices=(
                    other_agent_indices
                ),
            )
        )
        mean_metrics = (
            add_range_percentages(
                metrics=mean_metrics,
                true_range=true_range,
            )
        )
        final_mean_metrics = (
            mean_metrics
        )

        mean_prefix = (
            f"estimator/agent_{ego_agent_idx}/mean"
        )

        log_data[
            f"{mean_prefix}/position_rmse"
        ] = float(
            mean_metrics[
                "position_rmse"
            ]
        )
        log_data[
            f"{mean_prefix}/velocity_rmse"
        ] = float(
            mean_metrics[
                "velocity_rmse"
            ]
        )
        log_data[
            f"{mean_prefix}/position_nrmse_range_pct"
        ] = float(
            mean_metrics[
                "position_nrmse_range_pct"
            ]
        )
        log_data[
            f"{mean_prefix}/velocity_nrmse_range_pct"
        ] = float(
            mean_metrics[
                "velocity_nrmse_range_pct"
            ]
        )

        for dim_idx, dim_name in enumerate(
            DIM_NAMES
        ):
            log_data[
                f"{mean_prefix}/rmse_{dim_name}"
            ] = float(
                mean_metrics[
                    "per_dim_rmse"
                ][
                    dim_idx
                ]
            )

        # Direct true-vs-estimated trajectory values for one fixed env.
        # These are debug traces, not aggregate performance metrics.
        for agent_idx in (
            other_agent_indices
        ):
            trace_prefix = (
                f"trace/env_{debug_env_idx}/"
                f"agent_{ego_agent_idx}_estimates_"
                f"agent_{agent_idx}"
            )

            for dim_idx, dim_name in enumerate(
                DIM_NAMES
            ):
                true_value = (
                    true_joint_state[
                        debug_env_idx,
                        agent_idx,
                        dim_idx,
                    ]
                )
                estimated_value = (
                    estimated_joint_state[
                        debug_env_idx,
                        agent_idx,
                        dim_idx,
                    ]
                )

                log_data[
                    f"{trace_prefix}/true_{dim_name}"
                ] = float(
                    true_value
                )
                log_data[
                    f"{trace_prefix}/estimated_{dim_name}"
                ] = float(
                    estimated_value
                )
                log_data[
                    f"{trace_prefix}/error_{dim_name}"
                ] = float(
                    estimated_value
                    - true_value
                )

        if run is not None:
            wandb.log(
                log_data
            )

        if episode_step in (
            0,
            1,
            5,
            10,
            15,
            20,
            num_steps - 1,
        ):
            parts = [
                f"step {episode_step:2d}",
            ]

            for agent_idx in (
                other_agent_indices
            ):
                metrics = (
                    per_agent_metrics[
                        agent_idx
                    ]
                )
                parts.append(
                    (
                        f"A{agent_idx}: "
                        f"pos={float(metrics['position_rmse']):.6f}, "
                        f"vel={float(metrics['velocity_rmse']):.6f}"
                    )
                )

            parts.append(
                (
                    "mean: "
                    f"pos={float(mean_metrics['position_rmse']):.6f}, "
                    f"vel={float(mean_metrics['velocity_rmse']):.6f}"
                )
            )

            print(
                " | ".join(parts)
            )

    if (
        run is not None
        and final_mean_metrics is not None
    ):
        summary_prefix = (
            f"final/agent_{ego_agent_idx}/mean"
        )

        run.summary[
            f"{summary_prefix}/position_rmse"
        ] = float(
            final_mean_metrics[
                "position_rmse"
            ]
        )
        run.summary[
            f"{summary_prefix}/velocity_rmse"
        ] = float(
            final_mean_metrics[
                "velocity_rmse"
            ]
        )
        run.summary[
            f"{summary_prefix}/position_nrmse_range_pct"
        ] = float(
            final_mean_metrics[
                "position_nrmse_range_pct"
            ]
        )
        run.summary[
            f"{summary_prefix}/velocity_nrmse_range_pct"
        ] = float(
            final_mean_metrics[
                "velocity_nrmse_range_pct"
            ]
        )


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def main(
    cfg: DictConfig,
):
    ego_agent_idx = _cfg_int(
        cfg.ESTIMATOR,
        "EGO_AGENT_IDX",
        DEFAULT_EGO_AGENT_IDX,
    )
    debug_env_idx = _cfg_int(
        cfg.ESTIMATOR,
        "DEBUG_ENV_INDEX",
        DEFAULT_DEBUG_ENV_INDEX,
    )

    run = init_wandb(
        cfg=cfg,
        experiment_name=(
            EXPERIMENT_NAME
        ),
        group_name=str(
            cfg.ESTIMATOR.WANDB_GROUP
        ),
        extra_tags=[
            "state_estimator",
            "single_ego",
            f"ego_agent_{ego_agent_idx}",
            "shadow_rollout",
        ],
    )

    try:
        env = jaxmarl.make(
            cfg.ENV.NAME,
            action_type=(
                cfg.ENV.ACTION_TYPE
            ),
            contact_force=float(
                cfg.ENV.CONTACT_FORCE
            ),
            max_steps=int(
                cfg.ENV.EPISODE_HORIZON
            ),
        )

        num_agents = (
            env.num_agents
        )
        num_landmarks = (
            env.num_landmarks
        )
        num_eval_envs = int(
            cfg.ESTIMATOR.NUM_EVAL_ENVS
        )
        horizon = int(
            cfg.ESTIMATOR.ROLLOUT_HORIZON
        )

        if not (
            0
            <= ego_agent_idx
            < num_agents
        ):
            raise ValueError(
                f"EGO_AGENT_IDX={ego_agent_idx} "
                f"is invalid for {num_agents} agents."
            )

        if not (
            0
            <= debug_env_idx
            < num_eval_envs
        ):
            raise ValueError(
                f"DEBUG_ENV_INDEX={debug_env_idx} "
                f"is invalid for {num_eval_envs} eval envs."
            )

        if horizon > int(
            cfg.ENV.EPISODE_HORIZON
        ):
            raise ValueError(
                "Estimator rollout horizon cannot exceed "
                "the environment episode horizon."
            )

        other_agent_indices = [
            idx
            for idx in range(
                num_agents
            )
            if idx != ego_agent_idx
        ]

        print(
            "Single-ego shadow estimator evaluation"
        )
        print(
            "=" * 72
        )
        print(
            "ego agent:",
            ego_agent_idx,
        )
        print(
            "estimated agents:",
            other_agent_indices,
        )
        print(
            "reference execution:",
            "all agents use TRUE state + shared Beta policy + real env",
        )
        print(
            "shadow estimator:",
            "ego true state + estimated other states + shared Beta policy + 9D dynamics",
        )
        print(
            "contact force:",
            cfg.ENV.CONTACT_FORCE,
        )
        print(
            "num eval envs:",
            num_eval_envs,
        )
        print(
            "rollout horizon:",
            horizon,
        )
        print(
            "debug env index:",
            debug_env_idx,
        )
        print(
            "policy checkpoint:",
            cfg.ESTIMATOR.POLICY.CHECKPOINT_PATH,
        )
        print(
            "dynamics checkpoint:",
            cfg.DYNAMICS.CHECKPOINT_DIR,
        )

        (
            independent_ensembles,
            _,
            dynamics_metadata,
        ) = load_local_dynamics_checkpoint(
            cfg.DYNAMICS.CHECKPOINT_DIR
        )

        validate_checkpoint_environment(
            dynamics_metadata=(
                dynamics_metadata
            ),
            cfg=cfg,
        )

        (
            policy_network,
            policy_params,
        ) = load_beta_policy(
            checkpoint_path=(
                cfg.ESTIMATOR.POLICY
                .CHECKPOINT_PATH
            ),
            policy_cfg=(
                cfg.ESTIMATOR.POLICY
            ),
        )

        action_eps = float(
            cfg.ESTIMATOR.POLICY
            .ACTION_EPS
        )

        rng = jax.random.PRNGKey(
            int(
                cfg.ESTIMATOR.EVAL_SEED
            )
        )

        reset_keys = (
            jax.random.split(
                rng,
                num_eval_envs + 1,
            )
        )
        rng = reset_keys[0]

        _, env_state = jax.vmap(
            env.reset
        )(
            reset_keys[1:]
        )

        true_joint_state = (
            stack_physical_agent_state(
                env_state=env_state,
                num_agents=num_agents,
            )
        )

        animation_landmark_positions = (
            get_landmark_positions(
                env_state=env_state,
                num_agents=num_agents,
                num_landmarks=num_landmarks,
            )
        )

        # Initial condition is exactly known.
        estimated_joint_state = (
            true_joint_state
        )

        true_trajectory = [
            true_joint_state
        ]
        estimated_trajectory = [
            estimated_joint_state
        ]

        reference_return = jnp.zeros(
            (
                num_eval_envs,
                num_agents,
            ),
            dtype=jnp.float32,
        )

        for rollout_step in range(
            horizon
        ):
            # Agent 0 receives its current real state from the environment.
            # Agent 1/2 remain recursively estimated.
            estimated_joint_state = (
                estimated_joint_state.at[
                    :, ego_agent_idx, :
                ].set(
                    true_joint_state[
                        :, ego_agent_idx, :
                    ]
                )
            )

            landmark_positions = (
                get_landmark_positions(
                    env_state=env_state,
                    num_agents=num_agents,
                    num_landmarks=(
                        num_landmarks
                    ),
                )
            )

            # Use the actual environment step when constructing the 19D input.
            normalized_step = (
                env_state.step.astype(
                    jnp.float32
                )
                / float(
                    env.max_steps
                )
            )

            # ------------------------------------------------------------
            # 1) Fully real reference execution
            # ------------------------------------------------------------
            # All three agents see the TRUE joint state and use the same
            # saved shared Beta policy.
            true_actions = (
                deterministic_shared_policy_actions(
                    network=(
                        policy_network
                    ),
                    params=(
                        policy_params
                    ),
                    joint_state=(
                        true_joint_state
                    ),
                    landmark_positions=(
                        landmark_positions
                    ),
                    normalized_step=(
                        normalized_step
                    ),
                    action_eps=(
                        action_eps
                    ),
                )
            )

            # ------------------------------------------------------------
            # 2) Agent-0 shadow estimator
            # ------------------------------------------------------------
            # Build all policy inputs from Agent 0's internally maintained
            # state: real x0 + estimated x1/x2.
            estimated_actions = (
                deterministic_shared_policy_actions(
                    network=(
                        policy_network
                    ),
                    params=(
                        policy_params
                    ),
                    joint_state=(
                        estimated_joint_state
                    ),
                    landmark_positions=(
                        landmark_positions
                    ),
                    normalized_step=(
                        normalized_step
                    ),
                    action_eps=(
                        action_eps
                    ),
                )
            )

            # Only Agent 1 and Agent 2 are propagated by learned dynamics.
            predicted_next_joint_state = (
                propagate_estimated_other_agents(
                    independent_ensembles=(
                        independent_ensembles
                    ),
                    estimated_joint_state=(
                        estimated_joint_state
                    ),
                    predicted_actions=(
                        estimated_actions
                    ),
                    ego_agent_idx=(
                        ego_agent_idx
                    ),
                )
            )

            # ------------------------------------------------------------
            # Advance the fully real reference environment.
            # ------------------------------------------------------------
            action_dict = (
                joint_action_to_dict(
                    true_actions,
                    env.agents,
                )
            )

            rng, step_rng = (
                jax.random.split(rng)
            )
            step_keys = (
                jax.random.split(
                    step_rng,
                    num_eval_envs,
                )
            )

            (
                _,
                next_env_state,
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

            next_true_joint_state = (
                stack_physical_agent_state(
                    env_state=(
                        next_env_state
                    ),
                    num_agents=(
                        num_agents
                    ),
                )
            )

            reward_array = jnp.stack(
                [
                    rewards[agent]
                    for agent in env.agents
                ],
                axis=1,
            )
            reference_return = (
                reference_return
                + reward_array
            )

            # At t+1, refresh ONLY the ego state with truth.
            predicted_next_joint_state = (
                predicted_next_joint_state.at[
                    :, ego_agent_idx, :
                ].set(
                    next_true_joint_state[
                        :, ego_agent_idx, :
                    ]
                )
            )

            true_trajectory.append(
                next_true_joint_state
            )
            estimated_trajectory.append(
                predicted_next_joint_state
            )

            env_state = (
                next_env_state
            )
            true_joint_state = (
                next_true_joint_state
            )
            estimated_joint_state = (
                predicted_next_joint_state
            )

        log_shadow_rollout(
            run=run,
            true_trajectory=(
                true_trajectory
            ),
            estimated_trajectory=(
                estimated_trajectory
            ),
            ego_agent_idx=(
                ego_agent_idx
            ),
            debug_env_idx=(
                debug_env_idx
            ),
        )

        # ------------------------------------------------------------
        # Visualization diagnostic for one fixed evaluation environment.
        # ------------------------------------------------------------
        animation_enabled = bool(
            getattr(
                cfg.ESTIMATOR,
                "ANIMATION_ENABLED",
                True,
            )
        )

        if animation_enabled:
            animation_outputs = (
                create_agent0_shadow_animation(
                    true_trajectory=(
                        true_trajectory
                    ),
                    estimated_trajectory=(
                        estimated_trajectory
                    ),
                    landmark_positions=(
                        animation_landmark_positions
                    ),
                    ego_agent_idx=(
                        ego_agent_idx
                    ),
                    debug_env_idx=(
                        debug_env_idx
                    ),
                    output_dir=str(
                        getattr(
                            cfg.ESTIMATOR,
                            "ANIMATION_DIR",
                            "animations",
                        )
                    ),
                    fps=int(
                        getattr(
                            cfg.ESTIMATOR,
                            "ANIMATION_FPS",
                            4,
                        )
                    ),
                    trail_length=int(
                        getattr(
                            cfg.ESTIMATOR,
                            "ANIMATION_TRAIL_LENGTH",
                            8,
                        )
                    ),
                    velocity_scale=float(
                        getattr(
                            cfg.ESTIMATOR,
                            "ANIMATION_VELOCITY_SCALE",
                            0.35,
                        )
                    ),
                    save_gif=bool(
                        getattr(
                            cfg.ESTIMATOR,
                            "ANIMATION_SAVE_GIF",
                            True,
                        )
                    ),
                    save_mp4=bool(
                        getattr(
                            cfg.ESTIMATOR,
                            "ANIMATION_SAVE_MP4",
                            True,
                        )
                    ),
                    wandb_run=run,
                )
            )

            print(
                "\nAnimation outputs:"
            )
            for output_format, output_path in (
                animation_outputs.items()
            ):
                print(
                    f"  {output_format}: {output_path}"
                )

            if (
                bool(
                    getattr(
                        cfg.ESTIMATOR,
                        "ANIMATION_SAVE_MP4",
                        True,
                    )
                )
                and "mp4"
                not in animation_outputs
            ):
                print(
                    "  mp4: skipped because ffmpeg is not available; "
                    "GIF was still generated."
                )

        mean_reference_return = float(
            jnp.mean(
                reference_return
            )
        )

        print(
            "\nReference real-environment mean agent return:",
            mean_reference_return,
        )

        if run is not None:
            run.summary[
                "reference/mean_agent_return"
            ] = (
                mean_reference_return
            )
            run.summary[
                "diagnostic/ego_agent_idx"
            ] = (
                ego_agent_idx
            )
            run.summary[
                "diagnostic/debug_env_idx"
            ] = (
                debug_env_idx
            )

        print(
            "\nSingle-ego shadow rollout finished."
        )

    finally:
        finish_wandb(
            run
        )


if __name__ == "__main__":
    main()
