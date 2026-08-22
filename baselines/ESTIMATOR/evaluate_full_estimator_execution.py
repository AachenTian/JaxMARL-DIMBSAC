"""
Full three-agent estimator execution evaluation.

This is the first "real execution" comparison after the single-ego shadow
diagnostic.

Two paired real environments are evaluated from the SAME initial states:

1) reference_true_state
   Every agent reconstructs its 19D policy input from the TRUE joint state.
   The saved shared Beta policy produces all three actions.
   Those actions are executed in the real JaxMARL environment.

2) estimator_execution
   Every agent owns a separate estimator perspective.

   For ego i:
       S_hat_i,t =
           [estimated states of all agents]
       with the diagonal state x_hat_i,i,t overwritten every step by
           x_i,t^real
       from the estimator-execution environment.

   The actual action executed by ego i is:
       a_i,t =
           shared_policy(
               19D input reconstructed from S_hat_i,t
           )

   To predict another agent j inside ego i's estimator:
       S_hat_i,t
           -> reconstruct j's 19D policy input
           -> same shared policy
           -> predicted action a_hat_j,t^(i)
           -> local dynamics F_j([x_hat_j,t^(i), a_hat_j,t^(i)])
           -> x_hat_j,t+1^(i)

   The off-diagonal estimator states are NEVER corrected with truth.
   Only each estimator's own diagonal ego state is refreshed from the real
   environment every step.

The main comparison is:
    return(reference_true_state)
        vs
    return(estimator_execution)

The main estimator diagnostics are:
    position / velocity RMSE of all off-diagonal estimates over episode_step.

No PPO or dynamics training happens in this file.
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
from estimator.full_execution_animation import (
    create_full_execution_paired_animation,
)
from estimator.policy_beta import (
    deterministic_beta_action,
    load_beta_policy,
)
from estimator.state_rollout import (
    DIM_NAMES,
    build_estimated_policy_actions,
    build_policy_inputs_from_joint_state,
    get_landmark_positions,
    overwrite_ego_with_truth,
    stack_physical_agent_state,
)
from wandb_utils import finish_wandb, init_wandb


EXPERIMENT_NAME = "full_three_agent_estimator_execution"


def _cfg_value(
    section,
    key,
    default,
):
    try:
        return section[key]
    except Exception:
        return default


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
            "Full estimator execution expects the 9D local dynamics checkpoint."
        )


def deterministic_actions_from_true_joint_state(
    network,
    params,
    true_joint_state,
    landmark_positions,
    normalized_step,
    action_eps,
):
    """
    Shared-policy actions when every agent receives a TRUE 19D input.

    Returns:
        [num_envs, num_agents, 5]
    """
    policy_inputs = (
        build_policy_inputs_from_joint_state(
            joint_state=true_joint_state,
            landmark_positions=landmark_positions,
            normalized_step=normalized_step,
        )
    )

    num_envs = (
        policy_inputs.shape[0]
    )
    num_agents = (
        policy_inputs.shape[1]
    )

    flat_inputs = (
        policy_inputs.reshape(
            (
                num_envs
                * num_agents,
                -1,
            )
        )
    )

    flat_actions = (
        deterministic_beta_action(
            network=network,
            params=params,
            policy_inputs=flat_inputs,
            action_eps=action_eps,
        )
    )

    return flat_actions.reshape(
        (
            num_envs,
            num_agents,
            -1,
        )
    )


def extract_actual_actions_from_estimator_perspectives(
    actions_by_perspective,
):
    """
    actions_by_perspective:
        [N, E, A, action_dim]

    Perspective e predicts an action for every modeled agent.
    The real action executed by agent i must come from agent i's OWN estimator
    perspective, i.e. the diagonal:

        actual a_i = predicted action for agent i from perspective i

    Returns:
        [N, A, action_dim]
    """
    num_agents = (
        actions_by_perspective.shape[1]
    )

    return jnp.stack(
        [
            actions_by_perspective[
                :, agent_idx, agent_idx, :
            ]
            for agent_idx in range(
                num_agents
            )
        ],
        axis=1,
    )


def propagate_all_estimators(
    independent_ensembles,
    estimated_joint_states,
    actions_by_perspective,
):
    """
    Propagate every estimator's OFF-DIAGONAL modeled states.

    estimated_joint_states:
        [N, E, A, 4]

    actions_by_perspective:
        [N, E, A, 5]

    For each modeled physical agent j, its own local dynamics model F_j is used
    across every ego perspective e.

    The diagonal e == j is left untouched here because it will be replaced by
    the true next ego state after the real environment step.
    """
    num_envs = (
        estimated_joint_states.shape[0]
    )
    num_egos = (
        estimated_joint_states.shape[1]
    )
    num_agents = (
        estimated_joint_states.shape[2]
    )

    next_estimated = (
        estimated_joint_states
    )

    for modeled_agent_idx in range(
        num_agents
    ):
        modeled_states = (
            estimated_joint_states[
                :, :, modeled_agent_idx, :
            ]
        )
        modeled_actions = (
            actions_by_perspective[
                :, :, modeled_agent_idx, :
            ]
        )

        dynamics_inputs = (
            jnp.concatenate(
                [
                    modeled_states,
                    modeled_actions,
                ],
                axis=-1,
            )
        )

        if (
            dynamics_inputs.shape[-1]
            != 9
        ):
            raise ValueError(
                "Local dynamics input must be 9D, "
                f"got {dynamics_inputs.shape[-1]}D."
            )

        flat_inputs = (
            dynamics_inputs.reshape(
                (
                    num_envs
                    * num_egos,
                    9,
                )
            )
        )

        member_means, _ = (
            predict_dynamics_ensemble(
                ensemble_state=(
                    independent_ensembles
                    .agents[
                        modeled_agent_idx
                    ]
                ),
                dynamics_inputs=flat_inputs,
            )
        )

        mean_delta = (
            jnp.mean(
                member_means,
                axis=0,
            ).reshape(
                (
                    num_envs,
                    num_egos,
                    4,
                )
            )
        )

        predicted_next = (
            modeled_states
            + mean_delta
        )

        # Set predictions for all perspectives first.
        next_estimated = (
            next_estimated.at[
                :, :, modeled_agent_idx, :
            ].set(
                predicted_next
            )
        )

        # Do not use the model prediction for the modeled agent's own ego
        # perspective. Its true next ego state is injected after env.step().
        next_estimated = (
            next_estimated.at[
                :,
                modeled_agent_idx,
                modeled_agent_idx,
                :,
            ].set(
                estimated_joint_states[
                    :,
                    modeled_agent_idx,
                    modeled_agent_idx,
                    :,
                ]
            )
        )

    return next_estimated


def off_diagonal_errors(
    estimated_joint_states,
    true_joint_state,
):
    """
    Compare every ego estimator against the TRUE state of the same
    estimator-execution environment.

    Only off-diagonal entries are included.

    Returns:
        all_error:
            [N, E*(A-1), 4]

        per_ego_error:
            list with E arrays, each [N, A-1, 4]
    """
    num_egos = (
        estimated_joint_states.shape[1]
    )
    num_agents = (
        estimated_joint_states.shape[2]
    )

    per_ego = []
    all_errors = []

    for ego_idx in range(
        num_egos
    ):
        ego_errors = []

        for modeled_agent_idx in range(
            num_agents
        ):
            if (
                modeled_agent_idx
                == ego_idx
            ):
                continue

            error = (
                estimated_joint_states[
                    :,
                    ego_idx,
                    modeled_agent_idx,
                    :,
                ]
                - true_joint_state[
                    :,
                    modeled_agent_idx,
                    :,
                ]
            )

            ego_errors.append(
                error
            )
            all_errors.append(
                error
            )

        per_ego.append(
            jnp.stack(
                ego_errors,
                axis=1,
            )
        )

    all_error = (
        jnp.stack(
            all_errors,
            axis=1,
        )
    )

    return (
        all_error,
        per_ego,
    )


def summarize_state_error(
    error,
):
    """
    error shape:
        [..., 4]

    Aggregation is over every leading axis.
    """
    reduce_axes = tuple(
        range(
            error.ndim - 1
        )
    )

    per_dim_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error
            ),
            axis=reduce_axes,
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
        "per_dim_rmse": (
            per_dim_rmse
        ),
        "position_rmse": (
            position_rmse
        ),
        "velocity_rmse": (
            velocity_rmse
        ),
    }


def compute_true_state_range(
    estimator_true_trajectory,
):
    """
    Global [px, py, vx, vy] range from the TRUE estimator-execution
    trajectory. Used only for secondary NRMSE/range diagnostics.
    """
    stacked = jnp.stack(
        estimator_true_trajectory,
        axis=0,
    )
    flat = stacked.reshape(
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


def add_range_normalization(
    metrics,
    true_range,
    eps=1e-8,
):
    safe_range = (
        jnp.maximum(
            true_range,
            eps,
        )
    )

    per_dim_pct = (
        100.0
        * metrics[
            "per_dim_rmse"
        ]
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


def reward_dict_to_array(
    rewards,
    agent_list,
):
    """
    Convert reward dict -> [N, A].
    """
    return jnp.stack(
        [
            rewards[agent]
            for agent in agent_list
        ],
        axis=1,
    )


def action_error_metrics(
    estimator_actions,
    true_state_counterfactual_actions,
):
    """
    Compare the REAL actions executed in estimator execution against the
    counterfactual actions the same shared policy would have produced if those
    same agents had received TRUE joint-state inputs at the same timestep.

    This isolates the action change caused by estimated policy inputs.
    """
    error = (
        estimator_actions
        - true_state_counterfactual_actions
    )

    overall_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error
            )
        )
    )
    overall_mae = jnp.mean(
        jnp.abs(
            error
        )
    )

    per_agent_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error
            ),
            axis=(0, 2),
        )
    )

    return {
        "rmse": overall_rmse,
        "mae": overall_mae,
        "per_agent_rmse": (
            per_agent_rmse
        ),
    }


def make_initial_estimator_states(
    true_joint_state,
):
    """
    Every ego initially knows the exact joint state:

        S_hat_i,0 = S_0^true

    Shape:
        [N, E, A, 4]
    """
    num_agents = (
        true_joint_state.shape[1]
    )

    return jnp.repeat(
        true_joint_state[
            :, None, :, :
        ],
        repeats=num_agents,
        axis=1,
    )


def log_full_execution_results(
    run,
    estimator_true_trajectory,
    estimator_state_trajectory,
    reference_step_rewards,
    estimator_step_rewards,
    action_metric_history,
):
    """
    Log one W&B row per episode_step.

    episode_step 0:
        initial estimator error = 0
        cumulative returns = 0

    transition step t:
        step reward and action error correspond to t -> t+1.

    final episode_step H:
        final estimator state error and final cumulative returns.
    """
    horizon = len(
        reference_step_rewards
    )
    num_agents = (
        estimator_true_trajectory[
            0
        ].shape[1]
    )

    true_range = (
        compute_true_state_range(
            estimator_true_trajectory
        )
    )

    if run is not None:
        wandb.define_metric(
            "episode_step"
        )
        wandb.define_metric(
            "estimator/*",
            step_metric=(
                "episode_step"
            ),
        )
        wandb.define_metric(
            "performance/*",
            step_metric=(
                "episode_step"
            ),
        )
        wandb.define_metric(
            "policy/*",
            step_metric=(
                "episode_step"
            ),
        )

        for dim_idx, dim_name in enumerate(
            DIM_NAMES
        ):
            run.summary[
                (
                    "estimator_true_range/"
                    f"{dim_name}"
                )
            ] = float(
                true_range[
                    dim_idx
                ]
            )

    reference_cumulative = jnp.zeros(
        (
            reference_step_rewards[
                0
            ].shape[0],
            num_agents,
        ),
        dtype=jnp.float32,
    )
    estimator_cumulative = jnp.zeros_like(
        reference_cumulative
    )

    peak_position_rmse = 0.0
    peak_velocity_rmse = 0.0

    print(
        "\nFull three-agent estimator execution"
    )
    print(
        "=" * 104
    )

    for episode_step in range(
        horizon + 1
    ):
        true_joint_state = (
            estimator_true_trajectory[
                episode_step
            ]
        )
        estimated_joint_states = (
            estimator_state_trajectory[
                episode_step
            ]
        )

        (
            all_error,
            per_ego_error,
        ) = off_diagonal_errors(
            estimated_joint_states=(
                estimated_joint_states
            ),
            true_joint_state=(
                true_joint_state
            ),
        )

        all_metrics = (
            add_range_normalization(
                metrics=(
                    summarize_state_error(
                        all_error
                    )
                ),
                true_range=(
                    true_range
                ),
            )
        )

        peak_position_rmse = max(
            peak_position_rmse,
            float(
                all_metrics[
                    "position_rmse"
                ]
            ),
        )
        peak_velocity_rmse = max(
            peak_velocity_rmse,
            float(
                all_metrics[
                    "velocity_rmse"
                ]
            ),
        )

        log_data = {
            "episode_step": (
                episode_step
            ),
            (
                "estimator/all/"
                "position_rmse"
            ): float(
                all_metrics[
                    "position_rmse"
                ]
            ),
            (
                "estimator/all/"
                "velocity_rmse"
            ): float(
                all_metrics[
                    "velocity_rmse"
                ]
            ),
            (
                "estimator/all/"
                "position_nrmse_range_pct"
            ): float(
                all_metrics[
                    "position_nrmse_range_pct"
                ]
            ),
            (
                "estimator/all/"
                "velocity_nrmse_range_pct"
            ): float(
                all_metrics[
                    "velocity_nrmse_range_pct"
                ]
            ),
        }

        for dim_idx, dim_name in enumerate(
            DIM_NAMES
        ):
            log_data[
                (
                    "estimator/all/"
                    f"rmse_{dim_name}"
                )
            ] = float(
                all_metrics[
                    "per_dim_rmse"
                ][
                    dim_idx
                ]
            )

        for ego_idx in range(
            num_agents
        ):
            ego_metrics = (
                add_range_normalization(
                    metrics=(
                        summarize_state_error(
                            per_ego_error[
                                ego_idx
                            ]
                        )
                    ),
                    true_range=(
                        true_range
                    ),
                )
            )

            ego_prefix = (
                f"estimator/ego_{ego_idx}"
            )

            log_data[
                (
                    f"{ego_prefix}/"
                    "position_rmse"
                )
            ] = float(
                ego_metrics[
                    "position_rmse"
                ]
            )
            log_data[
                (
                    f"{ego_prefix}/"
                    "velocity_rmse"
                )
            ] = float(
                ego_metrics[
                    "velocity_rmse"
                ]
            )

        # Transition t -> t+1 data.
        if episode_step < horizon:
            reference_reward = (
                reference_step_rewards[
                    episode_step
                ]
            )
            estimator_reward = (
                estimator_step_rewards[
                    episode_step
                ]
            )

            reference_cumulative = (
                reference_cumulative
                + reference_reward
            )
            estimator_cumulative = (
                estimator_cumulative
                + estimator_reward
            )

            action_metrics = (
                action_metric_history[
                    episode_step
                ]
            )

            log_data[
                (
                    "performance/reference/"
                    "step_reward_mean"
                )
            ] = float(
                jnp.mean(
                    reference_reward
                )
            )
            log_data[
                (
                    "performance/estimator/"
                    "step_reward_mean"
                )
            ] = float(
                jnp.mean(
                    estimator_reward
                )
            )

            log_data[
                (
                    "policy/"
                    "action_rmse_estimated_input"
                    "_vs_true_state_input"
                )
            ] = float(
                action_metrics[
                    "rmse"
                ]
            )
            log_data[
                (
                    "policy/"
                    "action_mae_estimated_input"
                    "_vs_true_state_input"
                )
            ] = float(
                action_metrics[
                    "mae"
                ]
            )

            for agent_idx in range(
                num_agents
            ):
                log_data[
                    (
                        "policy/"
                        f"agent_{agent_idx}/"
                        "action_rmse_estimated_input"
                        "_vs_true_state_input"
                    )
                ] = float(
                    action_metrics[
                        "per_agent_rmse"
                    ][
                        agent_idx
                    ]
                )

        # Cumulative return after transition t if one exists.
        log_data[
            (
                "performance/reference/"
                "cumulative_mean_agent_return"
            )
        ] = float(
            jnp.mean(
                reference_cumulative
            )
        )
        log_data[
            (
                "performance/estimator/"
                "cumulative_mean_agent_return"
            )
        ] = float(
            jnp.mean(
                estimator_cumulative
            )
        )
        log_data[
            (
                "performance/"
                "cumulative_return_gap_"
                "estimator_minus_reference"
            )
        ] = float(
            jnp.mean(
                estimator_cumulative
            )
            - jnp.mean(
                reference_cumulative
            )
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
            horizon,
        ):
            print(
                (
                    f"step {episode_step:2d} | "
                    f"est pos={float(all_metrics['position_rmse']):.6f} | "
                    f"est vel={float(all_metrics['velocity_rmse']):.6f} | "
                    f"ref return={float(jnp.mean(reference_cumulative)):.3f} | "
                    f"est return={float(jnp.mean(estimator_cumulative)):.3f} | "
                    f"gap={float(jnp.mean(estimator_cumulative) - jnp.mean(reference_cumulative)):.3f}"
                )
            )

    final_reference_mean = float(
        jnp.mean(
            reference_cumulative
        )
    )
    final_estimator_mean = float(
        jnp.mean(
            estimator_cumulative
        )
    )
    final_gap = (
        final_estimator_mean
        - final_reference_mean
    )

    final_true_joint_state = (
        estimator_true_trajectory[
            -1
        ]
    )
    final_estimated_joint_states = (
        estimator_state_trajectory[
            -1
        ]
    )
    final_error, _ = (
        off_diagonal_errors(
            estimated_joint_states=(
                final_estimated_joint_states
            ),
            true_joint_state=(
                final_true_joint_state
            ),
        )
    )
    final_state_metrics = (
        summarize_state_error(
            final_error
        )
    )

    print(
        "\nFinal paired execution comparison"
    )
    print(
        "-" * 104
    )
    print(
        "reference true-state mean agent return:",
        f"{final_reference_mean:.6f}",
    )
    print(
        "estimator-execution mean agent return:",
        f"{final_estimator_mean:.6f}",
    )
    print(
        "return gap (estimator - reference):",
        f"{final_gap:.6f}",
    )
    print(
        "peak estimator position RMSE:",
        f"{peak_position_rmse:.6f}",
    )
    print(
        "peak estimator velocity RMSE:",
        f"{peak_velocity_rmse:.6f}",
    )
    print(
        "final estimator position RMSE:",
        f"{float(final_state_metrics['position_rmse']):.6f}",
    )
    print(
        "final estimator velocity RMSE:",
        f"{float(final_state_metrics['velocity_rmse']):.6f}",
    )

    if run is not None:
        run.summary[
            "final/reference_mean_agent_return"
        ] = (
            final_reference_mean
        )
        run.summary[
            "final/estimator_mean_agent_return"
        ] = (
            final_estimator_mean
        )
        run.summary[
            (
                "final/return_gap_"
                "estimator_minus_reference"
            )
        ] = (
            final_gap
        )
        run.summary[
            "final/peak_position_rmse"
        ] = (
            peak_position_rmse
        )
        run.summary[
            "final/peak_velocity_rmse"
        ] = (
            peak_velocity_rmse
        )
        run.summary[
            "final/position_rmse"
        ] = float(
            final_state_metrics[
                "position_rmse"
            ]
        )
        run.summary[
            "final/velocity_rmse"
        ] = float(
            final_state_metrics[
                "velocity_rmse"
            ]
        )

        for agent_idx in range(
            num_agents
        ):
            run.summary[
                (
                    "final/reference/"
                    f"agent_{agent_idx}_return"
                )
            ] = float(
                jnp.mean(
                    reference_cumulative[
                        :, agent_idx
                    ]
                )
            )
            run.summary[
                (
                    "final/estimator/"
                    f"agent_{agent_idx}_return"
                )
            ] = float(
                jnp.mean(
                    estimator_cumulative[
                        :, agent_idx
                    ]
                )
            )


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def main(
    cfg: DictConfig,
):
    full_group = str(
        _cfg_value(
            cfg.ESTIMATOR,
            "FULL_EXECUTION_WANDB_GROUP",
            "full-estimator-execution",
        )
    )

    run = init_wandb(
        cfg=cfg,
        experiment_name=(
            EXPERIMENT_NAME
        ),
        group_name=(
            full_group
        ),
        extra_tags=[
            "full_estimator_execution",
            "three_ego_estimators",
            "paired_reference",
            "true_state_vs_estimator",
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
            u_noise=jnp.asarray(
                list(cfg.ENV.U_NOISE),
                dtype=jnp.float32,
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

        if horizon > int(
            cfg.ENV.EPISODE_HORIZON
        ):
            raise ValueError(
                "Full execution horizon cannot exceed "
                "the environment episode horizon."
            )

        print(
            "Full three-agent estimator execution evaluation"
        )
        print(
            "=" * 88
        )
        print(
            "environment:",
            cfg.ENV.NAME,
        )
        print(
            "contact force:",
            cfg.ENV.CONTACT_FORCE,
        )
        print(
            "num agents:",
            num_agents,
        )
        print(
            "num eval envs:",
            num_eval_envs,
        )
        print(
            "horizon:",
            horizon,
        )
        print(
            "reference branch:",
            "TRUE 19D input -> shared Beta policy -> real env",
        )
        print(
            "estimator branch:",
            "3 ego estimators -> estimated 19D -> same shared Beta policy -> real env",
        )
        print(
            "policy execution:",
            "deterministic Beta mean in BOTH branches",
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
        ) = (
            load_local_dynamics_checkpoint(
                cfg.DYNAMICS.CHECKPOINT_DIR
            )
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

        _, initial_env_state = (
            jax.vmap(
                env.reset
            )(
                reset_keys[1:]
            )
        )

        # Paired comparison:
        # both branches start from exactly the same real initial states.
        reference_env_state = (
            initial_env_state
        )
        estimator_env_state = (
            initial_env_state
        )

        initial_true_joint_state = (
            stack_physical_agent_state(
                env_state=(
                    estimator_env_state
                ),
                num_agents=(
                    num_agents
                ),
            )
        )

        estimated_joint_states = (
            make_initial_estimator_states(
                initial_true_joint_state
            )
        )

        reference_true_trajectory = [
            initial_true_joint_state
        ]
        estimator_true_trajectory = [
            initial_true_joint_state
        ]
        estimator_state_trajectory = [
            estimated_joint_states
        ]

        animation_landmark_positions = (
            get_landmark_positions(
                env_state=(
                    initial_env_state
                ),
                num_agents=(
                    num_agents
                ),
                num_landmarks=(
                    num_landmarks
                ),
            )
        )

        reference_step_rewards = []
        estimator_step_rewards = []
        action_metric_history = []

        for rollout_step in range(
            horizon
        ):
            # ============================================================
            # REFERENCE BRANCH
            # ============================================================
            reference_true_joint_state = (
                stack_physical_agent_state(
                    env_state=(
                        reference_env_state
                    ),
                    num_agents=(
                        num_agents
                    ),
                )
            )
            reference_landmarks = (
                get_landmark_positions(
                    env_state=(
                        reference_env_state
                    ),
                    num_agents=(
                        num_agents
                    ),
                    num_landmarks=(
                        num_landmarks
                    ),
                )
            )
            reference_normalized_step = (
                reference_env_state.step.astype(
                    jnp.float32
                )
                / float(
                    env.max_steps
                )
            )

            reference_actions = (
                deterministic_actions_from_true_joint_state(
                    network=(
                        policy_network
                    ),
                    params=(
                        policy_params
                    ),
                    true_joint_state=(
                        reference_true_joint_state
                    ),
                    landmark_positions=(
                        reference_landmarks
                    ),
                    normalized_step=(
                        reference_normalized_step
                    ),
                    action_eps=(
                        action_eps
                    ),
                )
            )

            # ============================================================
            # ESTIMATOR-EXECUTION BRANCH
            # ============================================================
            estimator_true_joint_state = (
                stack_physical_agent_state(
                    env_state=(
                        estimator_env_state
                    ),
                    num_agents=(
                        num_agents
                    ),
                )
            )

            # Each ego receives ONLY its own current true state.
            estimated_joint_states = (
                overwrite_ego_with_truth(
                    estimated_joint_states=(
                        estimated_joint_states
                    ),
                    true_joint_state=(
                        estimator_true_joint_state
                    ),
                )
            )

            estimator_landmarks = (
                get_landmark_positions(
                    env_state=(
                        estimator_env_state
                    ),
                    num_agents=(
                        num_agents
                    ),
                    num_landmarks=(
                        num_landmarks
                    ),
                )
            )
            estimator_normalized_step = (
                estimator_env_state.step.astype(
                    jnp.float32
                )
                / float(
                    env.max_steps
                )
            )

            # Perspective e predicts actions for every agent using estimator e's
            # internally maintained joint state.
            actions_by_perspective = (
                build_estimated_policy_actions(
                    estimated_joint_states=(
                        estimated_joint_states
                    ),
                    landmark_positions=(
                        estimator_landmarks
                    ),
                    normalized_step=(
                        estimator_normalized_step
                    ),
                    network=(
                        policy_network
                    ),
                    policy_params=(
                        policy_params
                    ),
                    deterministic_action_fn=(
                        deterministic_beta_action
                    ),
                    action_eps=(
                        action_eps
                    ),
                )
            )

            # The actual real action of Agent i comes from Agent i's OWN
            # estimator perspective.
            estimator_actions = (
                extract_actual_actions_from_estimator_perspectives(
                    actions_by_perspective
                )
            )

            # Diagnostic only:
            # What would the exact same policy do at the SAME estimator-env
            # state if every agent had access to true joint-state information?
            true_state_counterfactual_actions = (
                deterministic_actions_from_true_joint_state(
                    network=(
                        policy_network
                    ),
                    params=(
                        policy_params
                    ),
                    true_joint_state=(
                        estimator_true_joint_state
                    ),
                    landmark_positions=(
                        estimator_landmarks
                    ),
                    normalized_step=(
                        estimator_normalized_step
                    ),
                    action_eps=(
                        action_eps
                    ),
                )
            )

            action_metric_history.append(
                action_error_metrics(
                    estimator_actions=(
                        estimator_actions
                    ),
                    true_state_counterfactual_actions=(
                        true_state_counterfactual_actions
                    ),
                )
            )

            # Predict the t+1 off-diagonal states BEFORE stepping the real env.
            predicted_next_estimator_states = (
                propagate_all_estimators(
                    independent_ensembles=(
                        independent_ensembles
                    ),
                    estimated_joint_states=(
                        estimated_joint_states
                    ),
                    actions_by_perspective=(
                        actions_by_perspective
                    ),
                )
            )

            # ============================================================
            # Advance BOTH real environments using paired step RNGs.
            # ============================================================
            rng, step_rng = (
                jax.random.split(
                    rng
                )
            )
            step_keys = (
                jax.random.split(
                    step_rng,
                    num_eval_envs,
                )
            )

            reference_action_dict = (
                joint_action_to_dict(
                    reference_actions,
                    env.agents,
                )
            )
            estimator_action_dict = (
                joint_action_to_dict(
                    estimator_actions,
                    env.agents,
                )
            )

            (
                _,
                next_reference_env_state,
                reference_rewards,
                _,
                _,
            ) = jax.vmap(
                env.step_env,
                in_axes=(0, 0, 0),
            )(
                step_keys,
                reference_env_state,
                reference_action_dict,
            )

            (
                _,
                next_estimator_env_state,
                estimator_rewards,
                _,
                _,
            ) = jax.vmap(
                env.step_env,
                in_axes=(0, 0, 0),
            )(
                step_keys,
                estimator_env_state,
                estimator_action_dict,
            )

            reference_step_rewards.append(
                reward_dict_to_array(
                    reference_rewards,
                    env.agents,
                )
            )
            estimator_step_rewards.append(
                reward_dict_to_array(
                    estimator_rewards,
                    env.agents,
                )
            )

            next_reference_true_joint_state = (
                stack_physical_agent_state(
                    env_state=(
                        next_reference_env_state
                    ),
                    num_agents=(
                        num_agents
                    ),
                )
            )

            next_estimator_true_joint_state = (
                stack_physical_agent_state(
                    env_state=(
                        next_estimator_env_state
                    ),
                    num_agents=(
                        num_agents
                    ),
                )
            )

            # Refresh ONLY the diagonal ego states at t+1.
            predicted_next_estimator_states = (
                overwrite_ego_with_truth(
                    estimated_joint_states=(
                        predicted_next_estimator_states
                    ),
                    true_joint_state=(
                        next_estimator_true_joint_state
                    ),
                )
            )

            reference_true_trajectory.append(
                next_reference_true_joint_state
            )
            estimator_true_trajectory.append(
                next_estimator_true_joint_state
            )
            estimator_state_trajectory.append(
                predicted_next_estimator_states
            )

            reference_env_state = (
                next_reference_env_state
            )
            estimator_env_state = (
                next_estimator_env_state
            )
            estimated_joint_states = (
                predicted_next_estimator_states
            )

        log_full_execution_results(
            run=run,
            estimator_true_trajectory=(
                estimator_true_trajectory
            ),
            estimator_state_trajectory=(
                estimator_state_trajectory
            ),
            reference_step_rewards=(
                reference_step_rewards
            ),
            estimator_step_rewards=(
                estimator_step_rewards
            ),
            action_metric_history=(
                action_metric_history
            ),
        )

        animation_enabled = bool(
            _cfg_value(
                cfg.ESTIMATOR,
                "FULL_ANIMATION_ENABLED",
                True,
            )
        )

        if animation_enabled:
            animation_outputs = (
                create_full_execution_paired_animation(
                    reference_true_trajectory=(
                        reference_true_trajectory
                    ),
                    estimator_true_trajectory=(
                        estimator_true_trajectory
                    ),
                    estimator_state_trajectory=(
                        estimator_state_trajectory
                    ),
                    landmark_positions=(
                        animation_landmark_positions
                    ),
                    reference_step_rewards=(
                        reference_step_rewards
                    ),
                    estimator_step_rewards=(
                        estimator_step_rewards
                    ),
                    debug_env_idx=int(
                        _cfg_value(
                            cfg.ESTIMATOR,
                            "FULL_ANIMATION_ENV_INDEX",
                            0,
                        )
                    ),
                    # The four-panel renderer shows Ego 0, Ego 1,
                    # and Ego 2 simultaneously, so no single ego index is
                    # selected here.
                    output_dir=str(
                        _cfg_value(
                            cfg.ESTIMATOR,
                            "FULL_ANIMATION_DIR",
                            "animations",
                        )
                    ),
                    fps=int(
                        _cfg_value(
                            cfg.ESTIMATOR,
                            "FULL_ANIMATION_FPS",
                            4,
                        )
                    ),
                    trail_length=int(
                        _cfg_value(
                            cfg.ESTIMATOR,
                            "FULL_ANIMATION_TRAIL_LENGTH",
                            8,
                        )
                    ),
                    velocity_scale=float(
                        _cfg_value(
                            cfg.ESTIMATOR,
                            "FULL_ANIMATION_VELOCITY_SCALE",
                            0.35,
                        )
                    ),
                    save_gif=bool(
                        _cfg_value(
                            cfg.ESTIMATOR,
                            "FULL_ANIMATION_SAVE_GIF",
                            True,
                        )
                    ),
                    save_mp4=bool(
                        _cfg_value(
                            cfg.ESTIMATOR,
                            "FULL_ANIMATION_SAVE_MP4",
                            True,
                        )
                    ),
                    wandb_run=run,
                )
            )

            print(
                "\nFour-panel estimator animation outputs:"
            )
            for output_format, output_path in (
                animation_outputs.items()
            ):
                print(
                    f"  {output_format}: {output_path}"
                )

            if (
                bool(
                    _cfg_value(
                        cfg.ESTIMATOR,
                        "FULL_ANIMATION_SAVE_MP4",
                        True,
                    )
                )
                and "mp4" not in animation_outputs
            ):
                print(
                    "  mp4: skipped because ffmpeg is unavailable; "
                    "GIF was still generated."
                )

        print(
            "\nFull estimator execution evaluation finished."
        )

    finally:
        finish_wandb(
            run
        )


if __name__ == "__main__":
    main()
