from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import wandb
from omegaconf import DictConfig

from data.collector import (
    joint_action_to_dict,
)
from dynamics.checkpoint import (
    load_local_dynamics_checkpoint,
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
    other_agent_error,
    overwrite_ego_with_truth,
    propagate_other_agents,
    stack_physical_agent_state,
    summarize_error,
)
from wandb_utils import (
    finish_wandb,
    init_wandb,
)


EXPERIMENT_NAME = (
    "state_estimator_rollout"
)


def deterministic_true_policy_actions(
    network,
    params,
    true_joint_state,
    landmark_positions,
    normalized_step,
    action_eps: float,
):
    """
    Compute the deterministic shared-policy action from TRUE 19D state input.
    These actions drive the real reference environment.
    """
    policy_inputs = (
        build_policy_inputs_from_joint_state(
            joint_state=(
                true_joint_state
            ),
            landmark_positions=(
                landmark_positions
            ),
            normalized_step=(
                normalized_step
            ),
        )
    )

    num_envs = (
        policy_inputs.shape[0]
    )
    num_agents = (
        policy_inputs.shape[1]
    )

    flat_inputs = policy_inputs.reshape(
        (
            num_envs
            * num_agents,
            -1,
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


def broadcast_oracle_actions(
    true_actions,
    num_egos: int,
):
    """
    The Oracle-action estimator receives the exact action that was executed by
    each real agent, but still propagates state with the learned dynamics.
    """
    return jnp.repeat(
        true_actions[:, None, :, :],
        repeats=num_egos,
        axis=1,
    )


def validate_checkpoint_environment(
    dynamics_metadata,
    cfg,
):
    expected_force = float(
        cfg.ENV.CONTACT_FORCE
    )
    saved_force = float(
        dynamics_metadata[
            "contact_force"
        ]
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
            "Estimator rollout expects the 9D local dynamics checkpoint."
        )


def compute_global_true_range(
    true_trajectory,
):
    """
    Range for [px,py,vx,vy], computed once over the complete real evaluation
    trajectory. The same denominator is used for all rollout modes.
    """
    true_trajectory = jnp.stack(
        true_trajectory,
        axis=0,
    )
    flat = true_trajectory.reshape(
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


def log_rollout_metrics(
    run,
    mode_name,
    errors,
    true_range,
):
    print(
        f"\n{mode_name}"
    )
    print(
        "-" * 64
    )

    final_metrics = None

    for episode_step, error in enumerate(
        errors
    ):
        metrics = summarize_error(
            error=error,
            true_range=true_range,
        )
        final_metrics = metrics

        log_data = {
            "episode_step": (
                episode_step
            ),
            (
                f"estimator/{mode_name}/"
                "position_rmse"
            ): float(
                metrics[
                    "position_rmse"
                ]
            ),
            (
                f"estimator/{mode_name}/"
                "velocity_rmse"
            ): float(
                metrics[
                    "velocity_rmse"
                ]
            ),
            (
                f"estimator/{mode_name}/"
                "position_nrmse_range_pct"
            ): float(
                metrics[
                    "position_nrmse_range_pct"
                ]
            ),
            (
                f"estimator/{mode_name}/"
                "velocity_nrmse_range_pct"
            ): float(
                metrics[
                    "velocity_nrmse_range_pct"
                ]
            ),
        }

        for dim_idx, dim_name in enumerate(
            DIM_NAMES
        ):
            log_data[
                (
                    f"estimator/{mode_name}/"
                    f"rmse_{dim_name}"
                )
            ] = float(
                metrics[
                    "per_dim_rmse"
                ][
                    dim_idx
                ]
            )
            log_data[
                (
                    f"estimator/{mode_name}/"
                    f"nrmse_range_pct_{dim_name}"
                )
            ] = float(
                metrics[
                    "per_dim_nrmse_range_pct"
                ][
                    dim_idx
                ]
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
            len(errors) - 1,
        ):
            print(
                f"step {episode_step:2d} | "
                f"pos RMSE={float(metrics['position_rmse']):.6f} | "
                f"vel RMSE={float(metrics['velocity_rmse']):.6f} | "
                f"pos range%={float(metrics['position_nrmse_range_pct']):.3f}% | "
                f"vel range%={float(metrics['velocity_nrmse_range_pct']):.3f}%"
            )

    if run is not None and final_metrics is not None:
        run.summary[
            (
                f"final/{mode_name}/"
                "position_rmse"
            )
        ] = float(
            final_metrics[
                "position_rmse"
            ]
        )
        run.summary[
            (
                f"final/{mode_name}/"
                "velocity_rmse"
            )
        ] = float(
            final_metrics[
                "velocity_rmse"
            ]
        )
        run.summary[
            (
                f"final/{mode_name}/"
                "position_nrmse_range_pct"
            )
        ] = float(
            final_metrics[
                "position_nrmse_range_pct"
            ]
        )
        run.summary[
            (
                f"final/{mode_name}/"
                "velocity_nrmse_range_pct"
            )
        ] = float(
            final_metrics[
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
    run = init_wandb(
        cfg=cfg,
        experiment_name=EXPERIMENT_NAME,
        group_name=str(
            cfg.ESTIMATOR.WANDB_GROUP
        ),
        extra_tags=[
            "state_estimator",
            "closed_loop_rollout",
            "oracle_action",
            "policy_action",
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

        if horizon > int(
            cfg.ENV.EPISODE_HORIZON
        ):
            raise ValueError(
                "Estimator rollout horizon cannot exceed the environment "
                "episode horizon."
            )

        print(
            "State estimator rollout evaluation"
        )
        print(
            "=" * 64
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
            "num eval envs:",
            num_eval_envs,
        )
        print(
            "rollout horizon:",
            horizon,
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
            cfg.ESTIMATOR.POLICY.ACTION_EPS
        )

        rng = jax.random.PRNGKey(
            int(
                cfg.ESTIMATOR.EVAL_SEED
            )
        )
        reset_keys = jax.random.split(
            rng,
            num_eval_envs + 1,
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

        # Each ego perspective starts with exact knowledge of every agent state.
        initial_estimated = jnp.repeat(
            true_joint_state[
                :, None, :, :
            ],
            repeats=num_agents,
            axis=1,
        )

        oracle_estimated = (
            initial_estimated
        )
        policy_estimated = (
            initial_estimated
        )

        true_trajectory = [
            true_joint_state
        ]
        errors = {
            "oracle_action": [
                other_agent_error(
                    estimated_joint_states=(
                        oracle_estimated
                    ),
                    true_joint_state=(
                        true_joint_state
                    ),
                )
            ],
            "policy_action": [
                other_agent_error(
                    estimated_joint_states=(
                        policy_estimated
                    ),
                    true_joint_state=(
                        true_joint_state
                    ),
                )
            ],
        }

        for rollout_step in range(
            horizon
        ):
            # At the beginning of each estimator step, the ego agent receives
            # its own TRUE state from the real environment.
            oracle_estimated = (
                overwrite_ego_with_truth(
                    estimated_joint_states=(
                        oracle_estimated
                    ),
                    true_joint_state=(
                        true_joint_state
                    ),
                )
            )
            policy_estimated = (
                overwrite_ego_with_truth(
                    estimated_joint_states=(
                        policy_estimated
                    ),
                    true_joint_state=(
                        true_joint_state
                    ),
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

            normalized_step = (
                jnp.full(
                    (num_eval_envs,),
                    rollout_step
                    / float(
                        cfg.ENV.EPISODE_HORIZON
                    ),
                    dtype=jnp.float32,
                )
            )

            # Reference environment actions use TRUE 19D inputs.
            true_actions = (
                deterministic_true_policy_actions(
                    network=(
                        policy_network
                    ),
                    params=policy_params,
                    true_joint_state=(
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

            # Oracle-action rollout:
            # exact executed action + learned 9D local dynamics.
            oracle_actions = (
                broadcast_oracle_actions(
                    true_actions=(
                        true_actions
                    ),
                    num_egos=num_agents,
                )
            )

            # Policy-action rollout:
            # reconstruct each modeled agent's 19D input from estimated state,
            # then query the same shared Beta policy.
            predicted_actions = (
                build_estimated_policy_actions(
                    estimated_joint_states=(
                        policy_estimated
                    ),
                    landmark_positions=(
                        landmark_positions
                    ),
                    normalized_step=(
                        normalized_step
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

            oracle_estimated = (
                propagate_other_agents(
                    independent_ensembles=(
                        independent_ensembles
                    ),
                    estimated_joint_states=(
                        oracle_estimated
                    ),
                    actions_by_perspective=(
                        oracle_actions
                    ),
                )
            )

            policy_estimated = (
                propagate_other_agents(
                    independent_ensembles=(
                        independent_ensembles
                    ),
                    estimated_joint_states=(
                        policy_estimated
                    ),
                    actions_by_perspective=(
                        predicted_actions
                    ),
                )
            )

            # Advance the real reference environment using the exact same
            # deterministic shared policy, but with true state information.
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
                _,
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

            # Correct ego state at t+1 before storing the next-step error.
            oracle_estimated = (
                overwrite_ego_with_truth(
                    estimated_joint_states=(
                        oracle_estimated
                    ),
                    true_joint_state=(
                        next_true_joint_state
                    ),
                )
            )
            policy_estimated = (
                overwrite_ego_with_truth(
                    estimated_joint_states=(
                        policy_estimated
                    ),
                    true_joint_state=(
                        next_true_joint_state
                    ),
                )
            )

            errors[
                "oracle_action"
            ].append(
                other_agent_error(
                    estimated_joint_states=(
                        oracle_estimated
                    ),
                    true_joint_state=(
                        next_true_joint_state
                    ),
                )
            )
            errors[
                "policy_action"
            ].append(
                other_agent_error(
                    estimated_joint_states=(
                        policy_estimated
                    ),
                    true_joint_state=(
                        next_true_joint_state
                    ),
                )
            )

            true_trajectory.append(
                next_true_joint_state
            )

            env_state = (
                next_env_state
            )
            true_joint_state = (
                next_true_joint_state
            )

        true_range = (
            compute_global_true_range(
                true_trajectory
            )
        )

        print(
            "\nTrue state range used for NRMSE/range:"
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

        # W&B charts use episode_step = 0...H.
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

        log_rollout_metrics(
            run=run,
            mode_name=(
                "oracle_action"
            ),
            errors=errors[
                "oracle_action"
            ],
            true_range=true_range,
        )

        log_rollout_metrics(
            run=run,
            mode_name=(
                "policy_action"
            ),
            errors=errors[
                "policy_action"
            ],
            true_range=true_range,
        )

        print(
            "\nEstimator rollout evaluation finished."
        )

    finally:
        finish_wandb(
            run
        )


if __name__ == "__main__":
    main()
