import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import wandb
from omegaconf import DictConfig

from data.collector import collect_random_transition_batch
from dynamics.checkpoint import save_local_dynamics_checkpoint
from dynamics.ensemble import (
    build_independent_dynamics_ensembles,
    create_bootstrap_indices,
    init_dynamics_ensemble,
    predict_dynamics_ensemble,
    replace_ensemble_member,
)
from dynamics.model import ProbabilisticDynamicsModel
from dynamics.metrics import compute_relative_error_metrics
from dynamics.normalization import (
    build_local_dynamics_data,
    compute_normalization_stats,
    normalize,
)
from dynamics.trainer import (
    evaluate_dynamics_model,
    update_dynamics_model,
)
from wandb_utils import (
    finish_wandb,
    init_wandb,
)


DIM_NAMES = ("px", "py", "vx", "vy")
EXPERIMENT_NAME = "local_9d"


def train_agent_ensemble(
    rng,
    agent_idx: int,
    real_batch,
    model,
    cfg: DictConfig,
):
    """
    Train one 9D -> 4D local independent dynamics ensemble.
    """
    dynamics_inputs, delta_x_targets = (
        build_local_dynamics_data(
            batch=real_batch,
            agent_idx=agent_idx,
        )
    )

    if dynamics_inputs.shape[-1] != 9:
        raise ValueError(
            "Local dynamics input must be 9D "
            f"for continuous SimpleSpread, got {dynamics_inputs.shape[-1]}D."
        )

    num_samples = dynamics_inputs.shape[0]

    # Keep the same RNG protocol as the 9D local baseline so that, with the
    # same ENV.SEED, the train/validation split and bootstrap index generation
    # are directly comparable.
    rng, split_rng = jax.random.split(rng)
    permutation = jax.random.permutation(
        split_rng,
        num_samples,
    )

    train_size = int(
        num_samples
        * cfg.DYNAMICS.TRAIN_FRACTION
    )

    train_indices = permutation[
        :train_size
    ]
    validation_indices = permutation[
        train_size:
    ]

    train_inputs = dynamics_inputs[
        train_indices
    ]
    train_targets = delta_x_targets[
        train_indices
    ]
    validation_inputs = dynamics_inputs[
        validation_indices
    ]
    validation_targets = delta_x_targets[
        validation_indices
    ]

    # Input and target statistics are computed from the training split only.
    # They remain fixed for the entire standalone training run.
    input_stats = compute_normalization_stats(
        train_inputs
    )
    target_stats = compute_normalization_stats(
        train_targets
    )

    normalized_train_inputs = normalize(
        train_inputs,
        input_stats,
    )
    normalized_train_targets = normalize(
        train_targets,
        target_stats,
    )

    rng, ensemble_rng = jax.random.split(
        rng
    )
    ensemble_state = init_dynamics_ensemble(
        rng=ensemble_rng,
        model=model,
        ensemble_size=int(
            cfg.DYNAMICS.ENSEMBLE_SIZE
        ),
        input_dim=train_inputs.shape[-1],
        input_stats=input_stats,
        target_stats=target_stats,
        learning_rate=float(
            cfg.DYNAMICS.LR
        ),
    )

    rng, bootstrap_rng = jax.random.split(
        rng
    )
    bootstrap_indices = create_bootstrap_indices(
        rng=bootstrap_rng,
        ensemble_size=int(
            cfg.DYNAMICS.ENSEMBLE_SIZE
        ),
        num_samples=train_size,
    )

    update_fn = jax.jit(
        update_dynamics_model
    )

    print(
        f"\nAgent {agent_idx}"
    )
    print(
        "=" * 64
    )
    print(
        "input dim:",
        train_inputs.shape[-1],
        "| target dim:",
        train_targets.shape[-1],
    )
    print(
        "real transitions:",
        num_samples,
        "| train:",
        train_size,
        "| validation:",
        validation_inputs.shape[0],
    )

    for member_idx in range(
        int(cfg.DYNAMICS.ENSEMBLE_SIZE)
    ):
        member_state = (
            ensemble_state.members[
                member_idx
            ]
        )

        member_bootstrap_indices = (
            bootstrap_indices[
                member_idx
            ]
        )

        bootstrap_inputs = (
            normalized_train_inputs[
                member_bootstrap_indices
            ]
        )
        bootstrap_targets = (
            normalized_train_targets[
                member_bootstrap_indices
            ]
        )

        initial_metrics = (
            evaluate_dynamics_model(
                state=member_state,
                dynamics_inputs=validation_inputs,
                delta_x_targets=validation_targets,
            )
        )

        best_validation_rmse = float(
            initial_metrics[
                "physical_rmse"
            ]
        )
        best_params = (
            member_state.model.params
        )
        best_step = 0

        print(
            f"\n  Member {member_idx}"
        )
        print(
            "  "
            f"initial RMSE={best_validation_rmse:.6f}"
        )

        if bool(cfg.WANDB.ENABLED):
            wandb.log(
                {
                    "experiment/input_dim": 9,
                    "experiment/agent_idx": agent_idx,
                    "experiment/member_idx": member_idx,
                    "dynamics/member_train_step": 0,
                    (
                        f"dynamics/agent_{agent_idx}/"
                        f"member_{member_idx}/val_rmse"
                    ): best_validation_rmse,
                    (
                        f"dynamics/agent_{agent_idx}/"
                        f"member_{member_idx}/val_nll"
                    ): float(
                        initial_metrics[
                            "nll"
                        ]
                    ),
                }
            )

        for train_step in range(
            1,
            int(cfg.DYNAMICS.TRAIN_STEPS) + 1,
        ):
            rng, batch_rng = (
                jax.random.split(rng)
            )

            batch_indices = (
                jax.random.randint(
                    batch_rng,
                    shape=(
                        int(
                            cfg.DYNAMICS.BATCH_SIZE
                        ),
                    ),
                    minval=0,
                    maxval=train_size,
                )
            )

            member_state, _ = (
                update_fn(
                    member_state,
                    bootstrap_inputs[
                        batch_indices
                    ],
                    bootstrap_targets[
                        batch_indices
                    ],
                )
            )

            if (
                train_step
                % int(
                    cfg.DYNAMICS.EVAL_INTERVAL
                )
                == 0
            ):
                validation_metrics = (
                    evaluate_dynamics_model(
                        state=member_state,
                        dynamics_inputs=validation_inputs,
                        delta_x_targets=validation_targets,
                    )
                )

                validation_rmse = float(
                    validation_metrics[
                        "physical_rmse"
                    ]
                )
                validation_nll = float(
                    validation_metrics[
                        "nll"
                    ]
                )

                print(
                    "  "
                    f"step {train_step:4d}: "
                    f"RMSE={validation_rmse:.6f}, "
                    f"NLL={validation_nll:.4f}"
                )

                if bool(
                    cfg.WANDB.ENABLED
                ):
                    wandb.log(
                        {
                            "dynamics/member_train_step": (
                                train_step
                            ),
                            (
                                f"dynamics/agent_{agent_idx}/"
                                f"member_{member_idx}/val_rmse"
                            ): validation_rmse,
                            (
                                f"dynamics/agent_{agent_idx}/"
                                f"member_{member_idx}/val_nll"
                            ): validation_nll,
                        }
                    )

                # Checkpoint selection follows physical validation RMSE.
                if (
                    validation_rmse
                    < best_validation_rmse
                ):
                    best_validation_rmse = (
                        validation_rmse
                    )
                    best_params = (
                        member_state.model.params
                    )
                    best_step = train_step

        member_state = (
            member_state._replace(
                model=(
                    member_state.model.replace(
                        params=best_params
                    )
                )
            )
        )

        final_metrics = (
            evaluate_dynamics_model(
                state=member_state,
                dynamics_inputs=validation_inputs,
                delta_x_targets=validation_targets,
            )
        )

        ensemble_state = (
            replace_ensemble_member(
                ensemble_state=ensemble_state,
                member_idx=member_idx,
                new_member_state=member_state,
            )
        )

        print(
            "  "
            f"best step={best_step}, "
            f"best RMSE={best_validation_rmse:.6f}"
        )
        print(
            "  per-dim RMSE:",
            {
                name: float(value)
                for name, value in zip(
                    DIM_NAMES,
                    final_metrics[
                        "per_dim_rmse"
                    ],
                )
            },
        )

        if bool(cfg.WANDB.ENABLED):
            final_log = {
                (
                    f"dynamics/agent_{agent_idx}/"
                    f"member_{member_idx}/best_step"
                ): best_step,
                (
                    f"dynamics/agent_{agent_idx}/"
                    f"member_{member_idx}/best_rmse"
                ): best_validation_rmse,
            }

            for dim_name, dim_value in zip(
                DIM_NAMES,
                final_metrics[
                    "per_dim_rmse"
                ],
            ):
                final_log[
                    (
                        f"dynamics/agent_{agent_idx}/"
                        f"member_{member_idx}/"
                        f"rmse_{dim_name}"
                    )
                ] = float(dim_value)

            wandb.log(
                final_log
            )

    return (
        rng,
        ensemble_state,
        validation_inputs,
        validation_targets,
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
    )

    try:
        env = jaxmarl.make(
            cfg.ENV.NAME,
            action_type=cfg.ENV.ACTION_TYPE,
            contact_force=float(
                cfg.ENV.CONTACT_FORCE
            ),
            max_steps=int(
                cfg.ENV.EPISODE_HORIZON
            ),
        )

        num_agents = env.num_agents
        action_dim = (
            env.action_space(
                env.agents[0]
            ).shape[0]
        )

        print(
            "ESTIMATOR: Local independent dynamics baseline"
        )
        print(
            "=" * 64
        )
        print(
            "environment:",
            cfg.ENV.NAME,
        )
        print(
            "num agents:",
            num_agents,
        )
        print(
            "num envs:",
            cfg.ENV.NUM_ENVS,
        )
        print(
            "contact force:",
            cfg.ENV.CONTACT_FORCE,
        )
        print(
            "input:",
            "4D own state + 5D own action = 9D",
        )
        print(
            "target:",
            "delta_x_i = 4D",
        )
        print(
            "ensemble size per agent:",
            cfg.DYNAMICS.ENSEMBLE_SIZE,
        )

        rng = jax.random.PRNGKey(
            int(cfg.ENV.SEED)
        )

        rng, real_batch = (
            collect_random_transition_batch(
                env=env,
                rng=rng,
                num_envs=int(
                    cfg.ENV.NUM_ENVS
                ),
                num_collect_steps=int(
                    cfg.DATA.NUM_COLLECT_STEPS
                ),
                episode_horizon=int(
                    cfg.ENV.EPISODE_HORIZON
                ),
            )
        )

        num_real_transitions = (
            real_batch.x.shape[0]
        )

        print(
            "\nCollected real transitions:",
            num_real_transitions,
        )

        if (
            num_real_transitions
            != int(
                cfg.DATA.EXPECTED_REAL_TRANSITIONS
            )
        ):
            raise ValueError(
                "Unexpected real dataset size: "
                f"{num_real_transitions}; expected "
                f"{cfg.DATA.EXPECTED_REAL_TRANSITIONS}."
            )

        model = ProbabilisticDynamicsModel(
            output_dim=4,
            hidden_dims=tuple(
                cfg.DYNAMICS.HIDDEN_DIMS
            ),
            log_var_min=float(
                cfg.DYNAMICS.LOG_VAR_MIN
            ),
            log_var_max=float(
                cfg.DYNAMICS.LOG_VAR_MAX
            ),
        )

        agent_ensembles = []
        validation_sets = []

        for agent_idx in range(
            num_agents
        ):
            (
                rng,
                ensemble_state,
                validation_inputs,
                validation_targets,
            ) = train_agent_ensemble(
                rng=rng,
                agent_idx=agent_idx,
                real_batch=real_batch,
                model=model,
                cfg=cfg,
            )

            agent_ensembles.append(
                ensemble_state
            )
            validation_sets.append(
                (
                    validation_inputs,
                    validation_targets,
                )
            )

        independent_ensembles = (
            build_independent_dynamics_ensembles(
                agent_ensembles
            )
        )

        print(
            "\n\nFinal ensemble diagnostics"
        )
        print(
            "=" * 64
        )

        agent_total_rmse = []
        agent_per_dim_rmse = []

        for agent_idx in range(
            num_agents
        ):
            (
                validation_inputs,
                validation_targets,
            ) = validation_sets[
                agent_idx
            ]

            member_means, _ = (
                predict_dynamics_ensemble(
                    ensemble_state=(
                        independent_ensembles
                        .agents[
                            agent_idx
                        ]
                    ),
                    dynamics_inputs=validation_inputs,
                )
            )

            ensemble_mean = jnp.mean(
                member_means,
                axis=0,
            )

            relative_metrics = compute_relative_error_metrics(
                prediction=ensemble_mean,
                target=validation_targets,
                target_std=(
                    independent_ensembles
                    .agents[agent_idx]
                    .members[0]
                    .target_stats
                    .std
                ),
            )

            per_dim_rmse = relative_metrics["per_dim_rmse"]
            total_rmse = relative_metrics["physical_rmse"]
            per_dim_nrmse_std = relative_metrics["per_dim_nrmse_std"]
            per_dim_zero_ratio = relative_metrics["per_dim_zero_baseline_ratio"]
            per_dim_zero_improvement_pct = relative_metrics[
                "per_dim_improvement_vs_zero_pct"
            ]

            agent_total_rmse.append(
                total_rmse
            )
            agent_per_dim_rmse.append(
                per_dim_rmse
            )

            print(
                f"\nAgent {agent_idx}"
            )
            print(
                "ensemble mean RMSE:",
                float(total_rmse),
            )
            print(
                "ensemble per-dim RMSE:",
                {
                    name: float(value)
                    for name, value in zip(
                        DIM_NAMES,
                        per_dim_rmse,
                    )
                },
            )

            print(
                "ensemble per-dim NRMSE/std:",
                {
                    name: float(value)
                    for name, value in zip(
                        DIM_NAMES,
                        per_dim_nrmse_std,
                    )
                },
            )
            print(
                "ensemble per-dim RMSE / zero-delta baseline:",
                {
                    name: float(value)
                    for name, value in zip(
                        DIM_NAMES,
                        per_dim_zero_ratio,
                    )
                },
            )

            if bool(cfg.WANDB.ENABLED):
                log_data = {
                    (
                        f"dynamics/agent_{agent_idx}/"
                        "ensemble/rmse"
                    ): float(total_rmse),
                }

                for dim_idx, dim_name in enumerate(
                    DIM_NAMES
                ):
                    log_data[
                        (
                            f"dynamics/agent_{agent_idx}/"
                            f"ensemble/rmse_{dim_name}"
                        )
                    ] = float(per_dim_rmse[dim_idx])
                    log_data[
                        (
                            f"dynamics/agent_{agent_idx}/"
                            f"ensemble/nrmse_std_{dim_name}"
                        )
                    ] = float(per_dim_nrmse_std[dim_idx])
                    log_data[
                        (
                            f"dynamics/agent_{agent_idx}/"
                            f"ensemble/zero_ratio_{dim_name}"
                        )
                    ] = float(per_dim_zero_ratio[dim_idx])

                wandb.log(
                    log_data
                )

        agent_total_rmse = jnp.stack(
            agent_total_rmse
        )
        agent_per_dim_rmse = jnp.stack(
            agent_per_dim_rmse,
            axis=0,
        )

        mean_per_dim_rmse = jnp.mean(
            agent_per_dim_rmse,
            axis=0,
        )

        mean_position_rmse = jnp.mean(
            mean_per_dim_rmse[
                :2
            ]
        )
        mean_velocity_rmse = jnp.mean(
            mean_per_dim_rmse[
                2:
            ]
        )

        print(
            "\nAcross-agent summary"
        )
        print(
            "-" * 64
        )
        print(
            "mean ensemble RMSE:",
            float(
                jnp.mean(
                    agent_total_rmse
                )
            ),
        )
        print(
            "mean position RMSE:",
            float(mean_position_rmse),
        )
        print(
            "mean velocity RMSE:",
            float(mean_velocity_rmse),
        )
        print(
            "mean per-dim RMSE:",
            {
                name: float(value)
                for name, value in zip(
                    DIM_NAMES,
                    mean_per_dim_rmse,
                )
            },
        )

        if bool(cfg.WANDB.ENABLED):
            wandb.log(
                {
                    "summary/input_dim": 9,
                    "summary/mean_rmse": float(
                        jnp.mean(
                            agent_total_rmse
                        )
                    ),
                    "summary/mean_position_rmse": float(
                        mean_position_rmse
                    ),
                    "summary/mean_velocity_rmse": float(
                        mean_velocity_rmse
                    ),
                    "summary/mean_rmse_px": float(
                        mean_per_dim_rmse[0]
                    ),
                    "summary/mean_rmse_py": float(
                        mean_per_dim_rmse[1]
                    ),
                    "summary/mean_rmse_vx": float(
                        mean_per_dim_rmse[2]
                    ),
                    "summary/mean_rmse_vy": float(
                        mean_per_dim_rmse[3]
                    ),
                }
            )

        print(
            "\n"
            "Local 9D dynamics experiment finished."
        )

        checkpoint_dir = save_local_dynamics_checkpoint(
            independent_ensembles=independent_ensembles,
            cfg=cfg,
        )
        print(
            "\nSaved local dynamics checkpoint:",
            checkpoint_dir,
        )

    finally:
        finish_wandb(
            run
        )


if __name__ == "__main__":
    main()
