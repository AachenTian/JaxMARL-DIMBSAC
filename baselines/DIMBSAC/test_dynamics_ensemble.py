import hydra
import jax
import jax.numpy as jnp
import jaxmarl
from omegaconf import DictConfig

from dynamics.ensemble import (
    build_independent_dynamics_ensembles,
    create_bootstrap_indices,
    init_dynamics_ensemble,
    predict_agent_dynamics_ensemble,
    replace_ensemble_member,
)
from dynamics.model import ProbabilisticDynamicsModel
from dynamics.normalization import (
    build_agent_dynamics_data,
    compute_normalization_stats,
    normalize,
)
from dynamics.trainer import (
    evaluate_dynamics_model,
    predict_dynamics,
    update_dynamics_model,
)
from real_collector import (
    collect_real_step,
    init_real_collector,
)
from replay_buffer import (
    RealTransitionBatch,
    add_real_batch,
    init_real_replay_buffer,
)


NUM_COLLECT_STEPS = 200
BUFFER_CAPACITY = 10_000


def get_valid_replay_batch(
    replay_buffer,
):
    """Extract all valid transitions currently stored in replay."""

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


def train_agent_ensemble(
    rng,
    agent_idx,
    real_batch,
    model,
    ensemble_size,
    dynamics_config,
):
    """Train one bootstrap dynamics ensemble for one agent."""

    # ========================================================
    # Build local dynamics dataset
    # ========================================================

    (
        dynamics_inputs,
        delta_x_targets,
    ) = build_agent_dynamics_data(
        real_batch,
        agent_idx,
    )

    num_samples = (
        dynamics_inputs.shape[0]
    )

    # ========================================================
    # Fixed train / validation split
    # ========================================================

    rng, split_rng = jax.random.split(
        rng
    )

    permutation = (
        jax.random.permutation(
            split_rng,
            num_samples,
        )
    )

    train_size = int(
        num_samples
        * dynamics_config.TRAIN_FRACTION
    )

    train_indices = (
        permutation[:train_size]
    )

    validation_indices = (
        permutation[train_size:]
    )

    validation_joint_x = (
        real_batch.x[
            validation_indices
        ]
    )

    train_inputs = (
        dynamics_inputs[
            train_indices
        ]
    )

    train_targets = (
        delta_x_targets[
            train_indices
        ]
    )

    validation_inputs = (
        dynamics_inputs[
            validation_indices
        ]
    )

    validation_targets = (
        delta_x_targets[
            validation_indices
        ]
    )

    print(
        "real transitions:",
        num_samples,
    )

    print(
        "train samples:",
        train_size,
    )

    print(
        "validation samples:",
        validation_inputs.shape[0],
    )

    # ========================================================
    # Agent-specific normalization
    # ========================================================

    input_stats = (
        compute_normalization_stats(
            train_inputs
        )
    )

    target_stats = (
        compute_normalization_stats(
            train_targets
        )
    )

    normalized_train_inputs = (
        normalize(
            train_inputs,
            input_stats,
        )
    )

    normalized_train_targets = (
        normalize(
            train_targets,
            target_stats,
        )
    )

    # ========================================================
    # Initialize ensemble
    # ========================================================

    rng, ensemble_rng = (
        jax.random.split(rng)
    )

    ensemble_state = (
        init_dynamics_ensemble(
            rng=ensemble_rng,
            model=model,
            ensemble_size=ensemble_size,
            input_dim=train_inputs.shape[-1],
            input_stats=input_stats,
            target_stats=target_stats,
            learning_rate=(
                dynamics_config.LR
            ),
        )
    )

    assert (
        len(ensemble_state.members)
        == ensemble_size
    )

    # ========================================================
    # Fixed bootstrap datasets
    # ========================================================

    rng, bootstrap_rng = (
        jax.random.split(rng)
    )

    bootstrap_indices = (
        create_bootstrap_indices(
            rng=bootstrap_rng,
            ensemble_size=ensemble_size,
            num_samples=train_size,
        )
    )

    assert (
        bootstrap_indices.shape
        == (
            ensemble_size,
            train_size,
        )
    )

    print("\nBootstrap datasets")
    print("------------------------------")

    for member_idx in range(
        ensemble_size
    ):
        unique_count = int(
            jnp.unique(
                bootstrap_indices[
                    member_idx
                ]
            ).shape[0]
        )

        print(
            f"member {member_idx}: "
            f"{unique_count} unique / "
            f"{train_size} samples"
        )

    # ========================================================
    # Train ensemble members independently
    # ========================================================

    update_fn = jax.jit(
        update_dynamics_model
    )

    for member_idx in range(
        ensemble_size
    ):
        print(
            f"\nMember {member_idx}"
        )
        print("------------------------------")

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

        # ----------------------------------------------------
        # Initial validation
        # ----------------------------------------------------

        initial_metrics = (
            evaluate_dynamics_model(
                member_state,
                validation_inputs,
                validation_targets,
            )
        )

        initial_rmse = float(
            initial_metrics[
                "physical_rmse"
            ]
        )

        initial_nll = float(
            initial_metrics["nll"]
        )

        print(
            "initial validation NLL:",
            initial_nll,
        )

        print(
            "initial RMSE:",
            initial_rmse,
        )

        # For now, checkpoint selection is based on
        # one-step physical prediction RMSE.
        best_validation_rmse = (
            initial_rmse
        )

        best_validation_nll = (
            initial_nll
        )

        best_params = (
            member_state.model.params
        )

        best_step = 0

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for train_step in range(
            1,
            dynamics_config.TRAIN_STEPS
            + 1,
        ):
            rng, batch_rng = (
                jax.random.split(rng)
            )

            batch_indices = (
                jax.random.randint(
                    batch_rng,
                    shape=(
                        dynamics_config.BATCH_SIZE,
                    ),
                    minval=0,
                    maxval=train_size,
                )
            )

            batch_inputs = (
                bootstrap_inputs[
                    batch_indices
                ]
            )

            batch_targets = (
                bootstrap_targets[
                    batch_indices
                ]
            )

            (
                member_state,
                _,
            ) = update_fn(
                member_state,
                batch_inputs,
                batch_targets,
            )

            if (
                train_step
                % dynamics_config.EVAL_INTERVAL
                == 0
            ):
                validation_metrics = (
                    evaluate_dynamics_model(
                        member_state,
                        validation_inputs,
                        validation_targets,
                    )
                )

                validation_nll = float(
                    validation_metrics[
                        "nll"
                    ]
                )

                validation_rmse = float(
                    validation_metrics[
                        "physical_rmse"
                    ]
                )

                print(
                    f"step {train_step}: "
                    f"validation NLL = "
                    f"{validation_nll:.4f}, "
                    f"RMSE = "
                    f"{validation_rmse:.4f}"
                )

                if (
                    validation_rmse
                    < best_validation_rmse
                ):
                    best_validation_rmse = (
                        validation_rmse
                    )

                    best_validation_nll = (
                        validation_nll
                    )

                    best_params = (
                        member_state.model.params
                    )

                    best_step = (
                        train_step
                    )

        # ----------------------------------------------------
        # Restore best checkpoint
        # ----------------------------------------------------

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
                member_state,
                validation_inputs,
                validation_targets,
            )
        )

        final_rmse = float(
            final_metrics[
                "physical_rmse"
            ]
        )

        ensemble_state = (
            replace_ensemble_member(
                ensemble_state,
                member_idx,
                member_state,
            )
        )

        if (
                agent_idx == 2
                and member_idx == 0
        ):
            (
                predicted_mean,
                predicted_variance,
            ) = predict_dynamics(
                member_state,
                validation_inputs,
            )

            squared_error = jnp.square(
                validation_targets
                - predicted_mean
            )

            standardized_squared_error = (
                    squared_error
                    / (
                            predicted_variance
                            + 1e-12
                    )
            )

            # Focus on velocity dimensions.
            velocity_score = jnp.mean(
                standardized_squared_error[
                    :, 2:4
                ],
                axis=-1,
            )

            # Top 10 most problematic transitions.
            worst_indices = jnp.argsort(
                velocity_score
            )[-10:][::-1]

            print(
                "\nWorst velocity uncertainty outliers"
            )
            print(
                "================================"
            )

            for idx in worst_indices:
                idx = int(idx)

                own_position = (
                    validation_joint_x[
                        idx,
                        agent_idx,
                        :2,
                    ]
                )

                other_distances = []

                for other_idx in range(
                        validation_joint_x.shape[1]
                ):
                    if (
                            other_idx
                            == agent_idx
                    ):
                        continue

                    other_position = (
                        validation_joint_x[
                            idx,
                            other_idx,
                            :2,
                        ]
                    )

                    distance = jnp.linalg.norm(
                        own_position
                        - other_position
                    )

                    other_distances.append(
                        distance
                    )

                min_other_distance = (
                    jnp.min(
                        jnp.stack(
                            other_distances
                        )
                    )
                )

                print(
                    f"\nsample {idx}"
                )

                print(
                    "velocity score:",
                    float(
                        velocity_score[idx]
                    ),
                )

                print(
                    "min other-agent distance:",
                    float(
                        min_other_distance
                    ),
                )

                print(
                    "target delta velocity:",
                    validation_targets[
                        idx,
                        2:4,
                    ],
                )

                print(
                    "predicted delta velocity:",
                    predicted_mean[
                        idx,
                        2:4,
                    ],
                )

                print(
                    "predicted velocity variance:",
                    predicted_variance[
                        idx,
                        2:4,
                    ],
                )

                print(
                    "standardized velocity error:",
                    standardized_squared_error[
                        idx,
                        2:4,
                    ],
                )

        print(
            "best step:",
            best_step,
        )

        print(
            "best validation RMSE:",
            best_validation_rmse,
        )

        print(
            "NLL at best-RMSE checkpoint:",
            best_validation_nll,
        )

        print(
            "final RMSE:",
            final_rmse,
        )

        print(
            "per-dim NLL:",
            final_metrics[
                "per_dim_nll"
            ],
        )

        print(
            "per-dim RMSE:",
            final_metrics[
                "per_dim_rmse"
            ],
        )

        print(
            "mean variance per dim:",
            final_metrics[
                "mean_variance_per_dim"
            ],
        )

        print(
            "MSE / variance ratio:",
            final_metrics[
                "error_variance_ratio"
            ],
        )

        print(
            "mean log_var per dim:",
            final_metrics[
                "mean_log_var_per_dim"
            ],
        )

        print(
            "min log_var per dim:",
            final_metrics[
                "min_log_var_per_dim"
            ],
        )

        print(
            "mean standardized squared error:",
            final_metrics[
                "mean_standardized_squared_error_per_dim"
            ],
        )

        print(
            "p95 standardized squared error:",
            final_metrics[
                "p95_standardized_squared_error_per_dim"
            ],
        )

        print(
            "p99 standardized squared error:",
            final_metrics[
                "p99_standardized_squared_error_per_dim"
            ],
        )

        assert jnp.isfinite(
            final_metrics["nll"]
        )

        assert jnp.isfinite(
            final_metrics[
                "physical_rmse"
            ]
        )

        assert (
            final_rmse
            <= initial_rmse
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
        action_type=(
            cfg.ENV.ACTION_TYPE
        ),
    )

    num_agents = (
        env.num_agents
    )

    num_landmarks = (
        env.num_landmarks
    )

    obs_dim = (
        4
        + 2 * num_landmarks
    )

    action_space = (
        env.action_space(
            env.agents[0]
        )
    )

    action_dim = (
        action_space.shape[0]
    )

    ensemble_size = int(
        cfg.DYNAMICS.ENSEMBLE_SIZE
    )

    print(
        "Independent dynamics ensemble test"
    )

    print(
        "================================"
    )

    print(
        "num agents:",
        num_agents,
    )

    print(
        "ensemble size per agent:",
        ensemble_size,
    )

    print(
        "total dynamics models:",
        num_agents * ensemble_size,
    )

    # ========================================================
    # RNG
    # ========================================================

    rng = jax.random.PRNGKey(
        cfg.ENV.SEED
    )

    # ========================================================
    # Collect shared real transitions
    # ========================================================

    rng, collector_rng = (
        jax.random.split(rng)
    )

    collector_state = (
        init_real_collector(
            env=env,
            num_envs=(
                cfg.ENV.NUM_ENVS
            ),
            rng=collector_rng,
        )
    )

    replay_buffer = (
        init_real_replay_buffer(
            capacity=BUFFER_CAPACITY,
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_landmarks=(
                num_landmarks
            ),
        )
    )

    print("\nCollecting real data")
    print("------------------------------")

    for _ in range(
        NUM_COLLECT_STEPS
    ):
        rng, action_rng = (
            jax.random.split(rng)
        )

        joint_action = (
            jax.random.uniform(
                action_rng,
                shape=(
                    cfg.ENV.NUM_ENVS,
                    num_agents,
                    action_dim,
                ),
                minval=(
                    action_space.low
                ),
                maxval=(
                    action_space.high
                ),
            )
        )

        (
            collector_state,
            transition,
            _,
        ) = collect_real_step(
            env=env,
            collector_state=(
                collector_state
            ),
            joint_action=joint_action,
            num_landmarks=(
                num_landmarks
            ),
            episode_horizon=(
                cfg.ENV.EPISODE_HORIZON
            ),
        )

        replay_buffer = (
            add_real_batch(
                replay_buffer,
                transition,
            )
        )

    real_batch = (
        get_valid_replay_batch(
            replay_buffer
        )
    )

    print(
        "shared replay size:",
        int(
            replay_buffer.size
        ),
    )

    # ========================================================
    # Shared model architecture
    # ========================================================

    model = (
        ProbabilisticDynamicsModel(
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
    )

    # ========================================================
    # Train one independent ensemble per agent
    # ========================================================

    agent_ensembles = []
    validation_sets = []

    print(
        "\n\nTraining independent ensembles"
    )

    print(
        "================================"
    )

    for agent_idx in range(
        num_agents
    ):
        print(
            f"\n\nAgent {agent_idx}"
        )

        print(
            "================================"
        )

        (
            rng,
            agent_ensemble,
            validation_inputs,
            validation_targets,
        ) = train_agent_ensemble(
            rng=rng,
            agent_idx=agent_idx,
            real_batch=real_batch,
            model=model,
            ensemble_size=(
                ensemble_size
            ),
            dynamics_config=(
                cfg.DYNAMICS
            ),
        )

        agent_ensembles.append(
            agent_ensemble
        )

        validation_sets.append(
            (
                validation_inputs,
                validation_targets,
            )
        )

    # ========================================================
    # Pack all independent ensembles
    # ========================================================

    dynamics_ensembles = (
        build_independent_dynamics_ensembles(
            agent_ensembles
        )
    )

    assert (
        len(
            dynamics_ensembles.agents
        )
        == num_agents
    )

    for agent_idx in range(
        num_agents
    ):
        assert (
            len(
                dynamics_ensembles
                .agents[agent_idx]
                .members
            )
            == ensemble_size
        )

    # ========================================================
    # Full-validation ensemble inference
    # ========================================================

    print(
        "\n\nIndependent ensemble inference"
    )

    print(
        "================================"
    )

    for agent_idx in range(
        num_agents
    ):
        (
            validation_inputs,
            validation_targets,
        ) = validation_sets[
            agent_idx
        ]

        (
            member_means,
            member_variances,
        ) = predict_agent_dynamics_ensemble(
            dynamics_ensembles=(
                dynamics_ensembles
            ),
            agent_idx=agent_idx,
            dynamics_inputs=(
                validation_inputs
            ),
        )

        num_validation_samples = (
            validation_inputs.shape[0]
        )

        print(
            f"\nAgent {agent_idx}"
        )

        print(
            "------------------------------"
        )

        print(
            "member means shape:",
            member_means.shape,
        )

        print(
            "member variances shape:",
            member_variances.shape,
        )

        assert (
            member_means.shape
            == (
                ensemble_size,
                num_validation_samples,
                4,
            )
        )

        assert (
            member_variances.shape
            == (
                ensemble_size,
                num_validation_samples,
                4,
            )
        )

        assert jnp.all(
            jnp.isfinite(
                member_means
            )
        )

        assert jnp.all(
            jnp.isfinite(
                member_variances
            )
        )

        assert jnp.all(
            member_variances > 0.0
        )

        # ----------------------------------------------------
        # Member diversity
        # ----------------------------------------------------

        max_prediction_difference = (
            jnp.max(
                jnp.abs(
                    member_means
                    - member_means[
                        0:1
                    ]
                )
            )
        )

        print(
            "max member prediction difference:",
            float(
                max_prediction_difference
            ),
        )

        assert (
            max_prediction_difference
            > 0.0
        )

        # ----------------------------------------------------
        # Member RMSE on the same full validation set
        # ----------------------------------------------------

        member_errors = (
            member_means
            - validation_targets[
                None,
                ...
            ]
        )

        member_rmse = (
            jnp.sqrt(
                jnp.mean(
                    jnp.square(
                        member_errors
                    ),
                    axis=(1, 2),
                )
            )
        )

        print(
            "member RMSE:",
            member_rmse,
        )

        assert jnp.all(
            jnp.isfinite(
                member_rmse
            )
        )

        # ----------------------------------------------------
        # Ensemble mean diagnostic
        # ----------------------------------------------------
        # This is only a diagnostic.
        # Model rollout will use one fixed member per trajectory.

        ensemble_mean = (
            jnp.mean(
                member_means,
                axis=0,
            )
        )

        ensemble_mean_rmse = (
            jnp.sqrt(
                jnp.mean(
                    jnp.square(
                        ensemble_mean
                        - validation_targets
                    )
                )
            )
        )

        print(
            "ensemble mean RMSE:",
            float(
                ensemble_mean_rmse
            ),
        )

        assert jnp.isfinite(
            ensemble_mean_rmse
        )

    # ========================================================
    # Final checks
    # ========================================================

    print(
        "\n\n================================"
    )

    print(
        "ALL INDEPENDENT ENSEMBLE TESTS PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()