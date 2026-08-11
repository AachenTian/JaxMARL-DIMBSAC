import jax
import jax.numpy as jnp
import jaxmarl

from dynamics.model import (
    ProbabilisticDynamicsModel,
)
from dynamics.normalization import (
    build_agent_dynamics_data,
    compute_normalization_stats,
    normalize,
)
from dynamics.trainer import (
    evaluate_dynamics_model,
    init_dynamics_train_state,
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


NUM_ENVS = 32
NUM_COLLECT_STEPS = 200

BUFFER_CAPACITY = 10_000
EPISODE_HORIZON = 25

TRAIN_FRACTION = 0.8
BATCH_SIZE = 256

TRAIN_STEPS = 1500
EVAL_INTERVAL = 300

DYNAMICS_LR = 1e-3


def get_valid_replay_batch(
    replay_buffer,
):
    """Extract all currently valid real transitions."""

    valid_size = int(
        replay_buffer.size
    )

    return RealTransitionBatch(
        obs=replay_buffer.obs[
            :valid_size
        ],
        x=replay_buffer.x[
            :valid_size
        ],
        actions=replay_buffer.actions[
            :valid_size
        ],
        rewards=replay_buffer.rewards[
            :valid_size
        ],
        next_obs=replay_buffer.next_obs[
            :valid_size
        ],
        next_x=replay_buffer.next_x[
            :valid_size
        ],
        dones=replay_buffer.dones[
            :valid_size
        ],
        episode_steps=(
            replay_buffer.episode_steps[
                :valid_size
            ]
        ),
        landmarks=replay_buffer.landmarks[
            :valid_size
        ],
    )


def main():

    env = jaxmarl.make(
        "MPE_simple_spread_v3",
        action_type="Continuous",
    )

    num_agents = env.num_agents
    num_landmarks = env.num_landmarks

    obs_dim = (
        4 + 2 * num_landmarks
    )

    action_space = env.action_space(
        env.agents[0]
    )

    action_dim = action_space.shape[0]

    rng = jax.random.PRNGKey(0)

    # ========================================================
    # Collect real data
    # ========================================================

    rng, collector_rng = jax.random.split(
        rng
    )

    collector_state = init_real_collector(
        env=env,
        num_envs=NUM_ENVS,
        rng=collector_rng,
    )

    replay_buffer = init_real_replay_buffer(
        capacity=BUFFER_CAPACITY,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
    )

    for _ in range(
        NUM_COLLECT_STEPS
    ):

        rng, action_rng = jax.random.split(
            rng
        )

        joint_action = jax.random.uniform(
            action_rng,
            shape=(
                NUM_ENVS,
                num_agents,
                action_dim,
            ),
            minval=action_space.low,
            maxval=action_space.high,
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
            episode_horizon=EPISODE_HORIZON,
        )

        replay_buffer = add_real_batch(
            replay_buffer,
            transition,
        )

    real_batch = get_valid_replay_batch(
        replay_buffer
    )

    num_samples = real_batch.x.shape[0]

    print("Real dynamics training")
    print("------------------------------")
    print(
        "real transitions:",
        num_samples,
    )

    # JIT is safe after the standalone model tests.
    update_fn = jax.jit(
        update_dynamics_model
    )

    # ========================================================
    # Train one independent model for each agent
    # ========================================================

    for agent_idx in range(
        num_agents
    ):

        print(
            f"\nAgent {agent_idx}"
        )
        print("------------------------------")

        (
            dynamics_inputs,
            delta_x_targets,
        ) = build_agent_dynamics_data(
            real_batch,
            agent_idx,
        )

        # Fixed random train/validation split.
        rng, split_rng = jax.random.split(
            rng
        )

        permutation = jax.random.permutation(
            split_rng,
            num_samples,
        )

        train_size = int(
            num_samples
            * TRAIN_FRACTION
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

        print(
            "train samples:",
            train_inputs.shape[0],
        )

        print(
            "validation samples:",
            validation_inputs.shape[0],
        )

        # Normalization statistics use real training data only.
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

        normalized_train_inputs = normalize(
            train_inputs,
            input_stats,
        )

        normalized_train_targets = normalize(
            train_targets,
            target_stats,
        )

        model = ProbabilisticDynamicsModel(
            output_dim=4,
            hidden_dims=(256, 256),
            log_var_min=-10.0,
            log_var_max=2.0,
        )

        rng, model_rng = jax.random.split(
            rng
        )

        state = init_dynamics_train_state(
            rng=model_rng,
            model=model,
            input_dim=4 + action_dim,
            input_stats=input_stats,
            target_stats=target_stats,
            learning_rate=DYNAMICS_LR,
        )

        initial_metrics = (
            evaluate_dynamics_model(
                state,
                validation_inputs,
                validation_targets,
            )
        )

        initial_rmse = float(
            initial_metrics[
                "physical_rmse"
            ]
        )

        # Keep the checkpoint with the best validation NLL.
        best_validation_nll = float(
            initial_metrics["nll"]
        )

        best_params = state.model.params
        best_step = 0

        print("\nBefore training")
        print(
            "validation NLL:",
            float(
                initial_metrics["nll"]
            ),
        )
        print(
            "physical RMSE:",
            initial_rmse,
        )
        print(
            "per-dim RMSE:",
            initial_metrics[
                "per_dim_rmse"
            ],
        )

        # ====================================================
        # Training
        # ====================================================

        for train_step in range(
            1,
            TRAIN_STEPS + 1,
        ):

            rng, batch_rng = (
                jax.random.split(rng)
            )

            batch_indices = (
                jax.random.randint(
                    batch_rng,
                    shape=(BATCH_SIZE,),
                    minval=0,
                    maxval=train_size,
                )
            )

            batch_inputs = (
                normalized_train_inputs[
                    batch_indices
                ]
            )

            batch_targets = (
                normalized_train_targets[
                    batch_indices
                ]
            )

            state, train_metrics = (
                update_fn(
                    state,
                    batch_inputs,
                    batch_targets,
                )
            )

            if (
                    train_step
                    % EVAL_INTERVAL
                    == 0
            ):

                validation_metrics = (
                    evaluate_dynamics_model(
                        state,
                        validation_inputs,
                        validation_targets,
                    )
                )

                validation_nll = float(
                    validation_metrics["nll"]
                )

                if (
                        validation_nll
                        < best_validation_nll
                ):
                    best_validation_nll = (
                        validation_nll
                    )

                    best_params = (
                        state.model.params
                    )

                    best_step = train_step

                print(
                    f"\ntrain_step = "
                    f"{train_step}"
                )

                print(
                    "train NLL:",
                    float(
                        train_metrics[
                            "nll_loss"
                        ]
                    ),
                )

                print(
                    "validation NLL:",
                    validation_nll,
                )

                print(
                    "physical RMSE:",
                    float(
                        validation_metrics[
                            "physical_rmse"
                        ]
                    ),
                )

        # ====================================================
        # Restore the checkpoint with the best validation NLL.
        # ====================================================
        state = state._replace(
            model=state.model.replace(
                params=best_params
            )
        )

        print(
            "\nbest validation step:",
            best_step,
        )

        print(
            "best validation NLL:",
            best_validation_nll,
        )

        # ====================================================
        # Final validation
        # ====================================================

        final_metrics = (
            evaluate_dynamics_model(
                state,
                validation_inputs,
                validation_targets,
            )
        )

        final_rmse = float(
            final_metrics[
                "physical_rmse"
            ]
        )

        print("\nFinal validation")
        print(
            "NLL:",
            float(
                final_metrics["nll"]
            ),
        )

        print(
            "physical RMSE:",
            final_rmse,
        )

        print(
            "per-dim RMSE:",
            final_metrics[
                "per_dim_rmse"
            ],
        )

        print(
            "mean physical variance:",
            float(
                final_metrics[
                    "mean_physical_variance"
                ]
            ),
        )

        assert jnp.isfinite(
            final_metrics["nll"]
        )

        assert final_rmse < initial_rmse

    print("\n==============================")
    print(
        "ALL REAL DYNAMICS TRAINING "
        "TESTS PASSED"
    )
    print("==============================")


if __name__ == "__main__":
    main()