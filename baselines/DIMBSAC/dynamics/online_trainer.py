from typing import NamedTuple

import jax
import jax.numpy as jnp

from dynamics.ensemble import (
    DynamicsEnsembleState,
    IndependentDynamicsEnsembles,
)
from dynamics.normalization import (
    build_agent_dynamics_data,
    normalize,
)
from dynamics.trainer import (
    evaluate_dynamics_model,
    update_dynamics_model,
)


class AgentRetrainMetrics(NamedTuple):
    """Summary and diagnostics for one dynamics retraining round."""

    epochs_trained: int
    stopped_early: bool

    # Checkpoint-selection statistics. Each member is restored to its own
    # best-RMSE checkpoint at the end of the round.
    best_mean_rmse: float
    member_best_rmse: jax.Array

    # Final held-out metrics after restoring the selected checkpoints and
    # evaluating them on this round's shared validation split.
    final_eval_mean_rmse: float
    final_eval_mean_nll: float
    final_eval_per_dim_rmse: jax.Array
    final_eval_per_dim_nll: jax.Array
    final_eval_zero_delta_rmse: float
    final_eval_relative_rmse: float
    member_final_eval_rmse: jax.Array
    member_final_eval_nll: jax.Array

    # Per-epoch held-out traces before checkpoint restoration. These are
    # logging diagnostics only and do not affect early stopping.
    epoch_mean_rmse: tuple
    epoch_mean_nll: tuple

    num_train: int
    num_validation: int


def _split_train_validation(
    rng,
    inputs,
    targets,
    train_fraction: float,
):
    """Create a fresh train/validation split for one retraining round."""

    num_samples = inputs.shape[0]

    if num_samples < 2:
        raise ValueError(
            "Dynamics retraining requires at least two samples."
        )

    num_train = int(
        num_samples * train_fraction
    )

    num_train = min(
        max(num_train, 1),
        num_samples - 1,
    )

    permutation = jax.random.permutation(
        rng,
        num_samples,
    )

    train_indices = permutation[
        :num_train
    ]

    validation_indices = permutation[
        num_train:
    ]

    return (
        inputs[train_indices],
        targets[train_indices],
        inputs[validation_indices],
        targets[validation_indices],
    )


def _create_bootstrap_indices(
    rng,
    ensemble_size: int,
    num_train: int,
):
    """
    Create one fresh bootstrap dataset per ensemble member.

    Sampling is with replacement.
    """

    member_keys = jax.random.split(
        rng,
        ensemble_size,
    )

    bootstrap_indices = []

    for member_idx in range(
        ensemble_size
    ):
        indices = jax.random.randint(
            member_keys[member_idx],
            shape=(num_train,),
            minval=0,
            maxval=num_train,
        )

        bootstrap_indices.append(
            indices
        )

    return tuple(
        bootstrap_indices
    )


def _evaluate_members(
    member_states,
    validation_inputs,
    validation_targets,
):
    """Evaluate all ensemble members on the same validation set."""

    member_rmse = []
    member_nll = []
    member_per_dim_rmse = []
    member_per_dim_nll = []

    for state in member_states:
        metrics = evaluate_dynamics_model(
            state=state,
            dynamics_inputs=validation_inputs,
            delta_x_targets=validation_targets,
        )

        member_rmse.append(
            metrics["physical_rmse"]
        )
        member_nll.append(
            metrics["nll"]
        )
        member_per_dim_rmse.append(
            metrics["per_dim_rmse"]
        )
        member_per_dim_nll.append(
            metrics["per_dim_nll"]
        )

    return (
        jnp.stack(member_rmse),
        jnp.stack(member_nll),
        jnp.stack(member_per_dim_rmse),
        jnp.stack(member_per_dim_nll),
    )


def retrain_agent_ensemble(
    rng,
    ensemble_state,
    real_batch,
    agent_idx: int,
    train_fraction: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float = 1e-5,
):
    """
    Periodically retrain one existing dynamics ensemble.

    Model parameters are NOT reinitialized.
    Normalization statistics remain frozen inside each member state.
    Fresh bootstrap datasets are created for this retraining round.
    """

    dynamics_inputs, delta_x_targets = (
        build_agent_dynamics_data(
            batch=real_batch,
            agent_idx=agent_idx,
        )
    )

    # --------------------------------------------------------
    # Fresh train/validation split for this retraining round
    # --------------------------------------------------------

    rng, split_rng = jax.random.split(
        rng
    )

    (
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
    ) = _split_train_validation(
        rng=split_rng,
        inputs=dynamics_inputs,
        targets=delta_x_targets,
        train_fraction=train_fraction,
    )

    num_train = train_inputs.shape[0]
    num_validation = (
        validation_inputs.shape[0]
    )

    ensemble_size = len(
        ensemble_state.members
    )

    # --------------------------------------------------------
    # Fresh bootstrap datasets
    # --------------------------------------------------------

    rng, bootstrap_rng = (
        jax.random.split(rng)
    )

    bootstrap_indices = (
        _create_bootstrap_indices(
            rng=bootstrap_rng,
            ensemble_size=ensemble_size,
            num_train=num_train,
        )
    )

    member_states = list(
        ensemble_state.members
    )

    # Initial states are valid early-stopping checkpoints.
    best_member_states = list(
        member_states
    )

    (
        initial_member_rmse,
        _,
        _,
        _,
    ) = _evaluate_members(
        member_states=member_states,
        validation_inputs=validation_inputs,
        validation_targets=(
            validation_targets
        ),
    )

    best_member_rmse = [
        float(x)
        for x in initial_member_rmse
    ]

    best_mean_rmse = float(
        jnp.mean(
            initial_member_rmse
        )
    )

    epochs_without_improvement = 0
    stopped_early = False
    epochs_trained = 0

    epoch_mean_rmse_history = []
    epoch_mean_nll_history = []

    update_fn = jax.jit(
        update_dynamics_model
    )

    # --------------------------------------------------------
    # Retraining epochs
    # --------------------------------------------------------

    for epoch in range(
        max_epochs
    ):
        epochs_trained = (
            epoch + 1
        )

        epoch_keys = jax.random.split(
            rng,
            ensemble_size + 1,
        )

        rng = epoch_keys[0]

        for member_idx in range(
            ensemble_size
        ):
            state = member_states[
                member_idx
            ]

            member_indices = (
                bootstrap_indices[
                    member_idx
                ]
            )

            bootstrap_inputs = (
                train_inputs[
                    member_indices
                ]
            )

            bootstrap_targets = (
                train_targets[
                    member_indices
                ]
            )

            # Frozen normalization statistics.
            normalized_inputs = normalize(
                bootstrap_inputs,
                state.input_stats,
            )

            normalized_targets = normalize(
                bootstrap_targets,
                state.target_stats,
            )

            permutation = (
                jax.random.permutation(
                    epoch_keys[
                        member_idx + 1
                    ],
                    num_train,
                )
            )

            normalized_inputs = (
                normalized_inputs[
                    permutation
                ]
            )

            normalized_targets = (
                normalized_targets[
                    permutation
                ]
            )

            effective_batch_size = min(
                batch_size,
                num_train,
            )

            num_batches = max(
                1,
                num_train
                // effective_batch_size,
            )

            # Use full fixed-size minibatches.
            num_used = (
                num_batches
                * effective_batch_size
            )

            normalized_inputs = (
                normalized_inputs[
                    :num_used
                ]
            )

            normalized_targets = (
                normalized_targets[
                    :num_used
                ]
            )

            for batch_idx in range(
                num_batches
            ):
                start = (
                    batch_idx
                    * effective_batch_size
                )

                end = (
                    start
                    + effective_batch_size
                )

                state, _ = update_fn(
                    state,
                    normalized_inputs[
                        start:end
                    ],
                    normalized_targets[
                        start:end
                    ],
                )

            member_states[
                member_idx
            ] = state

        # ----------------------------------------------------
        # Shared validation set
        # ----------------------------------------------------

        (
            member_rmse,
            member_nll,
            _,
            _,
        ) = _evaluate_members(
            member_states=member_states,
            validation_inputs=(
                validation_inputs
            ),
            validation_targets=(
                validation_targets
            ),
        )

        # Store each member's best checkpoint independently.
        for member_idx in range(
            ensemble_size
        ):
            current_rmse = float(
                member_rmse[
                    member_idx
                ]
            )

            if (
                current_rmse
                < best_member_rmse[
                    member_idx
                ]
                - min_delta
            ):
                best_member_rmse[
                    member_idx
                ] = current_rmse

                best_member_states[
                    member_idx
                ] = member_states[
                    member_idx
                ]

        mean_rmse = float(
            jnp.mean(
                member_rmse
            )
        )

        mean_nll = float(
            jnp.mean(
                member_nll
            )
        )

        epoch_mean_rmse_history.append(
            mean_rmse
        )
        epoch_mean_nll_history.append(
            mean_nll
        )

        print(
            f"epoch {epoch + 1}: "
            f"mean validation RMSE="
            f"{mean_rmse:.6f}, "
            f"mean NLL="
            f"{mean_nll:.4f}"
        )

        # Ensemble-level early stopping.
        if (
            mean_rmse
            < best_mean_rmse
            - min_delta
        ):
            best_mean_rmse = (
                mean_rmse
            )

            epochs_without_improvement = (
                0
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            stopped_early = True

            print(
                "early stopping at epoch",
                epoch + 1,
            )

            break

    # --------------------------------------------------------
    # Restore best checkpoints
    # --------------------------------------------------------

    final_ensemble = (
        DynamicsEnsembleState(
            members=tuple(
                best_member_states
            )
        )
    )

    # Re-evaluate the actually restored checkpoints on this round's held-out
    # validation split. This gives a matching RMSE/NLL pair for W&B rather
    # than mixing a selected RMSE with the NLL from the last training epoch.
    (
        final_member_rmse,
        final_member_nll,
        final_member_per_dim_rmse,
        final_member_per_dim_nll,
    ) = _evaluate_members(
        member_states=final_ensemble.members,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
    )

    final_eval_mean_rmse = float(
        jnp.mean(final_member_rmse)
    )
    final_eval_mean_nll = float(
        jnp.mean(final_member_nll)
    )
    final_eval_per_dim_rmse = jnp.mean(
        final_member_per_dim_rmse,
        axis=0,
    )
    final_eval_per_dim_nll = jnp.mean(
        final_member_per_dim_nll,
        axis=0,
    )

    # A zero-delta predictor is a useful scale reference for the physical
    # RMSE. It is evaluated on the exact same held-out validation targets.
    final_eval_zero_delta_rmse = float(
        jnp.sqrt(
            jnp.mean(
                jnp.square(validation_targets)
            )
        )
    )
    final_eval_relative_rmse = (
        final_eval_mean_rmse
        / (final_eval_zero_delta_rmse + 1e-8)
    )

    metrics = AgentRetrainMetrics(
        epochs_trained=epochs_trained,
        stopped_early=stopped_early,
        best_mean_rmse=float(
            jnp.mean(
                jnp.asarray(
                    best_member_rmse
                )
            )
        ),
        member_best_rmse=jnp.asarray(
            best_member_rmse
        ),
        final_eval_mean_rmse=(
            final_eval_mean_rmse
        ),
        final_eval_mean_nll=(
            final_eval_mean_nll
        ),
        final_eval_per_dim_rmse=(
            final_eval_per_dim_rmse
        ),
        final_eval_per_dim_nll=(
            final_eval_per_dim_nll
        ),
        final_eval_zero_delta_rmse=(
            final_eval_zero_delta_rmse
        ),
        final_eval_relative_rmse=(
            final_eval_relative_rmse
        ),
        member_final_eval_rmse=(
            final_member_rmse
        ),
        member_final_eval_nll=(
            final_member_nll
        ),
        epoch_mean_rmse=tuple(
            epoch_mean_rmse_history
        ),
        epoch_mean_nll=tuple(
            epoch_mean_nll_history
        ),
        num_train=num_train,
        num_validation=num_validation,
    )

    return (
        rng,
        final_ensemble,
        metrics,
    )


def retrain_independent_dynamics(
    rng,
    dynamics_ensembles,
    real_batch,
    train_fraction: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float = 1e-5,
):
    """
    Periodically retrain all independent agent dynamics ensembles.
    """

    num_agents = len(
        dynamics_ensembles.agents
    )

    updated_agents = []
    all_metrics = []

    for agent_idx in range(
        num_agents
    ):
        print(
            f"\nRetraining dynamics Agent {agent_idx}"
        )

        print(
            "================================"
        )

        (
            rng,
            updated_ensemble,
            metrics,
        ) = retrain_agent_ensemble(
            rng=rng,
            ensemble_state=(
                dynamics_ensembles.agents[
                    agent_idx
                ]
            ),
            real_batch=real_batch,
            agent_idx=agent_idx,
            train_fraction=train_fraction,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )

        updated_agents.append(
            updated_ensemble
        )

        all_metrics.append(
            metrics
        )

        print(
            "epochs trained:",
            metrics.epochs_trained,
        )

        print(
            "best mean RMSE:",
            metrics.best_mean_rmse,
        )

        print(
            "member best RMSE:",
            metrics.member_best_rmse,
        )
        print(
            "restored validation RMSE / NLL:",
            metrics.final_eval_mean_rmse,
            "/",
            metrics.final_eval_mean_nll,
        )
        print(
            "restored per-dim RMSE:",
            metrics.final_eval_per_dim_rmse,
        )

    updated_dynamics = (
        IndependentDynamicsEnsembles(
            agents=tuple(
                updated_agents
            )
        )
    )

    return (
        rng,
        updated_dynamics,
        tuple(all_metrics),
    )