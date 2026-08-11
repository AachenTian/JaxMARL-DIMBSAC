from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from dynamics.ensemble import (
    DynamicsEnsembleState,
    IndependentDynamicsEnsembles,
)
from dynamics.normalization import (
    build_agent_dynamics_data,
    compute_normalization_stats,
)
from dynamics.online_trainer import (
    _create_bootstrap_indices,
    retrain_agent_ensemble,
    retrain_independent_dynamics,
)
from dynamics.trainer import (
    evaluate_dynamics_model,
    init_dynamics_train_state,
)


# Test-only constants.
NUM_AGENTS = 3
ACTION_DIM = 5
STATE_DIM = 4
INPUT_DIM = STATE_DIM + ACTION_DIM

ENSEMBLE_SIZE = 2
LEARNING_RATE = 1e-3

WARMUP_SIZE = 768
ROUND_1_SIZE = 1024
ROUND_2_SIZE = 1536

BATCH_SIZE = 128


class SyntheticRealBatch(NamedTuple):
    """Minimal real batch required by dynamics training."""

    x: jax.Array
    actions: jax.Array
    next_x: jax.Array


class TinyDynamicsModel(nn.Module):
    """Small probabilistic model used only for this unit test."""

    hidden_dim: int = 64
    output_dim: int = STATE_DIM

    @nn.compact
    def __call__(
        self,
        inputs,
    ):
        x = nn.Dense(
            self.hidden_dim
        )(
            inputs
        )

        x = nn.silu(x)

        x = nn.Dense(
            self.hidden_dim
        )(
            x
        )

        x = nn.silu(x)

        mean = nn.Dense(
            self.output_dim
        )(
            x
        )

        raw_log_var = nn.Dense(
            self.output_dim
        )(
            x
        )

        log_var = jnp.clip(
            raw_log_var,
            -10.0,
            2.0,
        )

        return (
            mean,
            log_var,
        )


def build_synthetic_real_batch(
    rng,
    num_samples: int,
):
    """Create deterministic learnable multi-agent dynamics."""

    rng, x_rng, action_rng = (
        jax.random.split(
            rng,
            3,
        )
    )

    x = jax.random.normal(
        x_rng,
        shape=(
            num_samples,
            NUM_AGENTS,
            STATE_DIM,
        ),
    )

    actions = jax.random.uniform(
        action_rng,
        shape=(
            num_samples,
            NUM_AGENTS,
            ACTION_DIM,
        ),
        minval=0.0,
        maxval=1.0,
    )

    agent_offsets = jnp.arange(
        NUM_AGENTS,
        dtype=jnp.float32,
    )[None, :]

    delta_px = (
        0.05 * x[:, :, 2]
        + 0.02
        * (
            actions[:, :, 1]
            - actions[:, :, 2]
        )
    )

    delta_py = (
        0.05 * x[:, :, 3]
        + 0.02
        * (
            actions[:, :, 3]
            - actions[:, :, 4]
        )
    )

    delta_vx = (
        -0.08 * x[:, :, 2]
        + 0.25
        * (
            actions[:, :, 1]
            - actions[:, :, 2]
        )
        + 0.01 * agent_offsets
    )

    delta_vy = (
        -0.08 * x[:, :, 3]
        + 0.25
        * (
            actions[:, :, 3]
            - actions[:, :, 4]
        )
        - 0.01 * agent_offsets
    )

    delta_x = jnp.stack(
        [
            delta_px,
            delta_py,
            delta_vx,
            delta_vy,
        ],
        axis=-1,
    )

    next_x = (
        x + delta_x
    )

    return (
        rng,
        SyntheticRealBatch(
            x=x,
            actions=actions,
            next_x=next_x,
        ),
    )


def slice_batch(
    batch,
    end: int,
):
    return SyntheticRealBatch(
        x=batch.x[:end],
        actions=batch.actions[:end],
        next_x=batch.next_x[:end],
    )


def initialize_dynamics_once(
    rng,
    warmup_batch,
):
    """
    Initialize all ensembles once.

    Normalization statistics are computed only here.
    """

    model = TinyDynamicsModel()

    agent_ensembles = []

    for agent_idx in range(
        NUM_AGENTS
    ):
        (
            dynamics_inputs,
            delta_x_targets,
        ) = build_agent_dynamics_data(
            batch=warmup_batch,
            agent_idx=agent_idx,
        )

        input_stats = (
            compute_normalization_stats(
                dynamics_inputs
            )
        )

        target_stats = (
            compute_normalization_stats(
                delta_x_targets
            )
        )

        members = []

        for _ in range(
            ENSEMBLE_SIZE
        ):
            rng, member_rng = (
                jax.random.split(rng)
            )

            member_state = (
                init_dynamics_train_state(
                    rng=member_rng,
                    model=model,
                    input_dim=INPUT_DIM,
                    input_stats=input_stats,
                    target_stats=target_stats,
                    learning_rate=(
                        LEARNING_RATE
                    ),
                )
            )

            members.append(
                member_state
            )

        agent_ensembles.append(
            DynamicsEnsembleState(
                members=tuple(
                    members
                )
            )
        )

    dynamics = (
        IndependentDynamicsEnsembles(
            agents=tuple(
                agent_ensembles
            )
        )
    )

    return (
        rng,
        dynamics,
    )


def get_optimizer_steps(
    dynamics,
):
    """Return optimizer step for every agent/member."""

    steps = []

    for agent_ensemble in (
        dynamics.agents
    ):
        agent_steps = []

        for member in (
            agent_ensemble.members
        ):
            agent_steps.append(
                int(
                    member.model.step
                )
            )

        steps.append(
            agent_steps
        )

    return steps


def check_stats_unchanged(
    before,
    after,
):
    """Verify that normalization statistics stay frozen."""

    for agent_idx in range(
        NUM_AGENTS
    ):
        for member_idx in range(
            ENSEMBLE_SIZE
        ):
            before_member = (
                before.agents[
                    agent_idx
                ].members[
                    member_idx
                ]
            )

            after_member = (
                after.agents[
                    agent_idx
                ].members[
                    member_idx
                ]
            )

            assert jnp.array_equal(
                before_member
                .input_stats.mean,
                after_member
                .input_stats.mean,
            )

            assert jnp.array_equal(
                before_member
                .input_stats.std,
                after_member
                .input_stats.std,
            )

            assert jnp.array_equal(
                before_member
                .target_stats.mean,
                after_member
                .target_stats.mean,
            )

            assert jnp.array_equal(
                before_member
                .target_stats.std,
                after_member
                .target_stats.std,
            )


def evaluate_mean_rmse(
    dynamics,
    batch,
):
    """Evaluate mean member RMSE for every agent."""

    agent_rmse = []

    for agent_idx in range(
        NUM_AGENTS
    ):
        (
            dynamics_inputs,
            delta_x_targets,
        ) = build_agent_dynamics_data(
            batch=batch,
            agent_idx=agent_idx,
        )

        member_rmse = []

        for member in (
            dynamics.agents[
                agent_idx
            ].members
        ):
            metrics = (
                evaluate_dynamics_model(
                    state=member,
                    dynamics_inputs=(
                        dynamics_inputs
                    ),
                    delta_x_targets=(
                        delta_x_targets
                    ),
                )
            )

            member_rmse.append(
                metrics[
                    "physical_rmse"
                ]
            )

        agent_rmse.append(
            float(
                jnp.mean(
                    jnp.stack(
                        member_rmse
                    )
                )
            )
        )

    return agent_rmse


def main():
    rng = jax.random.PRNGKey(0)

    # ========================================================
    # Build growing real replay data
    # ========================================================

    (
        rng,
        full_batch,
    ) = build_synthetic_real_batch(
        rng=rng,
        num_samples=ROUND_2_SIZE,
    )

    warmup_batch = slice_batch(
        full_batch,
        WARMUP_SIZE,
    )

    round_1_batch = slice_batch(
        full_batch,
        ROUND_1_SIZE,
    )

    round_2_batch = slice_batch(
        full_batch,
        ROUND_2_SIZE,
    )

    # ========================================================
    # Initialize dynamics exactly once
    # ========================================================

    (
        rng,
        initial_dynamics,
    ) = initialize_dynamics_once(
        rng=rng,
        warmup_batch=warmup_batch,
    )

    initial_steps = (
        get_optimizer_steps(
            initial_dynamics
        )
    )

    initial_rmse = (
        evaluate_mean_rmse(
            dynamics=initial_dynamics,
            batch=round_2_batch,
        )
    )

    print(
        "Periodic dynamics retraining test"
    )

    print(
        "================================"
    )

    print(
        "\nInitial optimizer steps:"
    )

    print(
        initial_steps
    )

    print(
        "Initial mean RMSE:",
        initial_rmse,
    )

    for agent_steps in (
        initial_steps
    ):
        assert all(
            step == 0
            for step in agent_steps
        )

    # ========================================================
    # Bootstrap resampling test
    # ========================================================

    rng, bootstrap_rng_1, bootstrap_rng_2 = (
        jax.random.split(
            rng,
            3,
        )
    )

    bootstrap_1 = (
        _create_bootstrap_indices(
            rng=bootstrap_rng_1,
            ensemble_size=(
                ENSEMBLE_SIZE
            ),
            num_train=500,
        )
    )

    bootstrap_2 = (
        _create_bootstrap_indices(
            rng=bootstrap_rng_2,
            ensemble_size=(
                ENSEMBLE_SIZE
            ),
            num_train=500,
        )
    )

    different_bootstrap_found = any(
        not bool(
            jnp.array_equal(
                bootstrap_1[i],
                bootstrap_2[i],
            )
        )
        for i in range(
            ENSEMBLE_SIZE
        )
    )

    assert (
        different_bootstrap_found
    )

    print(
        "\nFresh bootstrap check: passed"
    )

    # ========================================================
    # Retraining round 1
    # ========================================================

    (
        rng,
        round_1_dynamics,
        round_1_metrics,
    ) = retrain_independent_dynamics(
        rng=rng,
        dynamics_ensembles=(
            initial_dynamics
        ),
        real_batch=round_1_batch,
        train_fraction=0.8,
        batch_size=BATCH_SIZE,
        max_epochs=8,
        patience=20,
        min_delta=1e-6,
    )

    round_1_steps = (
        get_optimizer_steps(
            round_1_dynamics
        )
    )

    round_1_rmse = (
        evaluate_mean_rmse(
            dynamics=round_1_dynamics,
            batch=round_2_batch,
        )
    )

    print(
        "\nAfter retraining round 1"
    )

    print(
        "--------------------------------"
    )

    print(
        "optimizer steps:",
        round_1_steps,
    )

    print(
        "mean RMSE:",
        round_1_rmse,
    )

    check_stats_unchanged(
        initial_dynamics,
        round_1_dynamics,
    )

    for agent_idx in range(
        NUM_AGENTS
    ):
        for member_idx in range(
            ENSEMBLE_SIZE
        ):
            assert (
                round_1_steps[
                    agent_idx
                ][
                    member_idx
                ]
                > 0
            )

    # ========================================================
    # Retraining round 2 with more real data
    # ========================================================

    (
        rng,
        round_2_dynamics,
        round_2_metrics,
    ) = retrain_independent_dynamics(
        rng=rng,
        dynamics_ensembles=(
            round_1_dynamics
        ),
        real_batch=round_2_batch,
        train_fraction=0.8,
        batch_size=BATCH_SIZE,
        max_epochs=4,
        patience=20,
        min_delta=1e-6,
    )

    round_2_steps = (
        get_optimizer_steps(
            round_2_dynamics
        )
    )

    round_2_rmse = (
        evaluate_mean_rmse(
            dynamics=round_2_dynamics,
            batch=round_2_batch,
        )
    )

    print(
        "\nAfter retraining round 2"
    )

    print(
        "--------------------------------"
    )

    print(
        "optimizer steps:",
        round_2_steps,
    )

    print(
        "mean RMSE:",
        round_2_rmse,
    )

    check_stats_unchanged(
        initial_dynamics,
        round_2_dynamics,
    )

    # Optimizer steps must never reset between retraining rounds.
    for agent_idx in range(
        NUM_AGENTS
    ):
        for member_idx in range(
            ENSEMBLE_SIZE
        ):
            assert (
                round_2_steps[
                    agent_idx
                ][
                    member_idx
                ]
                >= round_1_steps[
                    agent_idx
                ][
                    member_idx
                ]
            )

    # ========================================================
    # RMSE sanity checks
    # ========================================================

    assert all(
        jnp.isfinite(x)
        for x in round_1_rmse
    )

    assert all(
        jnp.isfinite(x)
        for x in round_2_rmse
    )

    print(
        "\nRMSE progression"
    )

    print(
        "--------------------------------"
    )

    for agent_idx in range(
        NUM_AGENTS
    ):
        print(
            f"Agent {agent_idx}: "
            f"{initial_rmse[agent_idx]:.6f}"
            f" -> "
            f"{round_1_rmse[agent_idx]:.6f}"
            f" -> "
            f"{round_2_rmse[agent_idx]:.6f}"
        )

    # The first training round should learn the simple
    # synthetic dynamics substantially better.
    for agent_idx in range(
        NUM_AGENTS
    ):
        assert (
            round_1_rmse[
                agent_idx
            ]
            < initial_rmse[
                agent_idx
            ]
        )

    # ========================================================
    # Forced early-stopping test
    # ========================================================

    print(
        "\nForced early-stopping check"
    )

    print(
        "--------------------------------"
    )

    (
        rng,
        _,
        early_stop_metrics,
    ) = retrain_agent_ensemble(
        rng=rng,
        ensemble_state=(
            round_2_dynamics.agents[0]
        ),
        real_batch=round_2_batch,
        agent_idx=0,
        train_fraction=0.8,
        batch_size=BATCH_SIZE,
        max_epochs=10,
        patience=2,

        # No realistic RMSE improvement can exceed this
        # threshold, so patience must trigger.
        min_delta=1e6,
    )

    print(
        "stopped early:",
        early_stop_metrics.stopped_early,
    )

    print(
        "epochs trained:",
        early_stop_metrics.epochs_trained,
    )

    assert (
        early_stop_metrics
        .stopped_early
    )

    assert (
        early_stop_metrics
        .epochs_trained
        == 2
    )

    # ========================================================
    # Final result
    # ========================================================

    print(
        "\nNormalization stats remained frozen."
    )

    print(
        "Optimizer steps did not reset."
    )

    print(
        "Fresh bootstrap sampling works."
    )

    print(
        "Early stopping works."
    )

    print(
        "\n================================"
    )

    print(
        "PERIODIC DYNAMICS RETRAINING TEST PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()