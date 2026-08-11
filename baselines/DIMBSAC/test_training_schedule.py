from training_schedule import (
    advance_environment_steps,
    init_dimbsac_schedule,
    mark_dynamics_initialized,
    mark_dynamics_retrained,
    should_initialize_dynamics,
    should_retrain_dynamics,
)


NUM_ENVS = 32

INIT_REAL_TRANSITIONS = 6400

RETRAIN_INTERVAL_ENV_STEPS = 250


def main():
    state = (
        init_dimbsac_schedule()
    )

    print(
        "DIMBSAC schedule test"
    )

    print(
        "================================"
    )

    # ========================================================
    # Dynamics warmup
    # ========================================================

    num_collects_to_init = (
        INIT_REAL_TRANSITIONS
        // NUM_ENVS
    )

    assert (
        num_collects_to_init
        == 200
    )

    for collect_idx in range(
        num_collects_to_init
    ):
        state = (
            advance_environment_steps(
                schedule_state=state,
                num_transitions=NUM_ENVS,
            )
        )

        if collect_idx < (
            num_collects_to_init - 1
        ):
            assert not (
                should_initialize_dynamics(
                    schedule_state=state,
                    init_real_transitions=(
                        INIT_REAL_TRANSITIONS
                    ),
                )
            )

    assert (
        state.total_env_steps
        == 6400
    )

    assert (
        should_initialize_dynamics(
            schedule_state=state,
            init_real_transitions=(
                INIT_REAL_TRANSITIONS
            ),
        )
    )

    print(
        "Dynamics initialization due at:",
        state.total_env_steps,
    )

    state = (
        mark_dynamics_initialized(
            state
        )
    )

    assert (
        state.dynamics_initialized
    )

    assert (
        state.last_dynamics_retrain_env_step
        == 6400
    )

    # ========================================================
    # First periodic retraining
    # ========================================================

    assert not (
        should_retrain_dynamics(
            schedule_state=state,
            retrain_interval_env_steps=(
                RETRAIN_INTERVAL_ENV_STEPS
            ),
        )
    )

    # 7 vectorized collects:
    # 7 * 32 = 224 transitions.
    for _ in range(7):
        state = (
            advance_environment_steps(
                schedule_state=state,
                num_transitions=NUM_ENVS,
            )
        )

    assert (
        state.total_env_steps
        == 6624
    )

    assert not (
        should_retrain_dynamics(
            schedule_state=state,
            retrain_interval_env_steps=(
                RETRAIN_INTERVAL_ENV_STEPS
            ),
        )
    )

    print(
        "After 224 new transitions:",
        state.total_env_steps,
        "-> no retrain"
    )

    # One more vectorized collect crosses 250:
    # 8 * 32 = 256.
    state = (
        advance_environment_steps(
            schedule_state=state,
            num_transitions=NUM_ENVS,
        )
    )

    assert (
        state.total_env_steps
        == 6656
    )

    assert (
        should_retrain_dynamics(
            schedule_state=state,
            retrain_interval_env_steps=(
                RETRAIN_INTERVAL_ENV_STEPS
            ),
        )
    )

    print(
        "After 256 new transitions:",
        state.total_env_steps,
        "-> retrain"
    )

    state = (
        mark_dynamics_retrained(
            state
        )
    )

    assert (
        state.last_dynamics_retrain_env_step
        == 6656
    )

    # ========================================================
    # Second periodic retraining
    # ========================================================

    for _ in range(8):
        state = (
            advance_environment_steps(
                schedule_state=state,
                num_transitions=NUM_ENVS,
            )
        )

    assert (
        state.total_env_steps
        == 6912
    )

    assert (
        should_retrain_dynamics(
            schedule_state=state,
            retrain_interval_env_steps=(
                RETRAIN_INTERVAL_ENV_STEPS
            ),
        )
    )

    print(
        "Second retraining due at:",
        state.total_env_steps,
    )

    state = (
        mark_dynamics_retrained(
            state
        )
    )

    # ========================================================
    # Final result
    # ========================================================

    print(
        "\nFinal schedule state:"
    )

    print(
        state
    )

    print(
        "\n================================"
    )

    print(
        "DIMBSAC TRAINING SCHEDULE TEST PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()