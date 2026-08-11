from typing import NamedTuple


class DIMBSACScheduleState(NamedTuple):
    """Bookkeeping for DIMBSAC training schedules."""

    total_env_steps: int
    dynamics_initialized: bool
    last_dynamics_retrain_env_step: int


def init_dimbsac_schedule():
    return DIMBSACScheduleState(
        total_env_steps=0,
        dynamics_initialized=False,
        last_dynamics_retrain_env_step=0,
    )


def advance_environment_steps(
    schedule_state: DIMBSACScheduleState,
    num_transitions: int,
):
    if num_transitions <= 0:
        raise ValueError("num_transitions must be positive.")

    return schedule_state._replace(
        total_env_steps=(
            schedule_state.total_env_steps + num_transitions
        )
    )


def should_initialize_dynamics(
    schedule_state: DIMBSACScheduleState,
    init_real_transitions: int,
):
    if schedule_state.dynamics_initialized:
        return False

    return (
        schedule_state.total_env_steps
        >= init_real_transitions
    )


def mark_dynamics_initialized(
    schedule_state: DIMBSACScheduleState,
):
    if schedule_state.dynamics_initialized:
        raise ValueError("Dynamics have already been initialized.")

    return schedule_state._replace(
        dynamics_initialized=True,
        last_dynamics_retrain_env_step=(
            schedule_state.total_env_steps
        ),
    )


def should_retrain_dynamics(
    schedule_state: DIMBSACScheduleState,
    retrain_interval_env_steps: int,
):
    if not schedule_state.dynamics_initialized:
        return False

    if retrain_interval_env_steps <= 0:
        raise ValueError(
            "retrain_interval_env_steps must be positive."
        )

    elapsed = (
        schedule_state.total_env_steps
        - schedule_state.last_dynamics_retrain_env_step
    )

    return elapsed >= retrain_interval_env_steps


def mark_dynamics_retrained(
    schedule_state: DIMBSACScheduleState,
):
    if not schedule_state.dynamics_initialized:
        raise ValueError(
            "Cannot retrain dynamics before initialization."
        )

    return schedule_state._replace(
        last_dynamics_retrain_env_step=(
            schedule_state.total_env_steps
        )
    )
