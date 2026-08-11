import jax
import jax.numpy as jnp

from dynamics.model_rollout import (
    ModelRolloutTrajectory,
)
from model_replay_buffer import (
    add_trajectory_to_model_replay_buffers,
    init_independent_model_replay_buffers,
    sample_model_batch,
    trajectory_to_agent_model_batch,
)


NUM_AGENTS = 3
OBS_DIM = 10
ACTION_DIM = 5

HORIZON = 3
BATCH_SIZE = 4
CAPACITY = 100


def main():
    # ========================================================
    # Build a small synthetic trajectory
    # ========================================================

    obs = jnp.arange(
        HORIZON
        * BATCH_SIZE
        * NUM_AGENTS
        * OBS_DIM,
        dtype=jnp.float32,
    ).reshape(
        HORIZON,
        BATCH_SIZE,
        NUM_AGENTS,
        OBS_DIM,
    )

    actions = jnp.zeros(
        (
            HORIZON,
            BATCH_SIZE,
            NUM_AGENTS,
            ACTION_DIM,
        ),
        dtype=jnp.float32,
    )

    rewards = jnp.arange(
        HORIZON
        * BATCH_SIZE
        * NUM_AGENTS,
        dtype=jnp.float32,
    ).reshape(
        HORIZON,
        BATCH_SIZE,
        NUM_AGENTS,
    )

    next_obs = (
        obs + 1.0
    )

    valid = jnp.array(
        [
            [True, True, True, True],
            [True, False, True, True],
            [False, False, True, True],
        ]
    )

    dones = jnp.array(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, False, False, False],
        ]
    )

    rollout_stops = jnp.array(
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, True, True],
        ]
    )

    x = jnp.zeros(
        (
            HORIZON,
            BATCH_SIZE,
            NUM_AGENTS,
            4,
        ),
        dtype=jnp.float32,
    )

    episode_steps = jnp.zeros(
        (
            HORIZON,
            BATCH_SIZE,
        ),
        dtype=jnp.int32,
    )

    trajectory = ModelRolloutTrajectory(
        obs=obs,
        x=x,
        actions=actions,
        rewards=rewards,
        dones=dones,
        rollout_stops=(
            rollout_stops
        ),
        valid=valid,
        episode_steps=(
            episode_steps
        ),
        next_episode_steps=(
            episode_steps + 1
        ),
        delta_x=x,
        next_obs=next_obs,
        next_x=x,
        member_indices=jnp.zeros(
            (
                BATCH_SIZE,
                NUM_AGENTS,
            ),
            dtype=jnp.int32,
        ),
    )

    expected_valid = int(
        jnp.sum(valid)
    )

    print(
        "Expected valid transitions:",
        expected_valid,
    )

    assert (
        expected_valid
        == 9
    )

    # ========================================================
    # Initialize independent buffers
    # ========================================================

    model_buffers = (
        init_independent_model_replay_buffers(
            num_agents=NUM_AGENTS,
            capacity=CAPACITY,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
        )
    )

    assert (
        len(model_buffers.agents)
        == NUM_AGENTS
    )

    # ========================================================
    # Add trajectory
    # ========================================================

    model_buffers = (
        add_trajectory_to_model_replay_buffers(
            model_buffers=(
                model_buffers
            ),
            trajectory=trajectory,
        )
    )

    print(
        "\nBuffer sizes"
    )

    print(
        "--------------------------------"
    )

    for agent_idx in range(
        NUM_AGENTS
    ):
        buffer = (
            model_buffers.agents[
                agent_idx
            ]
        )

        print(
            f"agent {agent_idx}:",
            int(buffer.size),
        )

        assert (
            int(buffer.size)
            == expected_valid
        )

    # ========================================================
    # Verify learner-specific rewards
    # ========================================================

    for agent_idx in range(
        NUM_AGENTS
    ):
        expected_batch = (
            trajectory_to_agent_model_batch(
                trajectory=trajectory,
                agent_idx=agent_idx,
            )
        )

        buffer = (
            model_buffers.agents[
                agent_idx
            ]
        )

        reward_error = jnp.max(
            jnp.abs(
                buffer.rewards[
                    :expected_valid
                ]
                - expected_batch.rewards
            )
        )

        print(
            f"agent {agent_idx} reward error:",
            float(reward_error),
        )

        assert (
            reward_error
            == 0.0
        )

    # ========================================================
    # Verify valid filtering
    # ========================================================

    first_batch = (
        trajectory_to_agent_model_batch(
            trajectory=trajectory,
            agent_idx=0,
        )
    )

    assert first_batch.obs.shape == (
        expected_valid,
        NUM_AGENTS,
        OBS_DIM,
    )

    assert first_batch.actions.shape == (
        expected_valid,
        NUM_AGENTS,
        ACTION_DIM,
    )

    assert first_batch.rewards.shape == (
        expected_valid,
    )

    assert first_batch.next_obs.shape == (
        expected_valid,
        NUM_AGENTS,
        OBS_DIM,
    )

    assert first_batch.dones.shape == (
        expected_valid,
    )

    print(
        "\nLearner batch shapes"
    )

    print(
        "obs:",
        first_batch.obs.shape,
    )

    print(
        "actions:",
        first_batch.actions.shape,
    )

    print(
        "rewards:",
        first_batch.rewards.shape,
    )

    print(
        "next_obs:",
        first_batch.next_obs.shape,
    )

    print(
        "dones:",
        first_batch.dones.shape,
    )

    # ========================================================
    # Sampling test
    # ========================================================

    rng = jax.random.PRNGKey(0)

    sampled_batch = (
        sample_model_batch(
            buffer=(
                model_buffers.agents[0]
            ),
            rng=rng,
            batch_size=4,
        )
    )

    assert sampled_batch.obs.shape == (
        4,
        NUM_AGENTS,
        OBS_DIM,
    )

    assert sampled_batch.actions.shape == (
        4,
        NUM_AGENTS,
        ACTION_DIM,
    )

    assert sampled_batch.rewards.shape == (
        4,
    )

    assert sampled_batch.next_obs.shape == (
        4,
        NUM_AGENTS,
        OBS_DIM,
    )

    assert sampled_batch.dones.shape == (
        4,
    )

    print(
        "\n================================"
    )

    print(
        "MODEL REPLAY BUFFER TEST PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()