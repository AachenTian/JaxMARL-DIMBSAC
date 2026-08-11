import jax.numpy as jnp

from dynamics.model_rollout import (
    ModelRolloutTrajectory,
)
from independent_rollout import (
    IndependentModelRollouts,
)
from model_replay_buffer import (
    add_independent_rollouts_to_model_buffers,
    init_independent_model_replay_buffers,
)


NUM_AGENTS = 3
OBS_DIM = 10
ACTION_DIM = 5

HORIZON = 2
BATCH_SIZE = 4
CAPACITY = 100


def build_trajectory(
    learner_idx: int,
):
    obs = jnp.full(
        (
            HORIZON,
            BATCH_SIZE,
            NUM_AGENTS,
            OBS_DIM,
        ),
        float(
            learner_idx + 1
        ),
        dtype=jnp.float32,
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

    rewards = jnp.zeros(
        (
            HORIZON,
            BATCH_SIZE,
            NUM_AGENTS,
        ),
        dtype=jnp.float32,
    )

    rewards = rewards.at[
        :,
        :,
        learner_idx,
    ].set(
        100.0 + learner_idx
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

    valid = jnp.ones(
        (
            HORIZON,
            BATCH_SIZE,
        ),
        dtype=jnp.bool_,
    )

    dones = jnp.zeros(
        (
            HORIZON,
            BATCH_SIZE,
        ),
        dtype=jnp.bool_,
    )

    rollout_stops = jnp.zeros(
        (
            HORIZON,
            BATCH_SIZE,
        ),
        dtype=jnp.bool_,
    )

    episode_steps = jnp.zeros(
        (
            HORIZON,
            BATCH_SIZE,
        ),
        dtype=jnp.int32,
    )

    return ModelRolloutTrajectory(
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
        next_obs=obs + 0.5,
        next_x=x,
        member_indices=jnp.zeros(
            (
                BATCH_SIZE,
                NUM_AGENTS,
            ),
            dtype=jnp.int32,
        ),
    )


def main():
    rollouts = (
        IndependentModelRollouts(
            agents=tuple(
                build_trajectory(i)
                for i in range(
                    NUM_AGENTS
                )
            )
        )
    )

    buffers = (
        init_independent_model_replay_buffers(
            num_agents=NUM_AGENTS,
            capacity=CAPACITY,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
        )
    )

    buffers = (
        add_independent_rollouts_to_model_buffers(
            model_buffers=buffers,
            independent_rollouts=rollouts,
        )
    )

    expected_size = (
        HORIZON
        * BATCH_SIZE
    )

    print(
        "Independent model buffer routing test"
    )

    print(
        "================================"
    )

    for agent_idx in range(
        NUM_AGENTS
    ):
        buffer = (
            buffers.agents[
                agent_idx
            ]
        )

        print(
            f"\nAgent {agent_idx}"
        )

        print(
            "buffer size:",
            int(
                buffer.size
            ),
        )

        print(
            "mean obs:",
            float(
                jnp.mean(
                    buffer.obs[
                        :expected_size
                    ]
                )
            ),
        )

        print(
            "mean reward:",
            float(
                jnp.mean(
                    buffer.rewards[
                        :expected_size
                    ]
                )
            ),
        )

        assert (
            int(
                buffer.size
            )
            == expected_size
        )

        assert jnp.allclose(
            buffer.obs[
                :expected_size
            ],
            float(
                agent_idx + 1
            ),
        )

        assert jnp.allclose(
            buffer.rewards[
                :expected_size
            ],
            100.0 + agent_idx,
        )

    print(
        "\n================================"
    )

    print(
        "INDEPENDENT MODEL BUFFER TEST PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()