import jax
import jax.numpy as jnp
import jaxmarl

from real_collector import (
    init_real_collector,
    collect_real_step,
)
from replay_buffer import (
    init_real_replay_buffer,
    add_real_batch,
    sample_real_batch,
)


NUM_ENVS = 8
BUFFER_CAPACITY = 1024
EPISODE_HORIZON = 25
NUM_COLLECT_STEPS = 60


def main():

    env = jaxmarl.make(
        "MPE_simple_spread_v3",
        action_type="Continuous",
    )

    agents = env.agents
    num_agents = env.num_agents
    num_landmarks = env.num_landmarks

    obs_dim = 4 + 2 * num_landmarks

    action_dim = env.action_space(
        agents[0]
    ).shape[0]

    print("Environment")
    print("------------------------------")
    print("num_agents:", num_agents)
    print("num_landmarks:", num_landmarks)
    print("local_obs_dim:", obs_dim)
    print("action_dim:", action_dim)

    rng = jax.random.PRNGKey(0)

    # Initialize the parallel environment collector.
    rng, collector_rng = jax.random.split(rng)

    collector_state = init_real_collector(
        env=env,
        num_envs=NUM_ENVS,
        rng=collector_rng,
    )

    # Initialize the shared real replay buffer.
    replay_buffer = init_real_replay_buffer(
        capacity=BUFFER_CAPACITY,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_landmarks=num_landmarks,
    )

    total_terminal_transitions = 0

    for collect_step in range(NUM_COLLECT_STEPS):

        rng, action_rng = jax.random.split(rng)

        # Random actions are used only for testing the data pipeline.
        joint_action = jax.random.uniform(
            action_rng,
            shape=(
                NUM_ENVS,
                num_agents,
                action_dim,
            ),
            minval=0.0,
            maxval=1.0,
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

        num_terminal = int(
            jnp.sum(transition.dones)
        )

        total_terminal_transitions += num_terminal

        # Inspect the episode boundary.
        if collect_step in [
            0,
            1,
            23,
            24,
            25,
            48,
            49,
            50,
        ]:
            print(
                f"\ncollect_step = {collect_step}"
            )

            print(
                "stored episode_steps:",
                transition.episode_steps,
            )

            print(
                "stored dones:",
                transition.dones,
            )

            print(
                "next collector episode_steps:",
                collector_state.episode_steps,
            )

    print("\nReplay Buffer")
    print("------------------------------")

    print(
        "size:",
        int(replay_buffer.size),
    )

    print(
        "position:",
        int(replay_buffer.position),
    )

    print(
        "terminal transitions:",
        total_terminal_transitions,
    )

    expected_size = (
        NUM_ENVS
        * NUM_COLLECT_STEPS
    )

    print(
        "expected size:",
        expected_size,
    )

    # Sample a training batch.
    rng, sample_rng = jax.random.split(rng)

    sample = sample_real_batch(
        replay_buffer,
        sample_rng,
        batch_size=32,
    )

    print("\nSample shapes")
    print("------------------------------")

    print("obs:", sample.obs.shape)
    print("x:", sample.x.shape)
    print("actions:", sample.actions.shape)
    print("rewards:", sample.rewards.shape)
    print("next_obs:", sample.next_obs.shape)
    print("next_x:", sample.next_x.shape)
    print("dones:", sample.dones.shape)
    print(
        "episode_steps:",
        sample.episode_steps.shape,
    )
    print(
        "landmarks:",
        sample.landmarks.shape,
    )

    # This will later become the dynamics target.
    delta_x = (
        sample.next_x
        - sample.x
    )

    print("\nDynamics target")
    print("------------------------------")

    print(
        "delta_x shape:",
        delta_x.shape,
    )

    print(
        "max |delta_x|:",
        float(
            jnp.max(
                jnp.abs(delta_x)
            )
        ),
    )

    print(
        "mean |delta_x|:",
        float(
            jnp.mean(
                jnp.abs(delta_x)
            )
        ),
    )

    # Basic data-contract checks.
    assert int(replay_buffer.size) == expected_size

    assert sample.obs.shape == (
        32,
        num_agents,
        obs_dim,
    )

    assert sample.x.shape == (
        32,
        num_agents,
        4,
    )

    assert sample.actions.shape == (
        32,
        num_agents,
        action_dim,
    )

    assert sample.rewards.shape == (
        32,
        num_agents,
    )

    assert sample.next_obs.shape == (
        32,
        num_agents,
        obs_dim,
    )

    assert sample.next_x.shape == (
        32,
        num_agents,
        4,
    )

    assert sample.landmarks.shape == (
        32,
        num_landmarks,
        2,
    )

    assert total_terminal_transitions > 0

    print("\n==============================")
    print("ALL REAL PIPELINE TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()