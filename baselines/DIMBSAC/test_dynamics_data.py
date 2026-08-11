import jax
import jax.numpy as jnp
import jaxmarl

from dynamics.normalization import (
    build_agent_dynamics_data,
    compute_normalization_stats,
    denormalize,
    denormalize_variance,
    normalize,
)
from real_collector import (
    collect_real_step,
    init_real_collector,
)
from replay_buffer import (
    add_real_batch,
    init_real_replay_buffer,
    sample_real_batch,
)


NUM_ENVS = 8
NUM_COLLECT_STEPS = 20
BATCH_SIZE = 64

BUFFER_CAPACITY = 1024
EPISODE_HORIZON = 25


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

    # Collect real transitions.
    for _ in range(NUM_COLLECT_STEPS):

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

    rng, sample_rng = jax.random.split(
        rng
    )

    batch = sample_real_batch(
        replay_buffer,
        sample_rng,
        batch_size=BATCH_SIZE,
    )

    print("Dynamics data test")
    print("------------------------------")

    for agent_idx in range(num_agents):

        (
            dynamics_input,
            delta_x,
        ) = build_agent_dynamics_data(
            batch,
            agent_idx,
        )

        print(
            f"\nAgent {agent_idx}"
        )

        print(
            "input shape:",
            dynamics_input.shape,
        )

        print(
            "target shape:",
            delta_x.shape,
        )

        assert dynamics_input.shape == (
            BATCH_SIZE,
            4 + action_dim,
        )

        assert delta_x.shape == (
            BATCH_SIZE,
            4,
        )

        input_stats = (
            compute_normalization_stats(
                dynamics_input
            )
        )

        target_stats = (
            compute_normalization_stats(
                delta_x
            )
        )

        normalized_input = normalize(
            dynamics_input,
            input_stats,
        )

        normalized_target = normalize(
            delta_x,
            target_stats,
        )

        reconstructed_target = denormalize(
            normalized_target,
            target_stats,
        )

        reconstruction_error = jnp.max(
            jnp.abs(
                reconstructed_target
                - delta_x
            )
        )

        print(
            "normalized input mean:",
            jnp.mean(
                normalized_input,
                axis=0,
            ),
        )

        print(
            "normalized target mean:",
            jnp.mean(
                normalized_target,
                axis=0,
            ),
        )

        print(
            "target reconstruction error:",
            float(reconstruction_error),
        )

        assert reconstruction_error < 1e-5

        # Test variance conversion.
        normalized_variance = jnp.ones(
            (BATCH_SIZE, 4),
            dtype=jnp.float32,
        )

        physical_variance = (
            denormalize_variance(
                normalized_variance,
                target_stats,
            )
        )

        expected_variance = jnp.square(
            target_stats.std
        )

        assert jnp.allclose(
            physical_variance[0],
            expected_variance,
        )

    print("\n==============================")
    print("ALL DYNAMICS DATA TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()