import jax
import jax.numpy as jnp

from networks.critic import SACCritic


BATCH_SIZE = 32

NUM_AGENTS = 3
OBS_DIM = 10
ACTION_DIM = 5


def main():

    print("Critic test")
    print("------------------------------")

    print("num_agents:", NUM_AGENTS)
    print("obs_dim:", OBS_DIM)
    print("action_dim:", ACTION_DIM)

    # Q_i receives joint observation and joint action.
    joint_obs = jnp.zeros(
        (
            BATCH_SIZE,
            NUM_AGENTS,
            OBS_DIM,
        ),
        dtype=jnp.float32,
    )

    joint_action = jnp.zeros(
        (
            BATCH_SIZE,
            NUM_AGENTS,
            ACTION_DIM,
        ),
        dtype=jnp.float32,
    )

    critic_1 = SACCritic(
        hidden_dims=(256, 256),
    )

    critic_2 = SACCritic(
        hidden_dims=(256, 256),
    )

    rng = jax.random.PRNGKey(0)

    rng, q1_rng, q2_rng = jax.random.split(
        rng,
        3,
    )

    # Q1 and Q2 must have independent parameters.
    q1_params = critic_1.init(
        q1_rng,
        joint_obs,
        joint_action,
    )

    q2_params = critic_2.init(
        q2_rng,
        joint_obs,
        joint_action,
    )

    q1 = critic_1.apply(
        q1_params,
        joint_obs,
        joint_action,
    )

    q2 = critic_2.apply(
        q2_params,
        joint_obs,
        joint_action,
    )

    print("\nCritic output")
    print("------------------------------")

    print("Q1 shape:", q1.shape)
    print("Q2 shape:", q2.shape)

    print(
        "Q1 mean:",
        float(jnp.mean(q1)),
    )

    print(
        "Q2 mean:",
        float(jnp.mean(q2)),
    )

    # Joint critic input dimension:
    #
    # 3 * 10 + 3 * 5 = 45
    expected_input_dim = (
        NUM_AGENTS * OBS_DIM
        + NUM_AGENTS * ACTION_DIM
    )

    print(
        "\nflattened critic input dim:",
        expected_input_dim,
    )

    assert expected_input_dim == 45

    assert q1.shape == (
        BATCH_SIZE,
    )

    assert q2.shape == (
        BATCH_SIZE,
    )

    assert jnp.all(
        jnp.isfinite(q1)
    )

    assert jnp.all(
        jnp.isfinite(q2)
    )

    # Also test a single transition without a batch dimension.
    single_obs = joint_obs[0]
    single_action = joint_action[0]

    single_q = critic_1.apply(
        q1_params,
        single_obs,
        single_action,
    )

    print(
        "single Q shape:",
        single_q.shape,
    )

    assert single_q.shape == ()

    print("\n==============================")
    print("ALL CRITIC TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()