import jax.numpy as jnp

from losses import (
    compute_critic_target,
    critic_loss,
    actor_loss,
)


def test_critic_target():

    rewards = jnp.array(
        [1.0, 2.0],
        dtype=jnp.float32,
    )

    dones = jnp.array(
        [False, True],
    )

    target_q1 = jnp.array(
        [10.0, 20.0],
        dtype=jnp.float32,
    )

    target_q2 = jnp.array(
        [8.0, 30.0],
        dtype=jnp.float32,
    )

    next_log_prob = jnp.array(
        [-0.5, -1.0],
        dtype=jnp.float32,
    )

    gamma = 0.99
    alpha = 0.2

    target = compute_critic_target(
        rewards=rewards,
        dones=dones,
        target_q1=target_q1,
        target_q2=target_q2,
        next_log_prob=next_log_prob,
        gamma=gamma,
        alpha=alpha,
    )

    # First transition:
    #
    # min Q = 8
    # soft value = 8 - 0.2 * (-0.5) = 8.1
    #
    # target = 1 + 0.99 * 8.1 = 9.019
    #
    # Second transition is terminal, so there is no bootstrap.
    expected_target = jnp.array(
        [9.019, 2.0],
        dtype=jnp.float32,
    )

    print("Critic target")
    print("------------------------------")
    print("target:", target)
    print("expected:", expected_target)

    assert jnp.allclose(
        target,
        expected_target,
        atol=1e-6,
    )


def test_critic_loss():

    target = jnp.array(
        [9.019, 2.0],
        dtype=jnp.float32,
    )

    q1 = jnp.array(
        [8.5, 2.5],
        dtype=jnp.float32,
    )

    q2 = jnp.array(
        [9.5, 1.5],
        dtype=jnp.float32,
    )

    loss, metrics = critic_loss(
        q1=q1,
        q2=q2,
        target=target,
    )

    expected_q1_loss = jnp.mean(
        (q1 - target) ** 2
    )

    expected_q2_loss = jnp.mean(
        (q2 - target) ** 2
    )

    expected_loss = (
        expected_q1_loss
        + expected_q2_loss
    )

    print("\nCritic loss")
    print("------------------------------")
    print("Q1 loss:", metrics["q1_loss"])
    print("Q2 loss:", metrics["q2_loss"])
    print("total loss:", loss)

    assert jnp.allclose(
        loss,
        expected_loss,
        atol=1e-6,
    )


def test_actor_loss():

    q1 = jnp.array(
        [3.0, 5.0],
        dtype=jnp.float32,
    )

    q2 = jnp.array(
        [4.0, 4.0],
        dtype=jnp.float32,
    )

    log_prob = jnp.array(
        [-0.2, -0.4],
        dtype=jnp.float32,
    )

    alpha = 0.2

    loss, metrics = actor_loss(
        q1=q1,
        q2=q2,
        log_prob=log_prob,
        alpha=alpha,
    )

    expected = jnp.mean(
        alpha * log_prob
        - jnp.minimum(q1, q2)
    )

    print("\nActor loss")
    print("------------------------------")
    print("actor loss:", loss)
    print(
        "mean log_prob:",
        metrics["log_prob_mean"],
    )
    print(
        "mean Q:",
        metrics["actor_q_mean"],
    )

    assert jnp.allclose(
        loss,
        expected,
        atol=1e-6,
    )


def main():

    test_critic_target()
    test_critic_loss()
    test_actor_loss()

    print("\n==============================")
    print("ALL SAC LOSS TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()