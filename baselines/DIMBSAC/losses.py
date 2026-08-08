import jax
import jax.numpy as jnp


def compute_critic_target(
    rewards,
    dones,
    target_q1,
    target_q2,
    next_log_prob,
    gamma: float,
    alpha: float,
):
    """
    Compute the SAC Bellman target for one agent.

    Only the entropy of the current agent is included:
        y_i = r_i + gamma * (1 - done) *
              [min(Q1_target, Q2_target) - alpha * log pi_i]
    """

    min_target_q = jnp.minimum(
        target_q1,
        target_q2,
    )

    soft_value = (
        min_target_q
        - alpha * next_log_prob
    )

    dones = dones.astype(
        rewards.dtype
    )

    target = (
        rewards
        + gamma
        * (1.0 - dones)
        * soft_value
    )

    return jax.lax.stop_gradient(target)


def critic_loss(
    q1,
    q2,
    target,
):
    """Twin-critic mean squared Bellman loss."""

    q1_loss = jnp.mean(
        jnp.square(q1 - target)
    )

    q2_loss = jnp.mean(
        jnp.square(q2 - target)
    )

    total_loss = q1_loss + q2_loss

    return total_loss, {
        "critic_loss": total_loss,
        "q1_loss": q1_loss,
        "q2_loss": q2_loss,
        "q1_mean": jnp.mean(q1),
        "q2_mean": jnp.mean(q2),
        "target_q_mean": jnp.mean(target),
    }


def actor_loss(
    q1,
    q2,
    log_prob,
    alpha: float,
):
    """
    SAC actor objective for one agent.

        J_pi = E[alpha * log pi_i(a_i|o_i) - min(Q1_i, Q2_i)]
    """

    min_q = jnp.minimum(
        q1,
        q2,
    )

    loss = jnp.mean(
        alpha * log_prob
        - min_q
    )

    return loss, {
        "actor_loss": loss,
        "log_prob_mean": jnp.mean(log_prob),
        "actor_q_mean": jnp.mean(min_q),
    }