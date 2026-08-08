import jax
import jax.numpy as jnp
import jaxmarl

from networks.actor import (
    SACActor,
    sample_squashed_gaussian,
    deterministic_action,
)

BATCH_SIZE = 32


def main():

    env = jaxmarl.make(
        "MPE_simple_spread_v3",
        action_type="Continuous",
    )

    agent = env.agents[0]

    obs_dim = 10

    action_space = env.action_space(
        agent
    )

    action_dim = action_space.shape[0]

    print("Actor test")
    print("------------------------------")
    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)
    print("action_low:", action_space.low)
    print("action_high:", action_space.high)

    actor = SACActor(
        action_dim=action_dim,
        hidden_dims=(256, 256),
        log_std_min=-5.0,
        log_std_max=2.0,
    )

    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(
        rng
    )

    dummy_obs = jnp.zeros(
        (BATCH_SIZE, obs_dim),
        dtype=jnp.float32,
    )

    params = actor.init(
        init_rng,
        dummy_obs,
    )

    mean, log_std = actor.apply(
        params,
        dummy_obs,
    )

    print("\nNetwork output")
    print("------------------------------")
    print("mean:", mean.shape)
    print("log_std:", log_std.shape)

    rng, action_rng = jax.random.split(
        rng
    )

    action, log_prob, _ = (
        sample_squashed_gaussian(
            rng=action_rng,
            mean=mean,
            log_std=log_std,
            action_low=action_space.low,
            action_high=action_space.high,
        )
    )

    print("\nStochastic action")
    print("------------------------------")
    print("action:", action.shape)
    print("log_prob:", log_prob.shape)
    print(
        "min action:",
        float(jnp.min(action)),
    )
    print(
        "max action:",
        float(jnp.max(action)),
    )

    eval_action = deterministic_action(
        mean=mean,
        action_low=action_space.low,
        action_high=action_space.high,
    )

    print("\nDeterministic action")
    print("------------------------------")
    print(
        "action:",
        eval_action.shape,
    )
    print(
        "min:",
        float(jnp.min(eval_action)),
    )
    print(
        "max:",
        float(jnp.max(eval_action)),
    )

    assert mean.shape == (
        BATCH_SIZE,
        action_dim,
    )

    assert log_std.shape == (
        BATCH_SIZE,
        action_dim,
    )

    assert action.shape == (
        BATCH_SIZE,
        action_dim,
    )

    assert log_prob.shape == (
        BATCH_SIZE,
    )

    assert jnp.all(
        action >= action_space.low
    )

    assert jnp.all(
        action <= action_space.high
    )

    assert jnp.all(
        jnp.isfinite(log_prob)
    )

    print("\n==============================")
    print("ALL ACTOR TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()