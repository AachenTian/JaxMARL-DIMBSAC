import jax
import jax.numpy as jnp
import jaxmarl

from networks.actor import SACActor
from networks.critic import SACCritic
from real_collector import (
    collect_real_step,
    init_real_collector,
)
from replay_buffer import (
    add_real_batch,
    init_real_replay_buffer,
    sample_real_batch,
)
from sac_agent import (
    init_sac_agent,
    update_sac_agent,
)


NUM_ENVS = 8
NUM_COLLECT_STEPS = 10
BATCH_SIZE = 32

BUFFER_CAPACITY = 1024
EPISODE_HORIZON = 25


def tree_difference_norm(
    params_before,
    params_after,
):
    """L2 norm of parameter changes."""

    leaves_before = jax.tree_util.tree_leaves(
        params_before
    )

    leaves_after = jax.tree_util.tree_leaves(
        params_after
    )

    squared_difference = 0.0

    for before, after in zip(
        leaves_before,
        leaves_after,
    ):
        squared_difference += jnp.sum(
            jnp.square(after - before)
        )

    return jnp.sqrt(
        squared_difference
    )


def main():

    env = jaxmarl.make(
        "MPE_simple_spread_v3",
        action_type="Continuous",
    )

    agents = env.agents

    num_agents = env.num_agents
    num_landmarks = env.num_landmarks

    obs_dim = 4 + 2 * num_landmarks

    action_space = env.action_space(
        agents[0]
    )

    action_dim = action_space.shape[0]

    print("SAC integration test")
    print("------------------------------")
    print("num_agents:", num_agents)
    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)

    rng = jax.random.PRNGKey(0)

    # ---------------------------------------------------------
    # Collect real transitions
    # ---------------------------------------------------------

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

    print(
        "real replay size:",
        int(replay_buffer.size),
    )

    # ---------------------------------------------------------
    # Sample real training data
    # ---------------------------------------------------------

    rng, sample_rng = jax.random.split(
        rng
    )

    batch = sample_real_batch(
        replay_buffer,
        sample_rng,
        batch_size=BATCH_SIZE,
    )

    # ---------------------------------------------------------
    # Networks
    # ---------------------------------------------------------

    actor = SACActor(
        action_dim=action_dim,
        hidden_dims=(256, 256),
        log_std_min=-5.0,
        log_std_max=2.0,
    )

    critic = SACCritic(
        hidden_dims=(256, 256),
    )

    dummy_local_obs = jnp.zeros(
        (1, obs_dim),
        dtype=jnp.float32,
    )

    dummy_joint_obs = jnp.zeros(
        (
            1,
            num_agents,
            obs_dim,
        ),
        dtype=jnp.float32,
    )

    dummy_joint_action = jnp.zeros(
        (
            1,
            num_agents,
            action_dim,
        ),
        dtype=jnp.float32,
    )

    # ---------------------------------------------------------
    # Initialize independent SAC agents
    # ---------------------------------------------------------

    agent_states = []

    for agent_idx in range(num_agents):

        rng, init_rng = jax.random.split(
            rng
        )

        state = init_sac_agent(
            rng=init_rng,
            actor=actor,
            critic=critic,
            dummy_local_obs=dummy_local_obs,
            dummy_joint_obs=dummy_joint_obs,
            dummy_joint_action=dummy_joint_action,
            actor_lr=3e-4,
            critic_lr=3e-4,
        )

        agent_states.append(state)

    # Freeze one actor snapshot for this update cycle.
    actor_param_snapshots = tuple(
        state.actor.params
        for state in agent_states
    )

    print(
        "number of independent agents:",
        len(agent_states),
    )

    # ---------------------------------------------------------
    # One SAC update for every agent
    # ---------------------------------------------------------

    for agent_idx in range(num_agents):

        old_actor_params = (
            agent_states[agent_idx].actor.params
        )

        old_q1_params = (
            agent_states[agent_idx].critic1.params
        )

        rng, update_rng = jax.random.split(
            rng
        )

        new_state, metrics = (
            update_sac_agent(
                agent_idx=agent_idx,
                agent_state=agent_states[
                    agent_idx
                ],
                actor=actor,
                critic=critic,
                actor_param_snapshots=(
                    actor_param_snapshots
                ),
                batch=batch,
                rng=update_rng,
                action_low=action_space.low,
                action_high=action_space.high,
                gamma=0.99,
                alpha=0.2,
                tau=0.005,
            )
        )

        actor_change = tree_difference_norm(
            old_actor_params,
            new_state.actor.params,
        )

        q1_change = tree_difference_norm(
            old_q1_params,
            new_state.critic1.params,
        )

        agent_states[agent_idx] = new_state

        print(
            f"\nAgent {agent_idx}"
        )
        print("------------------------------")

        print(
            "critic loss:",
            float(metrics["critic_loss"]),
        )

        print(
            "actor loss:",
            float(metrics["actor_loss"]),
        )

        print(
            "mean target Q:",
            float(metrics["target_q_mean"]),
        )

        print(
            "mean log_prob:",
            float(metrics["log_prob_mean"]),
        )

        print(
            "actor parameter change:",
            float(actor_change),
        )

        print(
            "Q1 parameter change:",
            float(q1_change),
        )

        assert jnp.isfinite(
            metrics["critic_loss"]
        )

        assert jnp.isfinite(
            metrics["actor_loss"]
        )

        assert actor_change > 0.0
        assert q1_change > 0.0

    print("\n==============================")
    print("ALL SAC AGENT TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()