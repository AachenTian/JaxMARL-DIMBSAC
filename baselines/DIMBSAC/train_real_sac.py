import jax
import jax.numpy as jnp
import jaxmarl

from networks.actor import (
    SACActor,
    deterministic_action,
    sample_squashed_gaussian,
)
from networks.critic import SACCritic
from observation import project_local_obs
from real_collector import (
    collect_real_step,
    init_real_collector,
    joint_action_to_dict,
    stack_rewards,
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


# ============================================================
# Debug configuration
# These values are not yet frozen as final experiment settings.
# ============================================================

NUM_ENVS = 32
NUM_EVAL_ENVS = 32

EPISODE_HORIZON = 25

BUFFER_CAPACITY = 100_000
LEARNING_STARTS = 2_000
BATCH_SIZE = 256

TOTAL_COLLECT_STEPS = 5000
UPDATES_PER_COLLECT_STEP = 4

EVAL_INTERVAL = 250
EVAL_SEED = 12345

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

GAMMA = 0.99
ALPHA = 0.2
TAU = 0.005


def sample_policy_actions(
    actor,
    agent_states,
    local_obs,
    rng,
    action_low,
    action_high,
):
    """Sample decentralized actions from all independent actors."""

    num_agents = local_obs.shape[1]

    action_keys = jax.random.split(
        rng,
        num_agents,
    )

    actions = []

    for agent_idx in range(num_agents):

        mean, log_std = actor.apply(
            {
                "params":
                    agent_states[
                        agent_idx
                    ].actor.params
            },
            local_obs[:, agent_idx],
        )

        action, _, _ = (
            sample_squashed_gaussian(
                rng=action_keys[agent_idx],
                mean=mean,
                log_std=log_std,
                action_low=action_low,
                action_high=action_high,
            )
        )

        actions.append(action)

    return jnp.stack(
        actions,
        axis=1,
    )


def evaluate_policy(
    env,
    actor,
    agent_states,
    rng,
    num_eval_envs,
    num_landmarks,
    episode_horizon,
    action_low,
    action_high,
):
    """Evaluate the current actors using deterministic actions."""

    reset_keys = jax.random.split(
        rng,
        num_eval_envs,
    )

    obs_dict, env_state = jax.vmap(
        env.reset
    )(reset_keys)

    num_agents = env.num_agents

    episode_returns = jnp.zeros(
        (
            num_eval_envs,
            num_agents,
        ),
        dtype=jnp.float32,
    )

    for _ in range(episode_horizon):

        local_obs = project_local_obs(
            obs_dict,
            env.agents,
            num_landmarks,
        )

        actions = []

        for agent_idx in range(num_agents):

            mean, _ = actor.apply(
                {
                    "params":
                        agent_states[
                            agent_idx
                        ].actor.params
                },
                local_obs[:, agent_idx],
            )

            action = deterministic_action(
                mean=mean,
                action_low=action_low,
                action_high=action_high,
            )

            actions.append(action)

        joint_action = jnp.stack(
            actions,
            axis=1,
        )

        action_dict = joint_action_to_dict(
            joint_action,
            env.agents,
        )

        rng, step_rng = jax.random.split(
            rng
        )

        step_keys = jax.random.split(
            step_rng,
            num_eval_envs,
        )

        (
            obs_dict,
            env_state,
            rewards,
            _,
            _,
        ) = jax.vmap(
            env.step_env,
            in_axes=(0, 0, 0),
        )(
            step_keys,
            env_state,
            action_dict,
        )

        episode_returns += stack_rewards(
            rewards,
            env.agents,
        )

    mean_agent_returns = jnp.mean(
        episode_returns,
        axis=0,
    )

    mean_team_return = jnp.mean(
        mean_agent_returns
    )

    return (
        mean_team_return,
        mean_agent_returns,
    )


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

    print("Pure real-data SAC training")
    print("------------------------------")
    print("num_agents:", num_agents)
    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)

    rng = jax.random.PRNGKey(0)

    # ========================================================
    # Networks
    # ========================================================

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

    agent_states = []

    for _ in range(num_agents):

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
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
        )

        agent_states.append(state)

    # ========================================================
    # Real environment and replay buffer
    # ========================================================

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

    replay_size = 0

    # JIT the update after the standalone tests have passed.
    update_fn = jax.jit(
        update_sac_agent,
        static_argnames=(
            "agent_idx",
            "actor",
            "critic",
            "gamma",
            "alpha",
            "tau",
        ),
    )

    # ========================================================
    # Initial evaluation
    # ========================================================

    eval_rng = jax.random.PRNGKey(
        EVAL_SEED
    )

    (
        initial_return,
        initial_agent_returns,
    ) = evaluate_policy(
        env=env,
        actor=actor,
        agent_states=agent_states,
        rng=eval_rng,
        num_eval_envs=NUM_EVAL_ENVS,
        num_landmarks=num_landmarks,
        episode_horizon=EPISODE_HORIZON,
        action_low=action_space.low,
        action_high=action_space.high,
    )

    print("\nInitial evaluation")
    print("------------------------------")
    print(
        "mean team return:",
        float(initial_return),
    )
    print(
        "agent returns:",
        initial_agent_returns,
    )

    # ========================================================
    # Training
    # ========================================================

    last_metrics = None

    for collect_step in range(
        1,
        TOTAL_COLLECT_STEPS + 1,
    ):

        local_obs = project_local_obs(
            collector_state.obs_dict,
            env.agents,
            num_landmarks,
        )

        rng, action_rng = jax.random.split(
            rng
        )

        # Random exploration before learning starts.
        if replay_size < LEARNING_STARTS:

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

        else:

            joint_action = sample_policy_actions(
                actor=actor,
                agent_states=agent_states,
                local_obs=local_obs,
                rng=action_rng,
                action_low=action_space.low,
                action_high=action_space.high,
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

        replay_size = min(
            replay_size + NUM_ENVS,
            BUFFER_CAPACITY,
        )

        # ====================================================
        # SAC updates using real replay only
        # ====================================================

        if replay_size >= max(
            LEARNING_STARTS,
            BATCH_SIZE,
        ):

            for _ in range(
                UPDATES_PER_COLLECT_STEP
            ):

                rng, sample_rng = (
                    jax.random.split(rng)
                )

                batch = sample_real_batch(
                    replay_buffer,
                    sample_rng,
                    BATCH_SIZE,
                )

                # All agents use the same pre-update actor snapshots.
                actor_snapshots = tuple(
                    state.actor.params
                    for state in agent_states
                )

                current_metrics = []

                for agent_idx in range(
                    num_agents
                ):

                    rng, update_rng = (
                        jax.random.split(rng)
                    )

                    (
                        new_state,
                        metrics,
                    ) = update_fn(
                        agent_idx=agent_idx,
                        agent_state=agent_states[
                            agent_idx
                        ],
                        actor=actor,
                        critic=critic,
                        actor_param_snapshots=(
                            actor_snapshots
                        ),
                        batch=batch,
                        rng=update_rng,
                        action_low=action_space.low,
                        action_high=action_space.high,
                        gamma=GAMMA,
                        alpha=ALPHA,
                        tau=TAU,
                    )

                    agent_states[
                        agent_idx
                    ] = new_state

                    current_metrics.append(
                        metrics
                    )

                last_metrics = (
                    current_metrics
                )

        # ====================================================
        # Deterministic evaluation
        # ====================================================

        if (
            collect_step % EVAL_INTERVAL == 0
            or collect_step
            == TOTAL_COLLECT_STEPS
        ):

            (
                mean_return,
                agent_returns,
            ) = evaluate_policy(
                env=env,
                actor=actor,
                agent_states=agent_states,
                rng=eval_rng,
                num_eval_envs=NUM_EVAL_ENVS,
                num_landmarks=num_landmarks,
                episode_horizon=EPISODE_HORIZON,
                action_low=action_space.low,
                action_high=action_space.high,
            )

            print(
                f"\ncollect_step = "
                f"{collect_step}"
            )

            print(
                "real transitions:",
                replay_size,
            )

            print(
                "mean team return:",
                float(mean_return),
            )

            print(
                "agent returns:",
                agent_returns,
            )

            if last_metrics is not None:

                for agent_idx in range(
                    num_agents
                ):
                    print(
                        f"agent {agent_idx} "
                        f"critic loss:",
                        float(
                            last_metrics[
                                agent_idx
                            ]["critic_loss"]
                        ),
                    )

                    print(
                        f"agent {agent_idx} "
                        f"actor loss:",
                        float(
                            last_metrics[
                                agent_idx
                            ]["actor_loss"]
                        ),
                    )

                    print(
                        f"agent {agent_idx} "
                        f"Q1 mean:",
                        float(
                            last_metrics[
                                agent_idx
                            ]["q1_mean"]
                        ),
                    )

                    print(
                        f"agent {agent_idx} "
                        f"Q2 mean:",
                        float(
                            last_metrics[
                                agent_idx
                            ]["q2_mean"]
                        ),
                    )

                    print(
                        f"agent {agent_idx} "
                        f"target Q mean:",
                        float(
                            last_metrics[
                                agent_idx
                            ]["target_q_mean"]
                        ),
                    )

                    print(
                        f"agent {agent_idx} "
                        f"log_prob mean:",
                        float(
                            last_metrics[
                                agent_idx
                            ]["log_prob_mean"]
                        ),
                    )

    print("\n==============================")
    print("REAL-DATA SAC TRAINING FINISHED")
    print("==============================")


if __name__ == "__main__":
    main()