import hydra
import jax
import jax.numpy as jnp
import jaxmarl
from omegaconf import DictConfig

from dynamics.ensemble import (
    build_independent_dynamics_ensembles,
)
from dynamics.model import (
    ProbabilisticDynamicsModel,
)
from dynamics.model_rollout import (
    generate_model_rollout,
)
from observation import (
    local_obs_to_x,
)
from real_collector import (
    collect_real_step,
    init_real_collector,
)
from replay_buffer import (
    add_real_batch,
    init_real_replay_buffer,
)
from test_dynamics_ensemble import (
    get_valid_replay_batch,
    train_agent_ensemble,
)
from model_action_sampler import (
    build_actor_action_sampler,
)
from networks.actor import SACActor
from networks.critic import SACCritic
from sac_agent import (
    init_sac_agent,
    update_sac_agent_with_batches,
)

from mixed_replay import (
    sample_mixed_critic_batch,
)


from model_replay_buffer import (
    add_trajectory_to_model_replay_buffers,
    init_independent_model_replay_buffers,
    trajectory_to_agent_model_batch,
)

from independent_rollout import (
    generate_independent_model_rollouts,
)

from model_replay_buffer import (
    add_independent_rollouts_to_model_buffers,
    init_independent_model_replay_buffers,
)

NUM_COLLECT_STEPS = 200
BUFFER_CAPACITY = 10_000


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="simple_spread",
)
def main(
    cfg: DictConfig,
):
    # ========================================================
    # Environment
    # ========================================================

    env = jaxmarl.make(
        cfg.ENV.NAME,
        action_type=(
            cfg.ENV.ACTION_TYPE
        ),
    )

    num_agents = env.num_agents
    num_landmarks = env.num_landmarks

    obs_dim = (
        4
        + 2 * num_landmarks
    )

    action_space = env.action_space(
        env.agents[0]
    )

    action_dim = (
        action_space.shape[0]
    )

    ensemble_size = int(
        cfg.DYNAMICS.ENSEMBLE_SIZE
    )

    rollout_batch_size = int(
        cfg.MODEL_ROLLOUT.BATCH_SIZE
    )

    rollout_horizon = int(
        cfg.MODEL_ROLLOUT.HORIZON
    )

    print(
        "Multi-step model rollout test"
    )

    print(
        "================================"
    )

    print(
        "num agents:",
        num_agents,
    )

    print(
        "ensemble size:",
        ensemble_size,
    )

    print(
        "rollout batch size:",
        rollout_batch_size,
    )

    print(
        "rollout horizon:",
        rollout_horizon,
    )

    # ========================================================
    # Collect real replay
    # ========================================================

    rng = jax.random.PRNGKey(
        cfg.ENV.SEED
    )

    # ========================================================
    # Initialize independent SAC actors
    # ========================================================

    actor = SACActor(
        action_dim=action_dim,
        hidden_dims=tuple(
            cfg.ACTOR.HIDDEN_DIMS
        ),
        log_std_min=(
            cfg.ACTOR.LOG_STD_MIN
        ),
        log_std_max=(
            cfg.ACTOR.LOG_STD_MAX
        ),
    )

    critic = SACCritic(
        hidden_dims=tuple(
            cfg.CRITIC.HIDDEN_DIMS
        ),
    )

    dummy_local_obs = jnp.zeros(
        (
            1,
            obs_dim,
        ),
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

    for _ in range(
            num_agents
    ):
        rng, init_rng = (
            jax.random.split(rng)
        )

        state = init_sac_agent(
            rng=init_rng,
            actor=actor,
            critic=critic,
            dummy_local_obs=(
                dummy_local_obs
            ),
            dummy_joint_obs=(
                dummy_joint_obs
            ),
            dummy_joint_action=(
                dummy_joint_action
            ),
            actor_lr=(
                cfg.ACTOR.LR
            ),
            critic_lr=(
                cfg.CRITIC.LR
            ),
        )

        agent_states.append(
            state
        )

    # Freeze the current actor parameters as rollout snapshots.
    actor_params = tuple(
        state.actor.params
        for state in agent_states
    )

    print(
        "\nInitialized actor snapshots:",
        len(actor_params),
    )

    assert (
            len(actor_params)
            == num_agents
    )

    rng, collector_rng = (
        jax.random.split(rng)
    )

    collector_state = (
        init_real_collector(
            env=env,
            num_envs=(
                cfg.ENV.NUM_ENVS
            ),
            rng=collector_rng,
        )
    )

    replay_buffer = (
        init_real_replay_buffer(
            capacity=BUFFER_CAPACITY,
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_landmarks=num_landmarks,
        )
    )

    print(
        "\nCollecting real data"
    )

    print(
        "--------------------------------"
    )

    for _ in range(
        NUM_COLLECT_STEPS
    ):

        rng, action_rng = (
            jax.random.split(rng)
        )

        joint_action = (
            jax.random.uniform(
                action_rng,
                shape=(
                    cfg.ENV.NUM_ENVS,
                    num_agents,
                    action_dim,
                ),
                minval=(
                    action_space.low
                ),
                maxval=(
                    action_space.high
                ),
            )
        )

        (
            collector_state,
            transition,
            _,
        ) = collect_real_step(
            env=env,
            collector_state=(
                collector_state
            ),
            joint_action=(
                joint_action
            ),
            num_landmarks=(
                num_landmarks
            ),
            episode_horizon=(
                cfg.ENV.EPISODE_HORIZON
            ),
        )

        replay_buffer = (
            add_real_batch(
                replay_buffer,
                transition,
            )
        )

    real_batch = (
        get_valid_replay_batch(
            replay_buffer
        )
    )

    print(
        "real replay size:",
        real_batch.x.shape[0],
    )

    # ========================================================
    # Train independent dynamics ensembles
    # ========================================================

    model = (
        ProbabilisticDynamicsModel(
            output_dim=4,
            hidden_dims=tuple(
                cfg.DYNAMICS.HIDDEN_DIMS
            ),
            log_var_min=(
                cfg.DYNAMICS.LOG_VAR_MIN
            ),
            log_var_max=(
                cfg.DYNAMICS.LOG_VAR_MAX
            ),
        )
    )

    agent_ensembles = []

    print(
        "\nTraining dynamics ensembles"
    )

    print(
        "================================"
    )

    for agent_idx in range(
        num_agents
    ):

        print(
            f"\nAgent {agent_idx}"
        )

        (
            rng,
            agent_ensemble,
            _,
            _,
        ) = train_agent_ensemble(
            rng=rng,
            agent_idx=agent_idx,
            real_batch=real_batch,
            model=model,
            ensemble_size=(
                ensemble_size
            ),
            dynamics_config=(
                cfg.DYNAMICS
            ),
        )

        agent_ensembles.append(
            agent_ensemble
        )

    dynamics_ensembles = (
        build_independent_dynamics_ensembles(
            agent_ensembles
        )
    )


    # ========================================================
    # Freeze actor and dynamics snapshots
    # ========================================================

    actor_snapshots = tuple(
        state.actor.params
        for state in agent_states
    )

    dynamics_snapshots = (
        dynamics_ensembles
    )

    assert (
            len(actor_snapshots)
            == num_agents
    )

    assert (
            len(dynamics_snapshots.agents)
            == num_agents
    )

    # ========================================================
    # Sample real rollout starting states
    # ========================================================

    rng, start_rng = (
        jax.random.split(rng)
    )

    # Sample from the full real replay so that some model
    # rollouts can encounter the true episode horizon.
    start_indices = (
        jax.random.permutation(
            start_rng,
            real_batch.x.shape[0],
        )[
            :rollout_batch_size
        ]
    )


    start_x = (
        real_batch.x[
            start_indices
        ]
    )

    landmarks = (
        real_batch.landmarks[
            start_indices
        ]
    )

    start_episode_steps = (
        real_batch.episode_steps[
            start_indices
        ]
    )

    assert start_x.shape == (
        rollout_batch_size,
        num_agents,
        4,
    )

    assert landmarks.shape == (
        rollout_batch_size,
        num_landmarks,
        2,
    )

    assert start_episode_steps.shape == (
        rollout_batch_size,
    )

    # ========================================================
    # Frozen actor action sampler
    # ========================================================

    actor_action_sampler = (
        build_actor_action_sampler(
            actor=actor,
            actor_params=actor_params,
            action_low=(
                action_space.low
            ),
            action_high=(
                action_space.high
            ),
        )
    )

    # ========================================================
    # Generate K-step synthetic trajectories
    # ========================================================

    rng, rollout_rng = (
        jax.random.split(rng)
    )

    trajectory = (
        generate_model_rollout(
            rng=rollout_rng,
            start_x=start_x,
            landmarks=landmarks,
            start_episode_steps=(
                start_episode_steps
            ),
            dynamics_ensembles=(
                dynamics_ensembles
            ),
            action_sampler=(
                actor_action_sampler
            ),
            env=env,
            ensemble_size=(
                ensemble_size
            ),
            horizon=(
                rollout_horizon
            ),
            episode_horizon=(
                cfg.ENV.EPISODE_HORIZON
            ),
        )
    )

    # ========================================================
    # Shape checks
    # ========================================================

    print(
        "\nTrajectory shapes"
    )

    print(
        "--------------------------------"
    )

    print(
        "obs:",
        trajectory.obs.shape,
    )

    print(
        "x:",
        trajectory.x.shape,
    )

    print(
        "actions:",
        trajectory.actions.shape,
    )

    print(
        "rewards:",
        trajectory.rewards.shape,
    )

    print(
        "episode_steps:",
        trajectory.episode_steps.shape,
    )

    print(
        "next_episode_steps:",
        trajectory.next_episode_steps.shape,
    )

    print(
        "delta_x:",
        trajectory.delta_x.shape,
    )

    print(
        "next_obs:",
        trajectory.next_obs.shape,
    )

    print(
        "next_x:",
        trajectory.next_x.shape,
    )

    print(
        "member_indices:",
        trajectory.member_indices.shape,
    )

    print(
        "dones:",
        trajectory.dones.shape,
    )

    print(
        "rollout_stops:",
        trajectory.rollout_stops.shape,
    )

    print(
        "valid:",
        trajectory.valid.shape,
    )

    assert trajectory.obs.shape == (
        rollout_horizon,
        rollout_batch_size,
        num_agents,
        obs_dim,
    )

    assert trajectory.x.shape == (
        rollout_horizon,
        rollout_batch_size,
        num_agents,
        4,
    )

    assert trajectory.actions.shape == (
        rollout_horizon,
        rollout_batch_size,
        num_agents,
        action_dim,
    )

    assert trajectory.delta_x.shape == (
        rollout_horizon,
        rollout_batch_size,
        num_agents,
        4,
    )

    assert trajectory.next_obs.shape == (
        rollout_horizon,
        rollout_batch_size,
        num_agents,
        obs_dim,
    )

    assert trajectory.next_x.shape == (
        rollout_horizon,
        rollout_batch_size,
        num_agents,
        4,
    )

    assert (
        trajectory.member_indices.shape
        == (
            rollout_batch_size,
            num_agents,
        )
    )

    assert trajectory.rewards.shape == (
        rollout_horizon,
        rollout_batch_size,
        num_agents,
    )

    assert trajectory.episode_steps.shape == (
        rollout_horizon,
        rollout_batch_size,
    )

    assert (
            trajectory.next_episode_steps.shape
            == (
                rollout_horizon,
                rollout_batch_size,
            )
    )

    assert trajectory.dones.shape == (
        rollout_horizon,
        rollout_batch_size,
    )

    assert trajectory.rollout_stops.shape == (
        rollout_horizon,
        rollout_batch_size,
    )

    assert trajectory.valid.shape == (
        rollout_horizon,
        rollout_batch_size,
    )

    # ========================================================
    # Numerical checks
    # ========================================================

    assert jnp.all(
        jnp.isfinite(
            trajectory.x
        )
    )

    assert jnp.all(
        jnp.isfinite(
            trajectory.next_x
        )
    )

    assert jnp.all(
        jnp.isfinite(
            trajectory.obs
        )
    )

    assert jnp.all(
        jnp.isfinite(
            trajectory.next_obs
        )
    )

    assert jnp.all(
        jnp.isfinite(
            trajectory.rewards
        )
    )

    action_min = jnp.min(
        trajectory.actions
    )

    action_max = jnp.max(
        trajectory.actions
    )

    print(
        "\nActor action diagnostics"
    )

    print(
        "--------------------------------"
    )

    print(
        "action min:",
        float(action_min),
    )

    print(
        "action max:",
        float(action_max),
    )

    assert jnp.all(
        trajectory.actions
        >= action_space.low - 1e-6
    )

    assert jnp.all(
        trajectory.actions
        <= action_space.high + 1e-6
    )

    print(
        "\nSynthetic reward diagnostics"
    )

    print(
        "--------------------------------"
    )

    print(
        "reward mean:",
        float(
            jnp.mean(
                trajectory.rewards
            )
        ),
    )

    print(
        "reward min:",
        float(
            jnp.min(
                trajectory.rewards
            )
        ),
    )

    print(
        "reward max:",
        float(
            jnp.max(
                trajectory.rewards
            )
        ),
    )

    # ========================================================
    # Temporal consistency
    # ========================================================

    episode_step_error = (
        jnp.max(
            jnp.abs(
                trajectory.next_episode_steps
                - (
                        trajectory.episode_steps
                        + 1
                )
            )
        )
    )

    print(
        "episode step increment error:",
        int(
            episode_step_error
        ),
    )

    assert (
            episode_step_error
            == 0
    )

    if rollout_horizon > 1:
        episode_recurrence_error = (
            jnp.max(
                jnp.abs(
                    trajectory.next_episode_steps[
                        :-1
                    ]
                    - trajectory.episode_steps[
                        1:
                    ]
                )
            )
        )

        print(
            "episode step recurrence error:",
            int(
                episode_recurrence_error
            ),
        )

        assert (
                episode_recurrence_error
                == 0
        )

    state_propagation_error = (
        jnp.max(
            jnp.abs(
                trajectory.next_x
                - (
                    trajectory.x
                    + trajectory.delta_x
                )
            )
        )
    )

    print(
        "\nTemporal consistency"
    )

    print(
        "--------------------------------"
    )

    print(
        "x + delta_x error:",
        float(
            state_propagation_error
        ),
    )

    assert (
        state_propagation_error
        < 1e-6
    )

    # next_x[t] must become x[t + 1].
    if rollout_horizon > 1:

        recurrence_error = (
            jnp.max(
                jnp.abs(
                    trajectory.next_x[:-1]
                    - trajectory.x[1:]
                )
            )
        )

        print(
            "x recurrence error:",
            float(
                recurrence_error
            ),
        )

        assert (
            recurrence_error
            < 1e-6
        )

    # next_obs[t] must become obs[t + 1].
    if rollout_horizon > 1:

        obs_recurrence_error = (
            jnp.max(
                jnp.abs(
                    trajectory.next_obs[:-1]
                    - trajectory.obs[1:]
                )
            )
        )

        print(
            "obs recurrence error:",
            float(
                obs_recurrence_error
            ),
        )

        assert (
            obs_recurrence_error
            < 1e-6
        )
    # ========================================================
    # Termination checks
    # ========================================================

    print(
        "\nTermination diagnostics"
    )

    print(
        "--------------------------------"
    )

    valid_transition_count = jnp.sum(
        trajectory.valid
    )

    env_terminal_count = jnp.sum(
        trajectory.dones
        & trajectory.valid
    )

    rollout_stop_count = jnp.sum(
        trajectory.rollout_stops
        & trajectory.valid
    )

    print(
        "valid transitions:",
        int(
            valid_transition_count
        ),
    )

    print(
        "env terminal transitions:",
        int(
            env_terminal_count
        ),
    )

    print(
        "model truncation transitions:",
        int(
            rollout_stop_count
        ),
    )

    expected_dones = (
            trajectory.next_episode_steps
            >= cfg.ENV.EPISODE_HORIZON
    )

    done_matches = (
            trajectory.dones
            == expected_dones
    )

    assert jnp.all(
        done_matches
        | (~trajectory.valid)
    )

    if rollout_horizon > 1:
        invalid_then_valid = (
                (~trajectory.valid[:-1])
                & trajectory.valid[1:]
        )

        assert not bool(
            jnp.any(
                invalid_then_valid
            )
        )

    done_and_rollout_stop = (
            trajectory.dones
            & trajectory.rollout_stops
            & trajectory.valid
    )

    assert not bool(
        jnp.any(
            done_and_rollout_stop
        )
    )

    assert (
            valid_transition_count
            <= rollout_horizon
            * rollout_batch_size
    )

    assert (
            env_terminal_count
            + rollout_stop_count
            > 0
    )

    # ========================================================
    # Observation-state consistency
    # ========================================================

    x_from_obs = local_obs_to_x(
        trajectory.obs
    )

    obs_state_error = (
        jnp.max(
            jnp.abs(
                x_from_obs
                - trajectory.x
            )
        )
    )

    next_x_from_obs = (
        local_obs_to_x(
            trajectory.next_obs
        )
    )

    next_obs_state_error = (
        jnp.max(
            jnp.abs(
                next_x_from_obs
                - trajectory.next_x
            )
        )
    )

    print(
        "obs -> x error:",
        float(
            obs_state_error
        ),
    )

    print(
        "next_obs -> next_x error:",
        float(
            next_obs_state_error
        ),
    )

    assert (
        obs_state_error
        < 1e-6
    )

    assert (
        next_obs_state_error
        < 1e-6
    )

    # ========================================================
    # Rollout magnitude diagnostic
    # ========================================================

    mean_delta_norm_per_step = (
        jnp.mean(
            jnp.linalg.norm(
                trajectory.delta_x,
                axis=-1,
            ),
            axis=(1, 2),
        )
    )

    print(
        "\nMean delta-x norm per step:"
    )

    print(
        mean_delta_norm_per_step
    )

    print(
        "\nFirst 8 fixed member selections:"
    )

    print(
        trajectory.member_indices[:8]
    )

    # ========================================================
    # Independent model replay buffers
    # ========================================================

    model_buffers = (
        init_independent_model_replay_buffers(
            num_agents=num_agents,
            capacity=(
                cfg.MODEL_REPLAY.CAPACITY
            ),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    )

    model_buffers = (
        add_trajectory_to_model_replay_buffers(
            model_buffers=model_buffers,
            trajectory=trajectory,
        )
    )

    expected_model_transitions = int(
        jnp.sum(
            trajectory.valid
        )
    )

    print(
        "\nModel replay buffers"
    )

    print(
        "--------------------------------"
    )

    print(
        "expected valid transitions:",
        expected_model_transitions,
    )

    for agent_idx in range(
            num_agents
    ):
        buffer = (
            model_buffers.agents[
                agent_idx
            ]
        )

        print(
            f"agent {agent_idx} buffer size:",
            int(buffer.size),
        )

        assert (
                int(buffer.size)
                == expected_model_transitions
        )

        expected_batch = (
            trajectory_to_agent_model_batch(
                trajectory=trajectory,
                agent_idx=agent_idx,
            )
        )

        reward_error = jnp.max(
            jnp.abs(
                buffer.rewards[
                    :expected_model_transitions
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
                < 1e-6
        )

    # ========================================================
    # Independent learner rollouts
    # ========================================================

    rng, independent_rollout_rng = (
        jax.random.split(rng)
    )

    (
        rng,
        independent_rollouts,
    ) = generate_independent_model_rollouts(
        rng=independent_rollout_rng,
        start_x=start_x,
        landmarks=landmarks,
        start_episode_steps=(
            start_episode_steps
        ),
        agent_states=agent_states,
        actor=actor,
        actor_snapshots=(
            actor_snapshots
        ),
        current_dynamics=(
            dynamics_ensembles
        ),
        dynamics_snapshots=(
            dynamics_snapshots
        ),
        env=env,
        action_low=(
            action_space.low
        ),
        action_high=(
            action_space.high
        ),
        ensemble_size=(
            ensemble_size
        ),
        horizon=(
            rollout_horizon
        ),
        episode_horizon=(
            cfg.ENV.EPISODE_HORIZON
        ),
    )

    assert (
            len(
                independent_rollouts.agents
            )
            == num_agents
    )

    print(
        "\nIndependent learner rollouts"
    )

    print(
        "--------------------------------"
    )

    for learner_idx in range(
            num_agents
    ):
        learner_trajectory = (
            independent_rollouts.agents[
                learner_idx
            ]
        )

        print(
            f"learner {learner_idx}:"
        )

        print(
            "  valid transitions:",
            int(
                jnp.sum(
                    learner_trajectory.valid
                )
            ),
        )

        print(
            "  own reward mean:",
            float(
                jnp.mean(
                    learner_trajectory.rewards[
                        :,
                        :,
                        learner_idx,
                    ]
                )
            ),
        )

        print(
            "  first member selection:",
            learner_trajectory.member_indices[
                0
            ],
        )

    # ========================================================
    # Independence checks
    # ========================================================

    member_difference_found = False

    for i in range(
            num_agents
    ):
        for j in range(
                i + 1,
                num_agents,
        ):
            same_members = jnp.array_equal(
                independent_rollouts
                .agents[i]
                .member_indices,
                independent_rollouts
                .agents[j]
                .member_indices,
            )

            if not bool(
                    same_members
            ):
                member_difference_found = True

    assert (
        member_difference_found
    )

    action_difference_found = False

    for i in range(
            num_agents
    ):
        for j in range(
                i + 1,
                num_agents,
        ):
            max_action_difference = (
                jnp.max(
                    jnp.abs(
                        independent_rollouts
                        .agents[i]
                        .actions
                        - independent_rollouts
                        .agents[j]
                        .actions
                    )
                )
            )

            print(
                f"max action difference "
                f"learner {i} vs {j}:",
                float(
                    max_action_difference
                ),
            )

            if float(
                    max_action_difference
            ) > 1e-6:
                action_difference_found = True

    assert (
        action_difference_found
    )

    # ========================================================
    # Independent model replay buffers
    # ========================================================

    independent_model_buffers = (
        init_independent_model_replay_buffers(
            num_agents=num_agents,
            capacity=(
                cfg.MODEL_REPLAY.CAPACITY
            ),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    )

    independent_model_buffers = (
        add_independent_rollouts_to_model_buffers(
            model_buffers=(
                independent_model_buffers
            ),
            independent_rollouts=(
                independent_rollouts
            ),
        )
    )

    print(
        "\nIndependent model replay buffers"
    )

    print(
        "--------------------------------"
    )

    for learner_idx in range(
            num_agents
    ):
        learner_trajectory = (
            independent_rollouts.agents[
                learner_idx
            ]
        )

        expected_size = int(
            jnp.sum(
                learner_trajectory.valid
            )
        )

        actual_size = int(
            independent_model_buffers
            .agents[learner_idx]
            .size
        )

        print(
            f"learner {learner_idx}: "
            f"expected={expected_size}, "
            f"buffer={actual_size}"
        )

        assert (
                actual_size
                == expected_size
        )

    for learner_idx in range(
            num_agents
    ):
        learner_trajectory = (
            independent_rollouts.agents[
                learner_idx
            ]
        )

        valid_indices = jnp.where(
            learner_trajectory.valid.reshape(
                -1
            )
        )[0]

        flat_obs = (
            learner_trajectory.obs.reshape(
                -1,
                num_agents,
                obs_dim,
            )
        )

        expected_first_obs = (
            flat_obs[
                valid_indices[0]
            ]
        )

        buffer_first_obs = (
            independent_model_buffers
            .agents[learner_idx]
            .obs[0]
        )

        routing_error = jnp.max(
            jnp.abs(
                expected_first_obs
                - buffer_first_obs
            )
        )

        print(
            f"learner {learner_idx} "
            f"routing error:",
            float(
                routing_error
            ),
        )

        assert (
                routing_error
                < 1e-6
        )

    # ========================================================
    # Mixed SAC update integration
    # ========================================================

    # Test-only ratio. This is not the final experiment setting.
    test_model_ratio = 0.5

    mixed_update_fn = jax.jit(
        update_sac_agent_with_batches,
        static_argnames=(
            "agent_idx",
            "actor",
            "critic",
            "gamma",
            "alpha",
            "tau",
        ),
    )

    # All learners use the same pre-update actor snapshots.
    update_actor_snapshots = tuple(
        state.actor.params
        for state in agent_states
    )

    updated_agent_states = list(
        agent_states
    )

    print(
        "\nMixed SAC update integration"
    )

    print(
        "--------------------------------"
    )

    for agent_idx in range(
            num_agents
    ):
        rng, batch_rng, update_rng = (
            jax.random.split(
                rng,
                3,
            )
        )

        (
            mixed_batch,
            batch_info,
        ) = sample_mixed_critic_batch(
            real_buffer=replay_buffer,
            model_buffer=(
                independent_model_buffers
                .agents[agent_idx]
            ),
            agent_idx=agent_idx,
            rng=batch_rng,
            batch_size=(
                cfg.REPLAY.BATCH_SIZE
            ),
            model_ratio=(
                test_model_ratio
            ),
        )

        print(
            f"\nAgent {agent_idx}"
        )

        print(
            "real samples:",
            batch_info.num_real,
        )

        print(
            "model samples:",
            batch_info.num_model,
        )

        print(
            "mixed obs shape:",
            mixed_batch.obs.shape,
        )

        print(
            "mixed rewards shape:",
            mixed_batch.rewards.shape,
        )

        assert mixed_batch.obs.shape == (
            cfg.REPLAY.BATCH_SIZE,
            num_agents,
            obs_dim,
        )

        assert mixed_batch.actions.shape == (
            cfg.REPLAY.BATCH_SIZE,
            num_agents,
            action_dim,
        )

        assert mixed_batch.rewards.shape == (
            cfg.REPLAY.BATCH_SIZE,
        )

        assert mixed_batch.next_obs.shape == (
            cfg.REPLAY.BATCH_SIZE,
            num_agents,
            obs_dim,
        )

        assert mixed_batch.dones.shape == (
            cfg.REPLAY.BATCH_SIZE,
        )

        assert (
                batch_info.num_real
                + batch_info.num_model
                == cfg.REPLAY.BATCH_SIZE
        )

        # Scheme B:
        # both critic and actor use the same mixed batch.
        (
            new_state,
            metrics,
        ) = mixed_update_fn(
            agent_idx=agent_idx,
            agent_state=(
                updated_agent_states[
                    agent_idx
                ]
            ),
            actor=actor,
            critic=critic,
            actor_param_snapshots=(
                update_actor_snapshots
            ),
            critic_batch=mixed_batch,
            actor_batch=mixed_batch,
            rng=update_rng,
            action_low=(
                action_space.low
            ),
            action_high=(
                action_space.high
            ),
            gamma=(
                cfg.SAC.GAMMA
            ),
            alpha=(
                cfg.SAC.ALPHA
            ),
            tau=(
                cfg.SAC.TAU
            ),
        )

        updated_agent_states[
            agent_idx
        ] = new_state

        print(
            "metrics:"
        )

        for name, value in (
                metrics.items()
        ):
            print(
                f"  {name}:",
                float(
                    jnp.mean(value)
                ),
            )

            assert jnp.all(
                jnp.isfinite(value)
            )

    # ========================================================
    # Updated parameter checks
    # ========================================================

    for agent_idx in range(
            num_agents
    ):
        state = (
            updated_agent_states[
                agent_idx
            ]
        )

        actor_leaves = (
            jax.tree_util.tree_leaves(
                state.actor.params
            )
        )

        critic1_leaves = (
            jax.tree_util.tree_leaves(
                state.critic1.params
            )
        )

        critic2_leaves = (
            jax.tree_util.tree_leaves(
                state.critic2.params
            )
        )

        assert all(
            bool(
                jnp.all(
                    jnp.isfinite(x)
                )
            )
            for x in actor_leaves
        )

        assert all(
            bool(
                jnp.all(
                    jnp.isfinite(x)
                )
            )
            for x in critic1_leaves
        )

        assert all(
            bool(
                jnp.all(
                    jnp.isfinite(x)
                )
            )
            for x in critic2_leaves
        )

    print(
        "\nAll mixed SAC updates are finite."
    )

    # ========================================================
    # Final result
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "MULTI-STEP MODEL ROLLOUT TEST PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()