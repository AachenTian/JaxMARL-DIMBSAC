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
from dynamics.rollout import (
    propagate_joint_x,
    sample_ensemble_members,
)
from observation import (
    local_obs_to_x,
    reconstruct_local_obs,
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


NUM_COLLECT_STEPS = 200
BUFFER_CAPACITY = 10_000
PROPAGATION_BATCH_SIZE = 256


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

    num_agents = (
        env.num_agents
    )

    num_landmarks = (
        env.num_landmarks
    )

    obs_dim = (
        4
        + 2 * num_landmarks
    )

    action_space = (
        env.action_space(
            env.agents[0]
        )
    )

    action_dim = (
        action_space.shape[0]
    )

    ensemble_size = int(
        cfg.DYNAMICS.ENSEMBLE_SIZE
    )

    print(
        "One-step model propagation test"
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
        "propagation batch size:",
        PROPAGATION_BATCH_SIZE,
    )

    # ========================================================
    # RNG
    # ========================================================

    rng = jax.random.PRNGKey(
        cfg.ENV.SEED
    )

    # ========================================================
    # Collect shared real replay
    # ========================================================

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
            num_landmarks=(
                num_landmarks
            ),
        )
    )

    print("\nCollecting real data")
    print("------------------------------")

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

    num_real_transitions = (
        real_batch.x.shape[0]
    )

    print(
        "real replay size:",
        num_real_transitions,
    )

    assert (
        num_real_transitions
        >= PROPAGATION_BATCH_SIZE
    )

    # ========================================================
    # Dynamics model architecture
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

    # ========================================================
    # Train one independent ensemble per agent
    # ========================================================

    print(
        "\nTraining independent dynamics ensembles"
    )

    print(
        "================================"
    )

    agent_ensembles = []

    for agent_idx in range(
        num_agents
    ):
        print(
            f"\nAgent {agent_idx}"
        )

        print(
            "--------------------------------"
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

    assert (
        len(
            dynamics_ensembles.agents
        )
        == num_agents
    )

    # ========================================================
    # Sample real starting transitions
    # ========================================================

    rng, sample_rng = (
        jax.random.split(rng)
    )

    sample_indices = (
        jax.random.permutation(
            sample_rng,
            num_real_transitions,
        )[
            :PROPAGATION_BATCH_SIZE
        ]
    )

    joint_x = (
        real_batch.x[
            sample_indices
        ]
    )

    joint_actions = (
        real_batch.actions[
            sample_indices
        ]
    )

    true_next_x = (
        real_batch.next_x[
            sample_indices
        ]
    )

    true_next_obs = (
        real_batch.next_obs[
            sample_indices
        ]
    )

    landmarks = (
        real_batch.landmarks[
            sample_indices
        ]
    )

    print(
        "\nStarting batch"
    )

    print(
        "--------------------------------"
    )

    print(
        "joint x shape:",
        joint_x.shape,
    )

    print(
        "joint actions shape:",
        joint_actions.shape,
    )

    print(
        "landmarks shape:",
        landmarks.shape,
    )

    assert joint_x.shape == (
        PROPAGATION_BATCH_SIZE,
        num_agents,
        4,
    )

    assert joint_actions.shape == (
        PROPAGATION_BATCH_SIZE,
        num_agents,
        action_dim,
    )

    assert landmarks.shape == (
        PROPAGATION_BATCH_SIZE,
        num_landmarks,
        2,
    )

    # ========================================================
    # Sample one member per trajectory and agent
    # ========================================================

    rng, member_rng = (
        jax.random.split(rng)
    )

    member_indices = (
        sample_ensemble_members(
            rng=member_rng,
            batch_size=(
                PROPAGATION_BATCH_SIZE
            ),
            num_agents=(
                num_agents
            ),
            ensemble_size=(
                ensemble_size
            ),
        )
    )

    print(
        "\nMember sampling"
    )

    print(
        "--------------------------------"
    )

    print(
        "member indices shape:",
        member_indices.shape,
    )

    print(
        "first 8 trajectory member selections:"
    )

    print(
        member_indices[:8]
    )

    assert member_indices.shape == (
        PROPAGATION_BATCH_SIZE,
        num_agents,
    )

    assert jnp.all(
        member_indices >= 0
    )

    assert jnp.all(
        member_indices
        < ensemble_size
    )

    # ========================================================
    # One-step model propagation
    # ========================================================

    (
        predicted_next_x,
        predicted_delta_x,
    ) = propagate_joint_x(
        dynamics_ensembles=(
            dynamics_ensembles
        ),
        joint_x=joint_x,
        joint_actions=(
            joint_actions
        ),
        member_indices=(
            member_indices
        ),
    )

    print(
        "\nModel propagation"
    )

    print(
        "--------------------------------"
    )

    print(
        "predicted delta-x shape:",
        predicted_delta_x.shape,
    )

    print(
        "predicted next-x shape:",
        predicted_next_x.shape,
    )

    assert (
        predicted_delta_x.shape
        == joint_x.shape
    )

    assert (
        predicted_next_x.shape
        == joint_x.shape
    )

    assert jnp.all(
        jnp.isfinite(
            predicted_delta_x
        )
    )

    assert jnp.all(
        jnp.isfinite(
            predicted_next_x
        )
    )

    propagation_identity_error = (
        jnp.max(
            jnp.abs(
                predicted_next_x
                - (
                    joint_x
                    + predicted_delta_x
                )
            )
        )
    )

    print(
        "propagation identity error:",
        float(
            propagation_identity_error
        ),
    )

    assert (
        propagation_identity_error
        < 1e-6
    )

    # ========================================================
    # Reconstruct local observations
    # ========================================================

    predicted_next_obs = (
        reconstruct_local_obs(
            predicted_next_x,
            landmarks,
        )
    )

    print(
        "\nObservation reconstruction"
    )

    print(
        "--------------------------------"
    )

    print(
        "predicted next-obs shape:",
        predicted_next_obs.shape,
    )

    assert (
        predicted_next_obs.shape
        == true_next_obs.shape
    )

    assert jnp.all(
        jnp.isfinite(
            predicted_next_obs
        )
    )

    # Reconstructed observation must contain exactly
    # the same physical state used to create it.
    x_from_predicted_obs = (
        local_obs_to_x(
            predicted_next_obs
        )
    )

    reconstruction_error = (
        jnp.max(
            jnp.abs(
                x_from_predicted_obs
                - predicted_next_x
            )
        )
    )

    print(
        "obs -> x reconstruction error:",
        float(
            reconstruction_error
        ),
    )

    assert (
        reconstruction_error
        < 1e-6
    )

    # ========================================================
    # Compare against real next state
    # ========================================================

    x_error = (
        predicted_next_x
        - true_next_x
    )

    x_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                x_error
            )
        )
    )

    per_dim_x_rmse = (
        jnp.sqrt(
            jnp.mean(
                jnp.square(
                    x_error
                ),
                axis=(0, 1),
            )
        )
    )

    obs_error = (
        predicted_next_obs
        - true_next_obs
    )

    obs_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                obs_error
            )
        )
    )

    print(
        "\nOne-step prediction diagnostics"
    )

    print(
        "--------------------------------"
    )

    print(
        "joint next-x RMSE:",
        float(
            x_rmse
        ),
    )

    print(
        "per-dim next-x RMSE "
        "[px, py, vx, vy]:",
        per_dim_x_rmse,
    )

    print(
        "local next-obs RMSE:",
        float(
            obs_rmse
        ),
    )

    # ========================================================
    # No-model baseline diagnostic
    # ========================================================

    no_model_error = (
        joint_x
        - true_next_x
    )

    no_model_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                no_model_error
            )
        )
    )

    print(
        "zero-delta baseline RMSE:",
        float(
            no_model_rmse
        ),
    )

    # Prediction quality is diagnostic here.
    # Generalization was already tested on held-out validation sets.
    assert jnp.isfinite(
        x_rmse
    )

    assert jnp.isfinite(
        obs_rmse
    )

    # ========================================================
    # Final result
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "ONE-STEP MODEL PROPAGATION TEST PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()