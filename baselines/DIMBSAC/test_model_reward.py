import hydra
import jax
import jax.numpy as jnp
import jaxmarl
from omegaconf import DictConfig

from dynamics.reward import (
    compute_model_rewards,
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
)


NUM_COLLECT_STEPS = 50
BUFFER_CAPACITY = 10_000
TEST_BATCH_SIZE = 256


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="simple_spread",
)
def main(
    cfg: DictConfig,
):
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

    rng = jax.random.PRNGKey(
        cfg.ENV.SEED
    )

    # ========================================================
    # Collect real transitions
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
            num_landmarks=num_landmarks,
        )
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
            joint_action=joint_action,
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

    # ========================================================
    # Select real transitions
    # ========================================================

    batch_size = min(
        TEST_BATCH_SIZE,
        real_batch.x.shape[0],
    )

    true_next_x = (
        real_batch.next_x[
            :batch_size
        ]
    )

    landmarks = (
        real_batch.landmarks[
            :batch_size
        ]
    )

    true_rewards = (
        real_batch.rewards[
            :batch_size
        ]
    )

    episode_steps = (
        real_batch.episode_steps[
            :batch_size
        ]
    )

    # Stored episode_steps refers to the transition start.
    next_episode_steps = (
        episode_steps + 1
    )

    # ========================================================
    # Reconstruct reward from true next physical state
    # ========================================================

    reconstructed_rewards = (
        compute_model_rewards(
            env=env,
            predicted_next_x=(
                true_next_x
            ),
            landmarks=landmarks,
            next_episode_steps=(
                next_episode_steps
            ),
            episode_horizon=(
                cfg.ENV.EPISODE_HORIZON
            ),
        )
    )

    print(
        "Reward reconstruction test"
    )

    print(
        "================================"
    )

    print(
        "true rewards shape:",
        true_rewards.shape,
    )

    print(
        "reconstructed rewards shape:",
        reconstructed_rewards.shape,
    )

    assert (
        reconstructed_rewards.shape
        == true_rewards.shape
    )

    reward_error = (
        reconstructed_rewards
        - true_rewards
    )

    max_reward_error = (
        jnp.max(
            jnp.abs(
                reward_error
            )
        )
    )

    mean_reward_error = (
        jnp.mean(
            jnp.abs(
                reward_error
            )
        )
    )

    print(
        "max reward error:",
        float(
            max_reward_error
        ),
    )

    print(
        "mean reward error:",
        float(
            mean_reward_error
        ),
    )

    print(
        "\nFirst 5 real rewards:"
    )

    print(
        true_rewards[:5]
    )

    print(
        "\nFirst 5 reconstructed rewards:"
    )

    print(
        reconstructed_rewards[:5]
    )

    assert jnp.all(
        jnp.isfinite(
            reconstructed_rewards
        )
    )

    assert (
        max_reward_error
        < 1e-5
    )

    print(
        "\n================================"
    )

    print(
        "MODEL REWARD RECONSTRUCTION TEST PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()