import jax
import jax.numpy as jnp

from mixed_replay import (
    compute_mixed_batch_sizes,
    sample_mixed_critic_batch,
)
from model_replay_buffer import (
    ModelTransitionBatch,
    add_model_batch,
    init_model_replay_buffer,
)
from replay_buffer import (
    init_real_replay_buffer,
)


NUM_AGENTS = 3
NUM_LANDMARKS = 3
OBS_DIM = 10
ACTION_DIM = 5

REAL_CAPACITY = 64
MODEL_CAPACITY = 64

REAL_SIZE = 32
MODEL_SIZE = 32

TEST_BATCH_SIZE = 20


def build_test_real_buffer():
    """Build real replay with easily identifiable values."""

    buffer = init_real_replay_buffer(
        capacity=REAL_CAPACITY,
        num_agents=NUM_AGENTS,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        num_landmarks=NUM_LANDMARKS,
    )

    # Real transitions are identified by obs == 1.
    obs = jnp.ones(
        (
            REAL_SIZE,
            NUM_AGENTS,
            OBS_DIM,
        ),
        dtype=jnp.float32,
    )

    actions = jnp.ones(
        (
            REAL_SIZE,
            NUM_AGENTS,
            ACTION_DIM,
        ),
        dtype=jnp.float32,
    )

    next_obs = (
        obs + 0.5
    )

    # Each learner has a distinct real reward.
    rewards = jnp.tile(
        jnp.array(
            [
                10.0,
                20.0,
                30.0,
            ],
            dtype=jnp.float32,
        ),
        (
            REAL_SIZE,
            1,
        ),
    )

    dones = jnp.zeros(
        (REAL_SIZE,),
        dtype=jnp.bool_,
    )

    buffer = buffer._replace(
        obs=buffer.obs.at[
            :REAL_SIZE
        ].set(
            obs
        ),
        actions=buffer.actions.at[
            :REAL_SIZE
        ].set(
            actions
        ),
        rewards=buffer.rewards.at[
            :REAL_SIZE
        ].set(
            rewards
        ),
        next_obs=buffer.next_obs.at[
            :REAL_SIZE
        ].set(
            next_obs
        ),
        dones=buffer.dones.at[
            :REAL_SIZE
        ].set(
            dones
        ),
        size=jnp.array(
            REAL_SIZE,
            dtype=jnp.int32,
        ),
        position=jnp.array(
            REAL_SIZE,
            dtype=jnp.int32,
        ),
    )

    return buffer


def build_test_model_buffer(
    agent_idx: int,
):
    """Build learner-specific model replay."""

    buffer = init_model_replay_buffer(
        capacity=MODEL_CAPACITY,
        num_agents=NUM_AGENTS,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
    )

    # Model transitions are identified by obs == 2.
    obs = jnp.full(
        (
            MODEL_SIZE,
            NUM_AGENTS,
            OBS_DIM,
        ),
        2.0,
        dtype=jnp.float32,
    )

    actions = jnp.full(
        (
            MODEL_SIZE,
            NUM_AGENTS,
            ACTION_DIM,
        ),
        2.0,
        dtype=jnp.float32,
    )

    next_obs = (
        obs + 0.5
    )

    model_reward = (
        100.0
        + agent_idx
    )

    rewards = jnp.full(
        (MODEL_SIZE,),
        model_reward,
        dtype=jnp.float32,
    )

    dones = jnp.zeros(
        (MODEL_SIZE,),
        dtype=jnp.bool_,
    )

    batch = ModelTransitionBatch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
    )

    return add_model_batch(
        buffer=buffer,
        batch=batch,
    )


def test_ratio(
    real_buffer,
    model_buffer,
    agent_idx: int,
    model_ratio: float,
    rng,
):
    (
        mixed_batch,
        info,
    ) = sample_mixed_critic_batch(
        real_buffer=real_buffer,
        model_buffer=model_buffer,
        agent_idx=agent_idx,
        rng=rng,
        batch_size=TEST_BATCH_SIZE,
        model_ratio=model_ratio,
    )

    print(
        f"\nmodel_ratio = {model_ratio}"
    )

    print(
        "--------------------------------"
    )

    print(
        "num real:",
        info.num_real,
    )

    print(
        "num model:",
        info.num_model,
    )

    assert (
        info.num_real
        + info.num_model
        == TEST_BATCH_SIZE
    )

    assert mixed_batch.obs.shape == (
        TEST_BATCH_SIZE,
        NUM_AGENTS,
        OBS_DIM,
    )

    assert mixed_batch.actions.shape == (
        TEST_BATCH_SIZE,
        NUM_AGENTS,
        ACTION_DIM,
    )

    assert mixed_batch.rewards.shape == (
        TEST_BATCH_SIZE,
    )

    assert mixed_batch.next_obs.shape == (
        TEST_BATCH_SIZE,
        NUM_AGENTS,
        OBS_DIM,
    )

    assert mixed_batch.dones.shape == (
        TEST_BATCH_SIZE,
    )

    # Real rows have obs == 1.
    real_rows = jnp.isclose(
        mixed_batch.obs[
            :,
            0,
            0,
        ],
        1.0,
    )

    # Model rows have obs == 2.
    model_rows = jnp.isclose(
        mixed_batch.obs[
            :,
            0,
            0,
        ],
        2.0,
    )

    actual_real = int(
        jnp.sum(
            real_rows
        )
    )

    actual_model = int(
        jnp.sum(
            model_rows
        )
    )

    print(
        "detected real rows:",
        actual_real,
    )

    print(
        "detected model rows:",
        actual_model,
    )

    assert (
        actual_real
        == info.num_real
    )

    assert (
        actual_model
        == info.num_model
    )

    # No row may come from an unknown source.
    assert jnp.all(
        real_rows
        | model_rows
    )

    expected_real_reward = (
        10.0
        * (
            agent_idx
            + 1
        )
    )

    expected_model_reward = (
        100.0
        + agent_idx
    )

    if info.num_real > 0:
        assert jnp.allclose(
            mixed_batch.rewards[
                real_rows
            ],
            expected_real_reward,
        )

    if info.num_model > 0:
        assert jnp.allclose(
            mixed_batch.rewards[
                model_rows
            ],
            expected_model_reward,
        )

    print(
        "learner-specific reward check: passed"
    )


def main():
    real_buffer = (
        build_test_real_buffer()
    )

    agent_idx = 1

    model_buffer = (
        build_test_model_buffer(
            agent_idx=agent_idx
        )
    )

    print(
        "Mixed replay test"
    )

    print(
        "================================"
    )

    # --------------------------------------------------------
    # Batch-size calculation
    # --------------------------------------------------------

    assert (
        compute_mixed_batch_sizes(
            20,
            0.0,
        )
        == (
            20,
            0,
        )
    )

    assert (
        compute_mixed_batch_sizes(
            20,
            0.25,
        )
        == (
            15,
            5,
        )
    )

    assert (
        compute_mixed_batch_sizes(
            20,
            0.50,
        )
        == (
            10,
            10,
        )
    )

    assert (
        compute_mixed_batch_sizes(
            20,
            0.75,
        )
        == (
            5,
            15,
        )
    )

    assert (
        compute_mixed_batch_sizes(
            20,
            1.0,
        )
        == (
            0,
            20,
        )
    )

    # --------------------------------------------------------
    # Mixed sampling
    # --------------------------------------------------------

    rng = jax.random.PRNGKey(0)

    test_ratios = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    )

    for model_ratio in test_ratios:

        rng, test_rng = (
            jax.random.split(rng)
        )

        test_ratio(
            real_buffer=real_buffer,
            model_buffer=model_buffer,
            agent_idx=agent_idx,
            model_ratio=model_ratio,
            rng=test_rng,
        )

    # --------------------------------------------------------
    # Invalid ratio
    # --------------------------------------------------------

    try:
        compute_mixed_batch_sizes(
            20,
            1.1,
        )

        raise AssertionError(
            "Expected ValueError for invalid model_ratio."
        )

    except ValueError:
        pass

    print(
        "\n================================"
    )

    print(
        "MIXED REPLAY TEST PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()