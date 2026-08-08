import jax
import jax.numpy as jnp
import jaxmarl

from observation import (
    project_local_obs,
    local_obs_to_x,
    extract_landmarks,
    reconstruct_local_obs,
)


NUM_ENVS = 8


def main():

    # ==========================================
    # Continuous Simple Spread
    # ==========================================

    env = jaxmarl.make(
        "MPE_simple_spread_v3",
        action_type="Continuous",
    )

    agents = env.agents
    num_agents = env.num_agents
    num_landmarks = env.num_landmarks

    print("agents:", agents)
    print("num_agents:", num_agents)
    print("num_landmarks:", num_landmarks)

    print(
        "full obs shape:",
        env.observation_space(agents[0]).shape,
    )

    print(
        "action shape:",
        env.action_space(agents[0]).shape,
    )

    # ==========================================
    # Parallel reset
    # ==========================================

    rng = jax.random.PRNGKey(0)

    rng, reset_rng = jax.random.split(rng)

    reset_keys = jax.random.split(
        reset_rng,
        NUM_ENVS,
    )

    obs, env_state = jax.vmap(
        env.reset
    )(reset_keys)

    # ==========================================
    # Full obs -> Local obs
    # ==========================================

    local_obs = project_local_obs(
        obs,
        agents,
        num_landmarks,
    )

    print(
        "\nlocal_obs.shape:",
        local_obs.shape,
    )

    # 应该是：
    # (8, 3, 10)

    # ==========================================
    # Local obs -> physical state x
    # ==========================================

    joint_x = local_obs_to_x(
        local_obs
    )

    print(
        "joint_x.shape:",
        joint_x.shape,
    )

    # 应该是：
    # (8, 3, 4)

    # ==========================================
    # Recover landmarks
    # ==========================================

    landmarks, landmarks_by_agent = (
        extract_landmarks(
            local_obs,
            num_landmarks,
        )
    )

    print(
        "landmarks.shape:",
        landmarks.shape,
    )

    # 应该是：
    # (8, 3, 2)

    # ==========================================
    # 检查所有 agent 恢复出的 landmark
    # 是否完全一致
    # ==========================================

    landmark_consistency_error = jnp.max(
        jnp.abs(
            landmarks_by_agent
            - landmarks[..., None, :, :]
        )
    )

    print(
        "landmark consistency error:",
        landmark_consistency_error,
    )

    # ==========================================
    # 用 JaxMARL 内部 state 做测试 oracle
    #
    # 注意：
    # 正式算法不会读取这些数据。
    #这里只用于验证我们有没有提取错。
    # ==========================================

    true_position = env_state.p_pos[
        :,
        :num_agents,
        :,
    ]

    true_velocity = env_state.p_vel[
        :,
        :num_agents,
        :,
    ]

    true_x = jnp.concatenate(
        [
            true_position,
            true_velocity,
        ],
        axis=-1,
    )

    x_error = jnp.max(
        jnp.abs(
            joint_x - true_x
        )
    )

    print(
        "physical x extraction error:",
        x_error,
    )

    # ==========================================
    # 检查 landmark 绝对位置
    # ==========================================

    true_landmarks = env_state.p_pos[
        :,
        num_agents:,
        :,
    ]

    landmark_error = jnp.max(
        jnp.abs(
            landmarks - true_landmarks
        )
    )

    print(
        "landmark extraction error:",
        landmark_error,
    )

    # ==========================================
    # x + landmarks -> local obs
    # ==========================================

    reconstructed_obs = reconstruct_local_obs(
        joint_x,
        landmarks,
    )

    reconstruction_error = jnp.max(
        jnp.abs(
            reconstructed_obs - local_obs
        )
    )

    print(
        "local obs reconstruction error:",
        reconstruction_error,
    )

    # ==========================================
    # Assertions
    # ==========================================

    assert local_obs.shape == (
        NUM_ENVS,
        num_agents,
        4 + 2 * num_landmarks,
    )

    assert joint_x.shape == (
        NUM_ENVS,
        num_agents,
        4,
    )

    assert landmarks.shape == (
        NUM_ENVS,
        num_landmarks,
        2,
    )

    assert landmark_consistency_error < 1e-6
    assert x_error < 1e-6
    assert landmark_error < 1e-6
    assert reconstruction_error < 1e-6

    print("\n==============================")
    print("ALL OBSERVATION TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()