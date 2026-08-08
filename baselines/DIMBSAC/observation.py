import jax.numpy as jnp


def project_local_obs(
    obs_dict,
    agents,
    num_landmarks: int,
):
    """
    from JaxMARL SimpleSpread 's complete observation，
    only save：

        own velocity
        own position
        landmark relative positions

    delete：
        other-agent positions
        communication

    默认 when 3 landmarks：
        local_obs_dim = 2 + 2 + 3*2 = 10

    return shape:
        (..., num_agents, local_obs_dim)
    """

    local_obs_dim = 4 + 2 * num_landmarks

    local_obs = jnp.stack(
        [
            obs_dict[agent][..., :local_obs_dim]
            for agent in agents
        ],
        axis=-2,
    )

    return local_obs


def local_obs_to_x(local_obs):
    """
    JaxMARL local observation 前四维：

        [vx, vy, px, py]

    我们定义 Dynamics 使用的 physical state 为：

        x = [px, py, vx, vy]

    输入：
        (..., num_agents, 10)

    输出：
        (..., num_agents, 4)
    """

    velocity = local_obs[..., 0:2]
    position = local_obs[..., 2:4]

    x = jnp.concatenate(
        [position, velocity],
        axis=-1,
    )

    return x


def extract_landmarks(
    local_obs,
    num_landmarks: int,
):
    """
    从 local observation 恢复 landmark 的绝对位置。

    local obs 中保存：
        landmark_position - agent_position

    所以：
        landmark_position
        =
        relative_position + agent_position
    """

    position = local_obs[..., 2:4]

    landmark_rel = local_obs[
        ...,
        4 : 4 + 2 * num_landmarks,
    ]

    landmark_rel = landmark_rel.reshape(
        landmark_rel.shape[:-1]
        + (num_landmarks, 2)
    )

    # 每个 agent 都可以根据自己的 position
    # 和 relative landmark position 恢复 landmarks
    landmarks_by_agent = (
        landmark_rel
        + position[..., :, None, :]
    )

    # 理论上所有 agent 恢复出的 landmark 坐标相同
    landmarks = landmarks_by_agent[..., 0, :, :]

    return landmarks, landmarks_by_agent


def reconstruct_local_obs(
    x,
    landmarks,
):
    """
    根据 Dynamics 预测出的 physical state x
    和固定 landmark 位置重新构造 local observation。

    x:
        (..., num_agents, 4)
        [px, py, vx, vy]

    landmarks:
        (..., num_landmarks, 2)

    输出顺序保持和 JaxMARL 一致：

        [vx, vy, px, py, landmark relative positions]
    """

    position = x[..., 0:2]
    velocity = x[..., 2:4]

    landmark_rel = (
        landmarks[..., None, :, :]
        - position[..., :, None, :]
    )

    landmark_rel = landmark_rel.reshape(
        landmark_rel.shape[:-2]
        + (landmark_rel.shape[-2] * 2,)
    )

    local_obs = jnp.concatenate(
        [
            velocity,
            position,
            landmark_rel,
        ],
        axis=-1,
    )

    return local_obs