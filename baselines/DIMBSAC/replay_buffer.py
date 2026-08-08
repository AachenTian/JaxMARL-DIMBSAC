"""
DIMBSAC Shared Real Replay Buffer
DIMBSAC 共享真实经验回放池

This module stores real transitions collected from the JaxMARL environment.
本模块用于存储从 JaxMARL 真实环境中采集得到的 transition。

Important design:
重要设计：

1. Real environment experience is shared among all agents.
   真实环境经验允许所有 agent 共享。

2. The agent dimension is preserved inside each transition.
   每条 transition 内部保留 agent 维度。

3. Actor, Critic, and Dynamics will access different slices of the same data.
   Actor、Critic 和 Dynamics 后续会读取同一份数据中的不同部分。

4. Synthetic model-generated data will NOT be stored here.
   模型生成的 synthetic data 不会存储在这里。
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ============================================================
# Replay buffer state
# Replay Buffer 状态
# ============================================================

class RealReplayBufferState(NamedTuple):
    """
    State of the shared real replay buffer.
    共享真实经验回放池的完整状态。

    Each buffer index corresponds to one environment transition.
    每个 buffer index 对应一条环境 transition。

    The agent dimension is preserved inside each transition.
    每条 transition 内部保留所有 agent 的维度。

    Example with 3 agents:
    以 3 个 agent 为例：

        obs[index].shape == (3, obs_dim)
        x[index].shape == (3, 4)
        actions[index].shape == (3, action_dim)
        rewards[index].shape == (3,)
    """

    # Joint local observations of all agents.
    # 所有 agent 的 joint local observations。
    #
    # Shape:
    #   (capacity, num_agents, obs_dim)
    obs: jax.Array

    # Joint 4-D physical states of all agents.
    # 所有 agent 的 4 维物理状态。
    #
    # x_i = [px, py, vx, vy]
    #
    # Shape:
    #   (capacity, num_agents, 4)
    x: jax.Array

    # Joint actions executed in the real environment.
    # 真实环境中实际执行的 joint actions。
    #
    # Shape:
    #   (capacity, num_agents, action_dim)
    actions: jax.Array

    # Reward of each agent.
    # 每个 agent 对应的 reward。
    #
    # Shape:
    #   (capacity, num_agents)
    rewards: jax.Array

    # Joint local observations after the transition.
    # transition 后所有 agent 的下一时刻 local observations。
    #
    # Shape:
    #   (capacity, num_agents, obs_dim)
    next_obs: jax.Array

    # Joint 4-D physical states after the transition.
    # transition 后所有 agent 的下一时刻 4 维物理状态。
    #
    # Shape:
    #   (capacity, num_agents, 4)
    next_x: jax.Array

    # Environment terminal flag.
    # 环境真正终止的标志。
    #
    # Important:
    # 重要：
    #
    # This is environment termination only.
    # 这里只表示真实环境 termination。
    #
    # It must NOT be set to True merely because a model rollout stops.
    # 不能因为 model rollout 达到长度上限而设置成 True。
    #
    # Shape:
    #   (capacity,)
    dones: jax.Array

    # Episode step corresponding to the START of each transition.
    # 每条 transition 起始状态对应的 episode step。
    #
    # Example:
    # 例如：
    #
    #   episode_step = 24
    #
    # represents:
    # 表示：
    #
    #   step 24 -> step 25
    #
    # For H = 25, this transition has done=True.
    # 对于 H=25，这条 transition 的 done=True。
    #
    # Shape:
    #   (capacity,)
    episode_steps: jax.Array

    # Absolute landmark positions.
    # Landmark 的绝对坐标。
    #
    # These are required later to reconstruct local observations
    # from model-predicted physical states.
    # 后续需要通过它们，从 dynamics 预测的物理状态重新构造 local observation。
    #
    # Shape:
    #   (capacity, num_landmarks, 2)
    landmarks: jax.Array

    # Next insertion position in the circular buffer.
    # Circular buffer 下一次写入的位置。
    #
    # Scalar int32.
    # 标量 int32。
    position: jax.Array

    # Number of currently valid transitions in the buffer.
    # 当前 buffer 中有效 transition 的数量。
    #
    # Scalar int32.
    # 标量 int32。
    size: jax.Array


# ============================================================
# Transition batch
# Transition Batch
# ============================================================

class RealTransitionBatch(NamedTuple):
    """
    A batch of real environment transitions.
    一批真实环境 transition。

    The first dimension is always the batch dimension.
    第一维始终是 batch dimension。

    During collection:
    真实采集时：

        batch_size = NUM_ENVS

    During replay sampling:
    Replay sampling 时：

        batch_size = training batch size
    """

    # Shape: (B, N, obs_dim)
    obs: jax.Array

    # Shape: (B, N, 4)
    x: jax.Array

    # Shape: (B, N, action_dim)
    actions: jax.Array

    # Shape: (B, N)
    rewards: jax.Array

    # Shape: (B, N, obs_dim)
    next_obs: jax.Array

    # Shape: (B, N, 4)
    next_x: jax.Array

    # Shape: (B,)
    dones: jax.Array

    # Shape: (B,)
    episode_steps: jax.Array

    # Shape: (B, num_landmarks, 2)
    landmarks: jax.Array


# ============================================================
# Buffer initialization
# Buffer 初始化
# ============================================================

def init_real_replay_buffer(
    capacity: int,
    num_agents: int,
    obs_dim: int,
    action_dim: int,
    num_landmarks: int,
) -> RealReplayBufferState:
    """
    Initialize an empty fixed-capacity circular real replay buffer.
    初始化一个固定容量的 circular shared real replay buffer。

    Parameters
    ----------
    capacity:
        Maximum number of environment transitions stored.
        最多保存多少条环境 transition。

    num_agents:
        Number of agents.
        Agent 数量。

    obs_dim:
        Local observation dimension for one agent.
        单个 agent 的 local observation 维度。

        For our current Simple Spread design:
        对当前 Simple Spread 设计：

            obs_dim = 10

    action_dim:
        Continuous action dimension for one agent.
        单个 agent 的 continuous action 维度。

    num_landmarks:
        Number of landmarks.
        Landmark 数量。

    Returns
    -------
    RealReplayBufferState
        Empty replay buffer.
        空 Replay Buffer。
    """

    return RealReplayBufferState(

        # ----------------------------------------------------
        # Current transition state
        # 当前 transition 的状态
        # ----------------------------------------------------

        obs=jnp.zeros(
            (
                capacity,
                num_agents,
                obs_dim,
            ),
            dtype=jnp.float32,
        ),

        x=jnp.zeros(
            (
                capacity,
                num_agents,
                4,
            ),
            dtype=jnp.float32,
        ),

        actions=jnp.zeros(
            (
                capacity,
                num_agents,
                action_dim,
            ),
            dtype=jnp.float32,
        ),

        rewards=jnp.zeros(
            (
                capacity,
                num_agents,
            ),
            dtype=jnp.float32,
        ),

        # ----------------------------------------------------
        # Next transition state
        # 下一时刻状态
        # ----------------------------------------------------

        next_obs=jnp.zeros(
            (
                capacity,
                num_agents,
                obs_dim,
            ),
            dtype=jnp.float32,
        ),

        next_x=jnp.zeros(
            (
                capacity,
                num_agents,
                4,
            ),
            dtype=jnp.float32,
        ),

        # ----------------------------------------------------
        # Episode information
        # Episode 信息
        # ----------------------------------------------------

        dones=jnp.zeros(
            (capacity,),
            dtype=jnp.bool_,
        ),

        episode_steps=jnp.zeros(
            (capacity,),
            dtype=jnp.int32,
        ),

        landmarks=jnp.zeros(
            (
                capacity,
                num_landmarks,
                2,
            ),
            dtype=jnp.float32,
        ),

        # ----------------------------------------------------
        # Circular buffer metadata
        # Circular Buffer 元数据
        # ----------------------------------------------------

        position=jnp.array(
            0,
            dtype=jnp.int32,
        ),

        size=jnp.array(
            0,
            dtype=jnp.int32,
        ),
    )


# ============================================================
# Add a batch into the replay buffer
# 向 Replay Buffer 写入一个 batch
# ============================================================

def add_real_batch(
    buffer: RealReplayBufferState,
    batch: RealTransitionBatch,
) -> RealReplayBufferState:
    """
    Add a batch of real transitions to the circular replay buffer.
    向 circular replay buffer 一次写入一批真实 transition。

    In our parallel collector:
    在并行真实环境采集器中：

        batch_size = NUM_ENVS

    Example:
    例如：

        NUM_ENVS = 32

    then one environment step produces:
    那么一次并行 environment step 会产生：

        32 transitions

    and these 32 transitions are inserted together.
    这 32 条 transition 会一次性写入 Replay Buffer。

    Notes
    -----
    The current implementation assumes:
    当前实现假设：

        batch_size <= buffer capacity

    which is always satisfied in our planned setup.
    在我们的训练设计中会始终满足这个条件。
    """

    capacity = buffer.obs.shape[0]
    batch_size = batch.obs.shape[0]

    # --------------------------------------------------------
    # Compute circular insertion indices.
    # 计算 circular buffer 中本次写入的 index。
    #
    # Example:
    # 例如：
    #
    # capacity = 100
    # position = 98
    # batch_size = 4
    #
    # indices = [98, 99, 0, 1]
    # --------------------------------------------------------

    indices = (
        jnp.arange(
            batch_size,
            dtype=jnp.int32,
        )
        + buffer.position
    ) % capacity

    # --------------------------------------------------------
    # Write all transition fields.
    # 写入所有 transition 字段。
    # --------------------------------------------------------

    buffer = buffer._replace(

        obs=buffer.obs.at[
            indices
        ].set(
            batch.obs
        ),

        x=buffer.x.at[
            indices
        ].set(
            batch.x
        ),

        actions=buffer.actions.at[
            indices
        ].set(
            batch.actions
        ),

        rewards=buffer.rewards.at[
            indices
        ].set(
            batch.rewards
        ),

        next_obs=buffer.next_obs.at[
            indices
        ].set(
            batch.next_obs
        ),

        next_x=buffer.next_x.at[
            indices
        ].set(
            batch.next_x
        ),

        dones=buffer.dones.at[
            indices
        ].set(
            batch.dones
        ),

        episode_steps=buffer.episode_steps.at[
            indices
        ].set(
            batch.episode_steps
        ),

        landmarks=buffer.landmarks.at[
            indices
        ].set(
            batch.landmarks
        ),

        # ----------------------------------------------------
        # Advance circular write position.
        # 更新 circular buffer 的下一次写入位置。
        # ----------------------------------------------------

        position=(
            buffer.position
            + batch_size
        ) % capacity,

        # ----------------------------------------------------
        # Buffer size grows until it reaches capacity.
        # Buffer 的有效数据量增加，直到达到 capacity。
        # ----------------------------------------------------

        size=jnp.minimum(
            buffer.size + batch_size,
            capacity,
        ),
    )

    return buffer


# ============================================================
# Uniform replay sampling
# Replay Buffer 均匀采样
# ============================================================

def sample_real_batch(
    buffer: RealReplayBufferState,
    rng: jax.Array,
    batch_size: int,
) -> RealTransitionBatch:
    """
    Uniformly sample a batch from valid real replay data.
    从当前有效的真实 replay 数据中均匀采样一个 batch。

    Important
    ---------
    This function should only be called when:
    只有在下面条件满足时才能调用：

        buffer.size > 0

    In training we will use a learning-start threshold before sampling.
    正式训练时我们会设置 learning-start threshold，
    因此不会在空 buffer 上采样。
    """

    # --------------------------------------------------------
    # Sample valid replay indices.
    # 从有效 replay 范围内随机采样 index。
    #
    # Only [0, buffer.size) is valid.
    # 只有 [0, buffer.size) 范围内的数据是有效的。
    # --------------------------------------------------------

    indices = jax.random.randint(
        rng,
        shape=(batch_size,),
        minval=0,
        maxval=buffer.size,
    )

    # --------------------------------------------------------
    # Gather transitions.
    # 根据 index 读取 transition。
    # --------------------------------------------------------

    return RealTransitionBatch(

        obs=buffer.obs[
            indices
        ],

        x=buffer.x[
            indices
        ],

        actions=buffer.actions[
            indices
        ],

        rewards=buffer.rewards[
            indices
        ],

        next_obs=buffer.next_obs[
            indices
        ],

        next_x=buffer.next_x[
            indices
        ],

        dones=buffer.dones[
            indices
        ],

        episode_steps=buffer.episode_steps[
            indices
        ],

        landmarks=buffer.landmarks[
            indices
        ],
    )