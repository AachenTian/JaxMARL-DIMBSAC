import jax
import jax.numpy as jnp

from jaxmarl.environments.mpe.simple import State


def build_predicted_mpe_state(
    predicted_x,
    landmarks,
    episode_step,
    num_agents: int,
    dim_c: int,
    episode_horizon: int,
):
    """
    Reconstruct one MPE state from predicted agent physical states.

    Args:
        predicted_x:
            (N, 4), ordered as [px, py, vx, vy]

        landmarks:
            (L, 2)

        episode_step:
            Scalar step index of the predicted next state.
    """

    agent_positions = predicted_x[
        :,
        :2,
    ]

    agent_velocities = predicted_x[
        :,
        2:4,
    ]

    num_landmarks = (
        landmarks.shape[0]
    )

    p_pos = jnp.concatenate(
        [
            agent_positions,
            landmarks,
        ],
        axis=0,
    )

    # Landmarks are fixed in SimpleSpread.
    landmark_velocities = jnp.zeros(
        (
            num_landmarks,
            2,
        ),
        dtype=predicted_x.dtype,
    )

    p_vel = jnp.concatenate(
        [
            agent_velocities,
            landmark_velocities,
        ],
        axis=0,
    )

    c = jnp.zeros(
        (
            num_agents,
            dim_c,
        ),
        dtype=predicted_x.dtype,
    )

    done = (
        episode_step
        >= episode_horizon
    )

    return State(
        p_pos=p_pos,
        p_vel=p_vel,
        c=c,
        done=done,
        step=episode_step,
    )


def compute_model_rewards(
    env,
    predicted_next_x,
    landmarks,
    next_episode_steps,
    episode_horizon: int,
):
    """
    Compute rewards from predicted next physical states.

    Args:
        predicted_next_x:
            (B, N, 4)

        landmarks:
            (B, L, 2)

        next_episode_steps:
            (B,)

    Returns:
        joint_rewards:
            (B, N)
    """

    num_agents = env.num_agents

    def reward_one(
        next_x,
        landmark_pos,
        next_step,
    ):
        predicted_state = (
            build_predicted_mpe_state(
                predicted_x=next_x,
                landmarks=landmark_pos,
                episode_step=next_step,
                num_agents=num_agents,
                dim_c=env.dim_c,
                episode_horizon=(
                    episode_horizon
                ),
            )
        )

        reward_dict = env.rewards(
            predicted_state
        )

        return jnp.stack(
            [
                reward_dict[agent]
                for agent in env.agents
            ],
            axis=0,
        )

    return jax.vmap(
        reward_one
    )(
        predicted_next_x,
        landmarks,
        next_episode_steps,
    )