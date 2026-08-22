import jax.numpy as jnp

from dynamics.ensemble import (
    predict_dynamics_ensemble,
)


DIM_NAMES = (
    "px",
    "py",
    "vx",
    "vy",
)


def stack_physical_agent_state(
    env_state,
    num_agents: int,
):
    """
    x_i = [p_x, p_y, v_x, v_y]
    Returns: [num_envs, num_agents, 4]
    """
    positions = env_state.p_pos[
        :, :num_agents, :
    ]
    velocities = env_state.p_vel[
        :, :num_agents, :
    ]
    return jnp.concatenate(
        [
            positions,
            velocities,
        ],
        axis=-1,
    )


def get_landmark_positions(
    env_state,
    num_agents: int,
    num_landmarks: int,
):
    return env_state.p_pos[
        :,
        num_agents:
        num_agents + num_landmarks,
        :,
    ]


def build_policy_inputs_from_joint_state(
    joint_state,
    landmark_positions,
    normalized_step,
):
    """
    Reconstruct the same 19D policy input used during Oracle policy training.

    joint_state:
        [N, A, 4], where x=[px,py,vx,vy].

    Output:
        [N, A, 19] for default 3-agent / 3-landmark SimpleSpread.

    Other-agent ordering is ascending agent index with ego removed, matching
    the continuous Beta IPPO training script.
    """
    num_envs = joint_state.shape[0]
    num_agents = joint_state.shape[1]

    if normalized_step.ndim == 1:
        normalized_step = (
            normalized_step[:, None]
        )

    per_agent_inputs = []

    for agent_idx in range(
        num_agents
    ):
        own_pos = joint_state[
            :, agent_idx, :2
        ]
        own_vel = joint_state[
            :, agent_idx, 2:4
        ]

        landmark_rel_pos = (
            landmark_positions
            - own_pos[:, None, :]
        )

        other_indices = [
            idx
            for idx in range(
                num_agents
            )
            if idx != agent_idx
        ]

        other_rel_pos = (
            joint_state[
                :, other_indices, :2
            ]
            - own_pos[:, None, :]
        )
        other_abs_vel = (
            joint_state[
                :, other_indices, 2:4
            ]
        )

        agent_input = jnp.concatenate(
            [
                own_vel,
                own_pos,
                landmark_rel_pos.reshape(
                    (num_envs, -1)
                ),
                other_rel_pos.reshape(
                    (num_envs, -1)
                ),
                other_abs_vel.reshape(
                    (num_envs, -1)
                ),
                normalized_step,
            ],
            axis=-1,
        )

        per_agent_inputs.append(
            agent_input
        )

    return jnp.stack(
        per_agent_inputs,
        axis=1,
    )


def overwrite_ego_with_truth(
    estimated_joint_states,
    true_joint_state,
):
    """
    estimated_joint_states: [N, E, A, 4]
      E is the ego perspective.

    For each ego e, overwrite x_hat_e,e with the real environment state.
    """
    num_egos = (
        estimated_joint_states.shape[1]
    )

    result = (
        estimated_joint_states
    )

    for ego_idx in range(
        num_egos
    ):
        result = result.at[
            :, ego_idx, ego_idx, :
        ].set(
            true_joint_state[
                :, ego_idx, :
            ]
        )

    return result


def build_estimated_policy_actions(
    estimated_joint_states,
    landmark_positions,
    normalized_step,
    network,
    policy_params,
    deterministic_action_fn,
    action_eps: float,
):
    """
    Build one full 19D policy input for every agent from every ego perspective.

    Returns:
        actions [N, E, A, 5]
    """
    num_egos = (
        estimated_joint_states.shape[1]
    )
    action_batches = []

    for ego_idx in range(
        num_egos
    ):
        policy_inputs = (
            build_policy_inputs_from_joint_state(
                joint_state=(
                    estimated_joint_states[
                        :, ego_idx, :, :
                    ]
                ),
                landmark_positions=(
                    landmark_positions
                ),
                normalized_step=(
                    normalized_step
                ),
            )
        )

        num_envs = (
            policy_inputs.shape[0]
        )
        num_agents = (
            policy_inputs.shape[1]
        )

        flat_inputs = (
            policy_inputs.reshape(
                (
                    num_envs
                    * num_agents,
                    -1,
                )
            )
        )

        flat_actions = (
            deterministic_action_fn(
                network=network,
                params=policy_params,
                policy_inputs=flat_inputs,
                action_eps=action_eps,
            )
        )

        actions = flat_actions.reshape(
            (
                num_envs,
                num_agents,
                -1,
            )
        )
        action_batches.append(
            actions
        )

    return jnp.stack(
        action_batches,
        axis=1,
    )


def propagate_other_agents(
    independent_ensembles,
    estimated_joint_states,
    actions_by_perspective,
):
    """
    One model step for every non-ego agent.

    estimated_joint_states:
        [N, E, A, 4]

    actions_by_perspective:
        [N, E, A, 5]

    Each agent j is propagated by its own 9D local dynamics ensemble:
        [x_hat_j(4), a_hat_j(5)] -> delta_x_hat_j(4)

    Ensemble prediction uses the mean of all members.
    """
    num_envs = (
        estimated_joint_states.shape[0]
    )
    num_egos = (
        estimated_joint_states.shape[1]
    )
    num_agents = (
        estimated_joint_states.shape[2]
    )

    next_estimated = (
        estimated_joint_states
    )

    for agent_idx in range(
        num_agents
    ):
        agent_states = (
            estimated_joint_states[
                :, :, agent_idx, :
            ]
        )
        agent_actions = (
            actions_by_perspective[
                :, :, agent_idx, :
            ]
        )

        dynamics_inputs = (
            jnp.concatenate(
                [
                    agent_states,
                    agent_actions,
                ],
                axis=-1,
            )
        )

        flat_inputs = (
            dynamics_inputs.reshape(
                (
                    num_envs
                    * num_egos,
                    -1,
                )
            )
        )

        member_means, _ = (
            predict_dynamics_ensemble(
                ensemble_state=(
                    independent_ensembles
                    .agents[
                        agent_idx
                    ]
                ),
                dynamics_inputs=flat_inputs,
            )
        )

        ensemble_mean_delta = (
            jnp.mean(
                member_means,
                axis=0,
            ).reshape(
                (
                    num_envs,
                    num_egos,
                    4,
                )
            )
        )

        predicted_next_state = (
            agent_states
            + ensemble_mean_delta
        )

        next_estimated = (
            next_estimated.at[
                :, :, agent_idx, :
            ].set(
                predicted_next_state
            )
        )

    return next_estimated


def other_agent_error(
    estimated_joint_states,
    true_joint_state,
):
    """
    Return only off-diagonal errors:
      for each ego perspective, compare its estimates of OTHER agents.

    Output shape:
        [N, E * (A-1), 4]
    """
    num_envs = (
        estimated_joint_states.shape[0]
    )
    num_egos = (
        estimated_joint_states.shape[1]
    )
    num_agents = (
        estimated_joint_states.shape[2]
    )

    errors = []

    for ego_idx in range(
        num_egos
    ):
        for agent_idx in range(
            num_agents
        ):
            if agent_idx == ego_idx:
                continue

            errors.append(
                estimated_joint_states[
                    :, ego_idx, agent_idx, :
                ]
                - true_joint_state[
                    :, agent_idx, :
                ]
            )

    return jnp.stack(
        errors,
        axis=1,
    ).reshape(
        (
            num_envs,
            -1,
            4,
        )
    )


def summarize_error(
    error,
    true_range,
    eps: float = 1e-8,
):
    """
    Compute physical RMSE and range-normalized RMSE.

    true_range is one global evaluation range for each state dimension,
    computed over the complete real trajectory.
    """
    per_dim_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error
            ),
            axis=(0, 1),
        )
    )

    safe_range = jnp.maximum(
        true_range,
        eps,
    )
    per_dim_nrmse_range_pct = (
        100.0
        * per_dim_rmse
        / safe_range
    )

    position_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error[..., :2]
            )
        )
    )
    velocity_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(
                error[..., 2:4]
            )
        )
    )

    position_nrmse_range_pct = (
        jnp.mean(
            per_dim_nrmse_range_pct[
                :2
            ]
        )
    )
    velocity_nrmse_range_pct = (
        jnp.mean(
            per_dim_nrmse_range_pct[
                2:4
            ]
        )
    )

    return {
        "per_dim_rmse": (
            per_dim_rmse
        ),
        "position_rmse": (
            position_rmse
        ),
        "velocity_rmse": (
            velocity_rmse
        ),
        "per_dim_nrmse_range_pct": (
            per_dim_nrmse_range_pct
        ),
        "position_nrmse_range_pct": (
            position_nrmse_range_pct
        ),
        "velocity_nrmse_range_pct": (
            velocity_nrmse_range_pct
        ),
    }
