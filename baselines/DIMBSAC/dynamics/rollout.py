import jax
import jax.numpy as jnp

from dynamics.ensemble import predict_dynamics_ensemble


def sample_ensemble_members(
    rng,
    batch_size: int,
    num_agents: int,
    ensemble_size: int,
):
    """
    Sample one dynamics member per agent and per trajectory.

    The sampled member indices should remain fixed throughout
    the entire synthetic trajectory.

    Returns:
        member_indices: (B, N)
    """

    return jax.random.randint(
        rng,
        shape=(
            batch_size,
            num_agents,
        ),
        minval=0,
        maxval=ensemble_size,
    )


def predict_joint_delta_x(
    dynamics_ensembles,
    joint_x,
    joint_actions,
    member_indices,
):
    """
    Predict physical delta-x for all agents.

    Args:
        joint_x:
            (B, N, 4)

        joint_actions:
            (B, N, A)

        member_indices:
            (B, N)

    Returns:
        joint_delta_x:
            (B, N, 4)
    """

    batch_size = joint_x.shape[0]
    num_agents = joint_x.shape[1]

    assert member_indices.shape == (
        batch_size,
        num_agents,
    )

    predicted_deltas = []

    batch_indices = jnp.arange(
        batch_size
    )

    for agent_idx in range(
        num_agents
    ):
        dynamics_inputs = jnp.concatenate(
            [
                joint_x[
                    :,
                    agent_idx,
                    :,
                ],
                joint_actions[
                    :,
                    agent_idx,
                    :,
                ],
            ],
            axis=-1,
        )

        # Evaluate all ensemble members for this agent.
        member_means, _ = (
            predict_dynamics_ensemble(
                ensemble_state=(
                    dynamics_ensembles
                    .agents[agent_idx]
                ),
                dynamics_inputs=(
                    dynamics_inputs
                ),
            )
        )

        # member_means:
        # (M, B, 4)
        selected_member_indices = (
            member_indices[
                :,
                agent_idx,
            ]
        )

        # Select one member prediction for each trajectory.
        selected_mean = (
            member_means[
                selected_member_indices,
                batch_indices,
                :,
            ]
        )

        predicted_deltas.append(
            selected_mean
        )

    return jnp.stack(
        predicted_deltas,
        axis=1,
    )


def propagate_joint_x(
    dynamics_ensembles,
    joint_x,
    joint_actions,
    member_indices,
):
    """
    Propagate one model step using selected member means.

    All inputs and outputs are in physical units.
    """

    joint_delta_x = (
        predict_joint_delta_x(
            dynamics_ensembles=(
                dynamics_ensembles
            ),
            joint_x=joint_x,
            joint_actions=joint_actions,
            member_indices=(
                member_indices
            ),
        )
    )

    next_joint_x = (
        joint_x
        + joint_delta_x
    )

    return (
        next_joint_x,
        joint_delta_x,
    )