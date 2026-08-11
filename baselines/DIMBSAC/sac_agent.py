from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from losses import (
    actor_loss,
    compute_critic_target,
    critic_loss,
)
from networks.actor import (
    SACActor,
    sample_squashed_gaussian,
)
from networks.critic import SACCritic

from sac_batch import (
    real_batch_to_sac_batch,
)


class SACAgentState(NamedTuple):
    """Train states for one independent SAC agent."""

    actor: TrainState
    critic1: TrainState
    critic2: TrainState

    target_critic1_params: dict
    target_critic2_params: dict


def init_sac_agent(
    rng,
    actor: SACActor,
    critic: SACCritic,
    dummy_local_obs,
    dummy_joint_obs,
    dummy_joint_action,
    actor_lr: float,
    critic_lr: float,
):
    """Initialize one independent actor and two independent critics."""

    rng, actor_rng, q1_rng, q2_rng = jax.random.split(
        rng,
        4,
    )

    actor_variables = actor.init(
        actor_rng,
        dummy_local_obs,
    )

    q1_variables = critic.init(
        q1_rng,
        dummy_joint_obs,
        dummy_joint_action,
    )

    q2_variables = critic.init(
        q2_rng,
        dummy_joint_obs,
        dummy_joint_action,
    )

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_variables["params"],
        tx=optax.adam(actor_lr),
    )

    critic1_state = TrainState.create(
        apply_fn=critic.apply,
        params=q1_variables["params"],
        tx=optax.adam(critic_lr),
    )

    critic2_state = TrainState.create(
        apply_fn=critic.apply,
        params=q2_variables["params"],
        tx=optax.adam(critic_lr),
    )

    return SACAgentState(
        actor=actor_state,
        critic1=critic1_state,
        critic2=critic2_state,
        target_critic1_params=q1_variables["params"],
        target_critic2_params=q2_variables["params"],
    )


def soft_update(
    target_params,
    source_params,
    tau: float,
):
    """Polyak update of target-network parameters."""

    return jax.tree_util.tree_map(
        lambda target, source: (
            (1.0 - tau) * target
            + tau * source
        ),
        target_params,
        source_params,
    )


def _stop_gradient_tree(params):
    return jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        params,
    )


def sample_joint_actions(
    actor: SACActor,
    actor_params: Sequence,
    joint_obs,
    rng,
    action_low,
    action_high,
):
    """
    Sample joint actions without policy gradients.

    Used when constructing the SAC critic target.
    """

    num_agents = joint_obs.shape[-2]

    action_keys = jax.random.split(
        rng,
        num_agents,
    )

    actions = []
    log_probs = []

    for agent_idx in range(num_agents):

        params = _stop_gradient_tree(
            actor_params[agent_idx]
        )

        mean, log_std = actor.apply(
            {"params": params},
            joint_obs[:, agent_idx],
        )

        action, log_prob, _ = (
            sample_squashed_gaussian(
                rng=action_keys[agent_idx],
                mean=mean,
                log_std=log_std,
                action_low=action_low,
                action_high=action_high,
            )
        )

        actions.append(
            jax.lax.stop_gradient(action)
        )

        log_probs.append(
            jax.lax.stop_gradient(log_prob)
        )

    joint_action = jnp.stack(
        actions,
        axis=1,
    )

    joint_log_prob = jnp.stack(
        log_probs,
        axis=1,
    )

    return joint_action, joint_log_prob


def sample_joint_actions_for_actor_update(
    actor: SACActor,
    own_actor_params,
    other_actor_params: Sequence,
    joint_obs,
    agent_idx: int,
    rng,
    action_low,
    action_high,
):
    """
    Construct joint actions for Actor_i update.

    Gradients flow only through Actor_i.
    Other agents' actions are treated as fixed snapshots.
    """

    num_agents = joint_obs.shape[-2]

    action_keys = jax.random.split(
        rng,
        num_agents,
    )

    actions = []

    own_log_prob = None

    for other_idx in range(num_agents):

        if other_idx == agent_idx:
            params = own_actor_params
        else:
            params = _stop_gradient_tree(
                other_actor_params[other_idx]
            )

        mean, log_std = actor.apply(
            {"params": params},
            joint_obs[:, other_idx],
        )

        action, log_prob, _ = (
            sample_squashed_gaussian(
                rng=action_keys[other_idx],
                mean=mean,
                log_std=log_std,
                action_low=action_low,
                action_high=action_high,
            )
        )

        if other_idx == agent_idx:
            own_log_prob = log_prob
        else:
            action = jax.lax.stop_gradient(
                action
            )

        actions.append(action)

    joint_action = jnp.stack(
        actions,
        axis=1,
    )

    return joint_action, own_log_prob

def _update_sac_agent_core(
    agent_idx: int,
    agent_state: SACAgentState,
    actor: SACActor,
    critic: SACCritic,
    actor_param_snapshots: Sequence,
    critic_batch,
    actor_batch,
    rng,
    action_low,
    action_high,
    gamma: float,
    alpha: float,
    tau: float,
):
    """
    Perform one SAC update for Agent_i.

    Critic and actor batches are separated so their data
    sources can be configured independently.
    """

    rng, target_action_rng, actor_action_rng = (
        jax.random.split(
            rng,
            3,
        )
    )

    # ---------------------------------------------------------
    # Critic target
    # ---------------------------------------------------------

    next_joint_action, next_log_probs = (
        sample_joint_actions(
            actor=actor,
            actor_params=actor_param_snapshots,
            joint_obs=(
                critic_batch.next_obs
            ),
            rng=target_action_rng,
            action_low=action_low,
            action_high=action_high,
        )
    )

    target_q1 = critic.apply(
        {
            "params":
                agent_state.target_critic1_params
        },
        critic_batch.next_obs,
        next_joint_action,
    )

    target_q2 = critic.apply(
        {
            "params":
                agent_state.target_critic2_params
        },
        critic_batch.next_obs,
        next_joint_action,
    )

    target = compute_critic_target(
        rewards=critic_batch.rewards,
        dones=critic_batch.dones,
        target_q1=target_q1,
        target_q2=target_q2,
        next_log_prob=next_log_probs[
            :,
            agent_idx,
        ],
        gamma=gamma,
        alpha=alpha,
    )

    # ---------------------------------------------------------
    # Twin critic update
    # ---------------------------------------------------------

    def critic_objective(
        q1_params,
        q2_params,
    ):
        q1 = critic.apply(
            {"params": q1_params},
            critic_batch.obs,
            critic_batch.actions,
        )

        q2 = critic.apply(
            {"params": q2_params},
            critic_batch.obs,
            critic_batch.actions,
        )

        return critic_loss(
            q1=q1,
            q2=q2,
            target=target,
        )

    (
        (
            critic_total_loss,
            critic_metrics,
        ),
        (
            q1_grads,
            q2_grads,
        ),
    ) = jax.value_and_grad(
        critic_objective,
        argnums=(0, 1),
        has_aux=True,
    )(
        agent_state.critic1.params,
        agent_state.critic2.params,
    )

    critic1_state = (
        agent_state.critic1.apply_gradients(
            grads=q1_grads
        )
    )

    critic2_state = (
        agent_state.critic2.apply_gradients(
            grads=q2_grads
        )
    )

    # ---------------------------------------------------------
    # Actor update
    # ---------------------------------------------------------

    def actor_objective(
        actor_params,
    ):
        (
            policy_joint_action,
            own_log_prob,
        ) = sample_joint_actions_for_actor_update(
            actor=actor,
            own_actor_params=actor_params,
            other_actor_params=(
                actor_param_snapshots
            ),
            joint_obs=actor_batch.obs,
            agent_idx=agent_idx,
            rng=actor_action_rng,
            action_low=action_low,
            action_high=action_high,
        )

        q1 = critic.apply(
            {
                "params":
                    critic1_state.params
            },
            actor_batch.obs,
            policy_joint_action,
        )

        q2 = critic.apply(
            {
                "params":
                    critic2_state.params
            },
            actor_batch.obs,
            policy_joint_action,
        )

        return actor_loss(
            q1=q1,
            q2=q2,
            log_prob=own_log_prob,
            alpha=alpha,
        )

    (
        (
            actor_total_loss,
            actor_metrics,
        ),
        actor_grads,
    ) = jax.value_and_grad(
        actor_objective,
        has_aux=True,
    )(
        agent_state.actor.params
    )

    actor_state = (
        agent_state.actor.apply_gradients(
            grads=actor_grads
        )
    )

    # ---------------------------------------------------------
    # Target critic update
    # ---------------------------------------------------------

    target_critic1_params = soft_update(
        agent_state.target_critic1_params,
        critic1_state.params,
        tau,
    )

    target_critic2_params = soft_update(
        agent_state.target_critic2_params,
        critic2_state.params,
        tau,
    )

    new_agent_state = SACAgentState(
        actor=actor_state,
        critic1=critic1_state,
        critic2=critic2_state,
        target_critic1_params=(
            target_critic1_params
        ),
        target_critic2_params=(
            target_critic2_params
        ),
    )

    metrics = {
        **critic_metrics,
        **actor_metrics,
    }

    return (
        new_agent_state,
        metrics,
    )

def update_sac_agent(
    agent_idx: int,
    agent_state: SACAgentState,
    actor: SACActor,
    critic: SACCritic,
    actor_param_snapshots: Sequence,
    batch,
    rng,
    action_low,
    action_high,
    gamma: float,
    alpha: float,
    tau: float,
):
    """
    Real-replay-only SAC update.

    This wrapper preserves the original baseline behavior.
    """

    real_sac_batch = (
        real_batch_to_sac_batch(
            real_batch=batch,
            agent_idx=agent_idx,
        )
    )

    return _update_sac_agent_core(
        agent_idx=agent_idx,
        agent_state=agent_state,
        actor=actor,
        critic=critic,
        actor_param_snapshots=(
            actor_param_snapshots
        ),
        critic_batch=(
            real_sac_batch
        ),
        actor_batch=(
            real_sac_batch
        ),
        rng=rng,
        action_low=action_low,
        action_high=action_high,
        gamma=gamma,
        alpha=alpha,
        tau=tau,
    )

def update_sac_agent_with_batches(
    agent_idx: int,
    agent_state: SACAgentState,
    actor: SACActor,
    critic: SACCritic,
    actor_param_snapshots: Sequence,
    critic_batch,
    actor_batch,
    rng,
    action_low,
    action_high,
    gamma: float,
    alpha: float,
    tau: float,
):
    """
    SAC update with independently specified critic
    and actor training batches.
    """

    return _update_sac_agent_core(
        agent_idx=agent_idx,
        agent_state=agent_state,
        actor=actor,
        critic=critic,
        actor_param_snapshots=(
            actor_param_snapshots
        ),
        critic_batch=critic_batch,
        actor_batch=actor_batch,
        rng=rng,
        action_low=action_low,
        action_high=action_high,
        gamma=gamma,
        alpha=alpha,
        tau=tau,
    )