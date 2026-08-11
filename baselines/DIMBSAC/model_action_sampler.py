from sac_agent import sample_joint_actions


def build_actor_action_sampler(
    actor,
    actor_params,
    action_low,
    action_high,
):
    """Build a frozen actor sampler for model rollouts."""

    def actor_action_sampler(
        rng,
        joint_obs,
    ):
        joint_action, _ = sample_joint_actions(
            actor=actor,
            actor_params=actor_params,
            joint_obs=joint_obs,
            rng=rng,
            action_low=action_low,
            action_high=action_high,
        )

        return joint_action

    return actor_action_sampler