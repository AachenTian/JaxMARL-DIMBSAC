"""
Four-panel animation for full three-agent estimator execution.

IMPORTANT execution semantics
-----------------------------
There is ONE estimator-controlled real environment trajectory:

    X_real[t] = [x0_real, x1_real, x2_real]

All three ego estimators observe / estimate that SAME real world. They do not
own separate real environments.

The four synchronized panels are:

    top-left:     ORACLE / REFERENCE execution
                  TRUE joint state -> shared policy -> real environment
    top-right:    Ego 0 view of the estimator-controlled real execution
    bottom-left:  Ego 1 view of the estimator-controlled real execution
    bottom-right: Ego 2 view of the estimator-controlled real execution

The three Ego panels share the SAME estimator-controlled real trajectory.
The top-left reference panel is a separate paired counterfactual trajectory
that starts from the same initial state and uses the same environment step
randomness, but feeds the policy TRUE joint-state information instead of
estimator-reconstructed information.

In each Ego panel:
    - all three REAL agents are drawn with solid filled markers / solid trails;
    - only OFF-DIAGONAL estimates are overlaid:
          Ego 0: hat{x1}^{(0)}, hat{x2}^{(0)}
          Ego 1: hat{x0}^{(1)}, hat{x2}^{(1)}
          Ego 2: hat{x0}^{(2)}, hat{x1}^{(2)}
    - an estimate is drawn as a hollow marker + dashed trail;
    - a dotted line connects the current REAL position to the current estimate;
    - ego's own diagonal estimate is not separately drawn because it is
      refreshed from truth every step and coincides with the REAL state.

This renderer does not run an environment. It visualizes the two already
stored paired branches for one selected debug_env_idx:
    - the oracle/reference branch in the top-left panel;
    - the estimator-execution branch in all three Ego panels.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _to_numpy(value):
    return np.asarray(value)


def _stack_time(sequence):
    return np.stack(
        [_to_numpy(value) for value in sequence],
        axis=0,
    )


def _selected_landmarks(
    landmark_positions,
    debug_env_idx,
):
    landmarks = _to_numpy(
        landmark_positions
    )

    if landmarks.ndim == 2:
        return landmarks

    if landmarks.ndim == 3:
        return landmarks[
            debug_env_idx
        ]

    raise ValueError(
        "landmark_positions must have shape [L, 2] "
        "or [N, L, 2], got "
        f"{landmarks.shape}."
    )


def _selected_step_rewards(
    step_rewards,
    debug_env_idx,
):
    """Selected-env cumulative mean-agent return, length H + 1."""
    cumulative = [0.0]
    running = 0.0

    for rewards in step_rewards:
        rewards_np = _to_numpy(
            rewards
        )
        running += float(
            np.mean(
                rewards_np[
                    debug_env_idx
                ]
            )
        )
        cumulative.append(
            running
        )

    return np.asarray(
        cumulative,
        dtype=np.float32,
    )


def _ego_rmse(
    ego_belief,
    true_state,
    ego_idx,
):
    """RMSE over ego's two off-diagonal agent estimates."""
    other_indices = [
        agent_idx
        for agent_idx in range(
            true_state.shape[0]
        )
        if agent_idx != ego_idx
    ]

    error = (
        ego_belief[
            other_indices
        ]
        - true_state[
            other_indices
        ]
    )

    position_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    error[:, :2]
                )
            )
        )
    )
    velocity_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    error[:, 2:4]
                )
            )
        )
    )

    return (
        position_rmse,
        velocity_rmse,
    )


def _compute_limits(
    reference_trajectory,
    true_trajectory,
    estimator_trajectory,
    landmarks,
):
    all_positions = np.concatenate(
        [
            reference_trajectory[
                ...,
                :2,
            ].reshape(
                -1,
                2,
            ),
            true_trajectory[
                ...,
                :2,
            ].reshape(
                -1,
                2,
            ),
            estimator_trajectory[
                ...,
                :2,
            ].reshape(
                -1,
                2,
            ),
            landmarks.reshape(
                -1,
                2,
            ),
        ],
        axis=0,
    )

    low = np.min(
        all_positions,
        axis=0,
    )
    high = np.max(
        all_positions,
        axis=0,
    )

    center = 0.5 * (
        low + high
    )
    span = float(
        np.max(
            high - low
        )
    )

    if span < 1e-6:
        span = 1.0

    half_extent = (
        0.5 * span
        + 0.18
    )

    return (
        (
            float(
                center[0]
                - half_extent
            ),
            float(
                center[0]
                + half_extent
            ),
        ),
        (
            float(
                center[1]
                - half_extent
            ),
            float(
                center[1]
                + half_extent
            ),
        ),
    )


def _draw_landmarks(
    ax,
    landmarks,
):
    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        marker="*",
        s=115,
        facecolors="none",
        edgecolors="0.25",
        linewidths=1.5,
        zorder=2,
    )


def _configure_axis(
    ax,
    x_limits,
    y_limits,
):
    ax.set_xlim(
        *x_limits
    )
    ax.set_ylim(
        *y_limits
    )
    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.set_xlabel(
        "x"
    )
    ax.set_ylabel(
        "y"
    )
    ax.grid(
        alpha=0.20
    )


def _draw_velocity_arrow(
    ax,
    position,
    velocity,
    color,
    velocity_scale,
    alpha=0.9,
    linestyle="-",
):
    delta = (
        velocity_scale
        * velocity
    )

    ax.annotate(
        "",
        xy=(
            position[0]
            + delta[0],
            position[1]
            + delta[1],
        ),
        xytext=(
            position[0],
            position[1],
        ),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "alpha": alpha,
            "lw": 1.2,
            "linestyle": linestyle,
        },
        zorder=5,
    )


def _draw_real_world(
    ax,
    frame_idx,
    true_trajectory,
    colors,
    trail_length,
    velocity_scale,
    show_agent_labels=True,
):
    """
    Draw the SAME estimator-controlled real trajectory.

    This function is called in all four panels, which guarantees that the real
    agents shown in Ego 0 / Ego 1 / Ego 2 panels are identical.
    """
    num_agents = (
        true_trajectory.shape[1]
    )

    trail_start = max(
        0,
        frame_idx
        - trail_length
        + 1,
    )

    for agent_idx in range(
        num_agents
    ):
        color = colors[
            agent_idx
            % len(colors)
        ]

        real_trail = (
            true_trajectory[
                trail_start:
                frame_idx + 1,
                agent_idx,
                :2,
            ]
        )

        ax.plot(
            real_trail[:, 0],
            real_trail[:, 1],
            linestyle="-",
            linewidth=2.0,
            alpha=0.78,
            color=color,
            zorder=3,
        )

        real_state = (
            true_trajectory[
                frame_idx,
                agent_idx,
            ]
        )

        ax.scatter(
            real_state[0],
            real_state[1],
            s=78,
            color=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=7,
        )

        _draw_velocity_arrow(
            ax=ax,
            position=(
                real_state[:2]
            ),
            velocity=(
                real_state[2:4]
            ),
            color=color,
            velocity_scale=(
                velocity_scale
            ),
            alpha=0.90,
            linestyle="-",
        )

        if show_agent_labels:
            ax.text(
                real_state[0],
                real_state[1],
                f" A{agent_idx} real",
                fontsize=7,
                color=color,
                va="bottom",
                ha="left",
                zorder=9,
            )


def _draw_reference_panel(
    ax,
    frame_idx,
    reference_trajectory,
    landmarks,
    colors,
    trail_length,
    velocity_scale,
    reference_return,
):
    _draw_landmarks(
        ax,
        landmarks,
    )

    _draw_real_world(
        ax=ax,
        frame_idx=(
            frame_idx
        ),
        true_trajectory=(
            reference_trajectory
        ),
        colors=(
            colors
        ),
        trail_length=(
            trail_length
        ),
        velocity_scale=(
            velocity_scale
        ),
    )

    ax.set_title(
        "ORACLE / REFERENCE: TRUE joint state -> policy\n"
        f"cumulative mean-agent return = {reference_return:.3f}"
    )

    agent_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markerfacecolor=(
                colors[
                    agent_idx
                    % len(colors)
                ]
            ),
            markeredgecolor="white",
            color=(
                colors[
                    agent_idx
                    % len(colors)
                ]
            ),
            label=f"A{agent_idx} real",
        )
        for agent_idx in range(
            reference_trajectory.shape[1]
        )
    ]

    ax.legend(
        handles=agent_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
    )


def _draw_off_diagonal_estimate(
    ax,
    frame_idx,
    ego_idx,
    modeled_agent_idx,
    true_trajectory,
    estimator_trajectory,
    color,
    trail_length,
    velocity_scale,
):
    """
    Overlay one off-diagonal estimate on top of the common real world.
    """
    trail_start = max(
        0,
        frame_idx
        - trail_length
        + 1,
    )

    estimated_trail = (
        estimator_trajectory[
            trail_start:
            frame_idx + 1,
            ego_idx,
            modeled_agent_idx,
            :2,
        ]
    )

    ax.plot(
        estimated_trail[:, 0],
        estimated_trail[:, 1],
        linestyle="--",
        linewidth=1.8,
        alpha=0.90,
        color=color,
        zorder=4,
    )

    true_position = (
        true_trajectory[
            frame_idx,
            modeled_agent_idx,
            :2,
        ]
    )

    estimated_state = (
        estimator_trajectory[
            frame_idx,
            ego_idx,
            modeled_agent_idx,
        ]
    )
    estimated_position = (
        estimated_state[:2]
    )

    ax.scatter(
        estimated_position[0],
        estimated_position[1],
        s=86,
        marker="o",
        facecolors="none",
        edgecolors=color,
        linewidths=2.0,
        zorder=10,
    )

    # Real -> estimated position error at the current time step.
    ax.plot(
        [
            true_position[0],
            estimated_position[0],
        ],
        [
            true_position[1],
            estimated_position[1],
        ],
        linestyle=":",
        linewidth=1.4,
        alpha=0.95,
        color=color,
        zorder=6,
    )

    _draw_velocity_arrow(
        ax=ax,
        position=(
            estimated_position
        ),
        velocity=(
            estimated_state[
                2:4
            ]
        ),
        color=color,
        velocity_scale=(
            velocity_scale
        ),
        alpha=0.78,
        linestyle="--",
    )

    ax.text(
        estimated_position[0],
        estimated_position[1],
        f" A{modeled_agent_idx} est",
        fontsize=7,
        color=color,
        va="top",
        ha="left",
        zorder=11,
    )


def _draw_ego_panel(
    ax,
    frame_idx,
    ego_idx,
    true_trajectory,
    estimator_trajectory,
    landmarks,
    colors,
    trail_length,
    velocity_scale,
):
    """
    Draw ONE common real world + ego-specific off-diagonal estimates.

    The real Agent 0/1/2 trajectories are therefore identical across
    Ego 0, Ego 1, and Ego 2 panels.
    """
    _draw_landmarks(
        ax,
        landmarks,
    )

    # First draw all three REAL agents. This is exactly the same call and the
    # same true_trajectory used by every ego panel.
    _draw_real_world(
        ax=ax,
        frame_idx=(
            frame_idx
        ),
        true_trajectory=(
            true_trajectory
        ),
        colors=(
            colors
        ),
        trail_length=(
            trail_length
        ),
        velocity_scale=(
            velocity_scale
        ),
    )

    num_agents = (
        true_trajectory.shape[1]
    )

    # Then overlay ONLY the two agents that this ego estimates.
    for modeled_agent_idx in range(
        num_agents
    ):
        if (
            modeled_agent_idx
            == ego_idx
        ):
            # The ego diagonal is truth-refreshed, so there is no separate
            # estimated marker/trajectory to draw.
            continue

        color = colors[
            modeled_agent_idx
            % len(colors)
        ]

        _draw_off_diagonal_estimate(
            ax=ax,
            frame_idx=(
                frame_idx
            ),
            ego_idx=(
                ego_idx
            ),
            modeled_agent_idx=(
                modeled_agent_idx
            ),
            true_trajectory=(
                true_trajectory
            ),
            estimator_trajectory=(
                estimator_trajectory
            ),
            color=color,
            trail_length=(
                trail_length
            ),
            velocity_scale=(
                velocity_scale
            ),
        )

    true_state = (
        true_trajectory[
            frame_idx
        ]
    )
    ego_belief = (
        estimator_trajectory[
            frame_idx,
            ego_idx,
        ]
    )

    (
        position_rmse,
        velocity_rmse,
    ) = _ego_rmse(
        ego_belief=(
            ego_belief
        ),
        true_state=(
            true_state
        ),
        ego_idx=(
            ego_idx
        ),
    )

    ax.set_title(
        f"Ego {ego_idx}: SAME real world + its two estimates\n"
        f"off-diagonal pos RMSE={position_rmse:.4f}, "
        f"vel RMSE={velocity_rmse:.4f}"
    )

    real_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        color="0.20",
        markerfacecolor="0.20",
        markeredgecolor="white",
        linewidth=2.0,
        label="real state / trail",
    )
    estimate_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="--",
        color="0.20",
        markerfacecolor="none",
        markeredgecolor="0.20",
        linewidth=1.8,
        label="off-diagonal estimate",
    )
    error_handle = Line2D(
        [0],
        [0],
        linestyle=":",
        color="0.20",
        linewidth=1.4,
        label="current position error",
    )

    ax.legend(
        handles=[
            real_handle,
            estimate_handle,
            error_handle,
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
    )


def create_full_execution_paired_animation(
    reference_true_trajectory,
    estimator_true_trajectory,
    estimator_state_trajectory,
    landmark_positions,
    reference_step_rewards,
    estimator_step_rewards,
    debug_env_idx=0,
    estimator_ego_idx=None,
    output_dir="animations",
    fps=4,
    trail_length=8,
    velocity_scale=0.35,
    save_gif=True,
    save_mp4=True,
    wandb_run=None,
):
    """
    Create one synchronized four-panel animation from stored rollout data.

    `estimator_ego_idx` is retained only for backward compatibility with the
    existing evaluator call. All three ego perspectives are always shown.
    """
    del estimator_ego_idx

    true_all = _stack_time(
        estimator_true_trajectory
    )
    estimator_all = _stack_time(
        estimator_state_trajectory
    )
    reference_all = _stack_time(
        reference_true_trajectory
    )

    if true_all.ndim != 4:
        raise ValueError(
            "estimator_true_trajectory must stack to [T, N, A, 4], "
            f"got {true_all.shape}."
        )

    if estimator_all.ndim != 5:
        raise ValueError(
            "estimator_state_trajectory must stack to [T, N, E, A, 4], "
            f"got {estimator_all.shape}."
        )

    num_frames = (
        true_all.shape[0]
    )
    num_envs = (
        true_all.shape[1]
    )
    num_agents = (
        true_all.shape[2]
    )
    num_egos = (
        estimator_all.shape[2]
    )

    if not (
        0 <= debug_env_idx
        < num_envs
    ):
        raise IndexError(
            f"FULL_ANIMATION_ENV_INDEX={debug_env_idx} is outside "
            f"[0, {num_envs - 1}]."
        )

    if (
        num_agents != 3
        or num_egos != 3
    ):
        raise ValueError(
            "The four-panel renderer expects exactly 3 agents and 3 ego "
            f"estimators, got agents={num_agents}, egos={num_egos}."
        )

    if (
        reference_all.shape[0]
        != num_frames
    ):
        raise ValueError(
            "Reference and estimator trajectories must have the same "
            "number of frames."
        )

    # ORACLE / REFERENCE branch selected from the vectorized paired run.
    # Its policy input is reconstructed from the TRUE joint state.
    reference_trajectory = (
        reference_all[
            :,
            debug_env_idx,
        ]
    )

    # ONE real estimator-execution trajectory selected from the same paired run.
    true_trajectory = (
        true_all[
            :,
            debug_env_idx,
        ]
    )

    # Three ego beliefs about that SAME estimator-controlled real trajectory.
    estimator_trajectory = (
        estimator_all[
            :,
            debug_env_idx,
        ]
    )

    landmarks = _selected_landmarks(
        landmark_positions=(
            landmark_positions
        ),
        debug_env_idx=(
            debug_env_idx
        ),
    )

    reference_cumulative = (
        _selected_step_rewards(
            step_rewards=(
                reference_step_rewards
            ),
            debug_env_idx=(
                debug_env_idx
            ),
        )
    )
    estimator_cumulative = (
        _selected_step_rewards(
            step_rewards=(
                estimator_step_rewards
            ),
            debug_env_idx=(
                debug_env_idx
            ),
        )
    )

    if (
        len(
            estimator_cumulative
        )
        != num_frames
    ):
        raise ValueError(
            "Expected reward history length H to match trajectory length H+1."
        )

    x_limits, y_limits = (
        _compute_limits(
            reference_trajectory=(
                reference_trajectory
            ),
            true_trajectory=(
                true_trajectory
            ),
            estimator_trajectory=(
                estimator_trajectory
            ),
            landmarks=(
                landmarks
            ),
        )
    )

    output_path = Path(
        output_dir
    )
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        "full_estimator_execution_"
        f"env{debug_env_idx}_"
        "oracle_vs_estimator_all_egos"
    )

    colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()
        .get(
            "color",
            [
                "C0",
                "C1",
                "C2",
            ],
        )
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.8,
            10.8,
        ),
        constrained_layout=True,
    )

    real_ax = (
        axes[
            0,
            0,
        ]
    )
    ego_axes = [
        axes[
            0,
            1,
        ],
        axes[
            1,
            0,
        ],
        axes[
            1,
            1,
        ],
    ]

    def draw_frame(
        frame_idx,
    ):
        for ax in axes.flat:
            ax.clear()
            _configure_axis(
                ax=ax,
                x_limits=(
                    x_limits
                ),
                y_limits=(
                    y_limits
                ),
            )

        _draw_reference_panel(
            ax=(
                real_ax
            ),
            frame_idx=(
                frame_idx
            ),
            reference_trajectory=(
                reference_trajectory
            ),
            landmarks=(
                landmarks
            ),
            colors=(
                colors
            ),
            trail_length=(
                trail_length
            ),
            velocity_scale=(
                velocity_scale
            ),
            reference_return=float(
                reference_cumulative[
                    frame_idx
                ]
            ),
        )

        for ego_idx, ax in enumerate(
            ego_axes
        ):
            _draw_ego_panel(
                ax=ax,
                frame_idx=(
                    frame_idx
                ),
                ego_idx=(
                    ego_idx
                ),
                true_trajectory=(
                    true_trajectory
                ),
                estimator_trajectory=(
                    estimator_trajectory
                ),
                landmarks=(
                    landmarks
                ),
                colors=(
                    colors
                ),
                trail_length=(
                    trail_length
                ),
                velocity_scale=(
                    velocity_scale
                ),
            )

        return_gap = (
            estimator_cumulative[
                frame_idx
            ]
            - reference_cumulative[
                frame_idx
            ]
        )

        fig.suptitle(
            (
                "Paired execution — ORACLE reference vs THREE estimator beliefs\n"
                f"env={debug_env_idx} | "
                f"episode step={frame_idx}/{num_frames - 1} | "
                f"reference return={reference_cumulative[frame_idx]:.3f} | "
                f"estimator return={estimator_cumulative[frame_idx]:.3f} | "
                f"gap={return_gap:.3f}\n"
                "Top-left = separate oracle/reference branch | "
                "Ego 0/1/2 panels share the SAME estimator-controlled real world\n"
                "In Ego panels: solid/filled = real agents | "
                "hollow/dashed = that ego's off-diagonal estimates"
            ),
            fontsize=12,
        )

        return []

    animation = (
        mpl_animation.FuncAnimation(
            fig=fig,
            func=draw_frame,
            frames=num_frames,
            interval=(
                1000.0
                / max(
                    fps,
                    1,
                )
            ),
            blit=False,
            repeat=True,
        )
    )

    outputs = {}

    if save_gif:
        gif_path = (
            output_path
            / f"{stem}.gif"
        )

        animation.save(
            str(
                gif_path
            ),
            writer=(
                mpl_animation.PillowWriter(
                    fps=fps
                )
            ),
        )

        outputs[
            "gif"
        ] = str(
            gif_path
        )

    if (
        save_mp4
        and mpl_animation.writers.is_available(
            "ffmpeg"
        )
    ):
        mp4_path = (
            output_path
            / f"{stem}.mp4"
        )

        animation.save(
            str(
                mp4_path
            ),
            writer=(
                mpl_animation.FFMpegWriter(
                    fps=fps,
                    bitrate=2400,
                )
            ),
        )

        outputs[
            "mp4"
        ] = str(
            mp4_path
        )

    plt.close(
        fig
    )

    if wandb_run is not None:
        try:
            import wandb

            if "mp4" in outputs:
                wandb_run.log(
                    {
                        (
                            "animation/"
                            "full_estimator_execution_"
                            "oracle_vs_estimator_all_egos"
                        ): wandb.Video(
                            outputs[
                                "mp4"
                            ],
                            fps=fps,
                            format="mp4",
                        )
                    }
                )
            elif "gif" in outputs:
                wandb_run.log(
                    {
                        (
                            "animation/"
                            "full_estimator_execution_"
                            "oracle_vs_estimator_all_egos"
                        ): wandb.Video(
                            outputs[
                                "gif"
                            ],
                            fps=fps,
                            format="gif",
                        )
                    }
                )
        except Exception as exc:
            print(
                "Warning: animation was saved locally but W&B upload failed:",
                exc,
            )

    return outputs
