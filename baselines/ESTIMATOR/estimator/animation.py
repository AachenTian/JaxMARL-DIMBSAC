from pathlib import Path

import jax
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import wandb


def _to_numpy(x):
    return np.asarray(
        jax.device_get(x)
    )


def create_agent0_shadow_animation(
    true_trajectory,
    estimated_trajectory,
    landmark_positions,
    ego_agent_idx,
    debug_env_idx,
    output_dir,
    fps=4,
    trail_length=8,
    velocity_scale=0.35,
    save_gif=True,
    save_mp4=True,
    wandb_run=None,
):
    """
    Visualize one fixed evaluation environment.

    The animation shows:
      - true positions of all three agents,
      - estimated positions of the two non-ego agents,
      - true and estimated short trajectory tails,
      - true and estimated velocity arrows,
      - static landmark positions,
      - current per-agent position / velocity L2 errors.

    True other-agent states are visualization-only and are never fed back to
    the estimator.
    """
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    true_states = _to_numpy(
        np.stack(
            [
                _to_numpy(x)
                for x in true_trajectory
            ],
            axis=0,
        )
    )
    estimated_states = _to_numpy(
        np.stack(
            [
                _to_numpy(x)
                for x in estimated_trajectory
            ],
            axis=0,
        )
    )
    landmarks = _to_numpy(
        landmark_positions
    )

    # Select one environment:
    #   states: [T, A, 4]
    #   landmarks: [L, 2]
    true_states = true_states[
        :, debug_env_idx, :, :
    ]
    estimated_states = (
        estimated_states[
            :, debug_env_idx, :, :
        ]
    )
    landmarks = landmarks[
        debug_env_idx
    ]

    num_frames = (
        true_states.shape[0]
    )
    num_agents = (
        true_states.shape[1]
    )

    other_agent_indices = [
        idx
        for idx in range(
            num_agents
        )
        if idx != ego_agent_idx
    ]

    # Determine a fixed axis range from the complete diagnostic trajectory.
    all_positions = [
        true_states[..., :2].reshape(
            (-1, 2)
        ),
        estimated_states[
            ..., :2
        ].reshape(
            (-1, 2)
        ),
        landmarks.reshape(
            (-1, 2)
        ),
    ]
    all_positions = np.concatenate(
        all_positions,
        axis=0,
    )

    xy_min = np.min(
        all_positions,
        axis=0,
    )
    xy_max = np.max(
        all_positions,
        axis=0,
    )
    span = np.maximum(
        xy_max - xy_min,
        0.25,
    )
    padding = (
        0.15 * span
    )

    xlim = (
        xy_min[0] - padding[0],
        xy_max[0] + padding[0],
    )
    ylim = (
        xy_min[1] - padding[1],
        xy_max[1] + padding[1],
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    def draw_frame(
        frame_idx,
    ):
        ax.clear()

        ax.set_title(
            (
                f"Agent {ego_agent_idx} shadow estimator | "
                f"env {debug_env_idx} | "
                f"step {frame_idx}/{num_frames - 1}"
            )
        )
        ax.set_xlabel(
            "x position"
        )
        ax.set_ylabel(
            "y position"
        )
        ax.set_xlim(
            *xlim
        )
        ax.set_ylim(
            *ylim
        )
        ax.set_aspect(
            "equal",
            adjustable="box",
        )
        ax.grid(
            alpha=0.25
        )

        # Static landmarks.
        ax.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            marker="X",
            s=120,
            label="Landmarks",
        )

        # Ego: only the real state exists in this diagnostic.
        ego_pos = true_states[
            frame_idx,
            ego_agent_idx,
            :2,
        ]
        ego_vel = true_states[
            frame_idx,
            ego_agent_idx,
            2:4,
        ]

        ax.scatter(
            ego_pos[0],
            ego_pos[1],
            marker="*",
            s=170,
            label=(
                f"Agent {ego_agent_idx} real (ego)"
            ),
        )
        ax.arrow(
            ego_pos[0],
            ego_pos[1],
            velocity_scale * ego_vel[0],
            velocity_scale * ego_vel[1],
            width=0.003,
            length_includes_head=True,
        )

        # Ego true trajectory tail.
        tail_start = max(
            0,
            frame_idx
            - trail_length
            + 1,
        )
        ego_tail = true_states[
            tail_start:
            frame_idx + 1,
            ego_agent_idx,
            :2,
        ]
        ax.plot(
            ego_tail[:, 0],
            ego_tail[:, 1],
            linestyle="-",
            alpha=0.6,
        )

        error_lines = []

        # Other agents: true vs estimator.
        for agent_idx in (
            other_agent_indices
        ):
            true_pos = true_states[
                frame_idx,
                agent_idx,
                :2,
            ]
            true_vel = true_states[
                frame_idx,
                agent_idx,
                2:4,
            ]
            est_pos = estimated_states[
                frame_idx,
                agent_idx,
                :2,
            ]
            est_vel = estimated_states[
                frame_idx,
                agent_idx,
                2:4,
            ]

            # Current positions.
            ax.scatter(
                true_pos[0],
                true_pos[1],
                marker="o",
                s=90,
                label=(
                    f"Agent {agent_idx} true"
                ),
            )
            ax.scatter(
                est_pos[0],
                est_pos[1],
                marker="x",
                s=100,
                linewidths=2.0,
                label=(
                    f"Agent {agent_idx} estimated"
                ),
            )

            # Connect current estimate to truth.
            ax.plot(
                [
                    true_pos[0],
                    est_pos[0],
                ],
                [
                    true_pos[1],
                    est_pos[1],
                ],
                linestyle=":",
                alpha=0.7,
            )

            # Velocity arrows.
            ax.arrow(
                true_pos[0],
                true_pos[1],
                velocity_scale
                * true_vel[0],
                velocity_scale
                * true_vel[1],
                width=0.002,
                length_includes_head=True,
                alpha=0.75,
            )
            ax.arrow(
                est_pos[0],
                est_pos[1],
                velocity_scale
                * est_vel[0],
                velocity_scale
                * est_vel[1],
                width=0.002,
                length_includes_head=True,
                alpha=0.75,
                linestyle="--",
            )

            # Short tails.
            true_tail = true_states[
                tail_start:
                frame_idx + 1,
                agent_idx,
                :2,
            ]
            est_tail = estimated_states[
                tail_start:
                frame_idx + 1,
                agent_idx,
                :2,
            ]

            ax.plot(
                true_tail[:, 0],
                true_tail[:, 1],
                linestyle="-",
                alpha=0.55,
            )
            ax.plot(
                est_tail[:, 0],
                est_tail[:, 1],
                linestyle="--",
                alpha=0.55,
            )

            pos_error = np.linalg.norm(
                est_pos - true_pos
            )
            vel_error = np.linalg.norm(
                est_vel - true_vel
            )

            error_lines.append(
                (
                    f"A{agent_idx}: "
                    f"|Δp|={pos_error:.5f}, "
                    f"|Δv|={vel_error:.5f}"
                )
            )

        ax.text(
            0.02,
            0.98,
            "\n".join(
                error_lines
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round",
                "alpha": 0.75,
            },
        )

        ax.legend(
            loc="lower right",
            fontsize=8,
        )

    animation = (
        mpl_animation.FuncAnimation(
            fig,
            draw_frame,
            frames=num_frames,
            interval=(
                1000.0 / float(fps)
            ),
            repeat=True,
        )
    )

    base_name = (
        f"agent{ego_agent_idx}_"
        f"estimator_env{debug_env_idx}"
    )
    outputs = {}

    if save_gif:
        gif_path = (
            output_dir
            / f"{base_name}.gif"
        )
        animation.save(
            gif_path,
            writer=(
                mpl_animation.PillowWriter(
                    fps=fps
                )
            ),
        )
        outputs[
            "gif"
        ] = gif_path

    if (
        save_mp4
        and mpl_animation.writers.is_available(
            "ffmpeg"
        )
    ):
        mp4_path = (
            output_dir
            / f"{base_name}.mp4"
        )
        animation.save(
            mp4_path,
            writer=(
                mpl_animation.FFMpegWriter(
                    fps=fps,
                    bitrate=1800,
                )
            ),
        )
        outputs[
            "mp4"
        ] = mp4_path

    plt.close(
        fig
    )

    if wandb_run is not None:
        if "mp4" in outputs:
            wandb.log(
                {
                    "animation/agent0_shadow_estimator": (
                        wandb.Video(
                            str(
                                outputs[
                                    "mp4"
                                ]
                            ),
                            fps=fps,
                            format="mp4",
                        )
                    )
                }
            )
        elif "gif" in outputs:
            # Keep the GIF in the W&B run files even if ffmpeg is unavailable.
            wandb.save(
                str(
                    outputs[
                        "gif"
                    ]
                ),
                base_path=str(
                    output_dir
                ),
                policy="now",
            )

    return outputs
