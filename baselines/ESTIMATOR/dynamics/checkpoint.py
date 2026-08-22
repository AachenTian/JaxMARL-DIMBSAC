import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from dynamics.ensemble import (
    DynamicsEnsembleState,
    build_independent_dynamics_ensembles,
    init_dynamics_ensemble,
    replace_ensemble_member,
)
from dynamics.model import ProbabilisticDynamicsModel
from dynamics.normalization import NormalizationStats


def save_local_dynamics_checkpoint(
    independent_ensembles,
    cfg,
):
    """
    Save inference-only local dynamics checkpoint.

    We save:
      - one msgpack parameter file per agent / ensemble member,
      - per-agent input / target normalization statistics,
      - model/environment metadata.

    Optimizer states are intentionally not required for estimator rollout.
    """
    checkpoint_dir = Path(
        str(cfg.DYNAMICS.CHECKPOINT_DIR)
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_agents = len(
        independent_ensembles.agents
    )
    ensemble_size = int(
        cfg.DYNAMICS.ENSEMBLE_SIZE
    )

    metadata = {
        "format": "local_dynamics_inference_v1",
        "env_name": str(cfg.ENV.NAME),
        "action_type": str(cfg.ENV.ACTION_TYPE),
        "contact_force": float(
            cfg.ENV.CONTACT_FORCE
        ),
        "episode_horizon": int(
            cfg.ENV.EPISODE_HORIZON
        ),
        "input_dim": 9,
        "output_dim": 4,
        "num_agents": num_agents,
        "ensemble_size": ensemble_size,
        "hidden_dims": [
            int(x)
            for x in cfg.DYNAMICS.HIDDEN_DIMS
        ],
        "log_var_min": float(
            cfg.DYNAMICS.LOG_VAR_MIN
        ),
        "log_var_max": float(
            cfg.DYNAMICS.LOG_VAR_MAX
        ),
        "state_layout": [
            "px",
            "py",
            "vx",
            "vy",
        ],
        "input_layout": [
            "px",
            "py",
            "vx",
            "vy",
            "a0",
            "a1",
            "a2",
            "a3",
            "a4",
        ],
    }

    for agent_idx, ensemble_state in enumerate(
        independent_ensembles.agents
    ):
        agent_dir = (
            checkpoint_dir
            / f"agent_{agent_idx}"
        )
        agent_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        reference_member = (
            ensemble_state.members[0]
        )

        np.savez(
            agent_dir / "normalization_stats.npz",
            input_mean=np.asarray(
                reference_member.input_stats.mean
            ),
            input_std=np.asarray(
                reference_member.input_stats.std
            ),
            target_mean=np.asarray(
                reference_member.target_stats.mean
            ),
            target_std=np.asarray(
                reference_member.target_stats.std
            ),
        )

        for member_idx, member_state in enumerate(
            ensemble_state.members
        ):
            member_path = (
                agent_dir
                / f"member_{member_idx}.msgpack"
            )
            member_path.write_bytes(
                serialization.to_bytes(
                    member_state.model.params
                )
            )

    metadata_path = (
        checkpoint_dir / "metadata.json"
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return checkpoint_dir


def load_local_dynamics_checkpoint(
    checkpoint_dir,
):
    """
    Restore local dynamics ensembles for inference.

    A fresh model/template is created from metadata, then each saved parameter
    tree is restored into the template. Normalization statistics are restored
    exactly from the checkpoint.
    """
    checkpoint_dir = Path(
        checkpoint_dir
    )
    metadata_path = (
        checkpoint_dir / "metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Dynamics metadata not found: "
            f"{metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    model = ProbabilisticDynamicsModel(
        output_dim=int(
            metadata["output_dim"]
        ),
        hidden_dims=tuple(
            int(x)
            for x in metadata[
                "hidden_dims"
            ]
        ),
        log_var_min=float(
            metadata["log_var_min"]
        ),
        log_var_max=float(
            metadata["log_var_max"]
        ),
    )

    input_dim = int(
        metadata["input_dim"]
    )
    ensemble_size = int(
        metadata["ensemble_size"]
    )
    num_agents = int(
        metadata["num_agents"]
    )

    agent_ensembles = []

    # Initialization RNG only creates compatible parameter/optimizer templates.
    # The saved parameter values fully replace the randomly initialized params.
    base_key = jax.random.PRNGKey(0)

    for agent_idx in range(
        num_agents
    ):
        agent_dir = (
            checkpoint_dir
            / f"agent_{agent_idx}"
        )
        stats_path = (
            agent_dir
            / "normalization_stats.npz"
        )

        if not stats_path.exists():
            raise FileNotFoundError(
                "Dynamics normalization stats not found: "
                f"{stats_path}"
            )

        stats = np.load(
            stats_path
        )
        input_stats = NormalizationStats(
            mean=jnp.asarray(
                stats["input_mean"]
            ),
            std=jnp.asarray(
                stats["input_std"]
            ),
        )
        target_stats = NormalizationStats(
            mean=jnp.asarray(
                stats["target_mean"]
            ),
            std=jnp.asarray(
                stats["target_std"]
            ),
        )

        agent_key = jax.random.fold_in(
            base_key,
            agent_idx,
        )
        ensemble_state = init_dynamics_ensemble(
            rng=agent_key,
            model=model,
            ensemble_size=ensemble_size,
            input_dim=input_dim,
            input_stats=input_stats,
            target_stats=target_stats,
            learning_rate=0.0,
        )

        for member_idx in range(
            ensemble_size
        ):
            member_path = (
                agent_dir
                / f"member_{member_idx}.msgpack"
            )

            if not member_path.exists():
                raise FileNotFoundError(
                    "Dynamics member checkpoint not found: "
                    f"{member_path}"
                )

            member_state = (
                ensemble_state.members[
                    member_idx
                ]
            )

            restored_params = (
                serialization.from_bytes(
                    member_state.model.params,
                    member_path.read_bytes(),
                )
            )

            restored_member = (
                member_state._replace(
                    model=(
                        member_state.model.replace(
                            params=restored_params
                        )
                    )
                )
            )

            ensemble_state = (
                replace_ensemble_member(
                    ensemble_state=ensemble_state,
                    member_idx=member_idx,
                    new_member_state=restored_member,
                )
            )

        agent_ensembles.append(
            ensemble_state
        )

    return (
        build_independent_dynamics_ensembles(
            agent_ensembles
        ),
        model,
        metadata,
    )
