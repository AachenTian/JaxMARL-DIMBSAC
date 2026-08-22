import os

import wandb
from omegaconf import OmegaConf


def init_wandb(
    cfg,
    experiment_name: str,
    group_name=None,
    extra_tags=None,
):
    """Initialize one W&B run from the single Hydra configuration."""
    if not bool(cfg.WANDB.ENABLED):
        return None

    configured_name = (
        None
        if cfg.WANDB.RUN_NAME is None
        else str(cfg.WANDB.RUN_NAME)
    )

    run_name = (
        os.environ.get("WANDB_NAME")
        or configured_name
        or experiment_name
    )

    tags = [
        str(tag)
        for tag in cfg.WANDB.TAGS
    ]

    if extra_tags is not None:
        tags.extend(
            str(tag)
            for tag in extra_tags
        )

    configured_group = (
        str(cfg.WANDB.GROUP)
        if group_name is None
        else str(group_name)
    )

    return wandb.init(
        entity=str(
            cfg.WANDB.ENTITY
        ),
        project=str(
            cfg.WANDB.PROJECT
        ),
        name=run_name,
        group=configured_group,
        tags=tags,
        mode=str(
            cfg.WANDB.MODE
        ),
        config=OmegaConf.to_container(
            cfg,
            resolve=True,
        ),
    )


def finish_wandb(run):
    if run is not None:
        run.finish()
