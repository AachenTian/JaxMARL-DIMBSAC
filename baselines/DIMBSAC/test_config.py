import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="simple_spread",
)
def main(cfg: DictConfig):

    print("Full configuration")
    print("------------------------------")
    print(
        OmegaConf.to_yaml(cfg)
    )

    print("\nSelected values")
    print("------------------------------")

    print(
        "env:",
        cfg.ENV.NAME,
    )

    print(
        "num envs:",
        cfg.ENV.NUM_ENVS,
    )

    print(
        "gamma:",
        cfg.SAC.GAMMA,
    )

    print(
        "dynamics hidden dims:",
        cfg.DYNAMICS.HIDDEN_DIMS,
    )

    print(
        "ensemble size:",
        cfg.DYNAMICS.ENSEMBLE_SIZE,
    )

    assert (
        cfg.DYNAMICS.ENSEMBLE_SIZE
        == 5
    )

    print("\n==============================")
    print("HYDRA CONFIG TEST PASSED")
    print("==============================")


if __name__ == "__main__":
    main()