import jax
import jax.numpy as jnp
import optax

from dynamics.losses import (
    gaussian_nll_loss,
)
from dynamics.model import (
    ProbabilisticDynamicsModel,
)


BATCH_SIZE = 64
INPUT_DIM = 9
OUTPUT_DIM = 4


def tree_difference_norm(
    params_before,
    params_after,
):
    leaves_before = jax.tree_util.tree_leaves(
        params_before
    )

    leaves_after = jax.tree_util.tree_leaves(
        params_after
    )

    squared_difference = 0.0

    for before, after in zip(
        leaves_before,
        leaves_after,
    ):
        squared_difference += jnp.sum(
            jnp.square(
                after - before
            )
        )

    return jnp.sqrt(
        squared_difference
    )


def main():

    rng = jax.random.PRNGKey(0)

    rng, input_rng, target_rng = (
        jax.random.split(
            rng,
            3,
        )
    )

    # Synthetic normalized data for network-level testing.
    inputs = jax.random.normal(
        input_rng,
        shape=(
            BATCH_SIZE,
            INPUT_DIM,
        ),
    )

    targets = jax.random.normal(
        target_rng,
        shape=(
            BATCH_SIZE,
            OUTPUT_DIM,
        ),
    )

    model = ProbabilisticDynamicsModel(
        output_dim=OUTPUT_DIM,
        hidden_dims=(256, 256),
        log_var_min=-10.0,
        log_var_max=2.0,
    )

    rng, init_rng = jax.random.split(
        rng
    )

    variables = model.init(
        init_rng,
        inputs,
    )

    params = variables["params"]

    mean, log_var = model.apply(
        {"params": params},
        inputs,
    )

    variance = jnp.exp(
        log_var
    )

    print("Probabilistic dynamics model")
    print("------------------------------")
    print("mean shape:", mean.shape)
    print("log_var shape:", log_var.shape)
    print(
        "variance min:",
        float(jnp.min(variance)),
    )
    print(
        "variance max:",
        float(jnp.max(variance)),
    )

    assert mean.shape == (
        BATCH_SIZE,
        OUTPUT_DIM,
    )

    assert log_var.shape == (
        BATCH_SIZE,
        OUTPUT_DIM,
    )

    assert jnp.all(
        jnp.isfinite(mean)
    )

    assert jnp.all(
        jnp.isfinite(log_var)
    )

    assert jnp.all(
        variance > 0.0
    )

    # ---------------------------------------------------------
    # Gaussian NLL
    # ---------------------------------------------------------

    loss, metrics = gaussian_nll_loss(
        mean=mean,
        log_var=log_var,
        target=targets,
    )

    print("\nLoss")
    print("------------------------------")
    print(
        "NLL:",
        float(loss),
    )
    print(
        "MSE:",
        float(metrics["mse"]),
    )
    print(
        "mean variance:",
        float(
            metrics["mean_variance"]
        ),
    )

    assert jnp.isfinite(loss)

    # ---------------------------------------------------------
    # Gradient test
    # ---------------------------------------------------------

    optimizer = optax.adam(
        1e-3
    )

    opt_state = optimizer.init(
        params
    )

    def loss_fn(model_params):

        pred_mean, pred_log_var = (
            model.apply(
                {
                    "params":
                        model_params
                },
                inputs,
            )
        )

        model_loss, _ = (
            gaussian_nll_loss(
                mean=pred_mean,
                log_var=pred_log_var,
                target=targets,
            )
        )

        return model_loss

    loss_before, grads = (
        jax.value_and_grad(
            loss_fn
        )(params)
    )

    updates, opt_state = (
        optimizer.update(
            grads,
            opt_state,
            params,
        )
    )

    new_params = optax.apply_updates(
        params,
        updates,
    )

    loss_after = loss_fn(
        new_params
    )

    parameter_change = (
        tree_difference_norm(
            params,
            new_params,
        )
    )

    print("\nGradient step")
    print("------------------------------")
    print(
        "loss before:",
        float(loss_before),
    )
    print(
        "loss after:",
        float(loss_after),
    )
    print(
        "parameter change:",
        float(parameter_change),
    )

    assert parameter_change > 0.0

    assert jnp.isfinite(
        loss_after
    )

    print("\n==============================")
    print("ALL DYNAMICS MODEL TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()