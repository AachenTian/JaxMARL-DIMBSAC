import jax.numpy as jnp


def gaussian_nll_loss(
    mean,
    log_var,
    target,
):
    """
    Gaussian negative log-likelihood loss.

    The constant term is omitted because it does not affect optimization.
    """

    log_var = jnp.clip(
        log_var,
        min=-6.0,
        max=2.0,
    )

    inv_var = jnp.exp(
        -log_var
    )


    elementwise_nll = 0.5 * (
        jnp.square(
            target - mean
        )
        * inv_var
        + log_var
    )

    loss = jnp.mean(
        elementwise_nll
    )

    variance = jnp.exp(
        log_var
    )

    mse = jnp.mean(
        jnp.square(
            target - mean
        )
    )

    metrics = {
        "nll_loss": loss,
        "mse": mse,
        "mean_variance": jnp.mean(
            variance
        ),
        "mean_log_var": jnp.mean(
            log_var
        ),
    }

    return loss, metrics