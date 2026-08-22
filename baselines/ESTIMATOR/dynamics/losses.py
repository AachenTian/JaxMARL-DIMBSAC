import jax.numpy as jnp


def gaussian_nll_loss(mean, log_var, target):
    """Diagonal-Gaussian NLL in normalized target space."""
    error = target - mean
    per_element_nll = 0.5 * (
        jnp.square(error) * jnp.exp(-log_var) + log_var
    )
    loss = jnp.mean(per_element_nll)
    return loss, {
        "nll": loss,
        "mean_log_var": jnp.mean(log_var),
    }
