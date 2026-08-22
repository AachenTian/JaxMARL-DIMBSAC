import jax.numpy as jnp


def compute_relative_error_metrics(
    prediction,
    target,
    target_std,
    eps: float = 1e-8,
):
    """
    Compute physical and relative RMSE diagnostics.

    Metrics:
      1) nrmse_std:
         physical RMSE divided by the training-target standard deviation.
         Because the target is normalized by this same std during training,
         this is the RMSE expressed in normalized target units.

      2) zero_baseline_ratio:
         model physical RMSE divided by the RMSE of predicting delta_x = 0.
         Values < 1 mean the learned model improves over the zero-delta
         baseline. improvement_vs_zero = 1 - ratio.

    MAPE is intentionally not used because delta-position / delta-velocity
    targets frequently cross or approach zero.
    """
    error = target - prediction
    squared_error = jnp.square(error)

    per_dim_rmse = jnp.sqrt(
        jnp.mean(
            squared_error,
            axis=0,
        )
    )
    total_rmse = jnp.sqrt(
        jnp.mean(
            squared_error
        )
    )

    safe_target_std = jnp.maximum(
        target_std,
        eps,
    )
    per_dim_nrmse_std = (
        per_dim_rmse
        / safe_target_std
    )

    zero_per_dim_rmse = jnp.sqrt(
        jnp.mean(
            jnp.square(target),
            axis=0,
        )
    )
    safe_zero_per_dim_rmse = jnp.maximum(
        zero_per_dim_rmse,
        eps,
    )
    per_dim_zero_baseline_ratio = (
        per_dim_rmse
        / safe_zero_per_dim_rmse
    )

    return {
        "physical_rmse": total_rmse,
        "per_dim_rmse": per_dim_rmse,
        "per_dim_nrmse_std": per_dim_nrmse_std,
        "per_dim_nrmse_std_pct": (
            100.0 * per_dim_nrmse_std
        ),
        "zero_per_dim_rmse": zero_per_dim_rmse,
        "per_dim_zero_baseline_ratio": (
            per_dim_zero_baseline_ratio
        ),
        "per_dim_improvement_vs_zero_pct": (
            100.0
            * (
                1.0
                - per_dim_zero_baseline_ratio
            )
        ),
    }
