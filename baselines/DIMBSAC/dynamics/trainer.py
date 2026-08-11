from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from dynamics.losses import gaussian_nll_loss
from dynamics.normalization import (
    NormalizationStats,
    denormalize,
    denormalize_variance,
    normalize,
)


class DynamicsTrainState(NamedTuple):
    """Training state for one probabilistic dynamics model."""

    model: TrainState
    input_stats: NormalizationStats
    target_stats: NormalizationStats


def init_dynamics_train_state(
    rng,
    model,
    input_dim: int,
    input_stats: NormalizationStats,
    target_stats: NormalizationStats,
    learning_rate: float,
):
    """Initialize one dynamics model and its optimizer."""

    dummy_input = jnp.zeros(
        (1, input_dim),
        dtype=jnp.float32,
    )

    variables = model.init(
        rng,
        dummy_input,
    )

    model_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adam(learning_rate),
    )

    return DynamicsTrainState(
        model=model_state,
        input_stats=input_stats,
        target_stats=target_stats,
    )


def update_dynamics_model(
    state: DynamicsTrainState,
    normalized_inputs,
    normalized_targets,
):
    """Perform one gradient update."""

    def loss_fn(params):

        mean, log_var = state.model.apply_fn(
            {"params": params},
            normalized_inputs,
        )

        return gaussian_nll_loss(
            mean=mean,
            log_var=log_var,
            target=normalized_targets,
        )

    (
        (loss, metrics),
        grads,
    ) = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(
        state.model.params
    )

    new_model_state = (
        state.model.apply_gradients(
            grads=grads
        )
    )

    new_state = state._replace(
        model=new_model_state
    )

    return new_state, metrics


def predict_dynamics(
    state: DynamicsTrainState,
    dynamics_inputs,
):
    """
    Predict delta_x in physical units.

    Returns:
        mean_delta_x
        variance_delta_x
    """

    normalized_inputs = normalize(
        dynamics_inputs,
        state.input_stats,
    )

    normalized_mean, log_var = (
        state.model.apply_fn(
            {"params": state.model.params},
            normalized_inputs,
        )
    )

    normalized_variance = jnp.exp(
        log_var
    )

    mean_delta_x = denormalize(
        normalized_mean,
        state.target_stats,
    )

    variance_delta_x = (
        denormalize_variance(
            normalized_variance,
            state.target_stats,
        )
    )

    return (
        mean_delta_x,
        variance_delta_x,
    )


def evaluate_dynamics_model(
    state,
    dynamics_inputs,
    delta_x_targets,
):
    """Evaluate dynamics prediction and variance calibration."""

    # Normalize inputs and targets using real training statistics.
    normalized_inputs = normalize(
        dynamics_inputs,
        state.input_stats,
    )

    normalized_targets = normalize(
        delta_x_targets,
        state.target_stats,
    )

    # Model outputs are in normalized target space.
    normalized_mean, log_var = (
        state.model.apply_fn(
            {"params": state.model.params},
            normalized_inputs,
        )
    )

    normalized_variance = jnp.exp(
        log_var
    )

    # --------------------------------------------------------
    # NLL diagnostics in normalized space
    # --------------------------------------------------------

    normalized_error = (
        normalized_targets
        - normalized_mean
    )

    standardized_squared_error = (
            jnp.square(normalized_error)
            * jnp.exp(-log_var)
    )

    mean_standardized_squared_error_per_dim = (
        jnp.mean(
            standardized_squared_error,
            axis=0,
        )
    )

    p95_standardized_squared_error_per_dim = (
        jnp.quantile(
            standardized_squared_error,
            0.95,
            axis=0,
        )
    )

    p99_standardized_squared_error_per_dim = (
        jnp.quantile(
            standardized_squared_error,
            0.99,
            axis=0,
        )
    )

    per_element_nll = 0.5 * (
        jnp.square(normalized_error)
        * jnp.exp(-log_var)
        + log_var
    )

    nll = jnp.mean(
        per_element_nll
    )

    per_dim_nll = jnp.mean(
        per_element_nll,
        axis=0,
    )

    # --------------------------------------------------------
    # Convert predictions back to physical units
    # --------------------------------------------------------

    physical_mean = denormalize(
        normalized_mean,
        state.target_stats,
    )

    physical_variance = (
        denormalize_variance(
            normalized_variance,
            state.target_stats,
        )
    )

    physical_error = (
        delta_x_targets
        - physical_mean
    )

    physical_squared_error = (
        jnp.square(
            physical_error
        )
    )

    physical_mse = jnp.mean(
        physical_squared_error
    )

    physical_rmse = jnp.sqrt(
        physical_mse
    )

    per_dim_mse = jnp.mean(
        physical_squared_error,
        axis=0,
    )

    per_dim_rmse = jnp.sqrt(
        per_dim_mse
    )

    # --------------------------------------------------------
    # Variance calibration diagnostics
    # --------------------------------------------------------

    mean_variance_per_dim = (
        jnp.mean(
            physical_variance,
            axis=0,
        )
    )

    error_variance_ratio = (
        per_dim_mse
        / (
            mean_variance_per_dim
            + 1e-8
        )
    )

    mean_log_var_per_dim = (
        jnp.mean(
            log_var,
            axis=0,
        )
    )

    min_log_var_per_dim = (
        jnp.min(
            log_var,
            axis=0,
        )
    )

    max_log_var_per_dim = (
        jnp.max(
            log_var,
            axis=0,
        )
    )

    return {
        "nll": nll,
        "per_dim_nll": per_dim_nll,
        "physical_mse": physical_mse,
        "physical_rmse": physical_rmse,
        "per_dim_mse": per_dim_mse,
        "per_dim_rmse": per_dim_rmse,
        "mean_physical_variance": (
            jnp.mean(
                physical_variance
            )
        ),
        "mean_variance_per_dim": (
            mean_variance_per_dim
        ),
        "error_variance_ratio": (
            error_variance_ratio
        ),
        "mean_log_var_per_dim": (
            mean_log_var_per_dim
        ),
        "min_log_var_per_dim": (
            min_log_var_per_dim
        ),
        "max_log_var_per_dim": (
            max_log_var_per_dim
        ),
        "mean_standardized_squared_error_per_dim": (
            mean_standardized_squared_error_per_dim
        ),
        "p95_standardized_squared_error_per_dim": (
            p95_standardized_squared_error_per_dim
        ),
        "p99_standardized_squared_error_per_dim": (
            p99_standardized_squared_error_per_dim
        ),
    }