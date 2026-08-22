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
    dummy_input = jnp.zeros((1, input_dim), dtype=jnp.float32)
    variables = model.init(rng, dummy_input)
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


def update_dynamics_model(state, normalized_inputs, normalized_targets):
    def loss_fn(params):
        mean, log_var = state.model.apply_fn(
            {"params": params}, normalized_inputs
        )
        return gaussian_nll_loss(mean, log_var, normalized_targets)

    (loss_aux, grads) = jax.value_and_grad(loss_fn, has_aux=True)(
        state.model.params
    )
    (loss, metrics) = loss_aux
    new_model_state = state.model.apply_gradients(grads=grads)
    new_state = state._replace(model=new_model_state)
    return new_state, {**metrics, "loss": loss}


def predict_dynamics(state, dynamics_inputs):
    normalized_inputs = normalize(dynamics_inputs, state.input_stats)
    normalized_mean, log_var = state.model.apply_fn(
        {"params": state.model.params}, normalized_inputs
    )
    normalized_variance = jnp.exp(log_var)
    mean_delta_x = denormalize(normalized_mean, state.target_stats)
    variance_delta_x = denormalize_variance(
        normalized_variance, state.target_stats
    )
    return mean_delta_x, variance_delta_x


def evaluate_dynamics_model(state, dynamics_inputs, delta_x_targets):
    normalized_inputs = normalize(dynamics_inputs, state.input_stats)
    normalized_targets = normalize(delta_x_targets, state.target_stats)

    normalized_mean, log_var = state.model.apply_fn(
        {"params": state.model.params}, normalized_inputs
    )

    normalized_error = normalized_targets - normalized_mean
    per_element_nll = 0.5 * (
        jnp.square(normalized_error) * jnp.exp(-log_var) + log_var
    )

    physical_mean = denormalize(normalized_mean, state.target_stats)
    physical_variance = denormalize_variance(
        jnp.exp(log_var), state.target_stats
    )
    physical_error = delta_x_targets - physical_mean
    physical_squared_error = jnp.square(physical_error)

    per_dim_mse = jnp.mean(physical_squared_error, axis=0)
    per_dim_rmse = jnp.sqrt(per_dim_mse)
    physical_rmse = jnp.sqrt(jnp.mean(physical_squared_error))

    return {
        "nll": jnp.mean(per_element_nll),
        "per_dim_nll": jnp.mean(per_element_nll, axis=0),
        "physical_rmse": physical_rmse,
        "per_dim_rmse": per_dim_rmse,
        "mean_variance_per_dim": jnp.mean(physical_variance, axis=0),
    }
