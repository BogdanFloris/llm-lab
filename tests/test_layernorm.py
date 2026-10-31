import jax
import jax.numpy as jnp
import numpy as np
import pytest

import layernorm


def test_parameter_shapes_and_initialization():
    params = layernorm.init(4)
    np.testing.assert_array_equal(params["gamma"], np.ones(4))
    np.testing.assert_array_equal(params["beta"], np.zeros(4))


def test_feature_statistics_and_jit():
    params = layernorm.init(4)
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    eager = layernorm.forward(x, params)
    compiled = jax.jit(layernorm.forward)(x, params)
    assert compiled.shape == x.shape
    np.testing.assert_allclose(compiled, eager, atol=1e-6)
    np.testing.assert_allclose(np.asarray(compiled).mean(axis=-1), 0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(compiled).var(axis=-1), 1, atol=2e-5)


def test_tokens_and_examples_are_independent():
    params = layernorm.init(4)
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    forward = jax.jit(layernorm.forward)
    original = forward(x, params)
    changed = forward(x.at[1, 2].add(jnp.array([10.0, -3.0, 2.0, 7.0])), params)
    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(original[1, :2], changed[1, :2])
    assert not np.array_equal(original[1, 2], changed[1, 2])


@pytest.mark.parametrize("width", [1, 4])
def test_constant_inputs_and_gradients_are_finite(width):
    params = layernorm.init(width)
    params["beta"] = jnp.arange(width, dtype=jnp.float32) + 2
    x = jnp.full((2, 3, width), 7.0)
    output = jax.jit(layernorm.forward)(x, params)
    np.testing.assert_array_equal(
        output, jnp.broadcast_to(params["beta"], x.shape),
    )
    grads = jax.jit(jax.grad(lambda inputs: layernorm.forward(inputs, params).sum()))(x)
    assert grads.shape == x.shape
    assert np.isfinite(np.asarray(grads)).all()


def test_learned_scale_and_bias_are_per_feature():
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    normalized = layernorm.forward(x, layernorm.init(4))
    params = {
        "gamma": jnp.array([2.0, -1.0, 0.5, 3.0]),
        "beta": jnp.array([1.0, 2.0, -1.0, 0.0]),
    }
    actual = jax.jit(layernorm.forward)(x, params)
    np.testing.assert_allclose(
        actual, normalized * params["gamma"] + params["beta"],
        rtol=1e-5, atol=1e-6,
    )
