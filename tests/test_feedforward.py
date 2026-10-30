import jax
import jax.numpy as jnp
import numpy as np
import pytest

import feedforward


@pytest.mark.parametrize("width", [16, 32])
def test_parameter_shapes_and_zero_biases(width):
    params = feedforward.init(jax.random.key(0), width)
    assert params["w1"].shape == (width, 4 * width)
    assert params["w2"].shape == (4 * width, width)
    np.testing.assert_array_equal(params["b1"], np.zeros(4 * width))
    np.testing.assert_array_equal(params["b2"], np.zeros(width))


def test_forward_is_position_wise_under_jit():
    params = feedforward.init(jax.random.key(0), 16)
    x = jax.random.normal(jax.random.key(1), (2, 3, 16))
    forward = jax.jit(feedforward.forward)
    original = forward(x, params)
    assert original.shape == x.shape
    np.testing.assert_allclose(
        original, feedforward.forward(x, params), rtol=1e-5, atol=1e-6,
    )
    changed = forward(x.at[1, 2].add(1.0), params)
    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(original[1, :2], changed[1, :2])
    assert not np.array_equal(original[1, 2], changed[1, 2])


def test_relu_and_biases_with_known_parameters():
    params = {
        "w1": jnp.array([[1.0, -1.0, 0.0, 0.0]]),
        "b1": jnp.array([0.0, 0.0, 1.0, -1.0]),
        "w2": jnp.array([[1.0], [1.0], [2.0], [9.0]]),
        "b2": jnp.array([3.0]),
    }
    x = jnp.array([[[-2.0], [0.0], [2.0]]])
    expected = jnp.array([[[7.0], [5.0], [7.0]]])
    np.testing.assert_array_equal(jax.jit(feedforward.forward)(x, params), expected)
