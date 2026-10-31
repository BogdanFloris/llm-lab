import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dropout import dropout


@pytest.mark.parametrize("rate,is_training", [(0.2, False), (0.0, True)])
def test_identity_paths_need_no_key(rate, is_training):
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    output = dropout(x, rate=rate, key=None, is_training=is_training)
    np.testing.assert_array_equal(output, x)


def test_retained_activations_are_scaled():
    x = jnp.full((2, 8, 16), 10.0)
    output = np.asarray(dropout(x, rate=0.2, key=jax.random.key(0), is_training=True))
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert np.all((output == 0.0) | (output == 12.5))
    assert np.any(output == 0.0)
    assert np.any(output == 12.5)


def test_key_reproducibility_and_independence():
    x = jnp.ones((2, 8, 16))
    key1, key2 = jax.random.split(jax.random.key(0))
    first = dropout(x, rate=0.5, key=key1, is_training=True)
    repeated = dropout(x, rate=0.5, key=key1, is_training=True)
    different = dropout(x, rate=0.5, key=key2, is_training=True)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_jit_with_static_mode_and_rate():
    compiled = jax.jit(dropout, static_argnames=("rate", "is_training"))
    x = jnp.ones((2, 3, 16))
    key = jax.random.key(0)
    np.testing.assert_array_equal(
        compiled(x, rate=0.2, key=key, is_training=True),
        dropout(x, rate=0.2, key=key, is_training=True),
    )
    np.testing.assert_array_equal(compiled(x, rate=0.2, key=None, is_training=False), x)


def test_gradients_follow_the_same_mask_and_scaling():
    x = jnp.full((2, 8, 16), 10.0)
    key = jax.random.key(0)
    output = dropout(x, rate=0.2, key=key, is_training=True)
    grads = jax.jit(jax.grad(
        lambda inputs: dropout(inputs, rate=0.2, key=key, is_training=True).sum()
    ))(x)
    np.testing.assert_array_equal(grads, output / 10.0)
