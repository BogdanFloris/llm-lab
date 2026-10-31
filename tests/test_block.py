import jax
import jax.numpy as jnp
import numpy as np
import pytest

import block


@pytest.mark.parametrize("width,heads", [(16, 2), (32, 4)])
def test_forward_preserves_shape_eager_and_jit(width, heads):
    params = block.init(jax.random.key(0), width, heads)
    x = jax.random.normal(jax.random.key(1), (2, 3, width))
    eager = block.forward(x, params, width, heads)
    compiled = jax.jit(lambda x, params: block.forward(x, params, width, heads))(x, params)
    assert compiled.shape == x.shape
    assert np.isfinite(np.asarray(compiled)).all()
    np.testing.assert_allclose(compiled, eager, rtol=1e-5, atol=1e-6)


def test_zero_branches_preserve_input_and_identity_gradient():
    params = jax.tree.map(jnp.zeros_like, block.init(jax.random.key(0), 32, 4))
    x = jax.random.normal(jax.random.key(1), (2, 3, 32))
    tangent = jnp.ones_like(x)
    forward = jax.jit(lambda x: block.forward(x, params, 32, 4))
    output, output_tangent = jax.jvp(forward, (x,), (tangent,))
    np.testing.assert_array_equal(output, x)
    np.testing.assert_allclose(output_tangent, tangent, rtol=0, atol=1e-6)
