import jax
import jax.numpy as jnp
import numpy as np
import pytest

import block


@pytest.mark.parametrize("width,heads", [(16, 2), (32, 4)])
def test_forward_preserves_shape_eager_and_jit(width, heads):
    params = block.init(jax.random.key(0), width, heads)
    x = jax.random.normal(jax.random.key(1), (2, 3, width))
    eager = block.forward(x, params, width, heads, key=None, rate=0.2)
    compiled = jax.jit(
        lambda x, params: block.forward(x, params, width, heads, key=None, rate=0.2)
    )(x, params)
    assert compiled.shape == x.shape
    assert np.isfinite(np.asarray(compiled)).all()
    np.testing.assert_allclose(compiled, eager, rtol=1e-5, atol=1e-6)


def test_zero_branches_preserve_input_and_identity_gradient():
    params = jax.tree.map(jnp.zeros_like, block.init(jax.random.key(0), 32, 4))
    x = jax.random.normal(jax.random.key(1), (2, 3, 32))
    tangent = jnp.ones_like(x)
    forward = jax.jit(lambda x: block.forward(x, params, 32, 4, key=None, rate=0.2))
    output, output_tangent = jax.jvp(forward, (x,), (tangent,))
    np.testing.assert_array_equal(output, x)
    np.testing.assert_allclose(output_tangent, tangent, rtol=0, atol=1e-6)


def test_attention_and_mlp_receive_distinct_dropout_keys(monkeypatch):
    seen_keys = {}

    def record_attention(x, params, head_size, rate, key=None, is_training=False):
        assert is_training is True
        assert rate == 0.2
        seen_keys["attention"] = np.asarray(jax.random.key_data(key))
        return jnp.zeros_like(x)

    def record_dropout(x, rate, key, is_training=False):
        assert is_training is True
        assert rate == 0.2
        seen_keys["mlp"] = np.asarray(jax.random.key_data(key))
        return x

    monkeypatch.setattr(block, "mh_forward", record_attention)
    monkeypatch.setattr(block, "dropout", record_dropout)
    params = block.init(jax.random.key(0), 16, 4)
    block.forward(
        jnp.ones((2, 3, 16)), params, 16, 4,
        rate=0.2, key=jax.random.key(1), is_training=True,
    )
    assert not np.array_equal(seen_keys["attention"], seen_keys["mlp"])
