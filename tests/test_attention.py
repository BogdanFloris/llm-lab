import jax
import jax.numpy as jnp
import numpy as np

import attention


def test_forward_shapes_eager_and_jit():
    params = attention.init_params(jax.random.key(0), 65)
    for batch_size, length in [(1, 1), (2, 3), (3, attention.T_MAX)]:
        tokens = jnp.zeros((batch_size, length), dtype=jnp.int32)
        eager = attention.forward(params, tokens)
        compiled = jax.jit(attention.forward)(params, tokens)
        assert eager.shape == (batch_size, length, 65)
        assert np.isfinite(np.asarray(eager)).all()
        np.testing.assert_allclose(eager, compiled, rtol=1e-5, atol=1e-5)


def test_future_tokens_cannot_change_prefix_logits():
    params = attention.init_params(jax.random.key(0), 65)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    forward = jax.jit(attention.forward)
    original = forward(params, tokens)
    changed = forward(params, tokens.at[0, 3].set(7))
    np.testing.assert_allclose(original[:, :3], changed[:, :3], rtol=1e-5, atol=1e-5)


def test_head_width_can_differ_from_embedding_width(monkeypatch):
    monkeypatch.setattr(attention, "HEAD_W", 16)
    params = attention.init_params(jax.random.key(0), 65)
    tokens = jnp.zeros((2, 3), dtype=jnp.int32)
    assert jax.jit(attention.forward)(params, tokens).shape == (2, 3, 65)


def test_loss_is_finite_scalar():
    params = attention.init_params(jax.random.key(0), 65)
    tokens = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    targets = jnp.array([[2, 3, 4]], dtype=jnp.int32)
    loss = jax.jit(attention.loss_fn)(params, tokens, targets)
    assert loss.shape == ()
    assert np.isfinite(float(loss))
