import jax
import jax.numpy as jnp
import numpy as np

import multihead


def test_projection_and_output_dropout_have_independent_keys(monkeypatch):
    params = multihead.init(jax.random.key(0), 16, 4, 4)
    params["w"] = 2 * jnp.eye(16)
    params["b"] = jnp.arange(16, dtype=jnp.float32)
    seen_keys = []

    def record_head(x, params, head_w, rate, key=None, is_training=False):
        assert is_training is True
        assert rate == 0.2
        seen_keys.append(np.asarray(jax.random.key_data(key)))
        return jnp.ones((*x.shape[:2], head_w))

    def record_output_dropout(x, rate, key, is_training=False):
        assert is_training is True
        assert rate == 0.2
        assert len(seen_keys) == 4
        seen_keys.append(np.asarray(jax.random.key_data(key)))
        expected = jnp.broadcast_to(2 + params["b"], x.shape)
        np.testing.assert_array_equal(x, expected)
        return x

    monkeypatch.setattr(multihead, "att_forward", record_head)
    monkeypatch.setattr(multihead, "dropout", record_output_dropout)
    x = jnp.ones((2, 3, 16))
    result = multihead.forward(x, params, 4, rate=0.2, key=jax.random.key(1), is_training=True)
    assert result.shape == x.shape
    assert len(seen_keys) == 5
    for i, key in enumerate(seen_keys):
        for other in seen_keys[i + 1:]:
            assert not np.array_equal(key, other)


def test_six_heads_evaluate_without_a_key():
    params = multihead.init(jax.random.key(0), 24, 6, 4)
    x = jnp.ones((2, 3, 24))
    forward = jax.jit(
        lambda x, params, key: multihead.forward(
            x, params, 4, rate=0.2, key=key, is_training=False,
        )
    )
    expected = forward(x, params, None)
    assert expected.shape == x.shape
    assert np.isfinite(np.asarray(expected)).all()
    np.testing.assert_array_equal(expected, forward(x, params, jax.random.key(1)))
