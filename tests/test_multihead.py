import jax
import jax.numpy as jnp
import numpy as np

import multihead


def test_heads_receive_independent_dropout_keys(monkeypatch):
    params = multihead.init(jax.random.key(0), 16, 4, 4)
    seen_keys = []

    def record_head(x, params, head_w, rate, key=None, is_training=False):
        assert is_training is True
        assert rate == 0.2
        seen_keys.append(np.asarray(jax.random.key_data(key)))
        return jnp.zeros((*x.shape[:2], head_w))

    monkeypatch.setattr(multihead, "att_forward", record_head)
    x = jnp.ones((2, 3, 16))
    result = multihead.forward(x, params, 4, rate=0.2, key=jax.random.key(1), is_training=True)
    assert result.shape == x.shape
    assert len(seen_keys) == 4
    for i, key in enumerate(seen_keys):
        for other in seen_keys[i + 1:]:
            assert not np.array_equal(key, other)
