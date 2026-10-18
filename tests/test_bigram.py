import jax
import jax.numpy as jnp
import numpy as np

from bigram import forward, init_params, loss_fn, optimizer, train_step


def test_shapes_eager_and_jit():
    params = init_params(jax.random.key(0), 65)
    for shape in [(2, 4), (1, 1), (3, 7)]:
        x = jnp.zeros(shape, dtype=jnp.int32)
        eager = forward(params, x)
        compiled = jax.jit(forward)(params, x)
        assert eager.shape == (*shape, 65)
        np.testing.assert_allclose(eager, compiled, atol=1e-6)
        loss = jax.jit(loss_fn)(params, x, jnp.ones_like(x))
        assert loss.shape == ()
        assert np.isfinite(float(loss))


def test_initial_loss_near_uniform():
    params = init_params(jax.random.key(0), 65)
    x = jax.random.randint(jax.random.key(1), (16, 32), 0, 65)
    y = jax.random.randint(jax.random.key(2), (16, 32), 0, 65)
    np.testing.assert_allclose(float(loss_fn(params, x, y)), np.log(65), atol=0.1)


def test_bigram_uses_only_current_token():
    params = init_params(jax.random.key(0), 8)
    x = jnp.array([[1, 2, 3, 4]])
    logits = forward(params, x)
    changed_future = forward(params, x.at[0, 3].set(7))
    np.testing.assert_allclose(logits[:, :3], changed_future[:, :3])
    changed_past = forward(params, x.at[0, 0].set(7))
    np.testing.assert_allclose(logits[:, 1:], changed_past[:, 1:])


def test_overfits_deterministic_batch():
    params = init_params(jax.random.key(0), 4)
    opt_state = optimizer.init(params)
    x = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    y = jnp.array([[1, 2, 3, 0]], dtype=jnp.int32)
    for _ in range(400):
        params, opt_state, loss = train_step(params, opt_state, x, y)
    assert float(loss) < 0.3
    np.testing.assert_array_equal(jnp.argmax(forward(params, x), axis=-1), y)
