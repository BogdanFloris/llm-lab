import jax
import jax.numpy as jnp


def dropout(x, rate, key, is_training=False):
    if not is_training or rate == 0.0:
        return x
    keep_rate = 1 - rate
    mask = jax.random.bernoulli(key, p=keep_rate, shape=x.shape)
    return jnp.where(mask, x / keep_rate, 0.0)

