import jax
import jax.numpy as jnp

from attention import forward as att_forward
from attention import init as att_init


def init(key, n_embed, n_head, head_size):
    keys = jax.random.split(key, n_head)
    params = {}
    for i in range(n_head):
        params[i] = att_init(keys[i], n_embed, head_size)
    return params


def forward(x, params, head_size, rate, key=None, is_training=False):
    keys = [None] * len(params)
    if key is not None:
        keys = jax.random.split(key, len(params))
    return jnp.concatenate(
        [
            att_forward(x, p, head_size, rate, keys[i], is_training)
            for i, p in enumerate(params.values())
        ],
        axis=2,
    )
