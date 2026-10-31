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


def forward(x, params, head_size):
    return jnp.concatenate(
        [att_forward(x, p, head_size) for p in params.values()], axis=2
    )
