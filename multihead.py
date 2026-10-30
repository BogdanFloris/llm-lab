import jax
import jax.numpy as jnp

from attention import forward as att_forward
from attention import init as att_init


def init(key, n_embed, no_heads, head_w):
    keys = jax.random.split(key, no_heads)
    params = {}
    for i in range(no_heads):
        params[i] = att_init(keys[i], n_embed, head_w // no_heads)
    return params


def forward(x, params, no_heads, head_w):
    return jnp.concatenate([att_forward(x, p, head_w // no_heads) for p in params.values()], axis=2)
