import jax
import jax.numpy as jnp

from attention import forward as att_forward
from attention import init as att_init
from dropout import dropout


def init(key, n_embed, n_head, head_size):
    key, block_keys = jax.random.split(key, 2)
    keys = jax.random.split(block_keys, n_head)
    block_params = {}
    for i in range(n_head):
        block_params[i] = att_init(keys[i], n_embed, head_size)
    params = {
        "blocks": block_params,
        "w": jax.random.normal(key, (head_size * n_head, n_embed)) * 0.02,
        "b": jnp.zeros((n_embed,)),
    }
    return params


def forward(x, params, head_size, rate, key=None, is_training=False):
    keys = [None] * len(params["blocks"])
    if key is not None:
        key, block_key = jax.random.split(key, 2)
        keys = jax.random.split(block_key, len(params["blocks"]))
    x = jnp.concatenate(
        [
            att_forward(x, p, head_size, rate, keys[i], is_training)
            for i, p in enumerate(params["blocks"].values())
        ],
        axis=2,
    )
    x = x @ params["w"] + params["b"]
    return dropout(x, rate, key, is_training)
