import jax
import jax.numpy as jnp

from dropout import dropout


def init(key, n_embed, head_w):
    keys = jax.random.split(key, 3)
    params = {
        "q": jax.random.normal(keys[0], (n_embed, head_w)) * 0.02,
        "k": jax.random.normal(keys[1], (n_embed, head_w)) * 0.02,
        "v": jax.random.normal(keys[2], (n_embed, head_w)) * 0.02,
    }
    return params


def forward(x, params, head_w, rate, key=None, is_training=False):
    _, t, _ = x.shape
    q = x @ params["q"]  # (B, T, HEAD_W)
    k = x @ params["k"]  # (B, T, HEAD_W)
    v = x @ params["v"]  # (B, T, HEAD_W)
    wei = q @ k.transpose(0, 2, 1) * head_w**-0.5  # (B, T, T)
    mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
    wei = jnp.where(mask, wei, -jnp.inf)
    wei = dropout(jax.nn.softmax(wei), rate, key, is_training)
    return wei @ v  # (B, T, HEAD_W)
