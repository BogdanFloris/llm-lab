import jax
import jax.numpy as jnp


def init(key, size):
    keys = jax.random.split(key, 2)
    params = {
        "w1": jax.random.normal(keys[0], (size, size * 4)) * 0.02,
        "b1": jnp.zeros((size * 4,)),
        "w2": jax.random.normal(keys[1], (size * 4, size)) * 0.02,
        "b2": jnp.zeros((size,)),
    }
    return params


def forward(x, params):
    x = x @ params["w1"] + params["b1"]
    x = jax.nn.relu(x)
    return x @ params["w2"] + params["b2"]
