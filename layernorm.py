import jax.numpy as jnp

EPS = 1e-5


def init(dim):
    params = {"gamma": jnp.ones((dim,)), "beta": jnp.zeros((dim,))}
    return params


def forward(x, params):
    xmean = x.mean(2, keepdims=True)
    xvar = x.var(2, keepdims=True)
    xhat = (x - xmean) / jnp.sqrt(xvar + EPS)
    return params["gamma"] * xhat + params["beta"]
