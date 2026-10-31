import jax

from dropout import dropout
from feedforward import forward as ffwd_forward
from feedforward import init as ffwd_init
from layernorm import forward as ln_forward
from layernorm import init as ln_init
from multihead import forward as mh_forward
from multihead import init as mh_init


def init(key, n_embed, n_head):
    keys = jax.random.split(key, 2)
    head_size = n_embed // n_head
    params = {
        "mh": mh_init(keys[0], n_embed, n_head, head_size),
        "ffwd": ffwd_init(keys[1], n_embed),
        "ln1": ln_init(n_embed),
        "ln2": ln_init(n_embed),
    }
    return params


def forward(x, params, n_embed, n_head, rate, key=None, is_training=False):
    head_size = n_embed // n_head
    attn_key = None
    ffwd_key = None
    if key is not None:
        attn_key, ffwd_key = jax.random.split(key)
    x = x + mh_forward(
        ln_forward(x, params["ln1"]),
        params["mh"],
        head_size,
        rate,
        attn_key,
        is_training,
    )
    x = x + dropout(
        ffwd_forward(ln_forward(x, params["ln2"]), params["ffwd"]),
        rate,
        ffwd_key,
        is_training,
    )
    return x
