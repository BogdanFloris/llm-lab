import jax
import jax.numpy as jnp
import optax

from block import forward as b_forward
from block import init as b_init
from layernorm import forward as ln_forward
from layernorm import init as ln_init

optimizer = optax.adam(learning_rate=1e-2)
N_EMBED = 32
T_MAX = 8
N_HEAD = 4
DROPOUT_RATE = 0.2


def init_params(key, vocab_size):
    """Return a parameter pytree. Start with small random logits."""
    keys = jax.random.split(key, 6)
    blocks = {
        "block1": b_init(keys[0], N_EMBED, N_HEAD),
        "block2": b_init(keys[1], N_EMBED, N_HEAD),
        "block3": b_init(keys[2], N_EMBED, N_HEAD),
    }
    params = {
        "token_embed": jax.random.normal(keys[3], (vocab_size, N_EMBED)) * 0.02,
        "pos_embed": jax.random.normal(keys[4], (T_MAX, N_EMBED)) * 0.02,
        "blocks": blocks,
        "ln": ln_init(N_EMBED),
        "lm_head": jax.random.normal(keys[5], (N_EMBED, vocab_size)) * 0.02,
    }
    return params


def forward(params, tokens, key=None, is_training=False):
    _, t = tokens.shape
    tok_emb = params["token_embed"][tokens]  # (B, T, N_EMBED)
    pos_emb = params["pos_embed"][jnp.arange(t)]  # (T, N_EMBED)
    x = tok_emb + pos_emb  # (B, T, N_EMBED)
    for block_params in params["blocks"].values():
        block_key = None
        if key is not None:
            key, block_key = jax.random.split(key)
        x = b_forward(x, block_params, N_EMBED, N_HEAD,  DROPOUT_RATE, block_key, is_training)
    x = ln_forward(x, params["ln"])
    logits = x @ params["lm_head"]
    return logits


def loss_fn(params, tokens, targets, key=None, is_training=False):
    """Return mean next-character cross-entropy as a scalar."""
    logits = forward(params, tokens, key, is_training)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
    return loss


@jax.jit
def train_step(params, opt_state, tokens, targets, key):
    """Return (updated_params, updated_opt_state, loss)."""
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets, key=key, is_training=True)
    updates, updated_opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, updated_opt_state, loss
