import jax
import jax.numpy as jnp
import optax

from feedforward import forward as ffwd_forward
from feedforward import init as ffwd_init
from multihead import forward as mh_forward
from multihead import init as mh_init

optimizer = optax.adam(learning_rate=1e-2)
N_EMBED = 32
HEAD_W = 32
T_MAX = 8
NO_HEADS = 4


def init_params(key, vocab_size):
    """Return a parameter pytree. Start with small random logits."""
    keys = jax.random.split(key, 5)
    params = {
        "token_embed": jax.random.normal(keys[0], (vocab_size, N_EMBED)) * 0.02,
        "pos_embed": jax.random.normal(keys[1], (T_MAX, N_EMBED)) * 0.02,
        "mh_attention": mh_init(keys[2], N_EMBED, NO_HEADS, HEAD_W),
        "ffwd": ffwd_init(keys[3], HEAD_W),
        "lm_head": jax.random.normal(keys[4], (HEAD_W, vocab_size)) * 0.02,
    }
    return params


def forward(params, tokens):
    _, t = tokens.shape
    tok_emb = params["token_embed"][tokens]  # (B, T, N_EMBED)
    pos_emb = params["pos_embed"][jnp.arange(t)]  # (T, N_EMBED)
    x = tok_emb + pos_emb  # (B, T, N_EMBED)
    attn_head_out = mh_forward(x, params["mh_attention"], NO_HEADS, HEAD_W)
    ffwd_out = ffwd_forward(attn_head_out, params["ffwd"])
    logits = ffwd_out @ params["lm_head"]
    return logits


def loss_fn(params, tokens, targets):
    """Return mean next-character cross-entropy as a scalar."""
    logits = forward(params, tokens)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
    return loss


@jax.jit
def train_step(params, opt_state, tokens, targets):
    """Return (updated_params, updated_opt_state, loss)."""
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets)
    updates, updated_opt_state = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, updated_opt_state, loss
