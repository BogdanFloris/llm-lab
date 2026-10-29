import jax
import jax.numpy as jnp
import optax

optimizer = optax.adam(learning_rate=1e-2)

N_EMBED = 32
HEAD_W = 32
T_MAX = 8


def init_params(key, vocab_size):
    keys = jax.random.split(key, 6)
    params = {
        "token_embed": jax.random.normal(keys[0], (vocab_size, N_EMBED)) * 0.02,
        "pos_embed": jax.random.normal(keys[1], (T_MAX, N_EMBED)) * 0.02,
        "q": jax.random.normal(keys[2], (N_EMBED, HEAD_W)) * 0.02,
        "k": jax.random.normal(keys[3], (N_EMBED, HEAD_W)) * 0.02,
        "v": jax.random.normal(keys[4], (N_EMBED, HEAD_W)) * 0.02,
        "lm_head": jax.random.normal(keys[5], (HEAD_W, vocab_size)) * 0.02,
    }
    return params


def forward(params, tokens): # (B, T)
    _, t = tokens.shape
    tok_emb = params["token_embed"][tokens] # (B, T, N_EMBED)
    pos_emb = params["pos_embed"][jnp.arange(t)] # (T, N_EMBED)
    x = tok_emb + pos_emb # (B, T, N_EMBED)
    q = x @ params["q"] # (B, T, HEAD_W)
    k = x @ params["k"] # (B, T, HEAD_W)
    v = x @ params["v"] # (B, T, HEAD_W)
    wei = q @ k.transpose(0, 2, 1) * HEAD_W**-0.5 # (B, T, T)
    mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
    wei = jnp.where(mask, wei, -jnp.inf)
    wei = jax.nn.softmax(wei)
    att_head_out = wei @ v # (B, T, HEAD_W)
    logits = att_head_out @ params["lm_head"]
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
