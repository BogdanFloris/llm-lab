import jax
import jax.numpy as jnp
import optax

optimizer = optax.adam(learning_rate=1e-2)


def init_params(key, vocab_size):
    """Return a parameter pytree. Start with small random logits."""
    params = {"embedding": jax.random.uniform(key, (vocab_size, vocab_size))}
    return params


def forward(params, tokens):
    """Integer tokens (B, T) -> logits (B, T, V), without softmax."""
    embedding_table = params["embedding"]
    logits = jnp.take(embedding_table, tokens, axis=0)
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
