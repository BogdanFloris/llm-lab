import optax


optimizer = optax.adam(learning_rate=1e-2)


def init_params(key, vocab_size):
    """Return a parameter pytree. Start with small random logits."""
    raise NotImplementedError("TODO(bogdan): initialize the bigram parameters")


def forward(params, tokens):
    """Integer tokens (B, T) -> logits (B, T, V), without softmax."""
    raise NotImplementedError("TODO(bogdan): look up next-character logits")


def loss_fn(params, tokens, targets):
    """Return mean next-character cross-entropy as a scalar."""
    raise NotImplementedError("TODO(bogdan): compute the training loss")


def train_step(params, opt_state, tokens, targets):
    """Return (updated_params, updated_opt_state, loss). Add jit yourself."""
    raise NotImplementedError("TODO(bogdan): differentiate and apply an update")
