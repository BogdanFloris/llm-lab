import jax
import jax.numpy as jnp

import sample
from data import Tokenizer


def test_sampling_uses_and_crops_context(monkeypatch):
    monkeypatch.setattr(sample.jax, "jit", lambda function: function)
    tokenizer = Tokenizer("abcde")
    contexts = []

    def fixed_logits(params, tokens, key, is_training):
        assert key is None
        assert is_training is False
        contexts.append(tokenizer.decode(tokens[0]))
        logits = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.0])
        return jnp.broadcast_to(logits, (*tokens.shape, len(tokenizer)))

    text = sample.generate(
        fixed_logits, {}, tokenizer, jax.random.key(0),
        block_size=3, prompt="abcde", max_new_tokens=3,
    )
    assert text == "abcdeeee"
    assert contexts == ["cde", "dee", "eee"]


def test_sampling_accepts_short_context():
    tokenizer = Tokenizer("ab")

    def fixed_logits(params, tokens, key, is_training):
        assert key is None
        assert is_training is False
        return jnp.broadcast_to(jnp.array([-jnp.inf, 0.0]), (*tokens.shape, 2))

    text = sample.generate(
        fixed_logits, {}, tokenizer, jax.random.key(0),
        block_size=3, prompt="a", max_new_tokens=4,
    )
    assert text == "abbbb"


def test_short_context_uses_last_real_position_and_fixed_shape(monkeypatch):
    monkeypatch.setattr(sample.jax, "jit", lambda function: function)
    tokenizer = Tokenizer("abc")
    shapes = []

    def cyclic_logits(params, tokens, key, is_training):
        assert key is None
        assert is_training is False
        shapes.append(tokens.shape)
        targets = (tokens + 1) % len(tokenizer)
        return jnp.where(
            jnp.arange(len(tokenizer)) == targets[..., None], 0.0, -jnp.inf,
        )

    text = sample.generate(
        cyclic_logits, {}, tokenizer, jax.random.key(0),
        block_size=3, prompt="a", max_new_tokens=4,
    )
    assert text == "abcab"
    assert shapes == [(1, 3)] * 4
