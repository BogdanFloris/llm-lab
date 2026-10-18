import jax
import jax.numpy as jnp
import numpy as np

from data import Tokenizer, get_batch


def test_tokenizer_roundtrip():
    tokenizer = Tokenizer("hello\n")
    assert tokenizer.decode(tokenizer.encode("hello\n")) == "hello\n"
    assert tokenizer.chars == sorted(set("hello\n"))


def test_batch_alignment_and_reproducibility():
    tokens = jnp.arange(100, dtype=jnp.int32)
    key = jax.random.key(42)
    x, y = get_batch(tokens, key, batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    np.testing.assert_array_equal(y, x + 1)
    np.testing.assert_array_equal(x[:, 1:], x[:, :-1] + 1)
    x_again, _ = get_batch(tokens, key, batch_size=4, block_size=8)
    np.testing.assert_array_equal(x, x_again)
