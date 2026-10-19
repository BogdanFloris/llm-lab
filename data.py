import hashlib
from pathlib import Path
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np

DATA_PATH = Path(__file__).parent / "data" / "input.txt"
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


class Tokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {char: i for i, char in enumerate(self.chars)}

    def encode(self, text):
        return np.array([self.stoi[char] for char in text], dtype=np.int32)

    def decode(self, tokens):
        return "".join(self.chars[int(token)] for token in tokens)

    def __len__(self):
        return len(self.chars)


def load_data(path=DATA_PATH):
    if not path.exists():
        content = urlopen(DATA_URL, timeout=60).read()
        if hashlib.sha256(content).hexdigest() != DATA_SHA256:
            raise ValueError("Unexpected Tiny Shakespeare download")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != DATA_SHA256:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    text = content.decode("utf-8")
    tokenizer = Tokenizer(text)
    tokens = tokenizer.encode(text)
    split = int(0.9 * len(tokens))
    return tokenizer, jnp.asarray(tokens[:split]), jnp.asarray(tokens[split:])


def get_batch(tokens, key, batch_size, block_size):
    starts = jax.random.randint(
        key, (batch_size,), minval=0, maxval=len(tokens) - block_size
    )
    positions = starts[:, None] + jnp.arange(block_size)[None, :]
    return tokens[positions], tokens[positions + 1]


if __name__ == "__main__":
    tokenizer, train, val = load_data()
    print(f"Vocabulary: {len(tokenizer)}; train: {len(train)}; validation: {len(val)}")
