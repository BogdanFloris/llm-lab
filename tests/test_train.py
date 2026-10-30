import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from checkpoint import load_checkpoint
from data import Tokenizer
import train


def test_checkpoint_survives_sampling_failure(tmp_path, monkeypatch):
    params = {"weight": jnp.ones((2, 3))}
    model = SimpleNamespace(
        __name__="test_model",
        N_EMBED=3, HEAD_W=3, T_MAX=8, NO_HEADS=1,
        init_params=lambda key, vocab_size: params,
        optimizer=SimpleNamespace(init=lambda params: None),
        loss_fn=None,
        forward=None,
    )
    tokenizer = Tokenizer("\nab")
    monkeypatch.setattr(train, "model", model)
    monkeypatch.setattr(train, "load_data", lambda: (tokenizer, None, None))
    monkeypatch.setattr(train, "evaluate", lambda *args: 0.0)

    def fail_sampling(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(train, "generate", fail_sampling)
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--steps", "0", "--out", str(tmp_path)],
    )
    with pytest.raises(RuntimeError, match="sampling failed"):
        train.main()

    checkpoint = load_checkpoint(tmp_path / "params.pkl")
    np.testing.assert_array_equal(checkpoint["params"]["weight"], params["weight"])
    assert checkpoint["config"]["vocab"] == tokenizer.chars
    assert checkpoint["config"]["model"]["module"] == "test_model"
