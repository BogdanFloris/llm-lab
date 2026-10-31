import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from checkpoint import load_checkpoint
from data import Tokenizer
import train


def test_checkpoint_survives_sampling_failure(tmp_path, monkeypatch):
    params = {"weight": jnp.ones((2, 3)), "blocks": {"first": {}, "second": {}}}
    model = SimpleNamespace(
        __name__="test_model",
        N_EMBED=3, T_MAX=8, N_HEAD=1, DROPOUT_RATE=0.2, LEARNING_RATE=3e-4,
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
    assert checkpoint["config"]["model"]["n_blocks"] == 2
    assert checkpoint["config"]["model"]["dropout_rate"] == 0.2
    assert checkpoint["config"]["learning_rate"] == 3e-4
    assert checkpoint["config"]["batch_size"] == 64
    assert checkpoint["config"]["block_size"] == model.T_MAX
    assert checkpoint["config"]["eval_every"] == 500
    assert checkpoint["config"]["eval_batches"] == 20
    best = load_checkpoint(tmp_path / "best_params.pkl")
    np.testing.assert_array_equal(best["params"]["weight"], params["weight"])
    assert best["config"]["checkpoint_step"] == 0
    assert best["config"]["validation_loss"] == 0.0


def test_best_checkpoint_is_not_overwritten_by_worse_final_step(tmp_path, monkeypatch):
    params = {"weight": jnp.array(0.0), "blocks": {"first": {}}}

    def step(params, state, tokens, targets, key):
        return {**params, "weight": params["weight"] + 1}, state, None

    model = SimpleNamespace(
        __name__="test_model",
        N_EMBED=3, T_MAX=8, N_HEAD=1, DROPOUT_RATE=0.2, LEARNING_RATE=3e-4,
        init_params=lambda key, vocab_size: params,
        optimizer=SimpleNamespace(init=lambda params: None),
        loss_fn=None, forward=None, train_step=step,
    )
    monkeypatch.setattr(train, "model", model)
    monkeypatch.setattr(train, "load_data", lambda: (Tokenizer("\nab"), None, None))
    monkeypatch.setattr(train, "get_batch", lambda *args: (None, None))
    monkeypatch.setattr(
        train, "evaluate", lambda loss_fn, params, *args: [2.0, 1.0, 1.5][int(params["weight"])],
    )
    monkeypatch.setattr(train, "generate", lambda *args, **kwargs: "\n")
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "--steps", "2", "--eval-every", "1", "--out", str(tmp_path)],
    )
    train.main()
    best = load_checkpoint(tmp_path / "best_params.pkl")
    final = load_checkpoint(tmp_path / "params.pkl")
    assert float(best["params"]["weight"]) == 1
    assert best["config"]["checkpoint_step"] == 1
    assert best["config"]["validation_loss"] == 1.0
    assert float(final["params"]["weight"]) == 2


def test_evaluation_disables_dropout_and_keeps_batch_keys(monkeypatch):
    seen_keys = []

    def get_batch(tokens, key, batch_size, block_size):
        seen_keys.append(np.asarray(jax.random.key_data(key)))
        x = jnp.zeros((batch_size, block_size), dtype=jnp.int32)
        return x, x

    def fixed_loss(params, tokens, targets, key, is_training):
        assert key is None
        assert is_training is False
        return jnp.array(0.0)

    monkeypatch.setattr(train, "get_batch", get_batch)
    key = jax.random.key(0)
    assert train.evaluate(fixed_loss, {}, None, key, 2, 3, 2) == 0.0
    expected = [jax.random.key_data(k) for k in jax.random.split(key, 2)]
    np.testing.assert_array_equal(seen_keys, expected)
