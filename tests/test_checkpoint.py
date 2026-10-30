import jax
import jax.numpy as jnp
import numpy as np

from checkpoint import load_checkpoint, save_checkpoint


def test_nested_parameter_roundtrip(tmp_path):
    params = {
        "embedding": jnp.arange(12, dtype=jnp.float32).reshape(4, 3),
        "heads": [{"q": jnp.ones((3, 2))}, {"q": jnp.zeros((3, 2))}],
    }
    config = {"steps": 3000, "vocab": ["a", "b", "c", "d"]}
    path = tmp_path / "params.pkl"
    save_checkpoint(path, params, config)
    restored = load_checkpoint(path)
    assert restored["config"] == config
    assert jax.tree.structure(restored["params"]) == jax.tree.structure(params)
    for expected, actual in zip(
        jax.tree.leaves(params), jax.tree.leaves(restored["params"]), strict=True,
    ):
        assert isinstance(actual, np.ndarray)
        assert expected.dtype == actual.dtype
        np.testing.assert_array_equal(actual, expected)
