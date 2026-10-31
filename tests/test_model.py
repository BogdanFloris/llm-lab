import jax
import jax.numpy as jnp
import numpy as np

import model


def test_parameters_are_an_array_pytree():
    shapes = jax.eval_shape(lambda key: model.init_params(key, 65), jax.random.key(0))
    assert len(shapes["blocks"]) == 3
    for block_params in shapes["blocks"].values():
        heads = block_params["mh"]
        assert len(heads) == model.N_HEAD
        for head in heads.values():
            assert set(head) == {"q", "k", "v"}
            for weight in head.values():
                assert weight.shape == (model.N_EMBED, model.N_EMBED // model.N_HEAD)
        assert block_params["ffwd"]["w1"].shape == (model.N_EMBED, 4 * model.N_EMBED)
        assert block_params["ffwd"]["w2"].shape == (4 * model.N_EMBED, model.N_EMBED)
        for name in ("ln1", "ln2"):
            assert block_params[name]["gamma"].shape == (model.N_EMBED,)
            assert block_params[name]["beta"].shape == (model.N_EMBED,)
    assert shapes["ln"]["gamma"].shape == (model.N_EMBED,)
    assert shapes["lm_head"].shape == (model.N_EMBED, 65)
    leaves = jax.tree.leaves(shapes)
    assert leaves
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in leaves)
    assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves)


def test_gradient_tree_matches_parameters():
    key = jax.random.key(0)
    params = jax.eval_shape(lambda key: model.init_params(key, 65), key)
    tokens = jax.ShapeDtypeStruct((2, 3), jnp.int32)
    grads = jax.eval_shape(jax.grad(model.loss_fn), params, tokens, tokens)
    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for param, grad in zip(jax.tree.leaves(params), jax.tree.leaves(grads), strict=True):
        assert grad.shape == param.shape
        assert grad.dtype == param.dtype


def test_forward_shapes_eager_and_jit():
    params = model.init_params(jax.random.key(0), 65)
    for batch_size, length in [(1, 1), (2, 3), (3, model.T_MAX)]:
        tokens = jnp.zeros((batch_size, length), dtype=jnp.int32)
        eager = model.forward(params, tokens)
        compiled = jax.jit(model.forward)(params, tokens)
        assert eager.shape == (batch_size, length, 65)
        assert np.isfinite(np.asarray(eager)).all()
        np.testing.assert_allclose(eager, compiled, rtol=1e-5, atol=1e-5)


def test_future_tokens_cannot_change_prefix_logits():
    params = model.init_params(jax.random.key(0), 65)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    forward = jax.jit(model.forward)
    original = forward(params, tokens)
    changed = forward(params, tokens.at[0, 3].set(7))
    np.testing.assert_allclose(original[:, :3], changed[:, :3], rtol=1e-5, atol=1e-5)


def test_stack_supports_another_embedding_width(monkeypatch):
    monkeypatch.setattr(model, "N_EMBED", 16)
    params = model.init_params(jax.random.key(0), 65)
    tokens = jnp.zeros((2, 3), dtype=jnp.int32)
    assert jax.jit(model.forward)(params, tokens).shape == (2, 3, 65)


def test_blocks_receive_distinct_random_keys():
    params = model.init_params(jax.random.key(0), 65)
    queries = [block_params["mh"][0]["q"] for block_params in params["blocks"].values()]
    for i, query in enumerate(queries):
        for other in queries[i + 1:]:
            assert not np.array_equal(query, other)


def test_loss_is_finite_scalar():
    params = model.init_params(jax.random.key(0), 65)
    tokens = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    targets = jnp.array([[2, 3, 4]], dtype=jnp.int32)
    loss = jax.jit(model.loss_fn)(params, tokens, targets)
    assert loss.shape == ()
    assert np.isfinite(float(loss))
