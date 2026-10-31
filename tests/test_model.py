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
    grads = jax.eval_shape(jax.grad(model.loss_fn), params, tokens, tokens, None)
    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for param, grad in zip(jax.tree.leaves(params), jax.tree.leaves(grads), strict=True):
        assert grad.shape == param.shape
        assert grad.dtype == param.dtype


def test_forward_shapes_eager_and_jit():
    params = model.init_params(jax.random.key(0), 65)
    for batch_size, length in [(1, 1), (2, 3), (3, model.T_MAX)]:
        tokens = jnp.zeros((batch_size, length), dtype=jnp.int32)
        eager = model.forward(params, tokens, key=None)
        compiled = jax.jit(model.forward)(params, tokens, None)
        assert eager.shape == (batch_size, length, 65)
        assert np.isfinite(np.asarray(eager)).all()
        np.testing.assert_allclose(eager, compiled, rtol=1e-5, atol=1e-5)


def test_future_tokens_cannot_change_prefix_logits():
    params = model.init_params(jax.random.key(0), 65)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    forward = jax.jit(model.forward)
    original = forward(params, tokens, None)
    changed = forward(params, tokens.at[0, 3].set(7), None)
    np.testing.assert_allclose(original[:, :3], changed[:, :3], rtol=1e-5, atol=1e-5)


def test_stack_supports_another_embedding_width(monkeypatch):
    monkeypatch.setattr(model, "N_EMBED", 16)
    params = model.init_params(jax.random.key(0), 65)
    tokens = jnp.zeros((2, 3), dtype=jnp.int32)
    assert jax.jit(model.forward)(params, tokens, None).shape == (2, 3, 65)


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
    loss = jax.jit(model.loss_fn)(params, tokens, targets, None)
    assert loss.shape == ()
    assert np.isfinite(float(loss))


def test_evaluation_is_independent_of_dropout_key():
    params = model.init_params(jax.random.key(0), 65)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    forward = jax.jit(model.forward)
    expected = forward(params, tokens, None)
    for key in jax.random.split(jax.random.key(1), 2):
        np.testing.assert_array_equal(expected, forward(params, tokens, key))


def test_training_forward_is_reproducible_with_a_key():
    params = model.init_params(jax.random.key(0), 65)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    forward = jax.jit(
        lambda params, tokens, key: model.forward(params, tokens, key, is_training=True)
    )
    key1, key2 = jax.random.split(jax.random.key(1))
    first = forward(params, tokens, key1)
    assert np.isfinite(np.asarray(first)).all()
    np.testing.assert_array_equal(first, forward(params, tokens, key1))
    assert not np.array_equal(first, forward(params, tokens, key2))


def test_training_step_traces_with_a_dropout_key():
    params = model.init_params(jax.random.key(0), 65)
    opt_state = model.optimizer.init(params)
    tokens = jax.ShapeDtypeStruct((2, 3), jnp.int32)
    updated, state, loss = jax.eval_shape(
        model.train_step, params, opt_state, tokens, tokens, jax.random.key(1),
    )
    assert jax.tree.structure(updated) == jax.tree.structure(params)
    for before, after in zip(jax.tree.leaves(params), jax.tree.leaves(updated), strict=True):
        assert before.shape == after.shape
        assert before.dtype == after.dtype
    assert jax.tree.structure(state) == jax.tree.structure(opt_state)
    assert loss.shape == ()


def test_dropout_keys_are_distinct_across_blocks(monkeypatch):
    params = model.init_params(jax.random.key(0), 65)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    seen_keys = []

    def record_block(x, params, n_embed, n_head, rate, key=None, is_training=False):
        assert is_training is True
        seen_keys.append(np.asarray(jax.random.key_data(key)))
        return x

    monkeypatch.setattr(model, "b_forward", record_block)
    model.forward(params, tokens, jax.random.key(1), is_training=True)
    assert len(seen_keys) == len(params["blocks"])
    for i, key in enumerate(seen_keys):
        for other in seen_keys[i + 1:]:
            assert not np.array_equal(key, other), "Blocks received the same dropout key"
