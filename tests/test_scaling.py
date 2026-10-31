import jax
import jax.numpy as jnp

import model


def test_configured_model_traces_at_full_context_and_batch():
    params = jax.eval_shape(lambda key: model.init_params(key, 65), jax.random.key(0))
    assert len(params["blocks"]) == model.N_LAYER
    tokens = jax.ShapeDtypeStruct((64, model.T_MAX), jnp.int32)
    logits = jax.eval_shape(model.forward, params, tokens)
    assert logits.shape == (64, model.T_MAX, 65)
