import jax
import jax.numpy as jnp

from bigram import forward


def generate(params, tokenizer, key, prompt="\n", max_new_tokens=300):
    next_logits = jax.jit(lambda params, tokens: forward(params, tokens)[0, -1])
    tokens = tokenizer.encode(prompt).tolist()
    for _ in range(max_new_tokens):
        key, draw_key = jax.random.split(key)
        context = jnp.asarray([[tokens[-1]]], dtype=jnp.int32)
        token = jax.random.categorical(draw_key, next_logits(params, context))
        tokens.append(int(token))
    return tokenizer.decode(tokens)
