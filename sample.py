import jax
import jax.numpy as jnp

def generate(
    forward, params, tokenizer, key, *,
    block_size, prompt="\n", max_new_tokens=300,
):
    next_logits = jax.jit(
        lambda params, tokens, position: forward(
            params, tokens, key=None, is_training=False
        )[0, position]
    )
    tokens = tokenizer.encode(prompt).tolist()
    for _ in range(max_new_tokens):
        key, draw_key = jax.random.split(key)
        context = tokens[-block_size:]
        position = len(context) - 1
        padded = context + [0] * (block_size - len(context))
        inputs = jnp.asarray([padded], dtype=jnp.int32)
        token = jax.random.categorical(draw_key, next_logits(params, inputs, position))
        tokens.append(int(token))
    return tokenizer.decode(tokens)
