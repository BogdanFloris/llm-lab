# Progress

## Module 1 — GPT from scratch

- Bogdan reports watching the bigram portion of Karpathy's video and reading some JAX documentation.
- Reading check: Bogdan identified the `(V, V)` table, one-character context, and same-key reproducibility/key splitting. Initially described a single probability vector as the output; after a shape hint, correctly gave `(B, T, V)` logits.
- Scaffold added: Nix/direnv, locked uv dependencies, checksum-verified Tiny Shakespeare, batching, training/evaluation, categorical sampling, and tests.
- Bogdan implemented the four model functions. `init_params` returns a dictionary containing a random `(V, V)` table; `forward` selects rows; loss uses integer labels; the jitted training step differentiates with respect to parameters and applies Optax updates.
- Blocker: Bogdan reported CUDA out of memory. JAX's default memory reservation was suspected given desktop GPU usage. Disabled preallocation in the dev shell and verified initialization succeeds; the original failing invocation was not reproduced.
- Verified: JAX sees the GPU; a jitted GPU matrix multiplication matches the CPU reference; both data tests pass; data vocabulary is 65; train CLI and Python syntax checks pass.
- Bogdan reports reaching step 3000 before the shape/loss fixes, followed by an IndexError. That run does not verify correct learning. ARC has not run training or the overfit test.
- Hints: first distinguished lookup from multiplication. Bogdan switched to `jnp.take`, but omitted its axis; it flattens the table and loses the vocabulary dimension. Pointed to the API's axis behavior and asked which axis selects whole rows.
- Loss hint: pointed out the mismatch between integer targets and the distribution-label API; Bogdan switched to the integer-label variant.
- Bogdan predicted initial cross-entropy `ln(65)`. Measured on the fixed random test batch: 4.250052 versus uniform 4.174387 (difference +0.075665). Initial logits are uniform in `[0, 1)`, so probabilities are not exactly uniform.
- Verified after the fixes: eager/jitted shapes, scalar finite loss, initial loss within 0.1 of `ln(65)`, and no past/future dependence beyond the current token (3 model tests pass). These check some module criteria, not module completion or understanding of transformations.
- Bogdan correctly identified parameters as the differentiated argument and the gradient array shape `(V, V)`; ARC clarified that the outer dictionary structure is preserved too. He predicted validation loss around 2.5.
- Verified training and sampling: a 3000-step GPU run completed with initial validation loss 4.2203, final train loss 2.4308, and final validation loss 2.4780 (prediction difference -0.022). Generated text was saved without error. Artifacts: `runs/20261019-175248-589080/`.
- Understanding check: Bogdan correctly said token-value changes should not recompile and that `train_step` contains no randomness. His JIT explanation was approximate; ARC clarified tracing of operations and specialization by shape/dtype, rather than treating XLA as a memory-fetching layer. Random batch selection lives outside `train_step`.
- Overfit check: Bogdan predicted zero because each input has a fixed successor. After 400 updates, measured loss was 0.064136 and all four next-token argmax predictions were correct. The existing overfit test also passed. Zero is the limiting target, not an expectation of exact zero after finite updates.
- Bigram milestone: all four model behavior tests have now passed across runs, Shakespeare validation loss reached 2.4780, sampling works, and reading/code questions have been discussed. No learner code was implemented by ARC.
- Next: Bogdan watches the video through single-head self-attention, then checks understanding before the next build. Attention, the full transformer, and trace/visualization work remain; module 1 is not complete.
