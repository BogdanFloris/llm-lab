# Progress

## Module 1 — GPT from scratch

- Bogdan reports watching the bigram portion of Karpathy's video and reading some JAX documentation.
- Reading check: Bogdan identified the `(V, V)` table, one-character context, and same-key reproducibility/key splitting. Initially described a single probability vector as the output; after a shape hint, correctly gave `(B, T, V)` logits.
- Scaffold added: Nix/direnv, locked uv dependencies, checksum-verified Tiny Shakespeare, batching, training/evaluation, categorical sampling, and tests. The four model functions in `bigram.py` remain unimplemented.
- Verified: JAX sees the GPU; a jitted GPU matrix multiplication matches the CPU reference; both data tests pass; data vocabulary is 65; train CLI and Python syntax checks pass.
- Untested: model behavior, losses, training, and end-to-end sampling. No training or loss experiment run. No module done criteria verified yet.
- Next: Bogdan implements `init_params` and `forward`; then reviews shapes before proceeding to loss and the jitted training step. Ask for loss predictions before loss checks or training.
