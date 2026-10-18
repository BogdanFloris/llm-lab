# LLM lab

Curriculum and ownership: `PLAN.md`. Learning notes: `PROGRESS.md`.

## Environment

```sh
direnv allow
uv sync --locked
uv run python -c 'import jax; print(jax.devices())'
```

Without direnv, run commands inside `nix develop`.

Nix supplies Python and native libraries. uv locks Python packages, including
JAX's CUDA wheels. The shell exposes the host NVIDIA driver, not a second CUDA
toolkit.

## Bigram

Start in `bigram.py`. The four unfinished functions are yours:

1. `init_params`: choose a parameter pytree; use small random initial logits.
2. `forward`: `(B, T)` integer token IDs to `(B, T, V)` logits.
3. `loss_fn`: mean next-token cross-entropy over all batch and time positions.
4. `train_step`: differentiate the loss, use the supplied Optax optimizer,
   return updated parameters, optimizer state, and scalar loss. You add `jit`.

You can add imports as needed. No parameter container is prescribed.
The harness initializes the optimizer state with `optimizer.init(params)`.

Agent-owned files:

- `data.py`: checksum-verified Tiny Shakespeare, sorted character vocabulary,
  contiguous 90/10 train/validation split, random next-token batches.
- `train.py`: training and evaluation, JSONL metrics and generated text in `runs/`.
  Evaluation reuses fixed batches for comparable checkpoints and has separate
  random keys from training.
- `sample.py`: categorical sampling, using one previous character for this
  bigram only. This is not the greedy-decoding exercise from module 2.
- `tests/`: data checks and model behavior checks.

The model tests intentionally fail with `NotImplementedError` until implemented.
Only the data tests can pass initially:

```sh
uv run pytest tests/test_data.py
```

Before executing loss checks or training, predict the initial loss and where you
expect training to end. Then:

```sh
uv run pytest tests/test_bigram.py
uv run python train.py
```

No model training or loss measurements have been run as part of scaffolding.
