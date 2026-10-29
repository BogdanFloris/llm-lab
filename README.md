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
toolkit. JAX preallocation is disabled so it can share the GPU with the desktop.

## Bigram

The four model functions in `bigram.py` are learner-owned:

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
- `sample.py`: categorical sampling using a cropped context of up to
  `--block-size` tokens. This is not the greedy-decoding exercise from module 2.
- `tests/`: data checks and model behavior checks.

Data tests:

```sh
uv run pytest tests/test_data.py
```

Before executing loss checks or training, predict the initial loss and where you
expect training to end. Then:

```sh
uv run pytest tests/test_bigram.py
uv run python train.py
```

Measured progress and understanding checks live in `PROGRESS.md`.

## Single-head attention

`attention.py` is also learner-owned, with the same model interface.
`train.py --model attention` selects it; the default is still `bigram`.
The block size must not exceed `attention.T_MAX`.

Sampling now passes the full available context, cropped to the block size.
Positions start at zero within each cropped window, as in training.
Short contexts may cause additional JIT compilations as sampling starts.

Before measuring attention loss or training, predict the relevant loss.
After that checkpoint:

```sh
uv run pytest tests/test_attention.py
uv run python train.py --model attention
```

Attention tests check eager/jitted output shapes, causality, unequal embedding
and head widths, and finite scalar loss. They do not establish that training
converges or that initialization is well scaled.
