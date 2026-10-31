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

## Model

The four model functions in `model.py` are learner-owned:

1. `init_params`: choose a parameter pytree; use small random initial logits.
2. `forward`: `(B, T)` integer token IDs to `(B, T, V)` logits.
3. `loss_fn`: mean next-token cross-entropy over all batch and time positions.
4. `train_step`: differentiate the loss, use the supplied Optax optimizer,
   return updated parameters, optimizer state, and scalar loss. You add `jit`.

You can add imports as needed. Every trainable array must be a dynamic leaf in
the parameter pytree; ordinary class instances do not automatically expose
their attributes to JAX.
The harness initializes the optimizer state with `optimizer.init(params)`.

Agent-owned files:

- `data.py`: checksum-verified Tiny Shakespeare, sorted character vocabulary,
  contiguous 90/10 train/validation split, random next-token batches.
- `train.py`: training and evaluation, JSONL metrics and generated text in `runs/`.
  Evaluation reuses fixed batches for comparable checkpoints and has separate
  random keys from training.
- `sample.py`: categorical sampling using a cropped context of up to
  `--block-size` tokens. This is not the greedy-decoding exercise from module 2.
- `checkpoint.py`: saves final parameters and run configuration to `params.pkl`
  before sampling. Only load trusted local pickle files. Optimizer state is not
  saved; this is for inspection/inference, not exact training resumption.
- `tests/`: data checks and model behavior checks.

Data tests:

```sh
uv run pytest tests/test_data.py
```

Before executing loss checks or training, predict the initial loss and where you
expect training to end. Then:

```sh
uv run pytest tests/test_model.py
uv run python train.py
```

Measured progress and understanding checks live in `PROGRESS.md`.

## Attention refactor

`model.py` is the active training entry point. `attention.py` and `multihead.py`
contain learner-owned components. The old `--model` selector is removed because
the component modules are no longer standalone models. Earlier baselines remain
in jj history; measured results remain in `PROGRESS.md`.
The block size must not exceed `model.T_MAX`.

Sampling now passes the full available context, cropped to the block size.
Positions start at zero within each cropped window, as in training.
Short contexts may cause additional JIT compilations as sampling starts.

Before measuring attention loss or training, predict the relevant loss.
After that checkpoint:

```sh
uv run pytest tests/test_model.py
uv run python train.py
```

Model tests check array-pytree compatibility, eager/jitted output shapes,
causality, alternate embedding widths, and finite scalar loss. They do not
establish that training converges or that initialization is well scaled.

## Dropout integration

The harness calls forward/loss with `key=None, is_training=False` for evaluation
and generation. Evaluation uses its random key for batch selection only.
Sampling still uses randomness to select tokens, but not inside the model.

Training splits independent continuation, batch, and dropout keys each step,
then passes the dropout key to `model.train_step`. The model is responsible
for splitting independent keys for its blocks and dropout locations.
The checkpoint configuration records the dropout rate.

Training mode and rate must be known when tracing Python conditionals.
No dropout training experiment has been run yet; current verification and
open issues are recorded in `PROGRESS.md`.
