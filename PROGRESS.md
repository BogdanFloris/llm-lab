# Progress

## Module 1 — GPT from scratch

### Current state

- Bogdan reports finishing Karpathy's video, including scaling/dropout, and reading some JAX documentation.
- Bogdan wrote the model/components. ARC owns the environment, data, training/evaluation, categorical sampling, checkpoints, and tests. Learner model code has not been implemented or replaced by ARC.
- Active model: `model.py`, three pre-LayerNorm residual blocks, embedding width 32, four width-8 heads, MLP width 128, context 8. Initial weights use normal draws scaled by 0.02; biases start at zero; LayerNorm uses gamma ones/beta zeros.
- Dropout rate 0.2 is integrated on attention weights after softmax and on MLP branch outputs before residual addition. Independent keys flow through steps, blocks, branches, and heads. Evaluation accepts `key=None` and disables model dropout.
- Latest verification: **38 non-loss tests pass; one scalar-loss test excluded**. Coverage includes parameter/gradient-tree shapes, JIT behavior, causality, residual identity and identity gradients for zeroed branches, LayerNorm/MLP behavior, dropout scaling and key independence at all levels, deterministic evaluation, sampling, and checkpoint handling. Training-step verification with dropout is abstract tracing only: **no numerical dropout training updates or new model loss measurements yet**.

### Measured experiments

All training runs below used 3000 steps, context 8, learning rate 0.01, and seed 1337. Validation uses fixed sampled batches, not the full held-out corpus.

| Model | Bogdan's final validation prediction | Final train loss | Final validation loss |
|---|---|---:|---:|
| Bigram | about 2.5 | 2.4308 | 2.4780 |
| Single attention head | 2-point-something, below bigram | 2.3834 | 2.4192 |
| Four heads + MLP | 2.3 | 2.1382 | 2.2078 |
| Three residual blocks + LayerNorm, no dropout | 2.1 | 2.1118 | 2.1741 |

- Residual-stack initial validation loss: **4.1436**, versus Bogdan's uniform-baseline prediction `ln(65) ≈ 4.174`. Vocabulary has 65 characters, not 64.
- Architecture/depth/normalization changed together across some runs; these are not isolated component ablations. New dropout key plumbing also changes training batch-key progression; evaluation batch selection is preserved.
- Initialization experiments on the original single-head model, using seed 1337 and 20 validation batches, without updates: uniform `[0,1)` weights gave **54.4932** (prediction 20); standard-normal weights gave **102.0121** (prediction 10); normal weights scaled by 0.02 gave **4.174362** (prediction 10), near uniform baseline **4.174387**. Centering and scale needed separate explanation; ARC's rough magnitude reasoning was guidance, not Bogdan's independent prediction.
- Bigram deterministic-batch overfit check: Bogdan predicted loss approaching zero; after 400 updates, measured **0.064136** and all four next-token predictions were correct. A full-transformer overfit check remains undone.

### Artifacts and harness

- Bigram: `runs/20261019-175248-589080/`; single head: `runs/20261030-164849-343829/`. These earlier runs saved metrics/samples, not weights.
- Four heads + MLP: `runs/20261030-182905-030836/`; lowest logged validation 2.1811 at step 2700, final 2.2078.
- Residual stack: `runs/20261031-130855-717286/`; final checkpoint validation 2.1741. This is the latest trained model; it has no dropout training.
- The latter two runs contain `params.pkl`; reload and finite jitted forward were verified. Checkpoints save **final**, not best-validation, parameters, vocabulary, and configuration before sampling. No optimizer state/exact resume; only load trusted local pickle files.
- Training now targets `model.py`; component modules are not standalone model choices. Earlier baselines remain in jj history.
- Nix/direnv + locked uv environment verified on the GPU. JAX preallocation was disabled after reported OOM with desktop GPU use; initialization then passed, though the original failing invocation was not reproduced.

### Understanding and difficulties

- Bogdan correctly explained next-token alignment, one-token bigram context, same-key reproducibility/key splitting, and `ln(V)` for uniform predictions.
- Shape hints were needed for token lookup versus multiplication, actual context length versus position-table capacity, projection weights versus activations, concatenation, and per-head versus total widths. He subsequently derived head activations `(B,T,H)`, attention scores `(B,T,T)`, and MLP weights `(32,128)`/`(128,32)` with biases `(128,)`/`(32,)`.
- Reinforce **container structure versus array shape**: one head's gradient subtree is a q/k/v dictionary with three `(32,8)` arrays, not one `(32,8,8)` tensor. Stateful class attributes were replaced by explicit array pytrees.
- Q/K routing versus V content and logits versus selected token IDs needed explanation. Bogdan correctly answered that identical Q/K with different V preserves attention weights but changes output information.
- He correctly explained that two affine layers collapse without ReLU; biases compose too. Hidden expansion/nonlinearity was explained using an absolute-value example.
- Residual skip uses the original unnormalized input; the MLP skip uses the updated state. Bogdan understood near-identity behavior. Clarified that the skip contributes an identity gradient term, not guaranteed unchanged total gradients.
- LayerNorm initially used batch/time axes and subtracted epsilon; corrected to feature axis with positive epsilon. For `(2,8,32)`, there are 16 independent groups of 32 values. He correctly chose gamma ones/beta zeros.
- JIT: changing ordinary array values does not require recompilation; the fixed Python block loop executes during tracing, while compiled block operations execute each step. Fresh same-shaped keys produce new dropout masks without per-key compilation; Bogdan guessed this correctly but was initially uncertain.
- Dropout: retained value is 12.5 for input 10 at drop rate 0.2 (after a keep-probability hint). Attention dropout belongs after softmax; row sums are one **in expectation**, not for every mask, and should not be renormalized. Repeated issues were key reuse, splitting `None` in evaluation, and positional rate/key swaps; latest checks pass after Bogdan's fixes.

### Next

- Ask for predicted validation loss after 3000 steps with dropout on the **small** model; compare with no-dropout baseline 2.1741 before scaling. No dropout training prediction recorded yet.
- Then scale separately toward the planned full GPT, with explicit predictions before measurements.
- Still required: full-model one-batch overfit, Perfetto train-step trace and fusion inspection, attention-pattern plot/explanation, and remaining done-criteria understanding checks. **Module 1 is not complete.**
