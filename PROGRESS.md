# Progress

## Module 1 — GPT from scratch

### Current state

- Bogdan reports finishing Karpathy's video, including scaling/dropout, and reading some JAX documentation.
- Bogdan wrote the model/components. ARC owns the environment, data, training/evaluation, categorical sampling, checkpoints, and tests. Learner model code has not been implemented or replaced by ARC.
- Latest trained architecture was three small pre-LayerNorm blocks (width 32, 4 heads, context 8). Bogdan has now configured `model.py` for width **384**, **6 heads**, **6 layers**, context **256**, and dropout 0.2. Initialization/forward code is his; normal draws scaled by 0.02, zero biases, and LayerNorm gamma ones/beta zeros remain.
- Dropout rate 0.2 is integrated on attention weights after softmax and on MLP branch outputs before residual addition. Independent keys flow through steps, blocks, branches, and heads. Evaluation accepts `key=None` and disables model dropout.
- Latest verification: **43 non-loss tests pass; one scalar-loss test excluded**. Full-size preflight and the 5000-step run completed, including fixed-shape sampling and checkpoint reload/full-context forward. Peak memory/throughput have not been measured.

### Measured experiments

All training runs below used 3000 steps, context 8, learning rate 0.01, and seed 1337. Validation uses fixed sampled batches, not the full held-out corpus.

| Model | Bogdan's final validation prediction | Final train loss | Final validation loss |
|---|---|---:|---:|
| Bigram | about 2.5 | 2.4308 | 2.4780 |
| Single attention head | 2-point-something, below bigram | 2.3834 | 2.4192 |
| Four heads + MLP | 2.3 | 2.1382 | 2.2078 |
| Three residual blocks + LayerNorm, no dropout | 2.1 | 2.1118 | 2.1741 |
| Same small stack, dropout 0.2 | 2.05 | 2.1631 | 2.2049 |

- Residual-stack initial validation loss: **4.1436**, versus Bogdan's uniform-baseline prediction `ln(65) ≈ 4.174`. Vocabulary has 65 characters, not 64.
- Small-model dropout run: initial validation still 4.1436; final 2.2049 versus prediction 2.05 and no-dropout 2.1741. Both train and validation losses are higher; the smaller train/validation gap is not itself a win. This single run and changed training batch-key progression do not isolate dropout's causal effect. Logged train loss is evaluated with dropout disabled too.
- Large-model predictions: Bogdan expects initial uniform-baseline loss about 4.174 and validation loss 1.6 after 5000 updates.
- Full-size preflight (batch 64, context 256, 5 updates, one evaluation batch): initial validation 4.1857, step-5 validation 3.3828; no allocation error/non-finite report. Artifacts: `runs/20261031-152805-249566/`.
- First full run `runs/20261031-152902-full/` was deliberately stopped at step 500 (exit 143), not an OOM: ARC caught that variable-length generation would cause up to 256 prefix-length JIT compilations. Replaced agent-owned sampling with a fixed-size right-padded context and dynamic last-real-position selection. Focused sampler/padding checks pass; strict padding equivalence uses highest matmul precision to avoid shape-dependent numerical differences. Model/training precision unchanged.
- Replacement full run completed successfully from seed 1337, not preflight/partial weights: `runs/20261031-153314-full/` (exit 0). Initial validation **4.1842** versus prediction about 4.174; final train **0.849040**, final validation **1.581641** versus prediction **1.6**. Best logged validation **1.505895 at step 2500** (train 1.122740). Training loss kept falling while held-out loss worsened later: clear late overfitting pattern on the fixed validation batches.
- Architecture/depth/normalization changed together across some runs; these are not isolated component ablations. New dropout key plumbing also changes training batch-key progression; evaluation batch selection is preserved.
- Initialization experiments on the original single-head model, using seed 1337 and 20 validation batches, without updates: uniform `[0,1)` weights gave **54.4932** (prediction 20); standard-normal weights gave **102.0121** (prediction 10); normal weights scaled by 0.02 gave **4.174362** (prediction 10), near uniform baseline **4.174387**. Centering and scale needed separate explanation; ARC's rough magnitude reasoning was guidance, not Bogdan's independent prediction.
- Bigram deterministic-batch overfit check: Bogdan predicted loss approaching zero; after 400 updates, measured **0.064136** and all four next-token predictions were correct. A full-transformer overfit check remains undone.

### Artifacts and harness

- Bigram: `runs/20261019-175248-589080/`; single head: `runs/20261030-164849-343829/`. These earlier runs saved metrics/samples, not weights.
- Four heads + MLP: `runs/20261030-182905-030836/`; lowest logged validation 2.1811 at step 2700, final 2.2078.
- Residual stack without dropout: `runs/20261031-130855-717286/`; final checkpoint validation 2.1741.
- Latest small-model run: `runs/20261031-145346-793448/`, small stack with dropout 0.2, final validation 2.2049. Checkpoint reload and deterministic finite forward verified; sample saved.
- Latest trained model: `runs/20261031-153314-full/params.pkl`, six layers/width 384/context 256, final validation 1.581641. Reload and finite jitted full-context forward verified; generated text in `sample.txt` is much more word-like but still semantically odd. The step-2500 weights were not saved and cannot be recovered from these artifacts.
- All three later runs contain `params.pkl`; reload and finite jitted forward were verified. Checkpoints save **final**, not best-validation, parameters, vocabulary, and configuration before sampling. No optimizer state/exact resume; only load trusted local pickle files.
- After the full run, ARC added best-validation saving for **future runs**: `best_params.pkl` includes actual checkpoint step/validation loss and is not overwritten when final validation worsens. Regression tested alongside final-checkpoint survival of sampling failures. No best checkpoint exists for the completed full run.
- Training now targets `model.py`; component modules are not standalone model choices. Earlier baselines remain in jj history.
- Nix/direnv + locked uv environment verified on the GPU. JAX preallocation was disabled after reported OOM with desktop GPU use; initialization then passed, though the original failing invocation was not reproduced.
- Bogdan delegated remaining run parameters. ARC set batch 64, 5000 steps, context defaulting to model capacity 256, evaluation every 500 steps over 20 batches, and seed 1337. ARC changed only the optimizer hyperparameter in learner-owned `model.py` to `LEARNING_RATE = 3e-4`; the harness records it. These defaults are not a measured fit/performance guarantee.
- Architecture-review correction: ARC previously missed the video's learned multi-head output projection and output dropout. Bogdan has now added both, including a nested head subtree, projection weight/bias, and independent output/head dropout keys. Exact large-model parameter count has not been measured.
- Placement clarification: feed-forward output dropout is already implemented by the wrapper in `block.forward`, after `feedforward.forward` returns its second linear output. It is equivalent in location to the reference's dropout inside the MLP; do not add a duplicate. Only the multi-head output projection/dropout is missing here.

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

- Bogdan fixed the multi-head constructor's split call, optional evaluation key-list length, and output dropout. Updated tests verify projection-before-dropout with known weights/biases, independent head/output keys, and six-head evaluation without a key; all 40 non-loss tests pass. ARC changed tests/docs only at this checkpoint.
- Next: discuss the best-versus-final validation curve and move to interpretability/trace checks rather than launch another training run. Full run is complete; no background job remains. Ask for predictions before any new experiment.
- Still required: full-model one-batch overfit, Perfetto train-step trace and fusion inspection, attention-pattern plot/explanation, and remaining done-criteria understanding checks. **Module 1 is not complete.**
