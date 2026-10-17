# LLM study plan

Goal: understand LLMs well enough to trace and visualize what happens inside them, and to read new architecture and interpretability papers properly. Two texts mark the finish line:

- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) (DeepSeek-AI, September 2026)
- [Neural Networks Are Already Clean](https://already-clean.goodfire.com) (Nick Cammarata, Goodfire; a 65-minute read)

This replaces the October JAX block (small transformer, then RL, then Pallas). RL and Pallas moved to [Later](#later) because neither text needs them.

## Shape

| # | Module | Read | Build | Hours |
|---|---|---|---|---|
| 1 | GPT from scratch | Karpathy's video, JAX docs | the JAX lab | ~15 |
| 2 | A real model | RoPE, GQA, transformer math | Qwen3-0.6B in plain JAX | ~12 |
| 3 | Inference and the KV cache | rooflines, MLA, sparse attention, then **DeepSeek-V4.1-Flash** | a KV cache, compression experiments | ~18 |
| 4 | Neurons | circuits, superposition, SAEs, then **Already Clean** | activation recorder, neuron viewer, mini-audit | ~18 |


- 1 and 2 are the base. 3 and 4 can go in either order.
- From module 2 on, everything runs on a model you wrote, so any activation or KV entry is one return value away. No hooks, no framework.
- Every module ends with something you trace or visualize. That's the moat: few people both read frontier architecture papers and build the tools to see inside models. You have the tooling half; this adds the model half and makes you use both every time.

## 1. GPT from scratch

The JAX lab: Karpathy's "Let's build GPT" in plain JAX, no Flax.

Read
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2 h). Pause before each step and write it in JAX first.
- JAX docs: [key concepts](https://docs.jax.dev/en/latest/key-concepts.html), [jit](https://docs.jax.dev/en/latest/jit-compilation.html), [random numbers](https://docs.jax.dev/en/latest/random-numbers.html), [pytrees](https://docs.jax.dev/en/latest/pytrees.html), [the sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

Build
- You: `init_params`, `forward`, `loss_fn` and a jitted `train_step`. Agent: data loading, training loop, sampling, tests.
- Milestones: bigram (val ≈ 2.48 at learning rate 1e-2), one attention head, multi-head attention plus an MLP, residual blocks with LayerNorm, dropout, then the video's 10.7M-parameter model on the 5070 (val ≈ 1.48 in the video).

Trace
- Wrap a few train steps in `jax.profiler.trace(dir, create_perfetto_link=True)` and open the trace in Perfetto. Find the matmuls and the softmax, and see what XLA fused.
- Plot one head's attention pattern over a line of Shakespeare.

Done when the tests pass (shapes under jit, initial loss ≈ ln 65, causality, overfits one batch) and you can explain why every `jit`, `grad` and key split is where it is.

## 2. A real model

Qwen3-0.6B in plain JAX. It has the modern block (RMSNorm, RoPE, SwiGLU, grouped-query attention, QK-norm), fits the 5070 easily, and is the family the Goodfire essay audits (Qwen3-32B).

Read
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) (EleutherAI).
- [GQA](https://arxiv.org/abs/2305.13245), sections 1–2.
- [All the Transformer Math You Need to Know](https://jax-ml.github.io/scaling-book/transformers/) (How to Scale Your Model). Count Qwen3-0.6B's parameters and FLOPs per token by hand.
- [`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) as the reference. Read it, don't copy it.

Build
- You: load the [weights](https://huggingface.co/Qwen/Qwen3-0.6B) into a pytree, write `forward` and greedy decoding. Agent: download, tokenizer, and a test against Hugging Face transformers.
- Let `forward` return any intermediate you ask for (residual stream, attention weights, MLP activations). Modules 3 and 4 tap them.

Trace: plot the residual-stream norm per layer for one prompt and explain what you see.

Done when greedy output matches transformers token for token on a few prompts (fp32) and you can draw the block from memory.

## 3. Inference and the KV cache

Read, in order
1. How to Scale Your Model: [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) and [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/). Why decode is memory-bound and why the KV cache dominates at long context.
2. [DeepSeek-V2](https://arxiv.org/abs/2405.04434) section 2.1, Multi-head Latent Attention: each token's KV becomes one small latent shared by all heads.
3. [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556), the DeepSeek Sparse Attention section: a cheap indexer picks the top-k entries each query attends to.
4. [DeepSeek-V4](https://arxiv.org/abs/2606.19348), the hybrid attention section: CSA and HCA compress every m tokens into one entry, next to a sliding window.
5. [YOCO](https://arxiv.org/abs/2405.05254), skim: the upper layers share one KV cache. V4.1's encoder-decoder split is inspired by it.
6. **DeepSeek-V4.1-Flash**, sections 1–2 and 3.2. Skip pre- and post-training on the first pass, and treat mHC, Engram, DSpark and Muon as black boxes.

V4.1's frame (section 2.3): KV cache size multiplies along three axes, entry size (GQA, MLA), sequence (CSA, HCA) and layers (cross-layer reuse). Precision is a fourth factor, which V4.1 pushes to FP4. The build touches all four.

Build, on your Qwen3
- You: a KV cache. Prefill once, then decode one token per step into a fixed-size cache so `jit` compiles once.
- Predict ms per decoded token at 1k and 8k context: bytes read per step (weights plus KV cache) ÷ 672 GB/s, the 5070's memory bandwidth. Then measure.
- A compression table: bytes per token and held-out perplexity for a naive version of each axis, on a model that wasn't trained for any of them:
  - precision: FP8, and FP4 E2M1 with one E4M3 scale per 16 channels, quantized after RoPE (V4.1 section 2.4.4), simulated by quantizing then dequantizing
  - sequence: a sliding window, or mean-pooling every m keys
  - layers: a layer reusing the previous layer's KV
  - entry (optional): cache a rank-r latent of K and V from an SVD of their projection weights, MLA-style
- You write the variants; the agent writes the perplexity loop.

Trace: prefill and decode side by side in one Perfetto trace, and your table plotted next to V4.1's Figure 1(b).

Done when you can explain where Qwen3-0.6B's 112 KiB per token (bf16) and V4.1-Flash's 890 bytes of global KV per token go, what the Full, Reindex and Reuse modes each store, and the gap between your naive numbers and DeepSeek's trained ones.

## 4. Neurons

Read, in order
1. [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) (Distill, 2020). Cammarata co-wrote it, and the essay revisits its examples.
2. [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913): what MLP neurons do in a language model.
3. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): the residual stream, QK and OV circuits, induction heads. The first half is enough.
4. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html), the introduction and key results. This is the view the essay argues against.
5. [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html), skim: sparse autoencoders, the field's answer to superposition.
6. **Neural Networks Are Already Clean**.

Build, on your Qwen3
- Activation recorder: stream about 10M tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) through the model and keep the top contexts for each MLP neuron (one channel of `silu(gate) * up`, 3,072 per layer), for its positive and its negative pole. Agent: data plumbing. You: the capture and the top-k bookkeeping.
- Neuron viewer: one static page per neuron, tokens shaded by activation, both poles side by side. Yours to write; it's your trade.
- Mini-audit: label 50 random neuron poles as one job, two related jobs or unclear, and compare with the essay's ~90 / 7 / 2–4 % for Qwen3-32B. Then have the agent label the same 50 without seeing yours, and compare.

Done when you can state the essay's claim and the evidence it re-reads, and say whether your 0.6B audit agrees.

## Later

- RL on a task with a checkable answer (GRPO), from the October plan.
- Pallas kernels. Mosaic GPU's fast paths need Hopper or datacenter Blackwell features the 5070 lacks, so use a free Kaggle or Colab TPU. Start at the [Pallas quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html).
- The rest of V4.1-Flash: MoE, mHC, Muon, Engram, DSpark.
- Circuit tracing: [methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).

## The tutor agent

What the AGENTS.md on erebor should tell it:

- It tutors. It doesn't write the parts marked "You" above unless you ask for a solution after trying.
- Hints escalate: a question, then a pointer to the doc or paper section, then pseudocode, then a snippet only on request.
- Before each build it asks two or three questions on the reading. After, it has you explain the code back.
- Before every run it asks you to predict the number: loss, ms per token, bytes, perplexity.
- It writes the harness: downloads, data plumbing, eval loops, tests, plots.
- It keeps PROGRESS.md: what's done, what was hard, open questions.

