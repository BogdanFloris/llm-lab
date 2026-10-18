import argparse
from datetime import datetime
import json
from pathlib import Path

import jax
import numpy as np

from bigram import init_params, loss_fn, optimizer, train_step
from data import get_batch, load_data
from sample import generate


def evaluate(params, tokens, key, batch_size, block_size, batches):
    evaluate_batch = jax.jit(loss_fn)
    losses = []
    for batch_key in jax.random.split(key, batches):
        x, y = get_batch(tokens, batch_key, batch_size, block_size)
        losses.append(float(evaluate_batch(params, x, y)))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tokenizer, train, val = load_data()
    init_key, train_key, eval_key, sample_key = jax.random.split(
        jax.random.key(args.seed), 4
    )
    params = init_params(init_key, len(tokenizer))
    opt_state = optimizer.init(params)
    output = args.out or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "out": str(output), "vocab_size": len(tokenizer)}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"Devices: {jax.devices()}")
    print(f"Output: {output}")
    with (output / "metrics.jsonl").open("w") as metrics:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                train_eval_key, val_eval_key = jax.random.split(eval_key)
                row = {
                    "step": step,
                    "train_loss": evaluate(
                        params, train, train_eval_key,
                        args.batch_size, args.block_size, args.eval_batches,
                    ),
                    "val_loss": evaluate(
                        params, val, val_eval_key,
                        args.batch_size, args.block_size, args.eval_batches,
                    ),
                }
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(
                    f"step {step:5d} | train {row['train_loss']:.4f}"
                    f" | val {row['val_loss']:.4f}"
                )
            if step == args.steps:
                break
            train_key, batch_key = jax.random.split(train_key)
            x, y = get_batch(train, batch_key, args.batch_size, args.block_size)
            params, opt_state, _ = train_step(params, opt_state, x, y)

    text = generate(params, tokenizer, sample_key, max_new_tokens=args.sample_tokens)
    (output / "sample.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
