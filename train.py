import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import numpy as np

import model
from checkpoint import save_checkpoint
from data import get_batch, load_data
from sample import generate


def evaluate(loss_fn, params, tokens, key, batch_size, block_size, batches):
    evaluate_batch = jax.jit(
        lambda params, x, y: loss_fn(params, x, y, key=None, is_training=False)
    )
    losses = []
    for batch_key in jax.random.split(key, batches):
        x, y = get_batch(tokens, batch_key, batch_size, block_size)
        losses.append(float(evaluate_batch(params, x, y)))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--steps", type=int, default=5000)
    _ = parser.add_argument("--batch-size", type=int, default=64)
    _ = parser.add_argument("--block-size", type=int, default=model.T_MAX)
    _ = parser.add_argument("--eval-every", type=int, default=500)
    _ = parser.add_argument("--eval-batches", type=int, default=20)
    _ = parser.add_argument("--seed", type=int, default=1337)
    _ = parser.add_argument("--sample-tokens", type=int, default=300)
    _ = parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 1 <= args.block_size <= model.T_MAX:
        parser.error(f"--block-size must be between 1 and model.T_MAX ({model.T_MAX})")

    tokenizer, train, val = load_data()
    init_key, train_key, eval_key, sample_key = jax.random.split(
        jax.random.key(args.seed), 4
    )
    params = model.init_params(init_key, len(tokenizer))
    opt_state = model.optimizer.init(params)
    output = args.out or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    _ = output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "out": str(output),
        "vocab_size": len(tokenizer),
        "vocab": tokenizer.chars,
        "learning_rate": model.LEARNING_RATE,
        "model": {
            "module": model.__name__,
            "n_embed": model.N_EMBED,
            "t_max": model.T_MAX,
            "no_heads": model.N_HEAD,
            "n_blocks": len(params["blocks"]),
            "dropout_rate": model.DROPOUT_RATE,
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"Devices: {jax.devices()}")
    print(f"Output: {output}")
    best_val_loss = float("inf")
    with (output / "metrics.jsonl").open("w") as metrics:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                train_eval_key, val_eval_key = jax.random.split(eval_key)
                row = {
                    "step": step,
                    "train_loss": evaluate(
                        model.loss_fn, params, train, train_eval_key,
                        args.batch_size, args.block_size, args.eval_batches,
                    ),
                    "val_loss": evaluate(
                        model.loss_fn, params, val, val_eval_key,
                        args.batch_size, args.block_size, args.eval_batches,
                    ),
                }
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(
                    f"step {step:5d} | train {row['train_loss']:.4f}"
                    f" | val {row['val_loss']:.4f}"
                )
                if row["val_loss"] < best_val_loss:
                    best_val_loss = row["val_loss"]
                    save_checkpoint(
                        output / "best_params.pkl", params,
                        {**config, "checkpoint_step": step, "validation_loss": best_val_loss},
                    )
            if step == args.steps:
                break
            train_key, batch_key, dropout_key = jax.random.split(train_key, 3)
            x, y = get_batch(train, batch_key, args.batch_size, args.block_size)
            params, opt_state, _ = model.train_step(params, opt_state, x, y, dropout_key)

    save_checkpoint(output / "params.pkl", params, config)
    text = generate(
        model.forward, params, tokenizer, sample_key,
        block_size=args.block_size, max_new_tokens=args.sample_tokens,
    )
    _ = (output / "sample.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
