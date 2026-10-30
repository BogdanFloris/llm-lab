import pickle

import jax


def save_checkpoint(path, params, config):
    checkpoint = {"params": jax.device_get(params), "config": config}
    with path.open("wb") as file:
        pickle.dump(checkpoint, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path):
    """Load a trusted local checkpoint; pickle is unsafe for untrusted files."""
    with path.open("rb") as file:
        return pickle.load(file)
