"""Checkpoint management for Bayesian optimization."""

import hashlib
import json
import pickle
from pathlib import Path


def create_model_hash(X_shape: tuple, feature_names: list[str], config: dict) -> str:
    """Create hash from model configuration.

    Args:
        X_shape: Shape of training data
        feature_names: List of feature names
        config: Model configuration dictionary

    Returns:
        MD5 hash string
    """
    hash_input = {
        "data_shape": X_shape,
        "n_features": len(feature_names),
        "feature_names": sorted(feature_names),
        "config": config,
    }
    hash_string = json.dumps(hash_input, sort_keys=True)
    return hashlib.md5(hash_string.encode()).hexdigest()


def save_checkpoint(
    checkpoint_data: dict, checkpoint_dir: Path, model_hash: str
) -> None:
    """Save optimization checkpoint.

    Args:
        checkpoint_data: Data to checkpoint
        checkpoint_dir: Directory for checkpoints
        model_hash: Model configuration hash
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"checkpoint_{model_hash}.pkl"

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_dir: Path, model_hash: str) -> dict | None:
    """Load optimization checkpoint if it exists.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_hash: Model configuration hash

    Returns:
        Checkpoint data or None if not found
    """
    checkpoint_file = checkpoint_dir / f"checkpoint_{model_hash}.pkl"

    if checkpoint_file.exists():
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)
    return None
