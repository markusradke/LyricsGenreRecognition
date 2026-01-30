from typing import Tuple

import pandas as pd
from pandas import DataFrame, Series


def _concat_with_group(X: DataFrame, y: Series, group: str) -> DataFrame:
    """Return X and y concatenated where y has column name `group`."""
    if y.name != group:
        y = y.rename(group)
    return pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1,
    )


def draw_stratified_subsample(
    X_train: DataFrame,
    y_train: Series,
    group: str = "label",
    frac: float = 1.0,
    random_state: int = 42,
) -> Tuple[DataFrame, Series]:
    """Draw a stratified subsample of training data.

    Args:
        X_train: Training features.
        y_train: Training labels (Series).
        group: Column name used for stratification. y_train will be renamed
            to this name if necessary.
        frac: Fraction in (0, 1]; 1.0 means no subsampling.
        random_state: Random seed for sampling.

    Returns:
        A tuple (X_train_sub, y_train_sub).
    """
    if frac >= 1.0:
        return X_train, y_train
    if frac <= 0.0:
        raise ValueError("frac must be > 0.0")

    Xy_train = _concat_with_group(X_train, y_train, group)

    if group not in Xy_train.columns:
        raise KeyError(f"group column '{group}' not found in training data")

    Xy_train_sub = (
        Xy_train.groupby(group)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )

    return Xy_train_sub.drop(columns=[group]), Xy_train_sub[group]
