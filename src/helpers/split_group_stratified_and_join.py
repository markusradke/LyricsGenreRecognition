import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


def split_group_stratified_and_join(
    labels_and_group: pd.DataFrame,
    X: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X and labels by groups ensuring stratification by group's dominant label.
    Dataset is split on group level and then joined back to features and labels.

    Args:
        labels_and_group: DataFrame with columns 'group' and 'label'.
        X: Feature DataFrame aligned with labels_and_group by row order.
        test_size: Fraction of groups to use for test.
        random_state: Random state for reproducibility (seed 42 by convention).

    Returns:
        X_train, X_test, y_train, y_test
    """
    group_train, group_test = _split_by_group(labels_and_group, test_size, random_state)

    train_mask, test_mask = _create_train_test_masks(
        labels_and_group, group_train, group_test
    )

    X_train, X_test, y_train, y_test = _split_X_labels(
        X, labels_and_group, train_mask, test_mask
    )
    return X_train, X_test, y_train, y_test


def _split_by_group(
    labels_and_group: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.Index, pd.Index]:
    """Return group indices for train and test using stratification by dominant label."""
    group_labels = _get_group_labels(labels_and_group)
    group_train, group_test = train_test_split(
        group_labels.index,
        test_size=test_size,
        stratify=group_labels.values,
        random_state=random_state,
    )
    return group_train, group_test


def _get_group_labels(labels_and_group: pd.DataFrame) -> pd.Series:
    """Map each group to its most frequent label (mode).

    Assumes labels_and_group has columns 'group' and 'label'.
    """
    return labels_and_group.groupby("group")["label"].agg(
        lambda x: x.value_counts().idxmax()
    )


def _create_train_test_masks(
    labels_and_group: pd.DataFrame, group_train: pd.Index, group_test: pd.Index
) -> Tuple[pd.Series, pd.Series]:
    """Create boolean masks for rows that belong to train/test groups."""
    train_mask = labels_and_group["group"].isin(group_train)
    test_mask = labels_and_group["group"].isin(group_test)
    return train_mask, test_mask


def _split_X_labels(
    X: pd.DataFrame,
    labels_and_group: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split and reset indices for features X and labels Series."""
    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = labels_and_group.loc[train_mask, "label"].reset_index(drop=True)
    y_test = labels_and_group.loc[test_mask, "label"].reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def plot_genre_distribution(y_train: pd.Series, y_test: pd.Series) -> plt.Figure:
    """Plot relative label frequencies for train and test as grouped horizontal bars.

    Args:
        y_train: Series of labels for the training set.
        y_test: Series of labels for the test set.

    Returns:
        Matplotlib Figure with horizontal grouped bars (train grey, test #c20d40).
    """
    train_rel = y_train.value_counts(normalize=True)
    test_rel = y_test.value_counts(normalize=True)

    labels = train_rel.index.union(test_rel.index)
    train_aligned = train_rel.reindex(labels, fill_value=0)
    test_aligned = test_rel.reindex(labels, fill_value=0)

    combined = train_aligned + test_aligned
    labels_sorted = combined.sort_values(ascending=False).index

    train_vals = train_aligned.reindex(labels_sorted, fill_value=0).to_numpy()
    test_vals = test_aligned.reindex(labels_sorted, fill_value=0).to_numpy()

    n = len(labels_sorted)
    y_pos = np.arange(n)
    bar_height = 0.4

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.4)))
    ax.barh(
        y_pos + bar_height / 2,
        test_vals,
        height=bar_height,
        color="#c20d40",
        label="test",
    )
    ax.barh(
        y_pos - bar_height / 2,
        train_vals,
        height=bar_height,
        color="#c1c1c1",
        label="train",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Relative frequency")
    ax.set_xlim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
