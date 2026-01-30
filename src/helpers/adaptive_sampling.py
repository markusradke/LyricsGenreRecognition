"""Adaptive sampling for class imbalance handling."""

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import NearMiss


def compute_adaptive_quantile(n_classes: int) -> float:
    """Compute quantile threshold based on number of classes.

    Args:
        n_classes: Number of classes in the dataset

    Returns:
        Quantile value (0.5, 0.6, or 0.65)
    """
    if n_classes <= 10:
        return 0.5
    elif n_classes <= 20:
        return 0.6
    else:
        return 0.65


def compute_sampling_strategy(
    y: np.ndarray, target_ratio: float = 1.5
) -> tuple[dict, dict, int]:
    """Compute sampling strategy for class balancing.

    Args:
        y: Target labels
        target_ratio: Target ratio between classes

    Returns:
        Tuple of (downsample_strategy, upsample_strategy, reference_count)
    """
    class_counts = Counter(y)
    n_classes = len(class_counts)

    counts = np.array(list(class_counts.values()))
    quantile = compute_adaptive_quantile(n_classes)
    reference_count = int(np.quantile(counts, quantile))

    downsample_strategy = {}
    upsample_strategy = {}

    for class_label, count in class_counts.items():
        if count > reference_count * target_ratio:
            downsample_strategy[class_label] = int(reference_count * target_ratio)
        elif count < reference_count / target_ratio:
            upsample_strategy[class_label] = int(reference_count / target_ratio)

    return downsample_strategy, upsample_strategy, reference_count


class AdaptiveSampler:
    """Adaptive sampler for handling class imbalance."""

    def __init__(self, target_ratio: float = 1.5, random_state: int = 42) -> None:
        """Initialize adaptive sampler.

        Args:
            target_ratio: Target ratio between classes
            random_state: Random seed for reproducibility
        """
        self.target_ratio = target_ratio
        self.random_state = random_state
        self.downsampler = None
        self.upsampler = None
        self.reference_count = None

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply adaptive sampling to balance classes.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        downsample_strategy, upsample_strategy, self.reference_count = (
            compute_sampling_strategy(y, self.target_ratio)
        )

        X_resampled, y_resampled = X, y

        if downsample_strategy:
            self.downsampler = NearMiss(sampling_strategy=downsample_strategy)
            X_resampled, y_resampled = self.downsampler.fit_resample(
                X_resampled, y_resampled
            )

        if upsample_strategy:
            try:
                self.upsampler = BorderlineSMOTE(
                    sampling_strategy=upsample_strategy,
                    random_state=self.random_state,
                )
                X_resampled, y_resampled = self.upsampler.fit_resample(
                    X_resampled, y_resampled
                )
            except ValueError:
                self.upsampler = SMOTE(
                    sampling_strategy=upsample_strategy,
                    random_state=self.random_state,
                )
                X_resampled, y_resampled = self.upsampler.fit_resample(
                    X_resampled, y_resampled
                )

        return X_resampled, y_resampled
