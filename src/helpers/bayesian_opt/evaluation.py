"""Cross-validation evaluation for Bayesian optimization."""

import warnings
import gc

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_params(
    pipeline_factory,
    params: dict,
    X,
    y,
    cv: int,
    scoring: str,
    random_state: int,
) -> tuple[float, float]:
    """Evaluate parameter configuration using cross-validation.

    CV always runs sequentially. RF parallelism is handled internally by
    RandomForestClassifier(n_jobs=-1); all other models are fast enough
    that outer parallelism is not needed.

    Args:
        pipeline_factory: Function that creates pipeline from parameters
        params: Parameter configuration to evaluate
        X: Training features
        y: Training labels
        cv: Number of cross-validation folds
        scoring: Scoring metric for evaluation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (mean_score, standard_error)
    """
    pipeline = pipeline_factory(params)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv_splitter,
            scoring=scoring,
        )

    mean_score = np.mean(scores)
    std_error = np.std(scores, ddof=1) / np.sqrt(len(scores))

    if np.isnan(mean_score) or np.isnan(std_error):
        print(f"WARNING: NaN score detected for params: {params}. Assigning -inf.")
        mean_score = -np.inf
        std_error = 0.0

    del pipeline
    gc.collect()

    return mean_score, std_error
