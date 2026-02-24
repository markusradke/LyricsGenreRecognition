"""Cross-validation evaluation for Bayesian optimization."""

import warnings

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
    cv_n_jobs: int,
) -> tuple[float, float]:
    """Evaluate parameter configuration using cross-validation.

    Args:
        pipeline_factory: Function that creates pipeline from parameters
        params: Parameter configuration to evaluate
        X: Training features
        y: Training labels
        cv: Number of cross-validation folds
        scoring: Scoring metric for evaluation
        random_state: Random seed for reproducibility
        cv_n_jobs: Number of parallel jobs for CV

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
            n_jobs=cv_n_jobs,
        )

    mean_score = np.mean(scores)
    std_error = np.std(scores, ddof=1) / np.sqrt(len(scores))
    return mean_score, std_error


def determine_cv_n_jobs(n_jobs: int, cv: int) -> int:
    """Determine CV n_jobs for the sequential (non-nested) outer path.

    When the outer optimizer loop is sequential (n_jobs=1), all parallelism
    budget can be given to cross_val_score. When an outer joblib.Parallel
    context exists (n_jobs>1), this must return 1: loky worker processes
    cannot spawn sub-processes, so cross_val_score's Python-level parallelism
    is silently serialized regardless. BLAS/OpenMP inner threads are
    auto-limited by loky to cpu_count() // effective_n_jobs per worker.

    Args:
        n_jobs: Number of parallel jobs for the outer optimizer loop
        cv: Number of cross-validation folds

    Returns:
        CV n_jobs: min(n_jobs, cv) when outer is sequential, else 1
    """
    if n_jobs is None or n_jobs == 1:
        return min(n_jobs or 1, cv)
    return 1
