"""Training pipeline for genre classification with Bayesian optimization."""

from typing import Callable

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from .adaptive_sampling import AdaptiveSampler
from .bayesian_optimization import BayesianOptimizer


def create_pipeline_factory(
    random_state: int = 42,
) -> Callable[[dict[str, float]], ImbPipeline]:
    """Create pipeline factory for model training.

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Factory function that creates pipeline from parameters
    """

    def factory(params: dict[str, float]) -> ImbPipeline:
        return ImbPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "sampler",
                    AdaptiveSampler(
                        target_ratio=params["target_ratio"],
                        random_state=random_state,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=params["C"],
                        l1_ratio=params["l1_ratio"],
                        penalty="elasticnet",
                        solver="saga",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    return factory


def train_model_with_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_space: dict[str, list[float]] | None = None,
    n_initial: int = 25,
    n_iterations: int = 50,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    checkpoint_dir: str | None = None,
) -> tuple[ImbPipeline, dict]:
    """Train model using Bayesian optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        param_space: Parameter search space (log scale for C and target_ratio)
        n_initial: Number of initial samples for Latin hypercube
        n_iterations: Number of Bayesian optimization iterations
        cv: Number of cross-validation folds
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs for initial phase (-1 for all cores)
        checkpoint_dir: Directory for checkpoints (None to disable)

    Returns:
        Tuple of (best_pipeline, optimization_results)
    """
    if param_space is None:
        param_space = {
            "C": [-3, 2],
            "l1_ratio": [0, 1],
            "target_ratio": [np.log10(1.0), np.log10(5.0)],
        }

    pipeline_factory = create_pipeline_factory(random_state)

    optimizer = BayesianOptimizer(
        param_space=param_space,
        n_initial=n_initial,
        n_iterations=n_iterations,
        cv=cv,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=n_jobs,
        checkpoint_dir=checkpoint_dir,
    )

    tuning_history = optimizer.run_search(pipeline_factory, X_train, y_train)

    print("Selecting best parameters according to 1-SE rule...")
    best_params = optimizer.select_best_one_se(param_parsim="C", ascending=True)
    best_pipeline = pipeline_factory(best_params)

    print("Retraining best pipeline on full training data...")
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, best_params, tuning_history
