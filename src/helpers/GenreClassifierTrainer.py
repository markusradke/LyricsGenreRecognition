"""Training pipeline for genre classification with Bayesian optimization."""

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .adaptive_sampling import AdaptiveSampler
from .bayesian_optimization import BayesianOptimizer


class GenreClassifierTrainer:
    """Train genre classifiers with optional hyperparameter optimization."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        """Initialize trainer.

        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_pipeline_ = None
        self.best_params_ = None
        self.coefficients_ = None
        self.tuning_history_ = None

    def _create_pipeline(self, params: dict[str, float]) -> ImbPipeline:
        """Create pipeline from parameters.

        Args:
            params: Pipeline parameters (C, l1_ratio, target_ratio)

        Returns:
            Configured pipeline
        """
        return ImbPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "sampler",
                    AdaptiveSampler(
                        target_ratio=params["target_ratio"],
                        random_state=self.random_state,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=params["C"],
                        l1_ratio=params["l1_ratio"],
                        solver="saga",
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _extract_coefficients(
        self, pipeline: ImbPipeline, feature_names: list[str]
    ) -> pd.DataFrame:
        """Extract model coefficients as DataFrame.

        Args:
            pipeline: Fitted pipeline
            feature_names: Feature column names

        Returns:
            Coefficient DataFrame
        """
        classifier: LogisticRegression = pipeline.named_steps["classifier"]
        return pd.DataFrame(
            classifier.coef_.T,
            columns=classifier.classes_,
            index=feature_names,
        )

    def fit_with_bayesian_optimization(
        self,
        param_space: dict[str, list[float]] | None = None,
        n_initial: int = 25,
        n_iterations: int = 50,
        cv: int = 5,
        checkpoint_dir: str | None = None,
        parsimony_param: str = "C",
    ) -> None:
        """Train model using Bayesian optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            param_space: Parameter search space (log scale for C and target_ratio)
            n_initial: Number of initial samples for Latin hypercube
            n_iterations: Number of Bayesian optimization iterations
            cv: Number of cross-validation folds
            checkpoint_dir: Directory for checkpoints
            parsimony_param: Parameter to use for 1-SE rule
        """
        if param_space is None:
            param_space = {
                "C": [-3, 2],
                "l1_ratio": [0, 1],
                "target_ratio": [np.log10(1.0), np.log10(5.0)],
            }

        optimizer = BayesianOptimizer(
            param_space=param_space,
            n_initial=n_initial,
            n_iterations=n_iterations,
            cv=cv,
            scoring="f1_macro",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            checkpoint_dir=checkpoint_dir,
        )

        self.tuning_history_ = optimizer.run_search(
            self._create_pipeline, self.X_train, self.y_train
        )

        print("Selecting best parameters according to 1-SE rule...")
        self.best_params_ = optimizer.select_best_one_se(
            param_parsim=parsimony_param, ascending=True
        )

        print("Retraining best pipeline on full training data...")
        self.best_pipeline_ = self._create_pipeline(self.best_params_)
        self.best_pipeline_.fit(self.X_train, self.y_train)

        self.coefficients_ = self._extract_coefficients(
            self.best_pipeline_, self.X_train.columns.tolist()
        )

    def fit_with_fixed_params(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.5,
        target_ratio: float = 3.0,
        cv: int | None = None,
    ) -> None:
        """Train model with fixed hyperparameters.

        Args:
            X_train: Training features
            y_train: Training labels
            C: Regularization strength
            l1_ratio: ElasticNet mixing parameter
            target_ratio: Sampling target ratio
        """
        self.best_params_ = {
            "C": C,
            "l1_ratio": l1_ratio,
            "target_ratio": target_ratio,
        }

        print("Training pipeline with fixed parameters...")
        self.best_pipeline_ = self._create_pipeline(self.best_params_)

        self.best_pipeline_.fit(self.X_train, self.y_train)
        self.coefficients_ = self._extract_coefficients(
            self.best_pipeline_, self.X_train.columns.tolist()
        )

    def get_results(self) -> dict:
        """Get training results.

        Returns:
            Dictionary containing pipeline, parameters, coefficients, and history
        """
        return {
            "pipeline": self.best_pipeline_,
            "params": self.best_params_,
            "coefficients": self.coefficients_,
            "tuning_history": self.tuning_history_,
        }
