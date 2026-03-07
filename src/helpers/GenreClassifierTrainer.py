"""Training pipeline for genre classification with Bayesian optimization."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from typing import Any

from .bayesian_optimization import BayesianOptimizer


class GenreClassifierTrainer:
    """Train genre classifiers with optional hyperparameter optimization."""

    def __init__(
        self,
        X_train: pd.DataFrame | pd.Series,
        y_train: pd.Series,
        X_test: pd.DataFrame | pd.Series,
        y_test: pd.Series,
        random_state: int = 42,
        n_jobs: int = 1,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            X_train: Training features (DataFrame) or raw lyrics
            y_train: Training labels
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            feature_names: Feature names for sparse matrices (auto-detected for DataFrames)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_pipeline_ = None
        self.best_params_ = None
        self.coefficients_ = None
        self.vip_ = None
        self.tuning_history_ = None

        if feature_names is not None:
            self.feature_names_ = feature_names
        elif hasattr(X_train, "columns"):
            self.feature_names_ = X_train.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X_train.shape[1])]

    def _create_pipeline(
        self, params: dict[str, Any], n_estimators: int = 1000
    ) -> Pipeline:
        """Create pipeline from parameters.

        Args:
            params: Pipeline parameters. The learner is chosen based on the params dictionary.
                If dictionary contains 'C' and 'l1_ratio', a LogisticRegression with elastic net penalty is created.
                If dictionary contains 'max_features' and 'min_samples_leaf' parameters, a RandomForestClassifier is created.
                n_estimators: Number of trees for Random Forest (ignored for Logistic Regression)

        Returns:
            Configured pipeline. Always employs class-weighted loss.
        """
        steps = []
        steps.append(("scaler", StandardScaler(with_mean=False)))

        if "C" in params and "l1_ratio" in params:
            steps.append(
                (
                    "classifier",
                    LogisticRegression(
                        C=params.get("C", 1.0),
                        l1_ratio=params.get("l1_ratio", 0.5),
                        solver="saga",
                        max_iter=10000,
                        random_state=self.random_state,
                        class_weight="balanced",
                        verbose=0,
                    ),
                )
            )

        if "max_features" in params and "min_samples_leaf" in params:
            steps.append(
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_features=params.get("max_features"),
                        min_samples_leaf=params.get("min_samples_leaf"),
                        random_state=self.random_state,
                        n_jobs=-1,
                        class_weight="balanced",
                        verbose=0,
                    ),
                )
            )

        return Pipeline(steps)

    def _extract_coefficients(
        self, pipeline: Pipeline, feature_names: list[str]
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
        param_space: dict[str, list[float]],
        parsimony_param: str,
        parsimony_ascending: bool,
        n_initial: int = 25,
        n_iterations: int = 100,
        n_points: int = 1,
        stop_iter: int = 20,
        uncertain_jump: int = 5,
        cv: int = 5,
        n_estimators_tuning: int = 500,
        n_estimators_final: int = 1000,
        checkpoint_dir: str | None = None,
    ) -> None:
        """Train model using Bayesian optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            param_space: Parameter search space (log scale for C and target_ratio)
            n_initial: Number of initial samples for Latin hypercube
            n_iterations: Number of Bayesian optimization iterations
            cv: Number of cross-validation folds
            n_estimators_tuning: Number of trees for RF during tuning phase
            n_estimators_final: Number of trees for RF in final model
            checkpoint_dir: Directory for checkpoints
            parsimony_param: Parameter to use for 1-SE rule
        """

        is_random_forest = (
            "max_features" in param_space and "min_samples_leaf" in param_space
        )

        if is_random_forest:
            print(f"Using {n_estimators_tuning} trees during hyperparameter tuning...")

            def pipeline_factory_tuning(params):
                return self._create_pipeline(params, n_estimators=n_estimators_tuning)

        else:
            pipeline_factory_tuning = self._create_pipeline

        optimizer = BayesianOptimizer(
            param_space=param_space,
            n_initial=n_initial,
            n_iterations=n_iterations,
            n_points=n_points,
            stop_iter=stop_iter,
            uncertain_jump=uncertain_jump,
            cv=cv,
            scoring="f1_macro",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            checkpoint_dir=checkpoint_dir,
        )

        self.tuning_history_ = optimizer.run_search(
            pipeline_factory_tuning, self.X_train, self.y_train
        )

        print("Selecting best parameters according to 1-SE rule...")
        self.best_params_ = optimizer.select_best_one_se(
            param_parsim=parsimony_param, ascending=parsimony_ascending
        )
        print(f"{pd.DataFrame(self.best_params_, index=['Best Parameters:'])}")

        if is_random_forest:
            print(
                f"Retraining best pipeline with {n_estimators_final} trees on full training data..."
            )
            self.best_pipeline_ = self._create_pipeline(
                self.best_params_, n_estimators=n_estimators_final
            )
        else:
            print("Retraining best pipeline on full training data...")
            self.best_pipeline_ = self._create_pipeline(self.best_params_)

        self.best_pipeline_.fit(self.X_train, self.y_train)

        if "C" in self.best_params_ and "l1_ratio" in self.best_params_:
            print("Extracting model coefficients...")
            self.coefficients_ = self._extract_coefficients(
                self.best_pipeline_, self.feature_names_
            )
        if (
            "max_features" in self.best_params_
            and "min_samples_leaf" in self.best_params_
        ):
            print("Calculating holdout permutation importance...")
            self.vip_ = permutation_importance(
                self.best_pipeline_,
                self.X_test,
                self.y_test,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

    def fit_logistic_regression_with_fixed_params(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.5,
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
        }

        print("Training pipeline with fixed parameters...")
        self.best_pipeline_ = self._create_pipeline(self.best_params_)

        self.best_pipeline_.fit(self.X_train, self.y_train)
        self.coefficients_ = self._extract_coefficients(
            self.best_pipeline_, self.feature_names_
        )

    def fit_random_forest_with_fixed_params(
        self,
        max_features: float = 0.5,
        min_samples_leaf: int = 2,
    ) -> None:
        """Train Random Forest with fixed hyperparameters.

        Args:
            X_train: Training features
            y_train: Training labels
            max_features: Max features for Random Forest
            min_samples_leaf: Min samples split for Random Forest
        """
        self.best_params_ = {
            "max_features": max_features,
            "min_samples_leaf": min_samples_leaf,
        }

        print("Training Random Forest pipeline with fixed parameters...")
        self.best_pipeline_ = self._create_pipeline(self.best_params_)

        self.best_pipeline_.fit(self.X_train, self.y_train)

    def get_results(self) -> dict:
        """Get training results.

        Returns:
            Dictionary containing pipeline, parameters, coefficients, and history
        """
        return {
            "pipeline": self.best_pipeline_,
            "params": self.best_params_,
            "coefficients": self.coefficients_,
            "holdout_permutation_importance": self.vip_,
            "tuning_history": self.tuning_history_,
        }
