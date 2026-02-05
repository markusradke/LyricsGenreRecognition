"""Bayesian optimization for hyperparameter tuning."""

import hashlib
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import StratifiedKFold, cross_val_score


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


def _transform_sample_to_params(
    sample: np.ndarray, param_space: dict[str, list[float]]
) -> dict[str, float]:
    """Transform normalized sample to parameter values.

    Args:
        sample: Normalized sample values [0, 1]
        param_space: Parameter search space with ranges

    Returns:
        Dictionary of parameter values
    """
    params = {}
    for i, (param_name, param_range) in enumerate(param_space.items()):
        value_range = param_range[1] - param_range[0]
        scaled_value = param_range[0] + sample[i] * value_range

        if param_name in ["C", "target_ratio"]:
            params[param_name] = 10**scaled_value
        else:
            params[param_name] = scaled_value

    return params


def latin_hypercube_sample(
    param_space: dict[str, list[float]],
    n_samples: int,
    random_state: int = 42,
) -> list[dict[str, float]]:
    """Generate parameter samples using Latin hypercube sampling.

    Args:
        param_space: Parameter search space with ranges
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        List of parameter dictionaries
    """
    sampler = qmc.LatinHypercube(d=len(param_space), seed=random_state)
    samples = sampler.random(n=n_samples)

    return [_transform_sample_to_params(sample, param_space) for sample in samples]


class BayesianOptimizer:
    """Bayesian optimizer for hyperparameter tuning."""

    def __init__(
        self,
        param_space: dict[str, list[float]],
        n_initial: int = 20,
        n_iterations: int = 30,
        cv: int = 5,
        scoring: str = "f1_macro",
        random_state: int = 42,
        n_jobs: int = 1,
        checkpoint_dir: str | None = None,
    ) -> None:
        """Initialize Bayesian optimizer.

        Args:
            param_space: Parameter search space with ranges
            n_initial: Number of initial Latin hypercube samples
            n_iterations: Number of Bayesian optimization iterations
            cv: Number of cross-validation folds
            scoring: Scoring metric for evaluation
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs for initial phase (-1 for all)
            checkpoint_dir: Directory for saving checkpoints (None to disable)
        """
        self.param_space = param_space
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.results: list[dict] = []
        self.tuning_history: pd.DataFrame | None = None
        self.model_hash: str | None = None

    def run_search(self, pipeline_factory, X, y) -> dict:
        """Run optimization procedure with checkpointing.

        Args:
            pipeline_factory: Function that creates pipeline from parameters
            X: Training features
            y: Training labels

        Returns:
            Dictionary with best parameters and all results
        """
        self._initialize_optimization(X)
        completed_initial = self._load_checkpoint_if_exists()
        initial_params = latin_hypercube_sample(
            self.param_space, self.n_initial, self.random_state
        )

        self._run_initial_phase(
            pipeline_factory, X, y, initial_params, completed_initial
        )
        self._run_bayesian_phase(pipeline_factory, X, y)
        self._build_tuning_history()

        return self.tuning_history

    def _build_tuning_history(self) -> None:
        """Build tuning history DataFrame from results."""
        records = []
        param_names = sorted(self.param_space.keys())

        for i, result in enumerate(self.results):
            iteration = 0 if i < self.n_initial else i - self.n_initial + 1
            record = {
                "iteration": iteration,
                "score_mean": result["score_mean"],
                "score_se": result["score_se"],
            }
            for param_name in param_names:
                record[param_name] = result["params"][param_name]
            records.append(record)

        self.tuning_history = pd.DataFrame(records)

    def _initialize_optimization(self, X) -> None:
        """Initialize optimization state and create model hash.

        Args:
            X: Training features
        """
        feature_names = (
            list(X.columns)
            if hasattr(X, "columns")
            else [f"f_{i}" for i in range(X.shape[1])]
        )

        config = {
            "param_space": self.param_space,
            "n_initial": self.n_initial,
            "n_iterations": self.n_iterations,
            "cv": self.cv,
            "scoring": self.scoring,
            "random_state": self.random_state,
        }

        self.model_hash = create_model_hash(X.shape, feature_names, config)

    def _load_checkpoint_if_exists(self) -> int:
        """Load checkpoint if available.

        Returns:
            Number of completed initial evaluations
        """
        if not self.checkpoint_dir:
            return 0

        checkpoint = load_checkpoint(self.checkpoint_dir, self.model_hash)
        if checkpoint:
            print(f"Loaded checkpoint with {len(checkpoint['results'])} results")
            self.results = checkpoint["results"]
            return min(len(self.results), self.n_initial)

        return 0

    def _run_initial_phase(
        self, pipeline_factory, X, y, initial_params, completed_initial
    ) -> None:
        """Execute initial phase with Latin hypercube sampling.

        Args:
            pipeline_factory: Function that creates pipeline from parameters
            X: Training features
            y: Training labels
            initial_params: List of initial parameter configurations
            completed_initial: Number of already completed evaluations
        """
        print("=" * 60)
        print("Starting Initial Phase:")
        if self.n_jobs != 1:
            print(f"Using {self.n_jobs} parallel jobs")
        print("=" * 60)

        if completed_initial >= self.n_initial:
            return

        remaining_params = initial_params[completed_initial:]

        if self.n_jobs == 1:
            self._run_initial_sequential(
                pipeline_factory, X, y, remaining_params, completed_initial
            )
        else:
            self._run_initial_parallel(
                pipeline_factory, X, y, remaining_params, completed_initial
            )

    def _run_initial_sequential(
        self, pipeline_factory, X, y, remaining_params, completed_initial
    ) -> None:
        """Run initial evaluations sequentially with checkpointing.

        Args:
            pipeline_factory: Function that creates pipeline from parameters
            X: Training features
            y: Training labels
            remaining_params: Parameter configurations to evaluate
            completed_initial: Number of already completed evaluations
        """
        for i, params in enumerate(remaining_params):
            idx = completed_initial + i + 1
            print(f"Initial evaluation {idx}/{self.n_initial}")
            mean_score, std_error = self._evaluate_params(
                pipeline_factory, params, X, y
            )
            self.results.append(
                {"params": params, "score_mean": mean_score, "score_se": std_error}
            )
            print(f"Score: {mean_score:.4f} ± {std_error:.4f}")
            print("-" * 60)

            if self.checkpoint_dir:
                self._save_checkpoint()

    def _run_initial_parallel(
        self, pipeline_factory, X, y, remaining_params, completed_initial
    ) -> None:
        """Run initial evaluations in parallel with batch checkpointing.

        Args:
            pipeline_factory: Function that creates pipeline from parameters
            X: Training features
            y: Training labels
            remaining_params: Parameter configurations to evaluate
            completed_initial: Number of already completed evaluations
        """
        results_parallel = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_params)(pipeline_factory, params, X, y)
            for params in remaining_params
        )

        for i, (params, (mean_score, std_error)) in enumerate(
            zip(remaining_params, results_parallel)
        ):
            idx = completed_initial + i + 1
            self.results.append(
                {"params": params, "score_mean": mean_score, "score_se": std_error}
            )
            print(f"Initial evaluation {idx}/{self.n_initial}")
            print(f"Score: {mean_score:.4f} +- {std_error:.4f} (std. err.)")
            print("-" * 60)

        if self.checkpoint_dir:
            self._save_checkpoint()

    def _run_bayesian_phase(self, pipeline_factory, X, y) -> None:
        """Execute Bayesian optimization phase.

        Args:
            pipeline_factory: Function that creates pipeline from parameters
            X: Training features
            y: Training labels
        """
        completed_bayesian = len(self.results) - self.n_initial
        print("=" * 60)
        print("Starting Bayesian Phase:")
        print("=" * 60)

        for iteration in range(completed_bayesian, self.n_iterations):
            print(f"Bayesian iteration {iteration + 1}/{self.n_iterations}")
            next_params = self._suggest_next_params()
            mean_score, std_error = self._evaluate_params(
                pipeline_factory, next_params, X, y
            )
            self.results.append(
                {"params": next_params, "score_mean": mean_score, "score_se": std_error}
            )

            print(f"Score: {mean_score:.4f} +- {std_error:.4f} (std. err.)")
            print("-" * 60)

            if self.checkpoint_dir:
                self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Save current optimization state to checkpoint."""
        checkpoint_data = {
            "results": self.results,
            "param_space": self.param_space,
            "n_initial": self.n_initial,
            "n_iterations": self.n_iterations,
            "cv": self.cv,
            "scoring": self.scoring,
            "random_state": self.random_state,
        }
        save_checkpoint(checkpoint_data, self.checkpoint_dir, self.model_hash)

    def _determine_cv_n_jobs(self) -> int:
        """Determine number of jobs for cross-validation.

        Returns:
            Number of parallel jobs for CV:
            - 1 if n_jobs is None or 1 (sequential processing)
            - min(n_jobs, cv) otherwise (cap at number of folds)
        """
        if self.n_jobs is None or self.n_jobs == 1:
            return 1
        return min(self.n_jobs, self.cv)

    def _evaluate_params(self, pipeline_factory, params, X, y) -> tuple[float, float]:
        """Evaluate parameter configuration using cross-validation.

        Args:
            pipeline_factory: Function that creates pipeline from parameters
            params: Parameter configuration to evaluate
            X: Training features
            y: Training labels

        Returns:
            Tuple of (mean_score, standard_error)
        """
        pipeline = pipeline_factory(params)

        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        cv_n_jobs = self._determine_cv_n_jobs()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_splitter,
                scoring=self.scoring,
                n_jobs=cv_n_jobs,
            )

        mean_score = np.mean(scores)
        std_error = np.std(scores, ddof=1) / np.sqrt(len(scores))
        return mean_score, std_error

    def _suggest_next_params(self) -> dict[str, float]:
        """Suggest next parameter configuration using Gaussian process.

        Returns:
            Parameter dictionary for next evaluation
        """
        X_train, y_train = self._extract_training_data()
        gp = self._fit_gaussian_process(X_train, y_train)
        candidates = self._generate_candidates()
        best_candidate = self._select_best_candidate(gp, candidates)

        return best_candidate

    def _extract_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract training data from optimization results.

        Returns:
            Tuple of (parameter_matrix, score_vector)
        """
        X_train = np.array(
            [
                [
                    r["params"]["C"],
                    r["params"]["l1_ratio"],
                    r["params"]["target_ratio"],
                ]
                for r in self.results
            ]
        )
        y_train = np.array([r["score_mean"] for r in self.results])
        return X_train, y_train

    def _fit_gaussian_process(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> GaussianProcessRegressor:
        """Fit Gaussian process model to training data.

        Args:
            X_train: Parameter configurations
            y_train: Corresponding scores

        Returns:
            Fitted Gaussian process model
        """
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
        gp.fit(X_train, y_train)
        return gp

    def _generate_candidates(self) -> list[dict[str, float]]:
        """Generate candidate parameter configurations.

        Returns:
            List of candidate parameter dictionaries
        """
        return latin_hypercube_sample(
            self.param_space, 100, self.random_state + len(self.results)
        )

    def _select_best_candidate(
        self, gp: GaussianProcessRegressor, candidates: list[dict[str, float]]
    ) -> dict[str, float]:
        """Select best candidate using acquisition function.

        Args:
            gp: Fitted Gaussian process model
            candidates: List of candidate parameter configurations

        Returns:
            Best candidate parameter dictionary
        """
        X_candidates = np.array(
            [[c["C"], c["l1_ratio"], c["target_ratio"]] for c in candidates]
        )

        mu, sigma = gp.predict(X_candidates, return_std=True)
        acquisition = mu + 2.0 * sigma

        best_idx = np.argmax(acquisition)
        return candidates[best_idx]

    def select_best_one_se(
        self, param_parsim: str, ascending: bool = False
    ) -> dict[str, float]:
        """Select best model using 1-SE rule.

        Args:
            param_parsim: Parameter name to use for parsimony selection
            ascending: If True, select lowest parameter value as most parsimonious.
                      If False (default), select highest value as most parsimonious.

        Returns:
            Dictionary with selected parameter configuration

        Raises:
            ValueError: If tuning_history is not available or param_parsim not in history
        """
        if self.tuning_history is None:
            raise ValueError("tuning_history not available. Run run_search() first.")

        if param_parsim not in self.tuning_history.columns:
            raise ValueError(
                f"Parameter '{param_parsim}' not found in tuning history. "
                f"Available: {list(self.tuning_history.columns)}"
            )

        best_idx = self.tuning_history["score_mean"].idxmax()
        best_score = self.tuning_history.loc[best_idx, "score_mean"]
        best_se = self.tuning_history.loc[best_idx, "score_se"]
        threshold = best_score - best_se

        eligible_mask = self.tuning_history["score_mean"] >= threshold
        eligible_models = self.tuning_history[eligible_mask]

        if ascending:
            selected_idx = eligible_models[param_parsim].idxmin()
        else:
            selected_idx = eligible_models[param_parsim].idxmax()

        param_names = sorted(self.param_space.keys())
        selected_params = {
            param: self.tuning_history.loc[selected_idx, param] for param in param_names
        }

        return selected_params
