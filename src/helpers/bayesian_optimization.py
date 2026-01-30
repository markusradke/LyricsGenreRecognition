"""Bayesian optimization for hyperparameter tuning."""

import hashlib
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
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


def _extract_eligible_penalties(
    scores: np.ndarray, penalties: np.ndarray, threshold: float
) -> tuple[np.ndarray, list[float]]:
    """Extract penalties for models meeting threshold.

    Args:
        scores: Cross-validation scores
        penalties: Penalty values for each model
        threshold: Minimum score threshold

    Returns:
        Tuple of (eligible_indices, eligible_penalties)
    """
    eligible_indices = np.where(scores >= threshold)[0]
    eligible_penalties = [penalties[i] for i in eligible_indices]
    return eligible_indices, eligible_penalties


def one_standard_error_rule(
    scores: np.ndarray,
    params: list[dict] | None,
    penalties: np.ndarray,
) -> tuple[int, int]:
    """Select model using 1-SE rule for better generalization.

    Args:
        scores: Cross-validation scores
        params: Parameter configurations (unused, kept for compatibility)
        penalties: Penalty values (C parameter) for each model

    Returns:
        Tuple of (selected_index, best_score_index)
    """
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    std_error = np.std(scores) / np.sqrt(len(scores))
    threshold = best_score - std_error

    eligible_indices, eligible_penalties = _extract_eligible_penalties(
        scores, penalties, threshold
    )

    max_penalty_idx = eligible_indices[np.argmax(eligible_penalties)]
    return max_penalty_idx, best_idx


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
        self.model_hash: str | None = None

    def optimize(self, pipeline_factory, X, y) -> dict:
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

        return self._select_best_params()

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
            score = self._evaluate_params(pipeline_factory, params, X, y)
            self.results.append({"params": params, "score": score})
            print(f"Score: {score:.4f}")
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

        for i, (params, score) in enumerate(zip(remaining_params, results_parallel)):
            idx = completed_initial + i + 1
            self.results.append({"params": params, "score": score})
            print(f"Initial evaluation {idx}/{self.n_initial}")
            print(f"Score: {score:.4f}")
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
            score = self._evaluate_params(pipeline_factory, next_params, X, y)
            self.results.append({"params": next_params, "score": score})

            print(f"Score: {score:.4f}")
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

    def _evaluate_params(self, pipeline_factory, params, X, y) -> float:
        """Evaluate parameter configuration using cross-validation.

        Args:
            pipeline_factory: Function that creates pipeline from parameters
            params: Parameter configuration to evaluate
            X: Training features
            y: Training labels

        Returns:
            Mean cross-validation score
        """
        pipeline = pipeline_factory(params)

        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_splitter,
                scoring=self.scoring,
                n_jobs=-1,
            )

        return np.mean(scores)

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
        y_train = np.array([r["score"] for r in self.results])
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

    def _select_best_params(self) -> dict:
        """Select best parameters using 1-SE rule.

        Returns:
            Dictionary with best parameters and scores
        """
        scores = np.array([r["score"] for r in self.results])
        penalties = np.array([r["params"]["C"] for r in self.results])

        selected_idx, best_idx = one_standard_error_rule(scores, None, penalties)

        return {
            "best_params": self.results[selected_idx]["params"],
            "best_score": self.results[selected_idx]["score"],
            "absolute_best_params": self.results[best_idx]["params"],
            "absolute_best_score": self.results[best_idx]["score"],
            "all_results": self.results,
        }
