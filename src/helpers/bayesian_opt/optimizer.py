"""Main Bayesian optimizer class."""

from pathlib import Path

import pandas as pd

from .checkpointing import create_model_hash, load_checkpoint
from .phases import run_bayesian_phase, run_initial_phase
from .sampling import latin_hypercube_sample


class BayesianOptimizer:
    """Bayesian optimizer for hyperparameter tuning."""

    def __init__(
        self,
        param_space: dict[str, list[float]],
        n_initial: int = 20,
        n_iterations: int = 30,
        stop_iter: int = 10,
        uncertain_jump: int = 5,
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
            stop_iter: Number of iterations without improvement before early stopping
            uncertain_jump: Interval for uncertainty-based exploration jumps
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
        self.stop_iter = stop_iter
        self.uncertain_jump = uncertain_jump
        self.results: list[dict] = []
        self.tuning_history: pd.DataFrame | None = None
        self.model_hash: str | None = None
        self.best_score: float | None = None
        self.iters_without_improvement: int = 0
        self.iters_since_jump: int = 0

    def run_search(self, pipeline_factory, X, y) -> pd.DataFrame:
        """Run optimization procedure with checkpointing.

        Args:
            pipeline_factory: Function that creates pipeline from parameters
            X: Training features
            y: Training labels

        Returns:
            DataFrame with tuning history
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

    def select_best_one_se(
        self, param_parsim: str, ascending: bool = False
    ) -> dict[str, float]:
        """Select best model using 1-SE rule.

        Args:
            param_parsim: Parameter name to use for parsimony selection
            ascending: If True, select lowest parameter value as most parsimonious

        Returns:
            Dictionary with selected parameter configuration
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

    def _initialize_optimization(self, X) -> None:
        """Initialize optimization state and create model hash."""
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
        """Load checkpoint if available."""
        if not self.checkpoint_dir:
            return 0

        checkpoint = load_checkpoint(self.checkpoint_dir, self.model_hash)
        if checkpoint:
            print(f"Loaded checkpoint with {len(checkpoint['results'])} results")
            self.results = checkpoint["results"]
            self.best_score = checkpoint.get("best_score")
            self.iters_without_improvement = checkpoint.get(
                "iters_without_improvement", 0
            )
            self.iters_since_jump = checkpoint.get("iters_since_jump", 0)
            return min(len(self.results), self.n_initial)

        return 0

    def _run_initial_phase(
        self, pipeline_factory, X, y, initial_params, completed_initial
    ) -> None:
        """Execute initial phase with Latin hypercube sampling."""
        run_initial_phase(
            self, pipeline_factory, X, y, initial_params, completed_initial
        )

    def _run_bayesian_phase(self, pipeline_factory, X, y) -> None:
        """Execute Bayesian optimization phase."""
        run_bayesian_phase(self, pipeline_factory, X, y)

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
