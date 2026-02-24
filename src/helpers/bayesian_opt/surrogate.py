"""Gaussian process surrogate model for Bayesian optimization."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .sampling import latin_hypercube_sample


def fit_gaussian_process(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int
) -> GaussianProcessRegressor:
    """Fit Gaussian process model to training data.

    Args:
        X_train: Parameter configurations
        y_train: Corresponding scores
        random_state: Random seed for reproducibility

    Returns:
        Fitted Gaussian process model
    """
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=random_state)
    gp.fit(X_train, y_train)
    return gp


def suggest_next_params(
    results: list[dict],
    param_space: dict[str, list[float]],
    random_state: int,
    use_uncertainty: bool = False,
) -> dict[str, float]:
    """Suggest next parameter configuration using Gaussian process.

    Args:
        results: List of evaluation results
        param_space: Parameter search space
        random_state: Random seed for reproducibility
        use_uncertainty: If True, select based on uncertainty rather than acquisition

    Returns:
        Parameter dictionary for next evaluation
    """
    X_train, y_train = _extract_training_data(results, param_space)
    gp = fit_gaussian_process(X_train, y_train, random_state)
    candidates = _generate_candidates(param_space, random_state, len(results))
    best_candidate = _select_best_candidate(
        gp, candidates, param_space, use_uncertainty
    )

    return best_candidate


def _extract_training_data(
    results: list[dict], param_space: dict[str, list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract training data from optimization results.

    Args:
        results: List of evaluation results
        param_space: Parameter search space

    Returns:
        Tuple of (parameter_matrix, score_vector)
    """
    param_names = sorted(param_space.keys())
    X_train = np.array([[r["params"][p] for p in param_names] for r in results])
    y_train = np.array([r["score_mean"] for r in results])
    return X_train, y_train


def _generate_candidates(
    param_space: dict[str, list[float]], random_state: int, n_results: int
) -> list[dict[str, float]]:
    """Generate candidate parameter configurations.

    Args:
        param_space: Parameter search space
        random_state: Base random seed
        n_results: Number of completed results (for seed offset)

    Returns:
        List of candidate parameter dictionaries
    """
    return latin_hypercube_sample(param_space, 10000, random_state + n_results)


def _select_best_candidate(
    gp: GaussianProcessRegressor,
    candidates: list[dict[str, float]],
    param_space: dict[str, list[float]],
    use_uncertainty: bool = False,
) -> dict[str, float]:
    """Select best candidate using acquisition function or uncertainty.

    Args:
        gp: Fitted Gaussian process model
        candidates: List of candidate parameter configurations
        param_space: Parameter search space
        use_uncertainty: If True, select region with highest uncertainty

    Returns:
        Best candidate parameter dictionary
    """
    param_names = sorted(param_space.keys())
    X_candidates = np.array([[c[p] for p in param_names] for c in candidates])

    mu, sigma = gp.predict(X_candidates, return_std=True)

    if use_uncertainty:
        # Select candidate with highest uncertainty (exploration)
        selection_criterion = sigma
    else:
        # Use standard acquisition function (exploitation + exploration)
        selection_criterion = mu + 2.0 * sigma

    best_idx = int(np.argmax(selection_criterion))
    return candidates[best_idx]
