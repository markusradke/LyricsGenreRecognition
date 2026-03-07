"""Execution logic for optimization phases."""

import pandas as pd
import gc

from .checkpointing import save_checkpoint
from .evaluation import evaluate_params
from .surrogate import suggest_next_params


def run_initial_phase(
    optimizer,
    pipeline_factory,
    X,
    y,
    initial_params: list[dict],
    completed_initial: int,
) -> None:
    """Execute initial phase with Latin hypercube sampling.

    Args:
        optimizer: BayesianOptimizer instance
        pipeline_factory: Function that creates pipeline from parameters
        X: Training features
        y: Training labels
        initial_params: List of initial parameter configurations
        completed_initial: Number of already completed evaluations
    """
    print("=" * 60)
    print("Starting Initial Phase:")
    print(f"Initial grid:\n{pd.DataFrame(initial_params)}")
    print("=" * 60)

    if completed_initial >= optimizer.n_initial:
        print(
            f"Initial phase already completed ({completed_initial}/{optimizer.n_initial}). Skipping."
        )
        print("=" * 60)
        return

    remaining_params = initial_params[completed_initial:]

    _run_initial_sequential(
        optimizer, pipeline_factory, X, y, remaining_params, completed_initial
    )


def _early_stopping_reached(optimizer) -> bool:
    """Return True if early stopping threshold has already been reached."""
    return optimizer.iters_without_improvement >= optimizer.stop_iter


def _sync_best_score_from_results(optimizer) -> None:
    """Initialise best_score from stored results if not yet set.

    Called at the start of the Bayesian phase so that the initial-phase
    results are taken into account before the first Bayesian evaluation.
    """
    if optimizer.best_score is None and optimizer.results:
        optimizer.best_score = max(r["score_mean"] for r in optimizer.results)


def run_bayesian_phase(optimizer, pipeline_factory, X, y) -> None:
    """Execute Bayesian optimization phase.

    Args:
        optimizer: BayesianOptimizer instance
        pipeline_factory: Function that creates pipeline from parameters
        X: Training features
        y: Training labels
    """
    completed_bayesian = len(optimizer.results) - optimizer.n_initial
    print("=" * 60)
    print("Starting Bayesian Phase:")
    print("=" * 60)

    if completed_bayesian >= optimizer.n_iterations:
        print(
            f"Bayesian phase already completed ({completed_bayesian}/{optimizer.n_iterations}). Skipping."
        )
        print("=" * 60)
        return

    if _early_stopping_reached(optimizer):
        print(
            f"Early stopping already reached from checkpoint ({optimizer.iters_without_improvement}/{optimizer.stop_iter}). Skipping."
        )
        print("=" * 60)
        return

    _sync_best_score_from_results(optimizer)

    for iteration in range(completed_bayesian, optimizer.n_iterations):
        params = _suggest_next(optimizer, iteration)
        mean_score, std_error = evaluate_params(
            pipeline_factory,
            params,
            X,
            y,
            optimizer.cv,
            optimizer.scoring,
            optimizer.random_state,
        )
        _process_result(optimizer, params, mean_score, std_error, iteration)

        if optimizer.checkpoint_dir:
            _save_optimizer_checkpoint(optimizer)

        if _early_stopping_reached(optimizer):
            print("=" * 60)
            print(
                f"Early stopping: No improvement for {optimizer.stop_iter} "
                f"individual evaluations"
            )
            print(f"Best score achieved: {optimizer.best_score:.4f}")
            print(f"Stopped after iteration {iteration + 1}/{optimizer.n_iterations}")
            print("=" * 60)
            break


def _suggest_next(optimizer, iteration: int) -> dict:
    """Suggest the next candidate from the surrogate."""
    use_uncertainty = _should_use_uncertainty(optimizer)
    if use_uncertainty:
        print("[Uncertainty Jump] Exploring high-uncertainty region")
    return suggest_next_params(
        optimizer.results,
        optimizer.param_space,
        optimizer.random_state + iteration,
        use_uncertainty=use_uncertainty,
    )


def _should_use_uncertainty(optimizer) -> bool:
    """Check whether an uncertainty-based exploration jump is due."""
    return (
        optimizer.uncertain_jump > 0
        and optimizer.iters_without_improvement >= optimizer.uncertain_jump
        and optimizer.iters_since_jump >= optimizer.uncertain_jump
    )


def _process_result(
    optimizer,
    params: dict,
    mean_score: float,
    std_error: float,
    iteration: int,
) -> None:
    """Append one result and update optimizer state."""
    params_str = "  ".join(f"{k}={v:.4g}" for k, v in sorted(params.items()))
    print(
        f"Bayesian evaluation {iteration + 1}/{optimizer.n_iterations}  [{params_str}]"
    )
    optimizer.results.append(
        {"params": params, "score_mean": mean_score, "score_se": std_error}
    )
    print(f"Score: {mean_score:.4f} +- {std_error:.4f} (std. err.)")
    gc.collect()
    if optimizer.best_score is None or mean_score > optimizer.best_score:
        optimizer.best_score = mean_score
        optimizer.iters_without_improvement = 0
        print("New best score!")
    else:
        optimizer.iters_without_improvement += 1
        print(
            f"No improvement for {optimizer.iters_without_improvement} / "
            f"{optimizer.stop_iter} evaluation(s)"
        )

    use_uncertainty = _should_use_uncertainty(optimizer)
    if use_uncertainty:
        optimizer.iters_since_jump = 0
    else:
        optimizer.iters_since_jump += 1

    print("-" * 60)


def _run_initial_sequential(
    optimizer, pipeline_factory, X, y, remaining_params, completed_initial
) -> None:
    """Run initial evaluations sequentially with checkpointing."""
    for i, params in enumerate(remaining_params):
        idx = completed_initial + i + 1
        params_str = "  ".join(f"{k}={v:.4g}" for k, v in sorted(params.items()))
        print(f"Initial evaluation {idx}/{optimizer.n_initial}  [{params_str}]")
        mean_score, std_error = evaluate_params(
            pipeline_factory,
            params,
            X,
            y,
            optimizer.cv,
            optimizer.scoring,
            optimizer.random_state,
        )
        optimizer.results.append(
            {"params": params, "score_mean": mean_score, "score_se": std_error}
        )
        print(f"Score: {mean_score:.4f} ± {std_error:.4f}")
        print("-" * 60)

        if optimizer.checkpoint_dir:
            _save_optimizer_checkpoint(optimizer)


def _save_optimizer_checkpoint(optimizer) -> None:
    """Save optimizer state to checkpoint."""
    checkpoint_data = {
        "results": optimizer.results,
        "param_space": optimizer.param_space,
        "n_initial": optimizer.n_initial,
        "n_iterations": optimizer.n_iterations,
        "cv": optimizer.cv,
        "scoring": optimizer.scoring,
        "random_state": optimizer.random_state,
        "best_score": optimizer.best_score,
        "iters_without_improvement": optimizer.iters_without_improvement,
        "iters_since_jump": optimizer.iters_since_jump,
        "stop_iter": optimizer.stop_iter,
        "uncertain_jump": optimizer.uncertain_jump,
    }
    save_checkpoint(checkpoint_data, optimizer.checkpoint_dir, optimizer.model_hash)
