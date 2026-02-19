"""Execution logic for optimization phases."""

import pandas as pd
from joblib import Parallel, delayed

from .checkpointing import save_checkpoint
from .evaluation import determine_cv_n_jobs, evaluate_params
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
    if optimizer.n_jobs != 1:
        print(f"Using {optimizer.n_jobs} parallel jobs")
    print(f"Initial grid:\n{pd.DataFrame(initial_params)}")
    print("=" * 60)

    if completed_initial >= optimizer.n_initial:
        return

    remaining_params = initial_params[completed_initial:]

    if optimizer.n_jobs == 1:
        _run_initial_sequential(
            optimizer, pipeline_factory, X, y, remaining_params, completed_initial
        )
    else:
        _run_initial_parallel(
            optimizer, pipeline_factory, X, y, remaining_params, completed_initial
        )


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

    cv_n_jobs = determine_cv_n_jobs(optimizer.n_jobs, optimizer.cv)

    for iteration in range(completed_bayesian, optimizer.n_iterations):
        print(f"Bayesian iteration {iteration + 1}/{optimizer.n_iterations}")

        use_uncertainty = (
            optimizer.uncertain_jump > 0
            and optimizer.iters_without_improvement >= optimizer.uncertain_jump
            and optimizer.iters_since_jump >= optimizer.uncertain_jump
        )

        if use_uncertainty:
            print("[Uncertainty Jump] Exploring high-uncertainty region")

        next_params = suggest_next_params(
            optimizer.results,
            optimizer.param_space,
            optimizer.random_state,
            use_uncertainty=use_uncertainty,
        )
        mean_score, std_error = evaluate_params(
            pipeline_factory,
            next_params,
            X,
            y,
            optimizer.cv,
            optimizer.scoring,
            optimizer.random_state,
            cv_n_jobs,
        )
        optimizer.results.append(
            {"params": next_params, "score_mean": mean_score, "score_se": std_error}
        )

        print(f"Score: {mean_score:.4f} +- {std_error:.4f} (std. err.)")

        if optimizer.best_score is None or mean_score > optimizer.best_score:
            optimizer.best_score = mean_score
            optimizer.iters_without_improvement = 0
            print("New best score!")
        else:
            optimizer.iters_without_improvement += 1
            print(
                f"No improvement for {optimizer.iters_without_improvement} / {optimizer.stop_iter} "
                f"iteration(s)"
            )

        if use_uncertainty:
            optimizer.iters_since_jump = 0
        else:
            optimizer.iters_since_jump += 1

        print("-" * 60)

        if optimizer.checkpoint_dir:
            _save_optimizer_checkpoint(optimizer)

        if optimizer.iters_without_improvement >= optimizer.stop_iter:
            print("=" * 60)
            print(
                f"Early stopping: No improvement for {optimizer.stop_iter} "
                f"iterations"
            )
            print(f"Best score achieved: {optimizer.best_score:.4f}")
            print(f"Stopped at iteration {iteration + 1}/{optimizer.n_iterations}")
            print("=" * 60)
            break


def _run_initial_sequential(
    optimizer, pipeline_factory, X, y, remaining_params, completed_initial
) -> None:
    """Run initial evaluations sequentially with checkpointing."""
    cv_n_jobs = determine_cv_n_jobs(optimizer.n_jobs, optimizer.cv)

    for i, params in enumerate(remaining_params):
        idx = completed_initial + i + 1
        print(f"Initial evaluation {idx}/{optimizer.n_initial}")
        mean_score, std_error = evaluate_params(
            pipeline_factory,
            params,
            X,
            y,
            optimizer.cv,
            optimizer.scoring,
            optimizer.random_state,
            cv_n_jobs,
        )
        optimizer.results.append(
            {"params": params, "score_mean": mean_score, "score_se": std_error}
        )
        print(f"Score: {mean_score:.4f} ± {std_error:.4f}")
        print("-" * 60)

        if optimizer.checkpoint_dir:
            _save_optimizer_checkpoint(optimizer)


def _run_initial_parallel(
    optimizer, pipeline_factory, X, y, remaining_params, completed_initial
) -> None:
    """Run initial evaluations in parallel with batch checkpointing."""
    cv_n_jobs = determine_cv_n_jobs(optimizer.n_jobs, optimizer.cv)

    results_parallel = Parallel(n_jobs=optimizer.n_jobs)(
        delayed(evaluate_params)(
            pipeline_factory,
            params,
            X,
            y,
            optimizer.cv,
            optimizer.scoring,
            optimizer.random_state,
            cv_n_jobs,
        )
        for params in remaining_params
    )

    for i, (params, (mean_score, std_error)) in enumerate(
        zip(remaining_params, results_parallel)
    ):
        idx = completed_initial + i + 1
        optimizer.results.append(
            {"params": params, "score_mean": mean_score, "score_se": std_error}
        )
        print(f"Initial evaluation {idx}/{optimizer.n_initial}")
        print(f"Score: {mean_score:.4f} +- {std_error:.4f} (std. err.)")
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
    }
    save_checkpoint(checkpoint_data, optimizer.checkpoint_dir, optimizer.model_hash)
