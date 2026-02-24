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

    n_points = getattr(optimizer, "n_points", 1)
    cv_n_jobs = _bayesian_cv_n_jobs(optimizer.n_jobs, optimizer.cv, n_points)

    for iteration in range(completed_bayesian, optimizer.n_iterations, n_points):
        remaining = optimizer.n_iterations - iteration
        batch_size = min(n_points, remaining)

        batch_params = _suggest_batch(optimizer, batch_size, iteration)
        batch_results = _evaluate_batch(
            optimizer, pipeline_factory, X, y, batch_params, cv_n_jobs
        )

        _process_batch_results(
            optimizer, batch_params, batch_results, iteration, batch_size
        )

        if optimizer.checkpoint_dir:
            _save_optimizer_checkpoint(optimizer)

        if optimizer.iters_without_improvement >= optimizer.stop_iter:
            print("=" * 60)
            print(
                f"Early stopping: No improvement for {optimizer.stop_iter} "
                f"individual evaluations"
            )
            print(f"Best score achieved: {optimizer.best_score:.4f}")
            print(
                f"Stopped after iteration {iteration + batch_size}/{optimizer.n_iterations}"
            )
            print("=" * 60)
            break


def _bayesian_cv_n_jobs(n_jobs: int, cv: int, n_points: int) -> int:
    """Determine CV n_jobs for the Bayesian phase.

    When n_points > 1, workers are assigned to the outer parallel evaluation
    of candidates, so CV must run sequentially to avoid over-subscription.
    """
    if n_points > 1:
        return 1
    return determine_cv_n_jobs(n_jobs, cv)


def _suggest_batch(optimizer, batch_size: int, iteration: int) -> list[dict]:
    """Suggest a batch of candidates from the surrogate.

    The surrogate is fitted once on results available before this batch.
    Each candidate after the first uses a different random seed offset so
    the 10 000-point LHS candidate set is reshuffled, increasing diversity.
    """
    batch_params = []
    for i in range(batch_size):
        use_uncertainty = _should_use_uncertainty(optimizer)
        if i == 0 and use_uncertainty:
            print("[Uncertainty Jump] Exploring high-uncertainty region")
        params = suggest_next_params(
            optimizer.results,
            optimizer.param_space,
            optimizer.random_state + iteration + i,
            use_uncertainty=use_uncertainty,
        )
        batch_params.append(params)
    return batch_params


def _should_use_uncertainty(optimizer) -> bool:
    """Check whether an uncertainty-based exploration jump is due."""
    return (
        optimizer.uncertain_jump > 0
        and optimizer.iters_without_improvement >= optimizer.uncertain_jump
        and optimizer.iters_since_jump >= optimizer.uncertain_jump
    )


def _evaluate_batch(
    optimizer,
    pipeline_factory,
    X,
    y,
    batch_params: list[dict],
    cv_n_jobs: int,
) -> list[tuple[float, float]]:
    """Evaluate a batch of candidates, in parallel when n_points > 1."""
    n_points = getattr(optimizer, "n_points", 1)

    if n_points == 1:
        return [
            evaluate_params(
                pipeline_factory,
                batch_params[0],
                X,
                y,
                optimizer.cv,
                optimizer.scoring,
                optimizer.random_state,
                cv_n_jobs,
            )
        ]

    return Parallel(n_jobs=n_points)(
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
        for params in batch_params
    )


def _process_batch_results(
    optimizer,
    batch_params: list[dict],
    batch_results: list[tuple[float, float]],
    iteration: int,
    batch_size: int,
) -> None:
    """Append results and update optimizer state for each point individually."""
    use_uncertainty = _should_use_uncertainty(optimizer)

    for i, (params, (mean_score, std_error)) in enumerate(
        zip(batch_params, batch_results)
    ):
        point_num = iteration + i + 1
        print(f"Bayesian evaluation {point_num}/{optimizer.n_iterations}")
        optimizer.results.append(
            {"params": params, "score_mean": mean_score, "score_se": std_error}
        )
        print(f"Score: {mean_score:.4f} +- {std_error:.4f} (std. err.)")

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

        if use_uncertainty and i == 0:
            optimizer.iters_since_jump = 0
        else:
            optimizer.iters_since_jump += 1

        print("-" * 60)


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
    results_parallel = Parallel(n_jobs=optimizer.n_jobs)(
        delayed(evaluate_params)(
            pipeline_factory,
            params,
            X,
            y,
            optimizer.cv,
            optimizer.scoring,
            optimizer.random_state,
            1,  # cv runs sequentially inside loky worker processes
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
