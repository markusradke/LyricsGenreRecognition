"""Bayesian optimization for hyperparameter tuning.

This module provides backward compatibility by re-exporting the refactored
Bayesian optimization components from the bayesian_opt subpackage.
"""

from .bayesian_opt import BayesianOptimizer, latin_hypercube_sample

__all__ = ["BayesianOptimizer", "latin_hypercube_sample"]
