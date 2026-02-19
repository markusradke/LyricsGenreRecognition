"""Bayesian optimization for hyperparameter tuning.

This module provides a modular Bayesian optimization framework with:
- Latin hypercube sampling for initial exploration
- Gaussian process surrogate models
- Checkpointing for resumable optimization
- Parallel evaluation support
"""

from .optimizer import BayesianOptimizer
from .sampling import latin_hypercube_sample

__all__ = ["BayesianOptimizer", "latin_hypercube_sample"]
