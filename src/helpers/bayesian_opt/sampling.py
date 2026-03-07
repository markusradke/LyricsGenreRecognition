"""Parameter sampling utilities for Bayesian optimization."""

import numpy as np
from scipy.stats import qmc


_LOG_SCALE_PARAMS = {"C"}
_INTEGER_PARAMS = {"max_features", "min_samples_leaf"}


def transform_sample_to_params(
    sample: np.ndarray, param_space: dict[str, list[float]]
) -> dict[str, float | int]:
    """Transform normalized sample to parameter values.

    Parameters listed in ``_LOG_SCALE_PARAMS`` are treated as log10-scale
    (the range defines exponents, and ``10**value`` is returned).
    Parameters listed in ``_INTEGER_PARAMS`` are rounded to the nearest
    integer after linear scaling.
    All other parameters are scaled linearly and returned as floats.

    Args:
        sample: Normalized sample values [0, 1]
        param_space: Parameter search space with ranges

    Returns:
        Dictionary of parameter values
    """
    params: dict[str, float | int] = {}
    for i, (param_name, param_range) in enumerate(param_space.items()):
        value_range = param_range[1] - param_range[0]
        scaled_value = param_range[0] + sample[i] * value_range

        if param_name in _LOG_SCALE_PARAMS:
            params[param_name] = 10**scaled_value
        elif param_name in _INTEGER_PARAMS:
            params[param_name] = int(round(scaled_value))
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

    return [transform_sample_to_params(sample, param_space) for sample in samples]
