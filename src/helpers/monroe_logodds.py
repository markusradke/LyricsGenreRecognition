"""
Monroe log-odds ratio computation with Dirichlet smoothing.

Optimized implementation of the "Fightin' Words" method from Monroe et al.
(2008) for identifying discriminating n-grams between groups (genres) using
Bayesian-smoothed log-odds ratios with one-vs-rest comparisons.

Reference:
    Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008).
    Fightin' Words: Lexical Feature Selection and Evaluation for
    Identifying the Content of Political Conflict.
    Political Analysis, 16(4), 372-403.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List


def compute_pvalues_from_zscores(z_scores: np.ndarray) -> np.ndarray:
    """
    Convert z-scores to one-sided p-values.

    Parameters
    ----------
    z_scores : np.ndarray
        Z-scores from statistical tests.

    Returns
    -------
    p_values : np.ndarray
        One-sided p-values (upper tail test).
    """
    return 1 - norm.cdf(z_scores)


def compute_monroe_statistics(
    y_gc: np.ndarray,
    n_c: np.ndarray,
    y_g: np.ndarray,
    n: int,
    m: int,
    alpha_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Monroe delta, variance, and z-scores in one pass.

    Implements one-vs-rest comparison: each genre vs. all other genres combined.

    Parameters
    ----------
    y_gc : np.ndarray, shape (n_ngrams, n_genres)
        Count of each n-gram in each genre.
    n_c : np.ndarray, shape (n_genres,)
        Total tokens per genre.
    y_g : np.ndarray, shape (n_ngrams,)
        Total count per n-gram across entire corpus.
    n : int
        Total tokens in corpus.
    m : int
        Vocabulary size (number of unique n-grams).
    alpha_g : np.ndarray, shape (n_ngrams,)
        Dirichlet prior per n-gram.

    Returns
    -------
    delta : np.ndarray, shape (n_ngrams, n_genres)
        Log-odds ratios (Monroe Equation 15).
    variance : np.ndarray, shape (n_ngrams, n_genres)
        Variances (Monroe Equation 17).
    z_scores : np.ndarray, shape (n_ngrams, n_genres)
        Standardized z-scores.
    """
    alpha_g_col = alpha_g.reshape(-1, 1)
    y_g_col = y_g.reshape(-1, 1)
    m_alpha = alpha_g.sum()

    focal_num = y_gc + alpha_g_col
    focal_denom = n_c - y_gc + m_alpha - alpha_g_col
    other_num = y_g_col - y_gc + alpha_g_col
    other_denom = n - n_c - y_g_col + y_gc + m_alpha - alpha_g_col

    log_odds_focal = np.log(focal_num / focal_denom)
    log_odds_other = np.log(other_num / other_denom)
    delta = log_odds_focal - log_odds_other

    variance = 1.0 / focal_num + 1.0 / focal_denom + 1.0 / other_num + 1.0 / other_denom

    z_scores = delta / np.sqrt(variance)

    return delta, variance, z_scores


def apply_benjamini_hochberg_correction(
    p_values: np.ndarray, fdr: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.

    Implements the BH procedure:
    1. Sort p-values from low to high: p_1 <= p_2 <= ... <= p_m
    2. Find largest rank r where p_r <= fdr * (r/m)
    3. Reject all hypotheses with ranks 1, 2, ..., r

    Parameters
    ----------
    p_values : np.ndarray, shape (n_tests,) or (n_ngrams, n_genres)
        P-values from statistical tests.
    fdr : float, default=0.01
        Desired false discovery rate (e.g., 0.01 = 1% FDR).

    Returns
    -------
    passes_bh : np.ndarray, same shape as p_values
        Boolean mask indicating which tests pass BH correction.
    bh_threshold : np.ndarray, same shape as p_values
        BH threshold value for each test (fdr * rank / m).

    Notes
    -----
    Benjamini Y, Hochberg Y (1995) Controlling the false discovery rate:
    a practical and powerful approach to multiple hypothesis testing.
    J R Stat Soc B 57:289-300
    """
    original_shape = p_values.shape
    p_flat = p_values.flatten()
    m = len(p_flat)

    # Sort p-values and track original indices
    sorted_idx = np.argsort(p_flat)
    sorted_p = p_flat[sorted_idx]

    # Compute BH thresholds for each rank: fdr * (r/m)
    ranks = np.arange(1, m + 1)
    bh_thresholds = fdr * ranks / m

    # Find largest rank where p_r <= fdr * (r/m)
    comparisons = sorted_p <= bh_thresholds

    if comparisons.any():
        # Find the highest rank that passes
        max_significant_rank = np.where(comparisons)[0][-1]
        # All ranks up to and including this rank are significant
        passes_sorted = np.zeros(m, dtype=bool)
        passes_sorted[: max_significant_rank + 1] = True
    else:
        # No tests pass correction
        passes_sorted = np.zeros(m, dtype=bool)

    # Unsort back to original order
    passes_bh = np.empty(m, dtype=bool)
    passes_bh[sorted_idx] = passes_sorted

    # Map thresholds back to original order
    bh_threshold_array = np.empty(m)
    bh_threshold_array[sorted_idx] = bh_thresholds

    return passes_bh.reshape(original_shape), bh_threshold_array.reshape(original_shape)
