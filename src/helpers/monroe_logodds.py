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


def apply_fdr_correction(z_scores: np.ndarray, fdr_level: float = 0.01) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to z-scores.

    Controls false discovery rate when performing many simultaneous tests
    (e.g., m n-grams x k genres tests).

    Parameters
    ----------
    z_scores : np.ndarray, shape (n_ngrams, n_genres)
        Z-scores for each (n-gram, genre) pair.
    fdr_level : float, default=0.01
        Desired false discovery rate (proportion of false positives).

    Returns
    -------
    passes_fdr : np.ndarray, shape (n_ngrams, n_genres)
        Boolean mask indicating which tests pass FDR correction.

    Notes
    -----
    Benjamini Y, Hochberg Y (1995) Controlling the false discovery rate:
    a practical and powerful approach to multiple hypothesis testing.
    J R Stat Soc B 57:289-300
    """
    p_values = 1 - norm.cdf(z_scores.flatten())

    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    comparisons = sorted_p <= fdr_level * np.arange(1, m + 1) / m

    if comparisons.any():
        threshold_p = sorted_p[comparisons.nonzero()[0][-1]]
        passes = p_values <= threshold_p
    else:
        passes = np.zeros(m, dtype=bool)

    return passes.reshape(z_scores.shape)


def filter_discriminating_ngrams(
    z_scores: np.ndarray,
    ngram_names: List[str],
    genre_names: List[str],
    threshold: float = 2.326,
    apply_fdr: bool = True,
    fdr_level: float = 0.01,
) -> pd.DataFrame:
    """
    Filter n-grams by z-score threshold (one-sided test).

    Default threshold 2.326 corresponds to one-sided test at alpha=0.01
    (uncorrected). With apply_fdr=True, uses Benjamini-Hochberg correction
    to control false discovery rate across multiple tests.

    Parameters
    ----------
    z_scores : np.ndarray, shape (n_ngrams, n_genres)
        Z-scores for each (n-gram, genre) pair.
    ngram_names : List[str]
        N-gram identifiers (e.g., ["love", "dark side"]).
    genre_names : List[str]
        Genre labels.
    threshold : float, default=2.326
        Z-score threshold (one-sided, alpha=0.01). Ignored if apply_fdr=True.
    apply_fdr : bool, default=True
        If True, apply Benjamini-Hochberg FDR correction instead of
        fixed threshold. Recommended for academic rigor with large
        feature spaces (m x k tests).
    fdr_level : float, default=0.01
        False discovery rate level (only used if apply_fdr=True).

    Returns
    -------
    discriminating : pd.DataFrame
        DataFrame with columns ['ngram', 'genre', 'z_score']
        for n-grams passing threshold in each genre.

    Notes
    -----
    Multiple testing: With m n-grams and k genres, we perform m×k tests.
    Without correction, expect α * m * k false positives. FDR correction
    controls the proportion of false positives among selected features.
    """
    if apply_fdr:
        passes_fdr = apply_fdr_correction(z_scores, fdr_level)
        ngram_idx, genre_idx = np.where(passes_fdr & (z_scores > 0))
    else:
        ngram_idx, genre_idx = np.where(z_scores > threshold)

    discriminating = pd.DataFrame(
        {
            "ngram": [ngram_names[i] for i in ngram_idx],
            "genre": [genre_names[j] for j in genre_idx],
            "z_score": z_scores[ngram_idx, genre_idx],
        }
    )

    return discriminating
