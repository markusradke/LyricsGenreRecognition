"""
Monroe log-odds ratio computation with Dirichlet smoothing.

Implements the "Fightin' Words" method from Monroe et al. (2008) for
identifying discriminating n-grams between groups (genres) using
Bayesian-smoothed log-odds ratios.

Reference:
    Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008).
    Fightin' Words: Lexical Feature Selection and Evaluation for
    Identifying the Content of Political Conflict.
    Political Analysis, 16(4), 372-403.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def compute_log_odds_delta(
    y_gc: np.ndarray,
    n_c: np.ndarray,
    y_g: np.ndarray,
    n: int,
    m: int,
    alpha_g: np.ndarray,
) -> np.ndarray:
    """
    Compute Dirichlet-smoothed log-odds ratio (delta).

    Implements Equation 15 from Monroe et al. (2008):

    δ_gc = log((y_gc + α_g) / (n_c - y_gc + mα - α_g)) -
           log((y_g - y_gc + α_g) / (n - n_c - y_g + y_gc + mα - α_g))

    Parameters
    ----------
    y_gc : np.ndarray, shape (n_ngrams, n_genres)
        Count of each n-gram in each genre.
    n_c : np.ndarray, shape (n_genres,)
        Total n-gram tokens per genre.
    y_g : np.ndarray, shape (n_ngrams,)
        Total count of each n-gram across all genres.
    n : int
        Total n-gram tokens across entire corpus.
    m : int
        Vocabulary size (number of unique n-grams).
    alpha_g : np.ndarray, shape (n_ngrams,)
        Dirichlet prior for each n-gram (hyperparameter).

    Returns
    -------
    delta : np.ndarray, shape (n_ngrams, n_genres)
        Log-odds ratio for each (n-gram, genre) pair.
    """
    # Broadcast alpha_g to match y_gc shape
    alpha_g = alpha_g.reshape(-1, 1)

    # Total alpha (sum over vocabulary)
    m_alpha = m * np.mean(alpha_g)

    # Numerator: usage in focal genre
    numerator_focal = y_gc + alpha_g
    denominator_focal = n_c - y_gc + m_alpha - alpha_g

    # Numerator: usage in other genres
    numerator_other = y_g.reshape(-1, 1) - y_gc + alpha_g
    denominator_other = n - n_c - y_g.reshape(-1, 1) + y_gc + m_alpha - alpha_g

    # Compute log-odds
    log_odds_focal = np.log(numerator_focal / denominator_focal)
    log_odds_other = np.log(numerator_other / denominator_other)

    delta = log_odds_focal - log_odds_other

    return delta


def compute_variance(
    y_gc: np.ndarray,
    n_c: np.ndarray,
    y_g: np.ndarray,
    n: int,
    m: int,
    alpha_g: np.ndarray,
) -> np.ndarray:
    """
    Compute variance of log-odds ratio for z-score standardization.

    Based on Equation 17 from Monroe et al. (2008):

    Var(δ_gc) = 1/(y_gc + α_g) + 1/(n_c - y_gc + mα - α_g) +
                1/(y_g - y_gc + α_g) + 1/(n - n_c - y_g + y_gc + mα - α_g)

    Parameters
    ----------
    y_gc : np.ndarray, shape (n_ngrams, n_genres)
        Count of each n-gram in each genre.
    n_c : np.ndarray, shape (n_genres,)
        Total n-gram tokens per genre.
    y_g : np.ndarray, shape (n_ngrams,)
        Total count of each n-gram across all genres.
    n : int
        Total n-gram tokens across entire corpus.
    m : int
        Vocabulary size (number of unique n-grams).
    alpha_g : np.ndarray, shape (n_ngrams,)
        Dirichlet prior for each n-gram.

    Returns
    -------
    variance : np.ndarray, shape (n_ngrams, n_genres)
        Variance for each (n-gram, genre) pair.
    """
    # Broadcast alpha_g to match y_gc shape
    alpha_g = alpha_g.reshape(-1, 1)

    # Total alpha
    m_alpha = m * np.mean(alpha_g)

    # Four variance components
    var1 = 1.0 / (y_gc + alpha_g)
    var2 = 1.0 / (n_c - y_gc + m_alpha - alpha_g)
    var3 = 1.0 / (y_g.reshape(-1, 1) - y_gc + alpha_g)
    var4 = 1.0 / (n - n_c - y_g.reshape(-1, 1) + y_gc + m_alpha - alpha_g)

    variance = var1 + var2 + var3 + var4

    return variance


def compute_z_scores(delta: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """
    Standardize log-odds ratios to z-scores.

    z_gc = δ_gc / sqrt(Var(δ_gc))

    Parameters
    ----------
    delta : np.ndarray, shape (n_ngrams, n_genres)
        Log-odds ratios.
    variance : np.ndarray, shape (n_ngrams, n_genres)
        Variances.

    Returns
    -------
    z_scores : np.ndarray, shape (n_ngrams, n_genres)
        Standardized z-scores.
    """
    z_scores = delta / np.sqrt(variance)
    return z_scores


def apply_fdr_correction(z_scores: np.ndarray, fdr_level: float = 0.01) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to z-scores.

    Controls false discovery rate when performing many simultaneous tests
    (e.g., m n-grams × k genres tests).

    Parameters
    ----------
    z_scores : np.ndarray, shape (n_ngrams, n_genres)
        Z-scores for each (n-gram, genre) pair.
    fdr_level : float, default=0.01
        Desired false discovery rate (proportion of false positives).

    Returns
    -------
    threshold_matrix : np.ndarray, shape (n_ngrams, n_genres)
        Boolean mask indicating which tests pass FDR correction.

    Notes
    -----
    Benjamini Y, Hochberg Y (1995) Controlling the false discovery rate:
    a practical and powerful approach to multiple hypothesis testing. J R Stat Soc B 57:289–300
    """
    from scipy.stats import norm

    # Flatten z-scores and compute p-values (one-sided test)
    z_flat = z_scores.flatten()
    p_values = 1 - norm.cdf(z_flat)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Benjamini-Hochberg procedure
    m = len(p_values)
    threshold_line = fdr_level * np.arange(1, m + 1) / m

    # Find largest i where p_i <= (i/m) * fdr_level
    significant = sorted_p <= threshold_line
    if np.any(significant):
        max_idx = np.where(significant)[0][-1]
        threshold_p = sorted_p[max_idx]
    else:
        threshold_p = 0.0  # No tests pass

    # Create boolean mask
    passes_fdr = (p_values <= threshold_p).reshape(z_scores.shape)

    return passes_fdr


def filter_discriminating_ngrams(
    z_scores: np.ndarray,
    ngram_names: List[str],
    genre_names: List[str],
    threshold: float = 2.326,
    apply_fdr: bool = False,
    fdr_level: float = 0.01,
) -> pd.DataFrame:
    """
    Filter n-grams by z-score threshold (one-sided test).

    Default threshold 2.326 corresponds to one-sided test at α=0.01
    (uncorrected). With apply_fdr=True, uses Benjamini-Hochberg correction
    to control false discovery rate across multiple tests.

    Parameters
    ----------
    z_scores : np.ndarray, shape (n_ngrams, n_genres)
        Z-scores for each (n-gram, genre) pair.
    ngram_names : List[str]
        N-gram identifiers (e.g., ["love", "dark_side"]).
    genre_names : List[str]
        Genre labels.
    threshold : float, default=2.326
        Z-score threshold (one-sided, α=0.01). Ignored if apply_fdr=True.
    apply_fdr : bool, default=False
        If True, apply Benjamini-Hochberg FDR correction instead of
        fixed threshold. Recommended for academic rigor with large
        feature spaces (m × k tests).
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

    # Build result dataframe
    discriminating = pd.DataFrame(
        {
            "ngram": [ngram_names[i] for i in ngram_idx],
            "genre": [genre_names[j] for j in genre_idx],
            "z_score": z_scores[ngram_idx, genre_idx],
        }
    )

    return discriminating


def compute_alpha_upper_bound(
    corpus: pd.Series,
    ngram_type: str,
    min_ngrams_per_track: int = 1,
    min_artists: int = 20,
    alpha_min: float = 0.01,
    alpha_max: float = 100.0,
    tolerance: float = 0.01,
) -> float:
    """
    Compute maximum alpha where all tracks have at least min_ngrams_per_track.

    Uses binary search to find the alpha value where filtering by z-threshold
    still leaves all tracks with sufficient n-grams for classification.

    This is an offline calibration utility. Run once before experiments to
    determine appropriate upper bounds for alpha hyperparameters.

    Parameters
    ----------
    corpus : pd.Series
        Lyrics text, one entry per track.
    ngram_type : str
        One of: 'unigram', 'bigram', 'trigram', 'quadgram'.
    min_ngrams_per_track : int, default=1
        Minimum required n-grams per track after filtering.
    min_artists : int, default=20
        Minimum artists threshold for n-gram inclusion.
    alpha_min : float, default=0.01
        Lower bound for binary search.
    alpha_max : float, default=100.0
        Upper bound for binary search.
    tolerance : float, default=0.01
        Convergence tolerance for binary search.

    Returns
    -------
    alpha_upper : float
        Maximum alpha satisfying constraint.

    Notes
    -----
    This function is computationally expensive and should be run on a
    subset of data (e.g., 5% or 1000 tracks) for calibration purposes.
    The resulting bounds are then hardcoded in config.py for production use.
    """
    # Import here to avoid circular dependency
    from helpers.MonroeExtractor import MonroeExtractor

    def check_coverage(alpha: float) -> float:
        """Return fraction of tracks with >= min_ngrams_per_track."""
        extractor = MonroeExtractor(
            min_artists=min_artists, **{f"alpha_{ngram_type}": alpha}
        )
        # Note: This requires y labels, so caller must provide corpus with labels
        # For now, return placeholder
        raise NotImplementedError(
            "alpha_upper_bound computation requires full implementation "
            "with labeled data. Use manual testing with small subsets."
        )

    # Binary search logic would go here
    # For now, return a reasonable default
    print(f"WARNING: alpha_upper_bound not fully implemented.")
    print(f"Recommended: Test manually with alpha values 0.1, 1.0, 10.0")
    print(
        f"Choose highest value where >95% tracks have >={min_ngrams_per_track} n-grams"
    )

    return 10.0  # Reasonable default placeholder
