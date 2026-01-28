import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd


class InterRaterResult:
    def __init__(
        self,
        n_mc_runs: int,
        overall_macro_f1: float,
        per_genre_f1: pd.Series,
        per_run_overall_macro_f1: np.ndarray,
    ):
        self.n_mc_runs = n_mc_runs
        self.overall_macro_f1 = overall_macro_f1
        self.per_genre_f1 = per_genre_f1
        self.per_run_overall_macro_f1 = per_run_overall_macro_f1

        self.overall_macro_f1_std = float(np.std(per_run_overall_macro_f1))
        self.overall_macro_f1_ci = (
            float(np.percentile(per_run_overall_macro_f1, 2.5)),
            float(np.percentile(per_run_overall_macro_f1, 97.5)),
        )

    def __str__(self):
        return (
            "Simulated Interrater Agreement:\n"
            "===============================\n"
            f"Number of MC runs: {self.n_mc_runs}\n"
            f"Overall Macro F1: {self.overall_macro_f1:.3f} (sd: {self.overall_macro_f1_std:.3f}, CI: {self.overall_macro_f1_ci[0]:.3f}-{self.overall_macro_f1_ci[1]:.3f})\n"
            "===============================\n"
            f"Per-Genre F1:\n{self.per_genre_f1.to_string()}\n"
        )


def monte_carlo_interrater_agreement_f1(
    df: pd.DataFrame,
    granularity: int,
    n_mc: int = 1000,
    seed: int = 42,
) -> InterRaterResult:
    """
    Monte Carlo estimate of inter-rater agreement under a multinomial model.

    Data:
      - vote counts are columns: cat[n]_n_[genre]
        e.g. cat12_n_Rock, cat12_n_Pop, ...

    Model per track i:
      - use propensities p_i = counts_i / sum(counts_i) (requires sum>0)
      - Two independent "raters" sample y1_i, y2_i ~ Categorical(p_i)
      - Compute macro-F1 between y1 and y2; also per-genre (one-vs-rest) F1 averaged over MC

    Returns:
      - overall_macro_f1: mean macro-F1 over MC runs
      - per_genre_f1: mean per-genre F1 over MC runs (index = genre)
      - per_run_overall_macro_f1: macro-F1 for each MC run
    """
    _, counts_df = _extract_count_columns(df, granularity=granularity)
    genres = list(counts_df.columns)
    counts = counts_df.to_numpy(dtype=float)  # (N, G)

    row_sums = counts.sum(axis=1)

    rng = np.random.default_rng(seed)
    _, G = counts.shape

    per_run_macro = np.empty(n_mc, dtype=float)
    per_run_perclass = np.empty((n_mc, G), dtype=float)

    for k in range(n_mc):
        P = counts / row_sums[:, None]

        y1 = _sample_categorical_from_probs_rowwise(P, rng)
        y2 = _sample_categorical_from_probs_rowwise(P, rng)

        macro_f1, per_f1 = _macro_and_per_class_f1(y_true=y1, y_pred=y2, n_classes=G)
        per_run_macro[k] = macro_f1
        per_run_perclass[k, :] = per_f1

    per_genre_f1 = pd.Series(
        per_run_perclass.mean(axis=0), index=pd.Index(genres, name="genre"), name="f1"
    )
    overall_macro_f1 = float(per_run_macro.mean())

    return InterRaterResult(
        n_mc_runs=n_mc,
        overall_macro_f1=overall_macro_f1,
        per_genre_f1=per_genre_f1,
        per_run_overall_macro_f1=per_run_macro,
    )


def _extract_count_columns(
    df: pd.DataFrame, granularity: int
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Returns (genre_to_col, counts_df) where counts_df columns are ordered by genre name.
    Expected column schema: cat[n]_n_[genre]
    """
    pat = re.compile(rf"^cat{granularity}_n_(.+)$")
    genre_to_col = {}
    for col in df.columns:
        m = pat.match(col)
        if m:
            genre = m.group(1)
            genre_to_col[genre] = col

    if not genre_to_col:
        raise ValueError(
            f"No columns found for granularity={granularity}. Expected columns like 'cat{granularity}_n_<genre>'."
        )

    genres_sorted = sorted(genre_to_col.keys())
    cols_sorted = [genre_to_col[g] for g in genres_sorted]
    counts_df = df[cols_sorted].copy()
    counts_df.columns = genres_sorted  # rename to genre names for convenience
    return genre_to_col, counts_df


def _sample_categorical_from_probs_rowwise(
    P: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Vectorized-ish categorical sampling per row.
    P: (N, G), rows sum to 1
    returns y: (N,), in [0..G-1]
    """
    cdf = np.cumsum(P, axis=1)
    u = rng.random((P.shape[0], 1))
    return (u <= cdf).argmax(axis=1)


def _macro_and_per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> Tuple[float, np.ndarray]:
    """
    Computes macro-F1 and per-class F1 for single-label multi-class classification.
    Simple implementation via confusion counts (no sklearn).
    """
    per_f1 = np.zeros(n_classes, dtype=float)

    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        denom = 2 * tp + fp + fn
        per_f1[c] = (2 * tp / denom) if denom > 0 else 0.0

    macro_f1 = float(per_f1.mean()) if n_classes > 0 else 0.0
    return macro_f1, per_f1
