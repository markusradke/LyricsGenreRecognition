import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def monte_carlo_interrater_agreement_f1(
    df: pd.DataFrame,
    granularity: int,
    n_mc: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
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
      - DataFrame with per-genre F1 scores (colums: iteration, genre, f1)
    """
    _, counts_df = _extract_count_columns(df, granularity=granularity)
    genres = list(counts_df.columns)
    counts = counts_df.to_numpy(dtype=float)  # (N, G)

    row_sums = counts.sum(axis=1)

    rng = np.random.default_rng(seed)
    _, G = counts.shape

    per_run_perclass = np.empty((n_mc, G), dtype=float)

    P = counts / row_sums[:, None]  # compute propensities once

    for k in range(n_mc):
        y1 = _sample_categorical_from_probs_rowwise(P, rng)
        y2 = _sample_categorical_from_probs_rowwise(P, rng)
        per_run_perclass[k, :] = _per_class_f1(y_true=y1, y_pred=y2, n_classes=G)

    iterations = np.repeat(np.arange(n_mc), G)
    genres_arr = np.tile(genres, n_mc)
    f1_flat = per_run_perclass.ravel()
    return pd.DataFrame({"iteration": iterations, "genre": genres_arr, "f1": f1_flat})


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


def _per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> Tuple[float, np.ndarray]:
    """
    Computes per-class F1 for single-label multi-class classification.
    Simple implementation via confusion counts.
    """
    per_f1 = np.zeros(n_classes, dtype=float)

    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        denom = 2 * tp + fp + fn
        per_f1[c] = (2 * tp / denom) if denom > 0 else 0.0

    return per_f1


def print_simulated_interrater_agreement(mc):
    n_mc = mc["iteration"].nunique()
    macro_f1_iter = mc.groupby("iteration")["f1"].mean()
    mean_macro_f1 = macro_f1_iter.mean()
    std_macro_f1 = macro_f1_iter.std()
    ci_macro_f1 = macro_f1_iter.quantile([0.025, 0.975]).values

    mean_f1_per_genre = mc.groupby("genre")["f1"].mean()
    std_f1_per_genre = mc.groupby("genre")["f1"].std()
    ci_lower_per_genre = mc.groupby("genre")["f1"].quantile(0.025)
    ci_upper_per_genre = mc.groupby("genre")["f1"].quantile(0.975)
    per_genre_stat = pd.concat(
        (mean_f1_per_genre, std_f1_per_genre, ci_lower_per_genre, ci_upper_per_genre),
        axis=1,
    )
    per_genre_stat.columns = ["mean", "std", "CI lower", "CI upper"]
    per_genre_stat.sort_values(by="mean", axis=0, ascending=False, inplace=True)
    pd.options.display.float_format = "{:.3f}".format

    print("SIMULATED INTERRATER AGREEMENT (MONTE CARLO):")
    print("=" * 60)
    print("MC iterations: %s" % format(n_mc, ","))
    print(f"Mean Macro F1: {mean_macro_f1:.3f} ± {std_macro_f1:.3f} (stddev)")
    print(f"95% CI: [{ci_macro_f1[0]:.3f}, {ci_macro_f1[1]:.3f}]")
    print("=" * 60)
    print(per_genre_stat)
