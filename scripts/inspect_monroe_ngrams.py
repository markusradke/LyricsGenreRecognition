"""Inspect top Monroe n-grams per genre by z-score.

Computes Monroe scores and displays highest-scoring n-grams for each genre.
"""

import pandas as pd
import numpy as np
from helpers import ngram_utils, monroe_logodds
from helpers.StopwordFilter import StopwordFilter
from helpers.config import MIN_ARTISTS, RANDOM_SEED


def compute_ngram_counts(X, y, ngrams, order, extract_within_lines=True):
    """Count n-gram occurrences per genre."""
    genres = y.unique()
    ngram_to_idx = {ng: i for i, ng in enumerate(ngrams)}

    y_gc = np.zeros((len(ngrams), len(genres)))
    n_c = np.zeros(len(genres))

    for genre_idx, genre in enumerate(genres):
        genre_mask = y == genre
        genre_texts = X[genre_mask]

        for text in genre_texts:
            tokens = ngram_utils.tokenize(text, extract_within_lines)
            text_ngrams = ngram_utils.extract_ngrams_by_order(
                tokens, [order], extract_within_lines
            )[order]

            for ng in text_ngrams:
                if ng in ngram_to_idx:
                    y_gc[ngram_to_idx[ng], genre_idx] += 1
                    n_c[genre_idx] += 1

    y_g = y_gc.sum(axis=1)
    n = n_c.sum()
    m = len(ngrams)

    return y_gc, n_c, y_g, int(n), m, list(genres)


def inspect_order(X, y, artist, order, alpha, stopword_filter, top_n=100):
    """Inspect top n-grams for given order."""
    all_ngrams_set = set()
    for text in X:
        tokens = ngram_utils.tokenize(text, preserve_lines=True)
        ngrams_dict = ngram_utils.extract_ngrams_by_order(tokens, [order], True)
        all_ngrams_set.update(ngrams_dict[order])

    all_ngrams_list = list(all_ngrams_set)
    if order > 1:
        all_ngrams_list = ngram_utils.strip_boundary_ngrams(all_ngrams_list)
    all_ngrams_list = ngram_utils.filter_stopword_only(all_ngrams_list, stopword_filter)

    artist_counts = ngram_utils.count_artists_per_ngram(
        set(all_ngrams_list), X, artist, extract_within_lines=True
    )
    filtered_ngrams = [
        ng for ng in all_ngrams_list if artist_counts.get(ng, 0) >= MIN_ARTISTS
    ]

    if len(filtered_ngrams) == 0:
        return pd.DataFrame()

    y_gc, n_c, y_g, n, m, genres = compute_ngram_counts(X, y, filtered_ngrams, order)

    alpha_array = np.full(len(filtered_ngrams), alpha)
    delta = monroe_logodds.compute_log_odds_delta(y_gc, n_c, y_g, n, m, alpha_array)
    variance = monroe_logodds.compute_variance(y_gc, n_c, y_g, n, m, alpha_array)
    z_scores = monroe_logodds.compute_z_scores(delta, variance)

    discriminating_df = monroe_logodds.filter_discriminating_ngrams(
        z_scores,
        ["_".join(ng) for ng in filtered_ngrams],
        genres,
        threshold=2.326,
        apply_fdr=True,
        fdr_level=0.01,
    )

    order_name = {1: "unigram", 2: "bigram", 3: "trigram", 4: "quadgram"}[order]
    discriminating_df["type"] = order_name

    return discriminating_df


def inspect_monroe_ngrams(granularity_col, top_n=100):
    """Display top n-grams per genre sorted by z-score."""
    print(f"\n{'='*70}")
    print(f"Monroe N-gram Inspection: {granularity_col}")
    print(f"{'='*70}\n")

    corpus = pd.read_csv(
        "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
    )
    corpus = corpus.sample(frac=0.1, random_state=RANDOM_SEED)
    stopword_filter = StopwordFilter()

    X = corpus["lyrics_lemmatized"]
    y = corpus[granularity_col]
    artist = corpus["track.s.firstartist.name"]

    print(f"Computing scores for {len(corpus):,} tracks...")
    all_results = []
    for order, alpha in [(1, 10.0), (2, 1.0), (3, 1.0), (4, 1.0)]:
        df = inspect_order(X, y, artist, order, alpha, stopword_filter, top_n)
        if len(df) > 0:
            all_results.append(df)
            print(f"  {order}-grams: {len(df)} discriminating")

    if not all_results:
        print("No discriminating n-grams found!")
        return

    results = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal: {len(results)} n-gram/genre pairs\n")

    for genre in sorted(y.unique()):
        genre_df = results[results["genre"] == genre].copy()
        genre_df = genre_df.sort_values("z_score", ascending=False).head(top_n)

        if len(genre_df) == 0:
            continue

        print(f"{genre}:")
        print(f"{'  Rank':<8}{'N-gram':<45}{'Type':<12}{'Z-score':>10}")
        print("  " + "-" * 72)

        for rank, row in enumerate(genre_df.itertuples(), 1):
            ngram_display = row.ngram.replace("_", " ")[:43]
            print(f"  {rank:<6}{ngram_display:<45}{row.type:<12}{row.z_score:>10.2f}")

        print()


if __name__ == "__main__":
    for cat in ["cat5", "cat12"]:  # , "cat25", "cat32"]:
        inspect_monroe_ngrams(cat, top_n=100)
