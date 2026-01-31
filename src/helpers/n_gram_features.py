import pandas as pd
import numpy as np
import random
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix


def build_ngram_features(corpus, min_artists=50, top_n=100):
    """Build n-gram feature table from a lyrics corpus.

    This runs the full n-gram extraction and selection pipeline:
    extract unigrams/bigrams/trigrams, compute genre-level TF-IDF,
    filter by artist diversity, select top n-grams per genre and count
    final n-grams in each track.

    Args:
        corpus (pandas.DataFrame or dict-like):
            Table-like object with at least "lyrics", a genre
            column named "genre", and an artist column names "artist".
        min_artists (int, optional):
            Minimum distinct artists an n-gram must appear in to be kept.
            Defaults to 50.
        top_n (int, optional):
            Number of top-ranked n-grams to select per genre per n-gram order.
            Defaults to 100.

    Returns:
        pandas.DataFrame:
            DataFrame (one row per track) containing counts of selected n-grams.

    Raises:
        KeyError: If required columns (e.g. "lyrics" or genre column)
            are missing from `corpus`.

    Example:
        >>> df = build_ngram_features(corpus_df, granularity="3", min_artists=50)
    """

    print("=" * 60)
    print("Extracting n-grams from all lyrics...")
    print("=" * 60)
    uni_vectorizer, uni_matrix, uni_features = extract_ngrams(
        corpus["lyrics"], 1, 1, "unigrams"
    )
    bi_vectorizer, bi_matrix, bi_features = extract_ngrams(
        corpus["lyrics"], 2, 2, "bigrams"
    )
    tri_vectorizer, tri_matrix, tri_features = extract_ngrams(
        corpus["lyrics"], 3, 3, "trigrams"
    )

    print("=" * 60)
    print("Calculating tf-idf for combinations of n-grams and genres...")
    print("=" * 60)
    uni_tfidf_cat = calculate_genre_tfidf(
        corpus, "genre", uni_matrix, uni_features, "unigrams"
    )
    bi_tfidf_cat = calculate_genre_tfidf(
        corpus, "genre", bi_matrix, bi_features, "bigrams"
    )
    tri_tfidf_cat = calculate_genre_tfidf(
        corpus, "genre", tri_matrix, tri_features, "trigrams"
    )

    print("=" * 60)
    print(
        "Counting artists per ngram (and saving dictionrary and loading precomputed if possible), takes a while..."
    )
    print("=" * 60)
    uni_artist_count = count_artists_per_ngram(corpus, uni_matrix, uni_features)
    bi_artist_count = count_artists_per_ngram(corpus, bi_matrix, bi_features)
    tri_artist_count = count_artists_per_ngram(corpus, tri_matrix, tri_features)

    print("=" * 60)
    print(f"Filtering ngrams occurring in at least {min_artists} artists...")
    print("=" * 60)

    uni_tfidf_cat_filtered = filter_by_min_artists(
        uni_tfidf_cat, uni_artist_count, min_artists
    )
    bi_tfidf_cat_filtered = filter_by_min_artists(
        bi_tfidf_cat, bi_artist_count, min_artists
    )
    tri_tfidf_cat_filtered = filter_by_min_artists(
        tri_tfidf_cat, tri_artist_count, min_artists
    )

    print("=" * 60)
    print("Ranking ngrams by genre and tfidf...")
    print("=" * 60)
    uni_ranked_cat = rank_ngrams_by_genre(uni_tfidf_cat_filtered)
    bi_ranked_cat = rank_ngrams_by_genre(bi_tfidf_cat_filtered)
    tri_ranked_cat = rank_ngrams_by_genre(tri_tfidf_cat_filtered)

    top_unigrams = get_top_ngrams(uni_ranked_cat, top_n)
    top_bigrams = get_top_ngrams(bi_ranked_cat, top_n)
    top_trigrams = get_top_ngrams(tri_ranked_cat, top_n)

    print("=" * 60)
    print("Counting final ngrams in each track's lyrics")
    print("=" * 60)

    final_ngrams = set(top_unigrams).union(set(top_bigrams)).union(set(top_trigrams))
    print(
        "Total unique ngrams in final feature set: %s" % format(len(final_ngrams), ",")
    )
    print(final_ngrams)

    n_gram_features = count_final_ngrams_lyrics(corpus["lyrics"], final_ngrams)
    return n_gram_features


def extract_ngrams(texts, n_min, n_max, name=""):
    """Extract n-grams from texts using CountVectorizer.

    The returned matrix contains raw counts per document.

    Args:
        texts (iterable): Iterable of text documents (strings).
        n_min (int): Minimum n-gram size.
        n_max (int): Maximum n-gram size.
        name (str, optional): Descriptive name used in printed output.

    Returns:
        tuple: (vectorizer, matrix, feature_names)
            vectorizer: fitted sklearn CountVectorizer
            matrix: sparse matrix of n-gram counts (documents x features)
            feature_names: array of feature name strings

    Example:
        >>> vec, mat, feats = extract_ngrams(["a b c", "b c d"], 1, 2, "demo")
    """
    vectorizer = CountVectorizer(
        ngram_range=(n_min, n_max),
        token_pattern=r"\b[\w']+\b",  # include apostrophes in tokens
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"✓ Extracted {name}:")
    print(f"  - Number of unique {name}: {len(feature_names):,}")
    print(f"  - Matrix shape: {matrix.shape}")
    print(f"  - Example {name}: {_get_random_sample(feature_names)}")
    return vectorizer, matrix, feature_names


def _get_random_sample(feature_names, sample_size=5, seed=42):
    """Return a reproducible random sample of feature names.

    Args:
        feature_names (Sequence[str]): Sequence of feature strings.
        sample_size (int, optional): Number of items to sample. Defaults to 5.
        seed (int, optional): RNG seed for reproducibility. Defaults to 42.

    Returns:
        list[str]: Random sample from feature_names.
    """
    rng = random.Random(seed)
    sample_count = min(sample_size, len(feature_names))
    return rng.sample(list(feature_names), k=sample_count)


def calculate_genre_tfidf(df, genre_col, ngram_matrix, ngram_features, ngram_name):
    """Compute genre-level TF, IDF and TF-IDF for n-grams.

    TF is computed as the proportion of tracks within a genre that contain
    the n-gram (counts are binary-per-track for presence). IDF uses a
    smoothed log formula across genres.

    Args:
        df (pandas.DataFrame): DataFrame containing the genre column.
        genre_col (str): Name of the genre column in df.
        ngram_matrix (scipy.sparse matrix): Sparse count matrix (tracks x ngrams).
        ngram_features (Sequence[str]): Sequence of n-gram strings.
        ngram_name (str): Descriptive name used in printed output.

    Returns:
        pandas.DataFrame:
            DataFrame with columns ["genre", "ngram", "count", "tf", "idf", "tfidf"].

    Raises:
        KeyError: If `genre_col` is not present in `df`.
    """
    print(
        f"Calculating genre-level TF-IDF for {ngram_name} with {genre_col} genres ..."
    )

    # Convert to binary sparse matrix (no conversion to dense)
    binary_matrix = (ngram_matrix > 0).astype(int)

    # Get genres as numpy array for faster access
    genres_array = df[genre_col].values
    unique_genres = df[genre_col].unique()
    num_genres = len(unique_genres)

    # Build genre-ngram counts using sparse matrix operations
    genre_ngram_counts = defaultdict(lambda: defaultdict(int))

    # Process in COO format for efficient iteration
    binary_coo = binary_matrix.tocoo()

    for track_idx, ngram_idx in zip(binary_coo.row, binary_coo.col):
        genre = genres_array[track_idx]
        ngram = ngram_features[ngram_idx]
        genre_ngram_counts[genre][ngram] += 1

    # Calculate IDF only for ngrams that actually appear
    ngram_idf = {}
    for genre_dict in genre_ngram_counts.values():
        for ngram in genre_dict.keys():
            if ngram not in ngram_idf:
                genres_with_ngram = sum(
                    1
                    for g in genre_ngram_counts.keys()
                    if ngram in genre_ngram_counts[g]
                )
                # add smoothing: log((N + 1) / (df + 1)) + 1
                ngram_idf[ngram] = np.log((num_genres + 1) / genres_with_ngram + 1) + 1

    # Build results using pre-calculated total counts
    results = []
    for genre in genre_ngram_counts.keys():
        total_ngrams_in_genre = sum(genre_ngram_counts[genre].values())

        for ngram, count in genre_ngram_counts[genre].items():
            tf = count / total_ngrams_in_genre
            results.append(
                {
                    "genre": genre,
                    "ngram": ngram,
                    "count": count,
                    "tf": tf,
                    "idf": ngram_idf[ngram],
                    "tfidf": tf * ngram_idf[ngram],
                }
            )

    genre_tfidf_df = pd.DataFrame(results)
    print(f"✓ Calculated TF-IDF for {len(genre_tfidf_df):,} genre-ngram pairs")
    return genre_tfidf_df


def rank_ngrams_by_genre(tfidf_df, top=100):
    """Sort TF-IDF DataFrame by genre and descending TF-IDF.

    Args:
        tfidf_df (pandas.DataFrame): DataFrame with 'genre' and 'tfidf' columns.
        top (int, optional): Not used in sorting; kept for API compatibility.

    Returns:
        pandas.DataFrame: Sorted DataFrame with highest TF-IDF per genre first.
    """
    ranked = tfidf_df.sort_values(
        by=["genre", "tfidf"], ascending=[True, False]
    ).reset_index(drop=True)
    return ranked


def display_top_ngrams(ranked_df, ngram_name, top_n=10):
    """Print top n-grams per genre to stdout.

    Args:
        ranked_df (pandas.DataFrame): Ranked TF-IDF DataFrame with 'genre' and
            'ngram' columns.
        ngram_name (str): Descriptive name for the n-gram group used in headers.
        top_n (int, optional): Number of top n-grams to display per genre.
    """
    print("=" * 60)
    print(f"Top {top_n} {ngram_name} per genre")
    print("=" * 60)
    for genre in sorted(ranked_df["genre"].unique()):
        print(f"\n{genre.upper()}:")
        genre_top = ranked_df[ranked_df["genre"] == genre].head(top_n)

        for idx, row in genre_top.iterrows():
            print(
                f"  {row['ngram']:30s} | TF-IDF: {row['tfidf']:.4f} | Count: {row['count']:3d}"
            )


def count_artists_per_ngram(df, ngram_matrix, ngram_features):
    """Count number of unique artists that use each n-gram.

    This is optimized to operate on a sparse matrix and returns a mapping
    from n-gram string to number of distinct artists that contain it.

    Args:
        df (pandas.DataFrame): DataFrame that must contain
            "artist".
        ngram_matrix (scipy.sparse matrix): Sparse count matrix (tracks x ngrams).
        ngram_features (Sequence[str]): Sequence of n-gram strings.

    Returns:
        dict: Mapping {ngram: unique_artist_count}.

    Raises:
        KeyError: If "artist" is missing from df.
    """
    print("Counting artists per n-gram...")

    binary_matrix = (ngram_matrix > 0).astype(int)
    binary_matrix = binary_matrix.tocsc()

    artist_series = df["artist"].reset_index(drop=True)
    artist_count = {}

    for ngram_idx in tqdm(range(len(ngram_features))):
        track_indices = binary_matrix[:, ngram_idx].nonzero()[0]
        unique_artists = artist_series.iloc[track_indices].nunique()
        ngram = ngram_features[ngram_idx]
        artist_count[ngram] = unique_artists
    print(f"✓ Calculated artist diversity for {len(artist_count):,} n-grams")

    return artist_count


def filter_by_min_artists(tfidf_df, artist_count, min_artists=10):
    """Filter TF-IDF rows by minimum artist diversity.

    Args:
        tfidf_df (pandas.DataFrame): DataFrame with an 'ngram' column.
        artist_count (dict): Mapping from ngram to number of unique artists.
        min_artists (int, optional): Minimum artists threshold. Defaults to 10.

    Returns:
        pandas.DataFrame: Filtered TF-IDF DataFrame.
    """
    filtered_df = tfidf_df[tfidf_df["ngram"].map(artist_count) >= min_artists].copy()
    return filtered_df


def get_top_ngrams(ranked_df, top_n):
    """Select top `top_n` unique n-grams per genre.

    Args:
        ranked_df (pandas.DataFrame): Ranked TF-IDF DataFrame with 'genre' and
            'ngram' columns.
        top_n (int): Number of top n-grams to take per genre.

    Returns:
        numpy.ndarray: Array of unique selected n-grams.
    """
    ngrams = ranked_df.groupby("genre").head(top_n)["ngram"].unique()
    print("Total unique ngrams selected: %s" % format(len(ngrams), ","))
    return ngrams


def count_final_ngrams_lyrics(lyrics, ngram_list):
    """Count occurrences of a final vocabulary of n-grams in lyrics.

    Args:
        lyrics (iterable): Iterable of lyric strings (one per track).
        ngram_list (iterable): Vocabulary of n-grams to count.

    Returns:
        pandas.DataFrame: DataFrame with one column per n-gram and counts per track.

    Example:
        >>> df = count_final_ngrams_lyrics(["hello world", "world hello"], ["hello", "world"])
    """
    vectorizer = CountVectorizer(
        vocabulary=ngram_list,
        token_pattern=r"\b[\w']+\b",
        lowercase=True,
        ngram_range=(
            1,
            3,
        ),
    )

    ngram_matrix = vectorizer.fit_transform(lyrics)
    ngram_df = pd.DataFrame(
        ngram_matrix.toarray(), columns=vectorizer.get_feature_names_out()
    )
    return ngram_df
