import pandas as pd
import numpy as np
import random
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix


def build_ngram_features(corpus, granularity, min_artists=50, top_n=100):
    """
    Build n-gram based features from a lyrics corpus replicating Fell and Spohrleder (2014).
    This function performs a multi-step pipeline to extract and select n-gram features
    from song lyrics, then counts the selected n-grams in every track and returns a
    table of feature counts.
    Steps performed (high-level):
    - Extract unigrams, bigrams and trigrams from corpus["full_lyrics"].
    - Compute TF-IDF aggregated by genre at specified FOLDAT granularity (genre column named "cat{granularity}").
    - Count the number of distinct artists in which each n-gram appears. These counts
        are cached to disk as pickle files when first computed and reused on subsequent runs due to long computation time.
    - Discard n-grams that appear in fewer than `min_artists` distinct artists.
    - Rank n-grams per genre by TF-IDF and select the top `top_n` n-grams for each genre.
    - Create a final union of selected unigrams, bigrams and trigrams.
    - Count occurrences of each final n-gram in every track's lyrics.
    - Save the resulting n-gram feature table to CSV and return it as a pandas DataFrame.
    Parameters
    - corpus (pandas.DataFrame or dict-like):
            A table-like object providing at least:
                - "full_lyrics": an iterable/Series of lyric strings (one entry per track).
                - A genre/category column named "cat{granularity}" (e.g. "cat3" if granularity is "3")
                    used to aggregate TF-IDF by genre.
            The function assumes corpus supports indexing by column name (like a pandas DataFrame).
    - granularity (str or convertible to str):
            The genre granularity identifier used to form the genre column name
            ("cat" + granularity) and file paths. Example values: "5", "12", "25", and "32".
    - min_artists (int, optional, default=50):
            Minimum number of distinct artists that must contain an n-gram for it to be
            considered as a candidate feature. N-grams appearing in fewer artists are filtered out.
    - top_n (int, optional, default=100):
            Number of top-ranked n-grams to select per genre for each n-gram order
            (unigram, bigram, trigram). Final feature set is the union of these selections.
    Returns
    - pandas.DataFrame:
            A DataFrame (one row per track) containing counts of the final selected n-grams
            in each track's lyrics. The function also writes this DataFrame to:
                ../data/FS_G{granularity}_lyrics_n_gram_features.csv
    Side effects
    - Reads/writes cached pickle files for artist counts:
            ../data/FS_G{granularity}_unigram_artist_count.pkl
            ../data/FS_G{granularity}_bigram_artist_count.pkl
            ../data/FS_G{granularity}_trigram_artist_count.pkl
        If these files exist they are loaded; otherwise artist counts are computed and saved.
    - Writes a CSV with the final n-gram feature table:
            ../data/FS_G{granularity}_lyrics_n_gram_features.csv
    - Prints progress and diagnostic messages to stdout.
    Notes and assumptions
    - The function expects helper functions to be available in the same module or scope:
        extract_ngrams, calculate_genre_tfidf, count_artists_per_ngram, filter_by_min_artists,
        rank_ngrams_by_genre, get_top_ngrams, count_final_ngrams_lyrics.
    - granularity should be provided such that "cat" + granularity matches a column
        in `corpus`. If granularity is passed as an int, it will be cast to str where needed.
    - File I/O paths are relative; ensure the working directory and ../data/ exist and are writable.
    Example
    - build_ngram_features(corpus_df, granularity="3", min_artists=50, top_n=100)
    """

    print("=" * 60)
    print("Extracting n-grams from all lyrics...")
    print("=" * 60)
    uni_vectorizer, uni_matrix, uni_features = extract_ngrams(
        corpus["full_lyrics"], 1, 1, "unigrams"
    )
    bi_vectorizer, bi_matrix, bi_features = extract_ngrams(
        corpus["full_lyrics"], 2, 2, "bigrams"
    )
    tri_vectorizer, tri_matrix, tri_features = extract_ngrams(
        corpus["full_lyrics"], 3, 3, "trigrams"
    )

    print("=" * 60)
    print(
        f"Calculating tf-idf for combinations of n-grams and G{granularity} genres..."
    )
    print("=" * 60)
    uni_tfidf_cat = calculate_genre_tfidf(
        corpus, "cat" + str(granularity), uni_matrix, uni_features, "unigrams"
    )
    bi_tfidf_cat = calculate_genre_tfidf(
        corpus, "cat" + str(granularity), bi_matrix, bi_features, "bigrams"
    )
    tri_tfidf_cat = calculate_genre_tfidf(
        corpus, "cat" + str(granularity), tri_matrix, tri_features, "trigrams"
    )

    print("=" * 60)
    print(
        "Counting artists per ngram (and saving dictionrary and loading precomputed if possible), takes a while..."
    )
    print("=" * 60)

    try:
        with open(f"data/FS_G{granularity}_unigram_artist_count.pkl", "rb") as f:
            uni_artist_count = pickle.load(f)
    except FileNotFoundError:
        uni_artist_count = count_artists_per_ngram(corpus, uni_matrix, uni_features)
        with open(f"data/FS_G{granularity}_unigram_artist_count.pkl", "wb") as f:
            pickle.dump(uni_artist_count, f)
    try:
        with open(f"data/FS_G{granularity}_bigram_artist_count.pkl", "rb") as f:
            bi_artist_count = pickle.load(f)
    except FileNotFoundError:
        bi_artist_count = count_artists_per_ngram(corpus, bi_matrix, bi_features)
        with open(f"data/FS_G{granularity}_bigram_artist_count.pkl", "wb") as f:
            pickle.dump(bi_artist_count, f)
    try:
        with open(f"data/FS_G{granularity}_trigram_artist_count.pkl", "rb") as f:
            tri_artist_count = pickle.load(f)
    except FileNotFoundError:
        tri_artist_count = count_artists_per_ngram(corpus, tri_matrix, tri_features)
        with open(f"data/FS_G{granularity}_trigram_artist_count.pkl", "wb") as f:
            pickle.dump(tri_artist_count, f)

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

    n_gram_features = count_final_ngrams_lyrics(corpus["full_lyrics"], final_ngrams)
    n_gram_features.to_csv(
        f"data/FS_G{granularity}_lyrics_n_gram_features.csv", index=False
    )
    print(
        f"✓ Saved n-gram features to ../data/FS_G{granularity}_lyrics_n_gram_features.csv"
    )
    return n_gram_features


def extract_ngrams(texts, n_min, n_max, name=""):
    """
    Extract n-grams from a collection of texts.

    Parameters:
    - texts: list or Series of text documents
    - n_min: minimum n-gram size (e.g., 1 for unigrams)
    - n_max: maximum n-gram size (e.g., 1 for unigrams, 2 for bigrams)
    - name: descriptive name for printing

    Returns:
    - vectorizer: fitted CountVectorizer object
    - matrix: sparse matrix of n-gram counts
    - feature_names: list of n-gram strings
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
    rng = random.Random(seed)
    sample_count = min(sample_size, len(feature_names))
    return rng.sample(list(feature_names), k=sample_count)


def calculate_genre_tfidf(df, genre_col, ngram_matrix, ngram_features, ngram_name):
    """
    Calculate TF-IDF scores for n-grams at the genre level. ngrams are counted only once per track

    Parameters:
    - df: DataFrame with 'genre' column
    - genre_col: name of the genre column in df
    - ngram_matrix: sparse matrix of n-gram counts per track
    - ngram_features: array of n-gram strings
    - ngram_name: name for printing (e.g., "unigrams")

    Returns:
    - genre_tfidf_df: DataFrame with genres, n_grams, and TF_IDF scores
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
    """
    Rank n-grams by TF-IDF score within each genre.

    Parameters:
    - tfidf_df: DataFrame with genre, ngram, and tfidf columns
    - top_n: number of top n-grams to display per genre

    Returns:
    - ranked_df: DataFrame sorted by genre and tfidf score
    """
    ranked = tfidf_df.sort_values(
        by=["genre", "tfidf"], ascending=[True, False]
    ).reset_index(drop=True)
    return ranked


def display_top_ngrams(ranked_df, ngram_name, top_n=10):
    """
    Display top n-grams per genre.
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
    """
    Count how many unique artists use each n-gram (optimized).

    Parameters:
    - df: DataFrame with 'artist' column
    - ngram_matrix: sparse matrix of n-gram counts
    - ngram_features: array of n-gram strings

    Returns:
    - artist_count: dict mapping n-gram to number of unique artists
    """
    print("Counting artists per n-gram...")

    binary_matrix = (ngram_matrix > 0).astype(int)
    binary_matrix = binary_matrix.tocsc()

    artist_series = df["track.s.firstartist.name"].reset_index(drop=True)
    artist_count = {}

    for ngram_idx in tqdm(range(len(ngram_features))):
        track_indices = binary_matrix[:, ngram_idx].nonzero()[0]
        unique_artists = artist_series.iloc[track_indices].nunique()
        ngram = ngram_features[ngram_idx]
        artist_count[ngram] = unique_artists
    print(f"✓ Calculated artist diversity for {len(artist_count):,} n-grams")

    return artist_count


def filter_by_min_artists(tfidf_df, artist_count, min_artists=10):
    """
    Filter tf-idf DataFrame to keep only n-grams used by at least min_artists unique artists.
    """
    filtered_df = tfidf_df[tfidf_df["ngram"].map(artist_count) >= min_artists].copy()
    return filtered_df


def get_top_ngrams(ranked_df, top_n):
    ngrams = ranked_df.groupby("genre").head(top_n)["ngram"].unique()
    print("Total unique ngrams selected: %s" % format(len(ngrams), ","))
    return ngrams


def count_final_ngrams_lyrics(lyrics, ngram_list):
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
