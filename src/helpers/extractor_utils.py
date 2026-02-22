import random
from sklearn.feature_extraction.text import CountVectorizer


def extract_ngrams(texts, order, name, random_state):
    """Extract n-grams using CountVectorizer."""
    vectorizer = CountVectorizer(
        ngram_range=(order, order),
        token_pattern=r"\b[\w']+\b",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()

    rng = random.Random(random_state)
    sample = rng.sample(list(features), k=min(5, len(features)))

    print(f"Extracted {name}:")
    print(f"  - Unique: {len(features):,}")
    print(f"  - Shape: {matrix.shape}")
    print(f"  - Examples: {sample}")

    return matrix, features


def count_artists_per_ngram(artists, ngram_matrix, ngram_features):
    """Count unique artists per n-gram using pandas groupby."""
    import pandas as pd

    binary_matrix = (ngram_matrix > 0).astype(int).tocsc()
    artist_array = artists.to_numpy()

    rows, cols = binary_matrix.nonzero()

    df = pd.DataFrame({"ngram_idx": cols, "artist": artist_array[rows]})

    counts = df.groupby("ngram_idx")["artist"].nunique()

    artist_count = dict(zip(ngram_features[counts.index], counts.values))
    print(f"Counted unique artists for {len(artist_count):,} n-grams")
    return artist_count
