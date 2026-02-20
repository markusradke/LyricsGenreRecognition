import pandas as pd
import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from scipy.sparse import csr_matrix


class NGramFeatureExtractorFS:

    def __init__(
        self, min_artists: int = 50, top_n: int = 100, random_state: int = 42
    ) -> None:
        self.min_artists = min_artists
        self.top_n = top_n
        self.random_state = random_state
        self.vectorizers: dict[str, CountVectorizer] = {}
        self.matrices: dict[str, csr_matrix] = {}
        self.features: dict[str, np.ndarray] = {}
        self.final_ngrams: set[str] = set()

    def fit_transform(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Extract and select n-gram features from corpus.

        Args:
            corpus: DataFrame with "lyrics", "genre", and "artist" columns.

        Returns:
            DataFrame with n-gram counts per track.
        """
        for order, name in [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]:
            vec, mat, feats = self._extract_ngrams(corpus["lyrics"], order, name)
            self.vectorizers[name] = vec
            self.matrices[name] = mat
            self.features[name] = feats

        tfidf_by_order = {
            name: self._calculate_genre_tfidf(
                corpus, self.matrices[name], self.features[name]
            )
            for name in ["unigrams", "bigrams", "trigrams"]
        }

        artist_counts_by_order = {
            name: self._count_artists_per_ngram(
                corpus, self.matrices[name], self.features[name]
            )
            for name in ["unigrams", "bigrams", "trigrams"]
        }

        filtered_tfidf = {}
        for name in ["unigrams", "bigrams", "trigrams"]:
            tfidf = tfidf_by_order[name]
            filtered = tfidf[
                tfidf["ngram"].map(artist_counts_by_order[name]) >= self.min_artists
            ].copy()
            filtered_tfidf[name] = filtered.sort_values(
                by=["genre", "tfidf"], ascending=[True, False]
            ).reset_index(drop=True)

        self.final_ngrams = self._select_top_ngrams(filtered_tfidf)
        return self._count_final_ngrams(corpus["lyrics"])

    def _extract_ngrams(
        self, texts: pd.Series, order: int, name: str
    ) -> tuple[CountVectorizer, csr_matrix, np.ndarray]:
        """Extract n-grams using CountVectorizer."""
        vectorizer = CountVectorizer(
            ngram_range=(order, order),
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(texts)
        features = vectorizer.get_feature_names_out()

        rng = random.Random(self.random_state)
        sample = rng.sample(list(features), k=min(5, len(features)))

        print(f"Extracted {name}:")
        print(f"  - Unique: {len(features):,}")
        print(f"  - Shape: {matrix.shape}")
        print(f"  - Examples: {sample}")

        return vectorizer, matrix, features

    def _calculate_genre_tfidf(
        self, corpus: pd.DataFrame, ngram_matrix: csr_matrix, ngram_features: np.ndarray
    ) -> pd.DataFrame:
        """Compute genre-level TF-IDF for n-grams."""
        binary_matrix = (ngram_matrix > 0).astype(int).tocoo()
        genres_array = corpus["genre"].values
        unique_genres = corpus["genre"].unique()
        num_genres = len(unique_genres)

        genre_ngram_counts = defaultdict(lambda: defaultdict(int))

        for track_idx, ngram_idx in zip(binary_matrix.row, binary_matrix.col):
            genre = genres_array[track_idx]
            ngram = ngram_features[ngram_idx]
            genre_ngram_counts[genre][ngram] += 1

        all_ngrams = set()
        for genre_dict in genre_ngram_counts.values():
            all_ngrams.update(genre_dict.keys())

        ngram_idf = {}
        for ngram in all_ngrams:
            genres_with_ngram = sum(
                1 for g_dict in genre_ngram_counts.values() if ngram in g_dict
            )
            ngram_idf[ngram] = np.log((num_genres + 1) / (genres_with_ngram + 1)) + 1

        results = []
        for genre, ngram_dict in genre_ngram_counts.items():
            total = sum(ngram_dict.values())
            for ngram, count in ngram_dict.items():
                tf = count / total
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

        return pd.DataFrame(results)

    def _count_artists_per_ngram(
        self, corpus: pd.DataFrame, ngram_matrix: csr_matrix, ngram_features: np.ndarray
    ) -> dict[str, int]:
        """Count unique artists per n-gram."""
        binary_matrix = (ngram_matrix > 0).astype(int).tocsc()
        artist_series = corpus["artist"].reset_index(drop=True)
        artist_count = {}

        for ngram_idx, ngram in enumerate(ngram_features):
            track_indices = binary_matrix[:, ngram_idx].nonzero()[0]
            artist_count[ngram] = artist_series.iloc[track_indices].nunique()

        return artist_count

    def _select_top_ngrams(self, ranked_tfidf: dict[str, pd.DataFrame]) -> set[str]:
        """Select top n-grams per genre across all orders."""
        top_sets = [
            set(ranked_tfidf[name].groupby("genre").head(self.top_n)["ngram"].unique())
            for name in ["unigrams", "bigrams", "trigrams"]
        ]

        final = set.union(*top_sets) if top_sets else set()
        print(f"Total unique n-grams: {len(final):,}")
        return final

    def _count_final_ngrams(self, lyrics: pd.Series) -> pd.DataFrame:
        """Count final n-grams in lyrics."""
        vectorizer = CountVectorizer(
            vocabulary=list(self.final_ngrams),
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
            ngram_range=(1, 3),
        )
        matrix = vectorizer.fit_transform(lyrics)
        return pd.DataFrame(
            matrix.toarray(), columns=vectorizer.get_feature_names_out()
        )

    def transform(self, lyrics: pd.Series) -> pd.DataFrame:
        """Count n-grams in new lyrics using fitted vocabulary.

        Args:
            lyrics: Series of lyrics text to tokenize and count.

        Returns:
            DataFrame with n-gram counts per track.

        Raises:
            ValueError: If transform called before fit.
        """
        if not self.final_ngrams:
            raise ValueError("Must call fit() before transform()")

        return self._count_final_ngrams(lyrics)
