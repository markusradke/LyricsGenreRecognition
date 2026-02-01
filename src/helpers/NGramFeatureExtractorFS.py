import pandas as pd
import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix


class NGramFeatureExtractorFS:
    """Extract and rank n-grams from lyrics by genre."""

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

    def fit(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Extract and select n-gram features from corpus.

        Args:
            corpus: DataFrame with "lyrics", "genre", and "artist" columns.

        Returns:
            DataFrame with n-gram counts per track.
        """
        self._extract_ngrams_all_orders(corpus["lyrics"])
        tfidf_by_order = self._calculate_tfidf_all_orders(corpus)
        artist_counts_by_order = self._count_artists_all_orders(corpus)

        filtered_tfidf = self._filter_and_rank_all_orders(
            tfidf_by_order, artist_counts_by_order
        )

        self.final_ngrams = self._select_top_ngrams(filtered_tfidf)
        return self._count_final_ngrams(corpus["lyrics"], self.final_ngrams)

    def _extract_ngrams_all_orders(self, texts: pd.Series) -> None:
        """Fit vectorizers and extract n-grams for orders 1, 2, 3."""
        for order, name in [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]:
            vec, mat, feats = self._extract_ngrams(texts, order, order, name)
            self.vectorizers[name] = vec
            self.matrices[name] = mat
            self.features[name] = feats

    def _calculate_tfidf_all_orders(
        self, corpus: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Calculate genre-level TF-IDF for all n-gram orders."""
        tfidf = {}
        for name in ["unigrams", "bigrams", "trigrams"]:
            tfidf[name] = self._calculate_genre_tfidf(
                corpus, self.matrices[name], self.features[name], name
            )
        return tfidf

    def _count_artists_all_orders(
        self, corpus: pd.DataFrame
    ) -> dict[str, dict[str, int]]:
        """Count unique artists per n-gram for all orders."""
        counts = {}
        for name in ["unigrams", "bigrams", "trigrams"]:
            counts[name] = self._count_artists_per_ngram(
                corpus, self.matrices[name], self.features[name]
            )
        return counts

    def _filter_and_rank_all_orders(
        self,
        tfidf_by_order: dict[str, pd.DataFrame],
        artist_counts_by_order: dict[str, dict[str, int]],
    ) -> dict[str, pd.DataFrame]:
        """Filter by artist diversity and rank all n-gram orders."""
        filtered = {}
        for name in ["unigrams", "bigrams", "trigrams"]:
            tfidf = tfidf_by_order[name]
            filtered_tfidf = tfidf[
                tfidf["ngram"].map(artist_counts_by_order[name]) >= self.min_artists
            ].copy()
            filtered[name] = filtered_tfidf.sort_values(
                by=["genre", "tfidf"], ascending=[True, False]
            ).reset_index(drop=True)
        return filtered

    def _select_top_ngrams(self, ranked_tfidf: dict[str, pd.DataFrame]) -> set[str]:
        """Select top n-grams per genre across all orders."""
        top_sets = []
        for name in ["unigrams", "bigrams", "trigrams"]:
            top_ngrams = (
                ranked_tfidf[name].groupby("genre").head(self.top_n)["ngram"].unique()
            )
            top_sets.append(set(top_ngrams))

        final = set.union(*top_sets) if top_sets else set()
        print(f"Total unique n-grams: {len(final):,}")
        return final

    def _extract_ngrams(
        self, texts: pd.Series, n_min: int, n_max: int, name: str
    ) -> tuple[CountVectorizer, csr_matrix, np.ndarray]:
        """Extract n-grams using CountVectorizer."""
        vectorizer = CountVectorizer(
            ngram_range=(n_min, n_max),
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(texts)
        features = vectorizer.get_feature_names_out()

        print(f"✓ Extracted {name}:")
        print(f"  - Unique: {len(features):,}")
        print(f"  - Shape: {matrix.shape}")
        print(f"  - Examples: {self._sample_features(features)}")

        return vectorizer, matrix, features

    def _sample_features(self, features: np.ndarray) -> list[str]:
        """Return reproducible random sample of features."""
        rng = random.Random(self.random_state)
        k = min(5, len(features))
        return rng.sample(list(features), k=k)

    def _calculate_genre_tfidf(
        self,
        corpus: pd.DataFrame,
        ngram_matrix: csr_matrix,
        ngram_features: np.ndarray,
        name: str,
    ) -> pd.DataFrame:
        """Compute genre-level TF-IDF for n-grams."""
        print(f"Calculating genre-level TF-IDF for {name} with genre ...")

        # Convert to binary sparse matrix (no conversion to dense)
        binary_matrix = (ngram_matrix > 0).astype(int)

        # Get genres as numpy array for faster access
        genres_array = corpus["genre"].values
        unique_genres = corpus["genre"].unique()
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
                    ngram_idf[ngram] = (
                        np.log((num_genres + 1) / genres_with_ngram + 1) + 1
                    )

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

    def _count_artists_per_ngram(
        self, corpus: pd.DataFrame, ngram_matrix: csr_matrix, ngram_features: np.ndarray
    ) -> dict[str, int]:
        """Count unique artists per n-gram."""
        print("Counting artists per n-gram...")

        binary_matrix = (ngram_matrix > 0).astype(int)
        binary_matrix = binary_matrix.tocsc()

        artist_series = corpus["artist"].reset_index(drop=True)
        artist_count = {}

        for ngram_idx in tqdm(range(len(ngram_features))):
            track_indices = binary_matrix[:, ngram_idx].nonzero()[0]
            unique_artists = artist_series.iloc[track_indices].nunique()
            ngram = ngram_features[ngram_idx]
            artist_count[ngram] = unique_artists
        print(f"✓ Calculated artist diversity for {len(artist_count):,} n-grams")

        return artist_count

    def _count_final_ngrams(
        self, lyrics: pd.Series, ngram_list: set[str]
    ) -> pd.DataFrame:
        """Count final n-grams in lyrics."""
        vectorizer = CountVectorizer(
            vocabulary=list(ngram_list),
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

    def _get_fitted_vocabulary(self) -> list[str]:
        """Extract vocabulary from all fitted vectorizers."""
        vocab = set()
        for vectorizer in self.vectorizers.values():
            vocab.update(vectorizer.get_feature_names_out())
        return sorted(list(vocab))
