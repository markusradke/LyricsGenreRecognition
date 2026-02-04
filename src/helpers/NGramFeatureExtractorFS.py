import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

from helpers.NGramFeatureExctractor import NGramFeatureExtractor


class NGramFeatureExtractorFS(NGramFeatureExtractor):
    """Extract and rank n-grams from lyrics by genre according to Fell & Spohrleder (2014)."""

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

        return self._count_final_ngrams(lyrics, self.final_ngrams)
