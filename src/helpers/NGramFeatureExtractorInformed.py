import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, vstack

from helpers.NGramFeatureExctractor import NGramFeatureExtractor


class NGramFeatureExtractorInformed(NGramFeatureExtractor):
    """Extract and rank n-grams from lyrics by genre according to a more informed method."""

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

        selected_ngrams = self._select_top_ngrams(filtered_tfidf)
        self.final_ngrams = (
            sorted(selected_ngrams)
            if isinstance(selected_ngrams, set)
            else selected_ngrams
        )
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
        """Extract n-grams using CountVectorizer, only within single lines."""
        vectorizer = CountVectorizer(
            ngram_range=(n_min, n_max),
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
        )

        all_lines = []
        doc_indices = []
        for doc_idx, text in enumerate(texts):
            lines = text.split("\n")
            all_lines.extend(lines)
            doc_indices.extend([doc_idx] * len(lines))

        line_matrix = vectorizer.fit_transform(all_lines)

        doc_matrices = []
        for doc_idx in range(len(texts)):
            line_mask = [i for i, idx in enumerate(doc_indices) if idx == doc_idx]
            if line_mask:
                doc_matrix = line_matrix[line_mask].sum(axis=0)
                doc_matrices.append(csr_matrix(doc_matrix))

        matrix = vstack(doc_matrices)
        features = vectorizer.get_feature_names_out()

        print(f"✓ Extracted {name}:")
        print(f"  - Unique: {len(features):,}")
        print(f"  - Shape: {matrix.shape}")
        print(f"  - Examples: {self._sample_features(features)}")

        return vectorizer, matrix, features

    def _count_final_ngrams(
        self, texts: pd.Series, selected_ngrams: list[str]
    ) -> pd.DataFrame:
        """Count selected n-grams in texts, only within single lines."""
        vectorizer = CountVectorizer(
            vocabulary=selected_ngrams,
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
        )

        all_lines = []
        doc_indices = []
        for doc_idx, text in enumerate(texts):
            lines = text.split("\n")
            all_lines.extend(lines)
            doc_indices.extend([doc_idx] * len(lines))

        line_matrix = vectorizer.fit_transform(all_lines)

        doc_matrices = []
        for doc_idx in range(len(texts)):
            line_mask = [i for i, idx in enumerate(doc_indices) if idx == doc_idx]
            if line_mask:
                doc_matrix = line_matrix[line_mask].sum(axis=0)
                doc_matrices.append(csr_matrix(doc_matrix))

        matrix = vstack(doc_matrices)

        return pd.DataFrame(matrix.toarray(), columns=selected_ngrams)

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


if __name__ == "__main__":
    from helpers.NGramFeatureExtractorFS import NGramFeatureExtractorFS

    data = pd.DataFrame(
        {
            "lyrics": ["Hello world\nThis is a test", "Another song\nWith some lyrics"],
            "genre": ["Pop", "Rock"],
            "artist": ["Artist1", "Artist2"],
        }
    )
    extractor_FS = NGramFeatureExtractorInformed(
        min_artists=1, top_n=200, random_state=42
    )
    features_FS = extractor_FS.fit(data)
    extractor_informed = NGramFeatureExtractorFS(
        min_artists=1, top_n=200, random_state=42
    )
    features_informed = extractor_informed.fit(data)
    print("=" * 60)
    print(f"{len(features_FS.columns)} features extracted using Informed method.")
    print(features_FS.columns.to_list())
    print("=" * 60)
    print(f"{len(features_informed.columns)} features extracted using FS method:")
    print(features_informed.columns.tolist())
