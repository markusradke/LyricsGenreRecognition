import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, vstack

from helpers.BoundaryStripper import BoundaryStripper
from helpers.NGramFeatureExctractor import NGramFeatureExtractor
from helpers.StopwordFilter import StopwordFilter


class NGramFeatureExtractorInformed(NGramFeatureExtractor):
    """Extract and rank n-grams from lyrics by genre."""

    def __init__(
        self,
        min_artists: int = 50,
        top_n: int = 100,
        random_state: int = 42,
    ):
        super().__init__(min_artists, top_n, random_state)
        self.stopword_filter = StopwordFilter()
        self.boundary_stripper = BoundaryStripper()

    def fit(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Extract and select n-gram features from corpus.

        Args:
            corpus: DataFrame with "lyrics", "genre", and "artist" columns.

        Returns:
            DataFrame with n-gram counts per track.
        """
        self._extract_ngrams_all_orders(corpus["lyrics"])
        self._apply_boundary_stripping_all_orders()

        artist_counts_by_order = self._count_artists_all_orders(corpus)
        tfidf_by_order = self._calculate_tfidf_all_orders(corpus)

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
        """Fit vectorizers and extract n-grams for orders 1-3."""
        for order, name in [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]:
            vec, mat, feats = self._extract_ngrams(texts, order, order, name)
            self.vectorizers[name] = vec
            self.matrices[name] = mat
            self.features[name] = feats

    def _apply_boundary_stripping_all_orders(self) -> None:
        """Drop ngrams where boundary stripping changes the ngram."""
        for name in ["bigrams", "trigrams"]:
            original_features = self.features[name]
            original_matrix = self.matrices[name]

            kept_indices = []
            for i, ngram in enumerate(original_features):
                tokens = ngram.split()
                stripped_tokens = self._strip_tokens(tokens)
                if stripped_tokens == tokens:
                    kept_indices.append(i)

            self.features[name] = original_features[kept_indices]
            self.matrices[name] = original_matrix[:, kept_indices]

            dropped = len(original_features) - len(kept_indices)
            print(
                f"Boundary stripping {name}: kept {len(kept_indices):,} / {len(original_features):,} (dropped {dropped:,})"
            )

    def _strip_tokens(self, tokens: list[str]) -> list[str]:
        """Strip boundary words from token list."""
        if tokens and tokens[0] in self.boundary_stripper.boundary_words:
            tokens = tokens[1:]
        if tokens and tokens[-1] in self.boundary_stripper.boundary_words:
            tokens = tokens[:-1]
        return tokens

    def _extract_ngrams(
        self, texts: pd.Series, n_min: int, n_max: int, name: str
    ) -> tuple[CountVectorizer, csr_matrix, np.ndarray]:
        """Extract n-grams within single lines only."""
        vectorizer = CountVectorizer(
            ngram_range=(n_min, n_max),
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
        )

        all_lines, doc_indices = self._split_texts_into_lines(texts)
        line_matrix = vectorizer.fit_transform(all_lines)

        matrix = self._aggregate_lines_to_docs(line_matrix, doc_indices, len(texts))
        features = vectorizer.get_feature_names_out()

        filtered_indices = np.array(
            [
                i
                for i, f in enumerate(features)
                if not self.stopword_filter.is_stopword_only(f)
            ]
        )
        matrix = matrix[:, filtered_indices]
        features = features[filtered_indices]

        print(f"✓ Extracted {name}:")
        print(f"  - Unique: {len(features):,}")
        print(f"  - Shape: {matrix.shape}")
        print(f"  - Examples: {self._sample_features(features)}")

        return vectorizer, matrix, features

    def _split_texts_into_lines(self, texts: pd.Series) -> tuple[list[str], list[int]]:
        """Split texts into lines and track document indices."""
        all_lines = []
        doc_indices = []
        for doc_idx, text in enumerate(texts):
            lines = text.split("\n")
            all_lines.extend(lines)
            doc_indices.extend([doc_idx] * len(lines))
        return all_lines, doc_indices

    def _aggregate_lines_to_docs(
        self,
        line_matrix: csr_matrix,
        doc_indices: list[int],
        n_docs: int,
    ) -> csr_matrix:
        """Aggregate line-level counts back to document level."""
        doc_matrices = []
        for doc_idx in range(n_docs):
            line_mask = [i for i, idx in enumerate(doc_indices) if idx == doc_idx]
            if line_mask:
                doc_matrix = line_matrix[line_mask].sum(axis=0)
                doc_matrices.append(csr_matrix(doc_matrix))
        return vstack(doc_matrices)

    def _filter_stopword_features(self, features: np.ndarray) -> np.ndarray:
        """Remove stopword-only n-grams from features."""
        filtered = [f for f in features if not self.stopword_filter.is_stopword_only(f)]
        return np.array(filtered)

    def _count_final_ngrams(
        self,
        texts: pd.Series,
        selected_ngrams: list[str],
    ) -> pd.DataFrame:
        """Count selected n-grams within single lines."""
        vectorizer = CountVectorizer(
            vocabulary=selected_ngrams,
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
        )

        all_lines, doc_indices = self._split_texts_into_lines(texts)
        line_matrix = vectorizer.fit_transform(all_lines)
        matrix = self._aggregate_lines_to_docs(line_matrix, doc_indices, len(texts))

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
