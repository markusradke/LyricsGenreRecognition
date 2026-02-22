"""
Fell & Sporleder (2014) n-gram feature extraction as sklearn transformer.

Implements the baseline method from:
    Fell, M., & Sporleder, C. (2014). Lyrics-based analysis and
    classification of music. COLING 2014.
"""

import pandas as pd
import numpy as np
import random
import pickle
from pathlib import Path
from joblib import hash as joblib_hash
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

from .config import MIN_ARTISTS, ENABLE_STOPWORD_FILTER


class FSExtractor(BaseEstimator, TransformerMixin):
    """
    Fell & Sporleder n-gram extractor with TF-IDF ranking.

    Extracts unigrams, bigrams, and trigrams, filters by minimum artist
    threshold, computes genre-level TF-IDF, and selects top-100 n-grams
    per genre per n-gram order.

    Parameters
    ----------
    min_artists : int, default=MIN_ARTISTS (from config)
        Minimum number of unique artists that must use an n-gram for
        it to be included in the vocabulary.
    top_vocab_per_genre : int, default=100
        Number of top-ranked n-grams to select per genre per n-gram
        order (unigrams, bigrams, trigrams).
    use_stopword_filter : bool, default=ENABLE_STOPWORD_FILTER
        Whether to filter stopword-only n-grams.
    random_state : int, default=42
        Random seed for reproducibility.
    checkpoint_dir : str or Path, optional
        Directory to store checkpoints of vocabulary and vectorizer.
        If None, no checkpointing is used.

    Attributes
    ----------
    vocabulary_ : list of str
        Final selected n-gram vocabulary (union of top n-grams).
    vectorizer_ : CountVectorizer
        Fitted vectorizer for transforming new data.
    _cache_key : str or None
        Joblib hash of input data for checkpointing (computed during fit).
    _is_fitted : bool
        Whether the extractor has been fitted.
    """

    def __init__(
        self,
        min_artists: int = MIN_ARTISTS,
        top_vocab_per_genre: int = 100,
        use_stopword_filter: bool = ENABLE_STOPWORD_FILTER,
        random_state: int = 42,
        checkpoint_dir: str = None,
    ):
        self.min_artists = min_artists
        self.top_vocab_per_genre = top_vocab_per_genre
        self.use_stopword_filter = use_stopword_filter
        self.random_state = random_state
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._is_fitted = False

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, X, y, artist=None):
        """
        Learn vocabulary from training data.

        Parameters
        ----------
        X : pd.Series
            Lyrics text, one entry per track.
        y : pd.Series or np.ndarray
            Genre labels for each track.
        artist : pd.Series, optional
            Artist names for each track. Required for min_artists
            filtering. If None, raises ValueError.

        Returns
        -------
        self : FSExtractor
            Fitted extractor.

        Raises
        ------
        ValueError
            If artist parameter is None.
        """
        if artist is None:
            raise ValueError(
                "FSExtractor requires 'artist' metadata for min_artists "
                "filtering. Pass as: fit(X, y, artist=artist_series)"
            )

        # Convert to pandas for consistent indexing
        X = pd.Series(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        artist = pd.Series(artist).reset_index(drop=True)

        self._cache_key = self._compute_cache_key(X, y, artist)

        if self.checkpoint_dir:
            if self._load_checkpoint():
                print(
                    f"Loaded vocabulary and vectorizer from checkpoint: {self._cache_key[:8]}..."
                )
                self._is_fitted = True
                return self
            print(
                f"No checkpoint found for hash {self._cache_key[:8]}..., computing vocabulary..."
            )
        else:
            print("Checkpoint directory not specified, computing vocabulary...")

        orders_to_extract = [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]
        order_names = [name for _, name in orders_to_extract]

        print("Extracting n-grams for all orders...")
        matrices = {}
        features = {}
        for order, name in orders_to_extract:
            mat, feats = self._extract_ngrams(X, order, name)
            matrices[name] = mat
            features[name] = feats

        print("Calculating genre-level TF-IDF for n-grams...")
        tfidf_by_order = {
            name: self._calculate_genre_tfidf(y, matrices[name], features[name])
            for name in order_names
        }

        print("Counting unique artists per n-gram...")
        artist_counts_by_order = {
            name: self._count_artists_per_ngram(artist, matrices[name], features[name])
            for name in order_names
        }

        print("Filtering n-grams by minimum artist threshold and ranking by TF-IDF...")
        filtered_tfidf = {}
        for name in order_names:
            tfidf = tfidf_by_order[name]
            filtered = tfidf[
                tfidf["ngram"].map(artist_counts_by_order[name]) >= self.min_artists
            ].copy()
            filtered_tfidf[name] = filtered.sort_values(
                by=["genre", "tfidf"], ascending=[True, False]
            ).reset_index(drop=True)

        print("Selecting top n-grams per genre and final vocabulary...")
        self.vocabulary_ = self._select_top_ngrams(filtered_tfidf)
        print(f"Final vocabulary size: {len(self.vocabulary_):,} n-grams")

        print("Fitting CountVectorizer with selected vocabulary...")
        self.vectorizer_ = CountVectorizer(
            vocabulary=list(self.vocabulary_),
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
            ngram_range=(1, 3),
        )
        self.vectorizer_.fit(X)

        self._is_fitted = True

        if self.checkpoint_dir:
            self._save_checkpoint()
            print(f"Saved checkpoint: {self._cache_key[:8]}...")

        return self

    def transform(self, X):
        """
        Transform lyrics to n-gram count matrix.

        Parameters
        ----------
        X : pd.Series or array-like
            Lyrics text to transform.

        Returns
        -------
        X_transformed : scipy.sparse.csr_matrix
            Sparse matrix of n-gram counts, shape (n_samples, n_features).

        Raises
        ------
        ValueError
            If transform called before fit.
        """
        if not hasattr(self, "vocabulary_") or not self._is_fitted:
            raise ValueError("Must call fit() before transform()")

        X = pd.Series(X).reset_index(drop=True)
        return self.vectorizer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for output features.

        Parameters
        ----------
        input_features : array-like, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        feature_names : np.ndarray of str
            N-gram feature names.
        """
        if not hasattr(self, "vectorizer_") or not self._is_fitted:
            raise ValueError("Must call fit() before get_feature_names_out()")

        return self.vectorizer_.get_feature_names_out()

    def _compute_cache_key(self, X, y, artist):
        """Compute hash for caching vocabulary."""
        data_tuple = (
            tuple(X.index),
            tuple(X.values),
            tuple(y.values),
            tuple(artist.values),
            self.min_artists,
            self.top_vocab_per_genre,
            self.use_stopword_filter,
            self.random_state,
        )
        return joblib_hash(data_tuple)

    def _extract_ngrams(self, texts, order, name):
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

        return matrix, features

    def _calculate_genre_tfidf(self, genres, ngram_matrix, ngram_features):
        """Compute genre-level TF-IDF for n-grams."""
        binary_matrix = (ngram_matrix > 0).astype(int).tocoo()
        genres_array = genres.values
        unique_genres = genres.unique()
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
        print(f"Calculated TF-IDF for {len(results):,} genre-ngram pairs")

        return pd.DataFrame(results)

    def _count_artists_per_ngram(self, artists, ngram_matrix, ngram_features):
        """Count unique artists per n-gram."""
        binary_matrix = (ngram_matrix > 0).astype(int).tocsc()
        artist_series = artists.reset_index(drop=True)
        artist_count = {}

        for ngram_idx, ngram in enumerate(ngram_features):
            track_indices = binary_matrix[:, ngram_idx].nonzero()[0]
            artist_count[ngram] = artist_series.iloc[track_indices].nunique()
        print(f"Counted unique artists for {len(artist_count):,} n-grams")

        return artist_count

    def _select_top_ngrams(self, ranked_tfidf):
        """Select top n-grams per genre across all orders."""
        top_sets = [
            set(
                ranked_tfidf[name]
                .groupby("genre")
                .head(self.top_vocab_per_genre)["ngram"]
                .unique()
            )
            for name in ["unigrams", "bigrams", "trigrams"]
        ]

        final = set.union(*top_sets) if top_sets else set()
        print(f"Total unique n-grams (FS): {len(final):,}")
        return final

    def _get_checkpoint_paths(self):
        """Get paths for vocabulary and vectorizer checkpoint files."""
        vocab_path = self.checkpoint_dir / f"vocab_{self._cache_key}.pkl"
        vectorizer_path = self.checkpoint_dir / f"vectorizer_{self._cache_key}.pkl"
        return vocab_path, vectorizer_path

    def _save_checkpoint(self):
        """Save vocabulary and vectorizer to checkpoint files."""
        vocab_path, vectorizer_path = self._get_checkpoint_paths()

        with open(vocab_path, "wb") as f:
            pickle.dump(self.vocabulary_, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer_, f)

    def _load_checkpoint(self):
        """Load vocabulary and vectorizer from checkpoint files if they exist."""
        vocab_path, vectorizer_path = self._get_checkpoint_paths()

        if vocab_path.exists() and vectorizer_path.exists():
            with open(vocab_path, "rb") as f:
                self.vocabulary_ = pickle.load(f)

            with open(vectorizer_path, "rb") as f:
                self.vectorizer_ = pickle.load(f)

            return True

        return False
