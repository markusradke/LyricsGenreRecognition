"""Monroe "Fightin' Words" n-gram extractor as sklearn transformer."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

from .config import (
    MIN_ARTISTS,
    MONROE_Z_THRESHOLD,
    EXTRACT_WITHIN_LINES,
    ENABLE_STOPWORD_FILTER,
    INCLUDE_UNIGRAMS,
)
from .StopwordFilter import StopwordFilter
from . import monroe_logodds
from . import ngram_utils


class MonroeExtractor(BaseEstimator, TransformerMixin):
    """
    Monroe log-odds n-gram extractor with Dirichlet smoothing.

    Extracts genre-discriminating n-grams using Bayesian-smoothed log-odds
    ratios. Addresses sparsity issues via Dirichlet priors.

    Parameters
    ----------
    min_artists : int, default=MIN_ARTISTS
        Minimum artists threshold for n-gram inclusion.
    alpha_unigram : float, default=1.0
        Dirichlet prior for unigrams.
    alpha_bigram : float, default=1.0
        Dirichlet prior for bigrams.
    alpha_trigram : float, default=1.0
        Dirichlet prior for trigrams.
    alpha_quadgram : float, default=1.0
        Dirichlet prior for quadgrams.
    z_threshold : float, default=2.326
        Z-score threshold for feature selection (one-sided, alpha=0.01).
    extract_within_lines : bool, default=True
        Extract n-grams within line boundaries only.
    apply_fdr : bool, default=True
        Apply Benjamini-Hochberg FDR correction for multiple testing.
    fdr_level : float, default=0.01
        False discovery rate level.
    use_stopword_filter : bool, default=ENABLE_STOPWORD_FILTER
        Whether to filter stopword-only n-grams.
    include_unigrams : bool, default=INCLUDE_UNIGRAMS
        Whether to include unigrams in vocabulary. Set False for phrase-only.
    random_state : int, default=42
        Random seed.
    """

    def __init__(
        self,
        min_artists: int = MIN_ARTISTS,
        alpha_unigram: float = 1.0,
        alpha_bigram: float = 1.0,
        alpha_trigram: float = 1.0,
        alpha_quadgram: float = 1.0,
        z_threshold: float = MONROE_Z_THRESHOLD,
        extract_within_lines: bool = EXTRACT_WITHIN_LINES,
        apply_fdr: bool = True,
        fdr_level: float = 0.01,
        use_stopword_filter: bool = ENABLE_STOPWORD_FILTER,
        include_unigrams: bool = INCLUDE_UNIGRAMS,
        random_state: int = 42,
    ):
        self.min_artists = min_artists
        self.alpha_unigram = alpha_unigram
        self.alpha_bigram = alpha_bigram
        self.alpha_trigram = alpha_trigram
        self.alpha_quadgram = alpha_quadgram
        self.z_threshold = z_threshold
        self.extract_within_lines = extract_within_lines
        self.apply_fdr = apply_fdr
        self.fdr_level = fdr_level
        self.use_stopword_filter = use_stopword_filter
        self.include_unigrams = include_unigrams
        self.random_state = random_state
        self._is_fitted = False
        self._cache_dir = Path("data/cache/monroe")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_cache_key = None

    def fit(self, X, y, artist=None):
        """Learn vocabulary from training data."""
        if artist is None:
            raise ValueError("MonroeExtractor requires 'artist' metadata")

        X = pd.Series(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        artist = pd.Series(artist).reset_index(drop=True)

        # Initialize stopword filter conditionally
        self.stopword_filter_ = StopwordFilter() if self.use_stopword_filter else None
        self.genres_ = sorted(y.unique())

        # Check for cached counts (cross-trial optimization)
        cache_key = self._compute_cache_key(X, y, artist)
        if cache_key == self._last_cache_key:
            print("Using cached counts from previous fit (same data/config)")
        else:
            cache_file = self._cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                print(f"Loading cached counts from disk (key: {cache_key[:8]}...)")
                cached_data = joblib.load(cache_file)
                self._cached_tokens = cached_data["tokens"]
                self._cached_ngrams_by_order = cached_data["ngrams"]
                self._cached_counts_by_order = cached_data["counts"]
                self._last_cache_key = cache_key
            else:
                print(
                    "Computing and caching filtered n-grams (first time for this config)..."
                )
                # Cache tokenization and n-gram extraction
                self._tokenize_corpus(X)
                self._extract_ngrams_once()
                # Cache filtered n-grams and counts
                self._cached_counts_by_order = {}
                self._cache_filtered_ngrams_and_counts(X, y, artist)
                # Save to disk
                joblib.dump(
                    {
                        "tokens": self._cached_tokens,
                        "ngrams": self._cached_ngrams_by_order,
                        "counts": self._cached_counts_by_order,
                    },
                    cache_file,
                )
                print(f"Cached to disk (key: {cache_key[:8]}...)")
                self._last_cache_key = cache_key

        self.vocabulary_ = self._extract_vocabulary(X, y, artist)

        # Create vectorizer
        vocab_strings = ["_".join(ng) for ng in self.vocabulary_]
        self.vectorizer_ = CountVectorizer(
            vocabulary=vocab_strings,
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
            ngram_range=(1, 4),
        )

        # Fit to learn mappings (required by sklearn)
        self._fit_vectorizer_with_replaced_ngrams(X)

        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform lyrics to n-gram count matrix."""
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")

        X = pd.Series(X).reset_index(drop=True)

        # Replace n-grams in each document
        replaced = X.apply(lambda text: self._replace_ngrams(text))

        return self.vectorizer_.transform(replaced)

    def get_feature_names_out(self, input_features=None):
        """Get feature names."""
        if not self._is_fitted:
            raise ValueError("Must call fit() before get_feature_names_out()")
        return self.vectorizer_.get_feature_names_out()

    def update_alpha_and_recompute_vocabulary(self, X, **new_alphas):
        """Quickly recompute vocabulary with new alpha values.

        Uses cached filtered n-grams and count statistics, only recomputes
        Monroe scores and thresholding. Useful for fast alpha exploration.

        Parameters
        ----------
        X : pd.Series
            Training corpus (must be same as original fit).
        **new_alphas : float
            New alpha values (e.g., alpha_unigram=0.5, alpha_bigram=2.0).
            Unspecified alphas retain their current values.

        Returns
        -------
        self : MonroeExtractor
            Fitted extractor with updated vocabulary.

        Example
        -------
        >>> extractor.fit(X_train, y_train, artist_train)
        >>> extractor.update_alpha_and_recompute_vocabulary(
        ...     X_train, alpha_unigram=0.1, alpha_bigram=5.0
        ... )
        """
        if not hasattr(self, "_cached_counts_by_order"):
            raise ValueError(
                "Must call fit() before update_alpha_and_recompute_vocabulary(). "
                "Cached counts not available."
            )

        # Update alpha values
        for key, value in new_alphas.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid alpha parameter: {key}")

        print(f"Recomputing vocabulary with updated alphas: {new_alphas}")

        # Recompute vocabulary using cached counts (fast!)
        all_ngrams = []
        for order in [1, 2, 3, 4]:
            alpha = getattr(
                self,
                f"alpha_{'unigram' if order == 1 else 'bigram' if order == 2 else 'trigram' if order == 3 else 'quadgram'}",
            )
            ngrams = self._score_cached_ngrams(order, alpha)
            all_ngrams.extend(ngrams)
            print(f"  {order}-grams: {len(ngrams)} discriminating")

        print(f"Total vocabulary (Monroe): {len(all_ngrams)}")
        self.vocabulary_ = all_ngrams

        # Rebuild vectorizer with new vocabulary
        vocab_strings = ["_".join(ng) for ng in self.vocabulary_]
        self.vectorizer_ = CountVectorizer(
            vocabulary=vocab_strings,
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
            ngram_range=(1, 4),
        )

        X_series = pd.Series(X).reset_index(drop=True)
        self._fit_vectorizer_with_replaced_ngrams(X_series)

        return self

    def _score_cached_ngrams(self, order, alpha):
        """Score cached n-grams with given alpha (fast path)."""
        cache = self._cached_counts_by_order.get(order)
        if cache is None:
            return []

        filtered_ngrams = cache["filtered_ngrams"]
        y_gc = cache["y_gc"]
        n_c = cache["n_c"]
        y_g = cache["y_g"]
        n = cache["n"]
        m = cache["m"]

        # Compute Monroe scores (only step that depends on alpha)
        alpha_array = np.full(len(filtered_ngrams), alpha)
        delta = monroe_logodds.compute_log_odds_delta(y_gc, n_c, y_g, n, m, alpha_array)
        variance = monroe_logodds.compute_variance(y_gc, n_c, y_g, n, m, alpha_array)
        z_scores = monroe_logodds.compute_z_scores(delta, variance)

        # Filter by threshold
        discriminating_df = monroe_logodds.filter_discriminating_ngrams(
            z_scores,
            ["_".join(ng) for ng in filtered_ngrams],
            self.genres_,
            threshold=self.z_threshold,
            apply_fdr=self.apply_fdr,
            fdr_level=self.fdr_level,
        )

        # Return unique n-grams
        unique_ngrams = discriminating_df["ngram"].unique()
        return [tuple(ng.split("_")) for ng in unique_ngrams]

    def _extract_vocabulary(self, X, y, artist):
        """Extract discriminating n-grams via Monroe method."""
        all_ngrams = []

        # Determine orders to process based on include_unigrams flag
        orders = [2, 3, 4] if not self.include_unigrams else [1, 2, 3, 4]

        for order in orders:
            alpha = getattr(
                self,
                f"alpha_{'unigram' if order == 1 else 'bigram' if order == 2 else 'trigram' if order == 3 else 'quadgram'}",
            )
            ngrams = self._extract_and_score_order(X, y, artist, order, alpha)
            all_ngrams.extend(ngrams)
            print(f"  {order}-grams: {len(ngrams)} discriminating")

        print(f"Total vocabulary (Monroe): {len(all_ngrams)}")
        return all_ngrams

    def _compute_cache_key(self, X, y, artist):
        """Compute cache key for persistent storage.

        Hash includes data content and configuration parameters that affect
        filtered n-grams and count statistics, but EXCLUDES alpha values
        (which can be quickly recomputed using cached counts).
        """
        cache_tuple = (
            tuple(X.index),
            tuple(X.values),
            tuple(y.values),
            tuple(artist.values),
            self.min_artists,
            self.extract_within_lines,
            self.z_threshold,
            self.apply_fdr,
            self.fdr_level,
            self.use_stopword_filter,
            self.include_unigrams,
            self.random_state,
        )
        return joblib.hash(cache_tuple)

    def _tokenize_corpus(self, X):
        """Tokenize corpus once and cache results."""
        self._cached_tokens = [
            ngram_utils.tokenize(text, self.extract_within_lines) for text in X
        ]

    def _extract_ngrams_once(self):
        """Extract n-grams for all orders in single pass."""
        self._cached_ngrams_by_order = {1: [], 2: [], 3: [], 4: []}

        for tokens in self._cached_tokens:
            ngrams_dict = ngram_utils.extract_ngrams_by_order(
                tokens, [1, 2, 3, 4], self.extract_within_lines
            )
            for order in [1, 2, 3, 4]:
                self._cached_ngrams_by_order[order].extend(ngrams_dict[order])

    def _cache_filtered_ngrams_and_counts(self, X, y, artist):
        """Cache filtered n-grams and count statistics per order.

        This separates alpha-independent steps (filtering, counting) from
        alpha-dependent steps (scoring). Enables fast alpha tuning.
        """
        for order in [1, 2, 3, 4]:
            # Extract and filter n-grams (alpha-independent)
            all_ngrams_set = set(self._cached_ngrams_by_order[order])
            all_ngrams_list = list(all_ngrams_set)

            if order > 1:
                all_ngrams_list = ngram_utils.strip_boundary_ngrams(all_ngrams_list)

            # Apply stopword filter conditionally
            if self.use_stopword_filter:
                all_ngrams_list = ngram_utils.filter_stopword_only(
                    all_ngrams_list, self.stopword_filter_
                )

            # Filter by min_artists
            artist_counts = ngram_utils.count_artists_per_ngram(
                set(all_ngrams_list),
                X,
                artist,
                self.extract_within_lines,
                tokens_cache=self._cached_tokens,
            )
            filtered_ngrams = [
                ng
                for ng in all_ngrams_list
                if artist_counts.get(ng, 0) >= self.min_artists
            ]

            if len(filtered_ngrams) == 0:
                self._cached_counts_by_order[order] = None
                continue

            # Compute counts (alpha-independent)
            y_gc, n_c, y_g, n, m = self._compute_counts(X, y, filtered_ngrams, order)

            # Store everything needed for scoring
            self._cached_counts_by_order[order] = {
                "filtered_ngrams": filtered_ngrams,
                "y_gc": y_gc,
                "n_c": n_c,
                "y_g": y_g,
                "n": n,
                "m": m,
            }
            print(f"  Cached {len(filtered_ngrams)} {order}-grams")

    def _extract_and_score_order(self, X, y, artist, order, alpha):
        """Extract and score n-grams of given order."""
        # Use cached counts (already filtered and computed)
        cache = self._cached_counts_by_order.get(order)
        if cache is None:
            return []

        filtered_ngrams = cache["filtered_ngrams"]
        y_gc = cache["y_gc"]
        n_c = cache["n_c"]
        y_g = cache["y_g"]
        n = cache["n"]
        m = cache["m"]

        # Compute Monroe scores
        alpha_array = np.full(len(filtered_ngrams), alpha)
        delta = monroe_logodds.compute_log_odds_delta(y_gc, n_c, y_g, n, m, alpha_array)
        variance = monroe_logodds.compute_variance(y_gc, n_c, y_g, n, m, alpha_array)
        z_scores = monroe_logodds.compute_z_scores(delta, variance)

        # Filter by threshold
        discriminating_df = monroe_logodds.filter_discriminating_ngrams(
            z_scores,
            ["_".join(ng) for ng in filtered_ngrams],
            self.genres_,
            threshold=self.z_threshold,
            apply_fdr=self.apply_fdr,
            fdr_level=self.fdr_level,
        )

        # Return unique n-grams
        unique_ngrams = discriminating_df["ngram"].unique()
        return [tuple(ng.split("_")) for ng in unique_ngrams]

    def _compute_counts(self, X, y, ngrams, order):
        """Compute count matrices for Monroe formula."""
        # Build genre-ngram count matrix
        ngram_to_idx = {ng: i for i, ng in enumerate(ngrams)}
        genre_to_idx = {g: i for i, g in enumerate(self.genres_)}

        y_gc = np.zeros((len(ngrams), len(self.genres_)))

        # Use cached tokens instead of re-tokenizing
        for tokens, genre in zip(self._cached_tokens, y):
            doc_ngrams = ngram_utils.extract_ngrams_by_order(
                tokens, [order], self.extract_within_lines
            )[order]

            for ng in doc_ngrams:
                if ng in ngram_to_idx:
                    y_gc[ngram_to_idx[ng], genre_to_idx[genre]] += 1

        # Compute other quantities
        n_c = y_gc.sum(axis=0)  # Total tokens per genre
        y_g = y_gc.sum(axis=1)  # Total count per n-gram
        n = int(y_gc.sum())  # Total tokens
        m = len(ngrams)  # Vocabulary size

        return y_gc, n_c, y_g, n, m

    def _fit_vectorizer_with_replaced_ngrams(self, X):
        """Fit vectorizer on corpus with n-grams replaced."""
        replaced = X.apply(lambda text: self._replace_ngrams(text))
        self.vectorizer_.fit(replaced)

    def _replace_ngrams(self, text):
        """Replace n-grams with underscore-joined tokens."""
        tokens = ngram_utils.tokenize(text, preserve_lines=False)

        # Greedy longest match: 4-grams → 3-grams → 2-grams
        for order in [4, 3, 2]:
            ngrams_to_replace = [ng for ng in self.vocabulary_ if len(ng) == order]
            if not ngrams_to_replace:
                continue

            i = 0
            new_tokens = []
            while i < len(tokens):
                matched = False
                # Try to match n-gram at position i
                for ng in ngrams_to_replace:
                    if i + order <= len(tokens):
                        candidate = tuple(tokens[i : i + order])
                        if candidate == ng:
                            new_tokens.append("_".join(ng))
                            i += order
                            matched = True
                            break
                if not matched:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return " ".join(tokens)
