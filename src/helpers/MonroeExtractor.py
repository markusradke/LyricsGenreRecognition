"""
Monroe et al. (2008) n-gram feature extraction with fighting words method.

Implements discriminating n-gram selection using Bayesian-smoothed log-odds
ratios with empirical Bayes priors estimated from the full corpus.

Reference:
    Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008).
    Fightin' Words: Lexical Feature Selection and Evaluation for
    Identifying the Content of Political Conflict.
    Political Analysis, 16(4), 372-403.
"""

import pandas as pd
import numpy as np
import random
import pickle

from pathlib import Path
from joblib import hash as joblib_hash
from scipy.stats import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from .config import MIN_ARTISTS, ENABLE_STOPWORD_FILTER
from .monroe_logodds import compute_monroe_statistics, apply_fdr_correction


class MonroeExtractor(BaseEstimator, TransformerMixin):
    """
    Monroe et al. n-gram extractor with fighting words z-scores.

    Extracts unigrams, bigrams, and trigrams using Dirichlet-smoothed log-odds
    ratios. Uses empirical Bayes prior estimated from full corpus frequencies.
    Checkpoints z-scores for all n-grams to allow p-value threshold exploration.

    Parameters
    ----------
    min_artists : int, default=MIN_ARTISTS
        Minimum number of unique artists that must use an n-gram for inclusion.
    p_value : float, default=0.01
        Significance level for one-sided z-test (FDR-corrected).
    prior_strength : float, default=0.01
        Dirichlet prior strength (alpha). Lower values = stronger smoothing.
    use_stopword_filter : bool, default=ENABLE_STOPWORD_FILTER
        Whether to filter stopword-only n-grams.
    random_state : int, default=42
        Random seed for reproducibility.
    checkpoint_dir : str or Path, optional
        Directory to store checkpoints. If None, no checkpointing.

    Attributes
    ----------
    vocabulary_ : list of str
        Selected n-gram vocabulary passing significance threshold.
    vectorizer_ : CountVectorizer
        Fitted vectorizer for transforming new data.
    z_scores_df_ : pd.DataFrame
        All computed z-scores with columns ['ngram', 'genre', 'z_score',
        'z_fdr_corrected', 'passes_fdr'] - checkpointed for p-value exploration.
    _cache_key : str or None
        Joblib hash of input data for checkpointing (computed during fit).
    _is_fitted : bool
        Whether the extractor has been fitted.
    """

    def __init__(
        self,
        min_artists: int = MIN_ARTISTS,
        p_value: float = 0.01,
        prior_strength: float = 0.01,
        use_stopword_filter: bool = ENABLE_STOPWORD_FILTER,
        random_state: int = 42,
        checkpoint_dir: str = None,
    ):
        self.min_artists = min_artists
        self.p_value = p_value
        self.prior_strength = prior_strength
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
        X : pd.Series or array-like
            Lyrics text, one entry per track.
        y : pd.Series or np.ndarray
            Genre labels for each track.
        artist : pd.Series, optional
            Artist names for each track. Required for min_artists filtering.

        Returns
        -------
        self : MonroeExtractor
            Fitted extractor.

        Raises
        ------
        ValueError
            If artist parameter is None.
        """
        if artist is None:
            raise ValueError(
                "MonroeExtractor requires 'artist' metadata for min_artists filtering"
            )

        X = pd.Series(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        artist = pd.Series(artist).reset_index(drop=True)

        self._cache_key = self._compute_cache_key(X, y, artist)

        if self.checkpoint_dir:
            if self._load_checkpoint():
                print(f"Loaded z-scores from checkpoint: {self._cache_key[:8]}...")
                self._is_fitted = True
                return self._select_vocabulary_by_pvalue(self.p_value)
            print("No checkpoint found, computing z-scores from scratch...")
        else:
            print("Checkpoint directory not specified, computing z-scores...")

        orders_to_extract = [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]
        order_names = [name for _, name in orders_to_extract]

        print("Extracting n-grams for all orders...")
        matrices = {}
        features = {}
        for order, name in orders_to_extract:
            mat, feats = self._extract_ngrams(X, order, name)
            matrices[name] = mat
            features[name] = feats

        print("Counting unique artists per n-gram...")
        artist_counts_by_order = {
            name: self._count_artists_per_ngram(artist, matrices[name], features[name])
            for name in order_names
        }

        print("Filtering n-grams by minimum artist threshold...")
        filtered_features = {}
        filtered_matrices = {}
        for name in order_names:
            mask = np.array(
                [
                    artist_counts_by_order[name][ng] >= self.min_artists
                    for ng in features[name]
                ]
            )
            filtered_features[name] = features[name][mask]
            filtered_matrices[name] = matrices[name][:, mask]
            print(f"  {name}: {len(filtered_features[name]):,} n-grams retained")

        print("Computing Monroe z-scores with empirical Bayes prior...")
        self.z_scores_df_ = self._compute_all_zscores(
            y, filtered_matrices, filtered_features
        )

        print(f"Total n-grams scored: {len(self.z_scores_df_['ngram'].unique()):,}")

        if self.checkpoint_dir:
            self._save_checkpoint()
            print(f"Saved checkpoint: {self._cache_key[:8]}...")

        return self._select_vocabulary_by_pvalue(self.p_value)

    def transform(self, X):
        """Transform lyrics to n-gram count matrix."""
        if not hasattr(self, "vocabulary_") or not self._is_fitted:
            raise ValueError("Must call fit() before transform()")

        X = pd.Series(X).reset_index(drop=True)
        return self.vectorizer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get feature names for output features."""
        if not hasattr(self, "vectorizer_") or not self._is_fitted:
            raise ValueError("Must call fit() before get_feature_names_out()")

        return self.vectorizer_.get_feature_names_out()

    def update_pvalue_threshold(self, new_p_value):
        """Reselect vocabulary with new p-value without recomputing z-scores.

        Uses checkpointed z-scores to quickly explore different significance
        thresholds without expensive recomputation.

        Parameters
        ----------
        new_p_value : float
            New significance level (e.g., 0.05, 0.01, 0.001).

        Returns
        -------
        self : MonroeExtractor
            Extractor with updated vocabulary.
        """
        if not hasattr(self, "z_scores_df_"):
            raise ValueError(
                "Must call fit() before update_pvalue_threshold(). "
                "Z-scores not computed."
            )

        print(f"Updating vocabulary with p-value threshold: {new_p_value}")
        self.p_value = new_p_value
        return self._select_vocabulary_by_pvalue(new_p_value)

    def _select_vocabulary_by_pvalue(self, p_value):
        """Select vocabulary based on p-value threshold."""
        z_threshold = norm.ppf(1 - p_value)

        significant = self.z_scores_df_[
            self.z_scores_df_["passes_fdr"]
            & (self.z_scores_df_["z_score"] > z_threshold)
        ]

        self.vocabulary_ = list(significant["ngram"].unique())
        print(
            f"Selected vocabulary size: {len(self.vocabulary_):,} n-grams (p={p_value})"
        )

        self.vectorizer_ = CountVectorizer(
            vocabulary=self.vocabulary_,
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
            ngram_range=(1, 3),
        )

        self._is_fitted = True
        return self

    def _compute_all_zscores(self, genres, matrices, features):
        """Compute Monroe z-scores for all n-grams across all genres.

        Implements multi-class extension of Monroe et al.'s method:
        - Empirical Bayes prior from corpus-wide frequencies
        - One-vs-rest comparison for each genre
        - FDR correction for multiple testing
        """
        unique_genres = sorted(genres.unique())
        num_genres = len(unique_genres)

        results = []

        for name in ["unigrams", "bigrams", "trigrams"]:
            matrix = matrices[name]
            ngrams = features[name]

            if len(ngrams) == 0:
                continue

            corpus_counts = np.array(matrix.sum(axis=0)).flatten()
            total_corpus_counts = corpus_counts.sum()

            priors = self.prior_strength * (corpus_counts / total_corpus_counts)

            genre_matrices = np.zeros((len(ngrams), num_genres))
            genre_totals = np.zeros(num_genres)

            for idx, genre in enumerate(unique_genres):
                genre_mask = (genres == genre).values
                genre_matrix = matrix[genre_mask, :]
                genre_matrices[:, idx] = np.array(genre_matrix.sum(axis=0)).flatten()
                genre_totals[idx] = genre_matrices[:, idx].sum()

            _, _, z_scores = compute_monroe_statistics(
                genre_matrices,
                genre_totals,
                corpus_counts,
                total_corpus_counts,
                len(ngrams),
                priors,
            )

            for ng_idx, ngram in enumerate(ngrams):
                for g_idx, genre in enumerate(unique_genres):
                    results.append(
                        {
                            "ngram": ngram,
                            "genre": genre,
                            "z_score": z_scores[ng_idx, g_idx],
                        }
                    )

        df = pd.DataFrame(results)

        passes_fdr = apply_fdr_correction(
            df["z_score"].values.reshape(-1, num_genres), self.p_value
        )
        df["passes_fdr"] = passes_fdr.flatten()
        df["z_fdr_corrected"] = df["z_score"] * df["passes_fdr"]

        return df

    def _compute_cache_key(self, X, y, artist):
        """Compute hash for caching (excludes p_value for flexibility)."""
        data_tuple = (
            tuple(X.index),
            tuple(X.values),
            tuple(y.values),
            tuple(artist.values),
            self.min_artists,
            self.prior_strength,
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

    def _get_checkpoint_paths(self):
        """Get path for z-scores checkpoint file."""
        zscores_path = self.checkpoint_dir / f"zscores_{self._cache_key}.pkl"
        return zscores_path

    def _save_checkpoint(self):
        """Save z-scores to checkpoint file."""
        zscores_path = self._get_checkpoint_paths()

        with open(zscores_path, "wb") as f:
            pickle.dump(self.z_scores_df_, f)

    def _load_checkpoint(self):
        """Load z-scores from checkpoint file if it exists."""
        zscores_path = self._get_checkpoint_paths()

        if zscores_path.exists():
            with open(zscores_path, "rb") as f:
                self.z_scores_df_ = pickle.load(f)
            return True

        return False
