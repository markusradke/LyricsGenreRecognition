"""
Structural Topic Model (STM) wrapper using R via rpy2.

Handles tuning the number of topics using held-out likelihood, fitting the
final model on full training data, and transforming new documents to topic
proportions. Uses the stm R package with genre as a prevalence covariate.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.sparse import issparse

try:
    import rpy2.rinterface_lib.embedded as r_embedded

    r_embedded.set_initoptions(
        ("rpy2", "--no-save", "--no-restore", "--max-ppsize=500000")
    )

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
except ImportError:
    raise ImportError(
        "rpy2 is required for STM topic modeling. " "Install with: pip install rpy2"
    )


class STMTopicModeler:
    """
    Structural Topic Model wrapper for Python-R integration.

    Uses the R stm package to fit topic models with genre as a prevalence
    covariate. Tunes the number of topics K using searchK with held-out
    likelihood, then fits the final model on full training data.

    Parameters
    ----------
    k_range : tuple of int, default=(2, 20)
        Range of topic numbers to search (min_K, max_K).
    random_state : int, default=42
        Random seed for reproducibility.
    model_dir : str or Path, optional
        Directory to save fitted model for later inspection in R. If None, model not saved.

    Attributes
    ----------
    K_ : int
        Selected number of topics after tuning.
    stm_model_ : R object
        Fitted STM model (R stm object).
    vocab_ : list of str
        Vocabulary (feature names) used by the model.
    search_results_ : R object
        searchK results with diagnostic metrics for all K values tested.
    _is_fitted : bool
        Whether the model has been fitted.
    """

    def __init__(
        self,
        k_range=(2, 20),
        random_state=42,
        model_dir=None,
    ):
        self.k_range = k_range
        self.random_state = random_state
        self.model_dir = Path(model_dir) if model_dir else None
        self._is_fitted = False

        self.stm = importr("stm")
        self.base = importr("base")

        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)

    def tune_and_fit(self, X_artist, artist_genres, vocab):
        """
        Tune number of topics and fit final model.

        Runs searchK to find optimal K using held-out likelihood (first local
        minimum), then fits the final model on full training data with selected K.

        Parameters
        ----------
        X_artist : sparse matrix, shape (n_artists, n_features)
            Artist-level document-term matrix.
        artist_genres : pd.Series
            Genre labels for each artist.
        vocab : array-like of str
            Feature names (vocabulary).

        Returns
        -------
        self : STMTopicModeler
            Fitted model.
        """
        self.vocab_ = list(vocab)

        print("Converting data to STM format...")
        documents, meta = self._prepare_stm_data(X_artist, artist_genres)

        print(f"Tuning number of topics (K range: {self.k_range})...")
        self.search_results_ = self._run_searchK(documents, meta)

        self.K_ = self._select_best_K(self.search_results_)
        print(
            f"Selected K = {self.K_} topics (first local maximum of held-out likelihood)"
        )

        print(f"Fitting final STM model with K = {self.K_}...")
        self.stm_model_ = self._fit_stm(documents, meta, self.K_)

        self._is_fitted = True

        if self.model_dir:
            self._save_model()
            print(f"Model saved to {self.model_dir}")

        return self

    def transform(self, X, vocab):
        """
        Transform documents to topic proportions.

        Parameters
        ----------
        X : sparse matrix, shape (n_docs, n_features)
            Document-term matrix (track-level or artist-level).
        vocab : array-like of str
            Feature names (must match training vocabulary).

        Returns
        -------
        theta : np.ndarray, shape (n_docs, K)
            Topic proportions for each document.
        """
        if not self._is_fitted:
            raise ValueError("Must call tune_and_fit() before transform()")

        if list(vocab) != self.vocab_:
            raise ValueError("Vocabulary mismatch between training and transform")

        print("Converting data to STM format for inference...")
        documents, _ = self._prepare_stm_data(X, genre=None)

        print("Estimating topic proportions for new documents...")
        fit_result = self.stm.fitNewDocuments(
            model=self.stm_model_,
            documents=documents,
            newData=ro.NULL,
            prevalence=ro.NULL,
        )

        theta = np.array(fit_result.rx2("theta"))
        return theta

    def _prepare_stm_data(self, X, genre=None):
        """
        Convert sparse matrix to STM document format.

        STM expects documents as a list of matrices, where each document is
        a 2xN matrix: row 1 = term indices (1-indexed), row 2 = counts.

        Parameters
        ----------
        X : sparse matrix, shape (n_docs, n_features)
            Document-term matrix.
        genre : pd.Series, optional
            Genre labels for metadata (only for training).

        Returns
        -------
        documents : R list
            STM-formatted documents.
        meta : R DataFrame or NULL
            Metadata with genre column, or NULL if genre is None.
        """
        if issparse(X):
            X = X.tocsr()
        else:
            from scipy.sparse import csr_matrix

            X = csr_matrix(X)

        ro.globalenv["._indptr"] = ro.IntVector((X.indptr).tolist())
        ro.globalenv["._indices"] = ro.FloatVector(
            (X.indices + 1).astype(float).tolist()
        )
        ro.globalenv["._data"] = ro.FloatVector(X.data.astype(float).tolist())
        ro.globalenv["._n_docs"] = ro.IntVector([X.shape[0]])

        documents = ro.r(
            """
local({
  n <- ._n_docs[1]
  docs <- vector("list", n)
  for (i in seq_len(n)) {
    start <- ._indptr[i] + 1L
    end   <- ._indptr[i + 1]
    if (start > end) {
      docs[[i]] <- matrix(numeric(0), nrow = 2L, ncol = 0L)
    } else {
      docs[[i]] <- matrix(
        c(._indices[start:end], ._data[start:end]),
        nrow = 2L, byrow = TRUE
      )
    }
  }
  docs
})
"""
        )
        for key in ("._indptr", "._indices", "._data", "._n_docs"):
            ro.r(f"rm({key})")

        if genre is not None:
            with localconverter(ro.default_converter + pandas2ri.converter):
                meta = ro.conversion.py2rpy(pd.DataFrame({"genre": genre.values}))
        else:
            meta = ro.NULL

        return documents, meta

    def _run_searchK(self, documents, meta):
        """
        Run searchK to evaluate different numbers of topics.

        Parameters
        ----------
        documents : R list
            STM-formatted documents.
        meta : R DataFrame
            Metadata with genre column.

        Returns
        -------
        search_results : R object
            searchK results with diagnostic metrics.
        """
        vocab_r = ro.StrVector(self.vocab_)
        k_values = ro.IntVector(range(self.k_range[0], self.k_range[1] + 1))

        search_results = self.stm.searchK(
            documents=documents,
            vocab=vocab_r,
            K=k_values,
            prevalence=ro.Formula("~genre"),
            data=meta,
            init_type="Spectral",
            heldout_seed=self.random_state,
            verbose=True,
        )

        return search_results

    def _select_best_K(self, search_results):
        """
        Select K with first local maximum of held-out likelihood.

        Parameters
        ----------
        search_results : R object
            searchK results.

        Returns
        -------
        best_K : int
            Selected number of topics.
        """
        results = search_results.rx2("results")
        K_values = np.array(results.rx2("K")).flatten()
        heldout = np.array(results.rx2("heldout")).flatten()

        for i in range(1, len(heldout) - 1):
            if heldout[i] > heldout[i - 1] and heldout[i] > heldout[i + 1]:
                return int(K_values[i])

        best_idx = np.argmax(heldout)
        return int(K_values[best_idx])

    def _fit_stm(self, documents, meta, K):
        """
        Fit STM model with specified K.

        Parameters
        ----------
        documents : R list
            STM-formatted documents.
        meta : R DataFrame
            Metadata with genre column.
        K : int
            Number of topics.

        Returns
        -------
        model : R object
            Fitted STM model.
        """
        vocab_r = ro.StrVector(self.vocab_)

        model = self.stm.stm(
            documents=documents,
            vocab=vocab_r,
            K=K,
            prevalence=ro.Formula("~genre"),
            data=meta,
            max_em_its=500,
            init_type="Spectral",
            seed=self.random_state,
            verbose=True,
        )

        return model

    def _save_model(self):
        """Save fitted model and metadata to model directory."""
        model_path = self.model_dir / "stm_model.rds"
        search_results_path = self.model_dir / "stm_search_results.rds"
        metadata_path = self.model_dir / "stm_metadata.pkl"

        self.base.saveRDS(self.stm_model_, str(model_path))
        self.base.saveRDS(self.search_results_, str(search_results_path))

        metadata = {
            "K": self.K_,
            "vocab": self.vocab_,
            "k_range": self.k_range,
            "random_state": self.random_state,
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
