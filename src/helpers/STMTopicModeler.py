"""
Structural Topic Model (STM) wrapper using R via rpy2.

Handles tuning the number of topics using held-out likelihood, fitting the
final model on full training data, and transforming new documents to topic
proportions. Uses the stm R package with genre as a prevalence covariate.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
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
        Directory to save fitted model and checkpoints. If None, checkpointing
        is disabled.

    Attributes
    ----------
    K_ : int
        Selected number of topics after tuning.
    stm_model_ : R object
        Fitted STM model (R stm object).
    vocab_ : list of str
        Vocabulary (feature names) used by the model.
    search_results_ : dict
        Tuning results with "K" and "heldout" arrays.
    _is_fitted : bool
        Whether the model has been fitted.
    """

    def __init__(
        self,
        k_range=(2, 20),
        use_genre_prevalence=True,
        random_state=42,
        model_dir=None,
    ):
        self.k_range = k_range
        self.random_state = random_state
        self.use_genre_prevalence = use_genre_prevalence
        self.model_dir = Path(model_dir) if model_dir else None
        self._is_fitted = False

        self.stm = importr("stm")
        self.base = importr("base")

        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)

    def _compute_search_hash(self, X_artist, vocab):
        """
        Compute a joblib hash of the search inputs for checkpointing.

        Parameters
        ----------
        X_artist : sparse matrix
            Artist-level document-term matrix.
        vocab : array-like
            Vocabulary.

        Returns
        -------
        hash_str : str
            Hex hash string.
        """
        return joblib.hash(
            (
                X_artist,
                list(vocab),
                self.k_range,
                self.use_genre_prevalence,
                self.random_state,
            )
        )

    def _search_checkpoint_path(self, input_hash):
        """Return Path for the search checkpoint file, or None if model_dir unset."""
        if self.model_dir is None:
            return None
        return self.model_dir / f"stm_search_{input_hash[:12]}.pkl"

    def _load_search_checkpoint(self, checkpoint_path):
        """
        Load incremental search results from a checkpoint file.

        Returns
        -------
        results : dict with "K" list and "heldout" list, or None if not found.
        """
        if checkpoint_path is None or not checkpoint_path.exists():
            return None
        results = joblib.load(checkpoint_path)
        print(
            f"Resuming search from checkpoint (hash: {checkpoint_path.stem[-12:]}, "
            f"{len(results['K'])} K values already evaluated: {results['K']})"
        )
        return results

    def _save_search_checkpoint(self, checkpoint_path, results):
        """Save incremental search results dict to checkpoint file."""
        if checkpoint_path is None:
            return
        joblib.dump(results, checkpoint_path)

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
        self.search_results_ = self._run_searchK(
            documents, meta, X_artist=X_artist, vocab=vocab
        )

        self.K_ = self._select_best_K(self.search_results_)
        print(
            f"Selected K = {self.K_} topics (highest local maximum of held-out likelihood)"
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

    def _run_searchK(self, documents, meta, X_artist=None, vocab=None):
        """
        Evaluate different numbers of topics, skipping K values that fail.

        Manually replicates searchK by creating a held-out set once, then
        fitting each K individually with R tryCatch so that numerical failures
        (e.g. Cholesky decomposition errors) are caught and skipped.

        Checkpoints results incrementally after each K so that interrupted
        runs can resume from where they left off.

        Parameters
        ----------
        documents : R list
            STM-formatted documents.
        meta : R DataFrame
            Metadata with genre column.
        X_artist : sparse matrix, optional
            Original matrix used to compute the checkpoint hash.
        vocab : array-like, optional
            Vocabulary used to compute the checkpoint hash.

        Returns
        -------
        search_results : dict with keys "K" and "heldout"
            Diagnostic metrics for all successfully fitted K values.
        """
        vocab_r = ro.StrVector(self.vocab_)

        if self.use_genre_prevalence:
            prevalence_formula = ro.Formula("~genre")
        else:
            prevalence_formula = ro.NULL

        # Checkpointing setup
        input_hash = (
            self._compute_search_hash(X_artist, vocab)
            if (X_artist is not None and vocab is not None)
            else None
        )
        checkpoint_path = (
            self._search_checkpoint_path(input_hash) if input_hash else None
        )
        prior = self._load_search_checkpoint(checkpoint_path)

        already_done = set(prior["K"]) if prior else set()
        results_K = list(prior["K"]) if prior else []
        results_heldout = list(prior["heldout"]) if prior else []

        if input_hash:
            print(f"Search hash: {input_hash[:12]}")

        ro.globalenv["._documents"] = documents
        ro.globalenv["._vocab"] = vocab_r
        ro.globalenv["._meta"] = meta
        ro.globalenv["._seed"] = ro.IntVector([self.random_state])

        heldout = ro.r(
            """
local({
  set.seed(._seed[1])
  make.heldout(._documents, ._vocab)
})
"""
        )
        ro.globalenv["._heldout"] = heldout

        k_values = list(range(self.k_range[0], self.k_range[1] + 1))

        for k in k_values:
            if k in already_done:
                print(f"  K={k}: skipped (cached)")
                continue

            ro.globalenv["._k"] = ro.IntVector([k])
            if self.use_genre_prevalence:
                ro.globalenv["._prevalence"] = prevalence_formula
            else:
                ro.globalenv["._prevalence"] = ro.NULL

            result = ro.r(
                """
local({
  tryCatch({
    model <- stm(
      documents  = ._heldout$documents,
      vocab      = ._heldout$vocab,
      K          = ._k[1],
      prevalence = ._prevalence,
      data       = ._meta,
      max.em.its = 500,
      init.type  = "Spectral",
      seed       = ._seed[1],
      verbose    = TRUE
    )
    ho_lik <- eval.heldout(model, ._heldout$missing)$expected.heldout
    list(success = TRUE, heldout = ho_lik)
  }, error = function(e) {
    message(sprintf("K=%d failed: %s", ._k[1], conditionMessage(e)))
    list(success = FALSE, heldout = NA_real_)
  })
})
"""
            )

            success = bool(result.rx2("success")[0])
            if success:
                ho_val = float(result.rx2("heldout")[0])
                results_K.append(k)
                results_heldout.append(ho_val)
                print(f"  K={k}: heldout likelihood = {ho_val:.4f}")
            else:
                results_K.append(k)
                results_heldout.append(float("nan"))
                print(f"  K={k}: skipped (model fit failed)")

            self._save_search_checkpoint(
                checkpoint_path, {"K": results_K, "heldout": results_heldout}
            )

        for key in (
            "._documents",
            "._vocab",
            "._meta",
            "._seed",
            "._heldout",
            "._k",
            "._prevalence",
        ):
            ro.r(f"if (exists('{key}')) rm({key})")

        if not results_K:
            raise RuntimeError(
                "All K values failed during searchK. Cannot select a model."
            )

        return {"K": np.array(results_K), "heldout": np.array(results_heldout)}

    def _select_best_K(self, search_results):
        """
        Select K with highest local maximum of held-out likelihood.

        Failed K values (NaN heldout likelihood) are excluded from selection.

        Parameters
        ----------
        search_results : dict
            searchK results with "K" and "heldout" arrays.

        Returns
        -------
        best_K : int
            Selected number of topics.
        """
        K_values = np.array(search_results["K"]).flatten()
        heldout = np.array(search_results["heldout"]).flatten()

        valid_mask = ~np.isnan(heldout)
        K_values = K_values[valid_mask]
        heldout = heldout[valid_mask]

        if len(K_values) == 0:
            raise RuntimeError("No valid K values to select from (all failed).")

        local_max_indices = [
            i
            for i in range(1, len(heldout) - 1)
            if heldout[i] > heldout[i - 1] and heldout[i] > heldout[i + 1]
        ]

        if local_max_indices:
            best_idx = max(local_max_indices, key=lambda i: heldout[i])
            return int(K_values[best_idx])

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
        if self.use_genre_prevalence:
            prevalence = ro.Formula("~genre")
        else:
            prevalence = ro.NULL

        model = self.stm.stm(
            documents=documents,
            vocab=vocab_r,
            K=K,
            prevalence=prevalence,
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
        metadata_path = self.model_dir / "stm_metadata.pkl"

        self.base.saveRDS(self.stm_model_, str(model_path))

        metadata = {
            "K": self.K_,
            "vocab": self.vocab_,
            "k_range": self.k_range,
            "random_state": self.random_state,
            "search_results": self.search_results_,
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
