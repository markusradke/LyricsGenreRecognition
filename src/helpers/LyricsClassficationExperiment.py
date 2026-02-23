import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # for the time being

from helpers.split_group_stratified_and_join import (
    split_group_stratified_and_join,
    plot_comparison_genre_distributions,
)
from helpers.LyricsClassificationMetrics import LyricsClassificationMetrics
from helpers.FSExtractor import FSExtractor
from helpers.MonroeExtractor import MonroeExtractor
from helpers.GenreClassifierTrainer import GenreClassifierTrainer
from helpers.aggregate_artist_dtm import aggregate_dtm_by_artist
from helpers.STMTopicModeler import STMTopicModeler


class LyricsClassificationExperiment:
    def __init__(
        self,
        corpus,
        genrecol,
        lyricscol,
        artistcol,
        output_dir,
        test_size=0.2,
        random_state=42,
        subsample_debug=None,
    ):
        self.random_state = random_state
        self.output_dir = output_dir
        self.test_size = test_size
        self.subsample_debug = subsample_debug if subsample_debug is not None else 1.0
        self.corpus_train, self.corpus_test = self._prepare_corpus(
            corpus, genrecol, lyricscol, artistcol
        )
        self.granularity = self.corpus_train["genre"].nunique()
        self.random_performance_baseline = self._compute_random_baseline()
        self.X_train = None
        self.y_train = self.corpus_train["genre"]
        self.y_test = self.corpus_test["genre"]
        self.model = None

    def __str__(self):
        out = (
            f"LyricsClassificationExperiment with {self.granularity} genres\n"
            + "=============================================\n"
            + f"Train size: {self.corpus_train.shape[0]} samples\n"
            + f"Test size: {self.corpus_test.shape[0]} samples\n"
        )
        if self.X_train is None:
            out += "Features not yet computed.\n"
        else:
            out += (
                f"# of features: {self.X_train.shape[1]}\n"
                + f"Feature type: {self.feature_type}\n"
            )
        if self.model is None:
            out += "Model not yet trained.\n"
        out += (
            "=============================================\n"
            + f"Output directory: {self.output_dir}\n"
        )

        return out

    def _prepare_corpus(self, corpus, genrecol, lyricscol, artistcol):
        selected = corpus[[genrecol, lyricscol, artistcol]].rename(
            {
                genrecol: "genre",
                lyricscol: "lyrics",
                artistcol: "artist",
            },
            axis=1,
        )

        if self.subsample_debug < 1.0:
            selected, _ = train_test_split(
                selected,
                train_size=self.subsample_debug,
                random_state=self.random_state,
                stratify=selected["genre"],
                shuffle=True,
            )

        labels_and_group = selected.rename(
            columns={"genre": "label", "artist": "group"}
        )
        train, test, _, _ = split_group_stratified_and_join(
            labels_and_group,
            selected,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return train, test

    def _compute_random_baseline(self):
        p = self.corpus_train["genre"].value_counts(normalize=True)
        y_test = self.corpus_test["genre"]
        rng = np.random.default_rng(self.random_state)
        labels = p.index.to_numpy()
        weights = p.to_numpy()
        sampled = rng.choice(labels, size=self.corpus_test.shape[0], p=weights)
        y_pred = pd.Series(sampled, index=y_test.index, name="pred")
        return LyricsClassificationMetrics(y_test, y_pred)

    def compute_fs_ngram_features(
        self,
        min_artists=20,
        top_vocab_per_genre=100,
        use_stopword_filter=True,
        include_unigrams=True,
    ):
        """Extract FS ngram features from corpus."""
        self.extractor = FSExtractor(
            min_artists=min_artists,
            top_vocab_per_genre=top_vocab_per_genre,
            use_stopword_filter=use_stopword_filter,
            checkpoint_dir=self.output_dir + "/FSExtractor_checkpoints",
            random_state=self.random_state,
        )

        self.extractor.fit(
            self.corpus_train["lyrics"],
            self.corpus_train["genre"],
            self.corpus_train["artist"],
        )
        self.X_train = self.extractor.transform(self.corpus_train["lyrics"])
        self.X_test = self.extractor.transform(self.corpus_test["lyrics"])

        unigrams_str = "with unigrams" if include_unigrams else "phrases only"
        stopwords_str = (
            "stopwords filtered" if use_stopword_filter else "stopwords kept"
        )
        self.feature_type = f"FS N-grams (pipeline, top {top_vocab_per_genre}/genre, min {min_artists} artists, {unigrams_str}, {stopwords_str})"
        print(f"FSExtractor configured: {self.feature_type}")

    def compute_monroe_ngram_features(
        self,
        min_artists=20,
        use_stopword_filter=True,
        use_bigram_boundary_filter=True,
        include_unigrams=True,
        p_value=0.001,
        prior_concentration=0.5,
    ):
        """Create MonroeExtractor for pipeline-based training.

        Args:
            min_artists: Minimum artists threshold
            include_unigrams: Whether to include unigrams (False = phrases only)
            use_stopword_filter: Whether to filter stopword-only n-grams
            p_value: P-value threshold for feature selection (default 0.001)
            prior_concentration: Strength of Bayesian prior (default 0.5)
        """

        self.extractor = MonroeExtractor(
            min_artists=min_artists,
            p_value=p_value,
            prior_concentration=prior_concentration,
            checkpoint_dir=self.output_dir + "/MonroeExtractor_checkpoints",
            use_stopword_filter=use_stopword_filter,
            use_bigram_boundary_filter=use_bigram_boundary_filter,
            include_unigrams=include_unigrams,
        )
        self.X_train = self.extractor.fit(
            self.corpus_train["lyrics"],
            self.corpus_train["genre"],
            self.corpus_train["artist"],
        )
        self.X_train = self.extractor.transform(self.corpus_train["lyrics"])
        self.X_test = self.extractor.transform(self.corpus_test["lyrics"])

        unigrams_str = "with unigrams" if include_unigrams else "phrases only"
        stopwords_str = (
            "stopwords filtered" if use_stopword_filter else "stopwords kept"
        )
        self.feature_type = f"Monroe N-grams (min {min_artists} artists, {unigrams_str}, {stopwords_str}, p={p_value} (FDR correction), prior_concentration={prior_concentration})"
        print(f"MonroeExtractor configured: {self.feature_type}")

    def compute_stm_topic_features(self, k_range=(2, 20)):
        """Compute topic features using Structural Topic Model.

        Uses existing n-gram vocabulary to build artist-level DTM, tunes and fits
        STM model with genre as prevalence covariate, then transforms tracks to
        topic proportions.

        Args:
            k_range: Tuple (min_K, max_K) for topic number search
        """
        if not hasattr(self, "extractor") or self.X_train is None:
            raise ValueError(
                "No features computed. Call compute_monroe_ngram_features() or "
                "compute_fs_ngram_features() first."
            )

        vocab = self.extractor.get_feature_names_out()
        X_train_dtm = self.X_train

        print("Aggregating track-level DTM to artist-level...")
        X_artist, artist_genres = aggregate_dtm_by_artist(
            X_train_dtm,
            self.corpus_train["artist"],
            self.corpus_train["genre"],
        )
        print(
            f"Artist-level DTM: {X_artist.shape[0]} artists, {X_artist.shape[1]} features"
        )

        print("Initializing STM topic modeler...")
        self.stm_modeler = STMTopicModeler(
            k_range=k_range,
            random_state=self.random_state,
            model_dir=self.output_dir + "/stm_model",
        )

        print("Tuning and fitting STM model...")
        self.stm_modeler.tune_and_fit(X_artist, artist_genres, vocab)

        print("Transforming tracks to topic proportions...")
        self.X_train = self.stm_modeler.transform(X_train_dtm, vocab)
        X_test_dtm = self.extractor.transform(self.corpus_test["lyrics"])
        self.X_test = self.stm_modeler.transform(X_test_dtm, vocab)

        vocab_source = "Monroe" if "Monroe" in self.feature_type else "FS"
        self.feature_type = (
            f"STM Topics (K={self.stm_modeler.K_}, {vocab_source} vocab, "
            f"artist-level training, genre prevalence covariate)"
        )
        print(f"STM features computed: {self.feature_type}")
        print(
            f"Feature matrix shape: Train {self.X_train.shape}, Test {self.X_test.shape}"
        )

    def _ensure_features(self):
        if self.X_train is None:
            raise ValueError(
                "Features not computed. Call compute_fs_ngram_features() first."
            )

    def _create_trainer(self, n_jobs=None):
        """Create trainer with optional pipeline mode.

        Args:
            n_jobs: Number of parallel jobs
        """
        feature_names = self.extractor.get_feature_names_out().tolist()
        if n_jobs is None:
            return GenreClassifierTrainer(
                self.X_train,
                self.y_train,
                self.random_state,
                feature_names=feature_names,
            )
        return GenreClassifierTrainer(
            self.X_train,
            self.y_train,
            self.random_state,
            n_jobs,
            feature_names=feature_names,
        )

    def _fit_and_store_results(self, trainer, method_name, *args, **kwargs):
        fit_fn = getattr(trainer, method_name)
        fit_fn(*args, **kwargs)
        results = trainer.get_results()
        self.model = results.get("pipeline", None)
        self.model_parameters = results.get("params", {})
        self.model_coefficients = results.get("coefficients", pd.DataFrame())
        self.cv_tuning_history = results.get("tuning_history", pd.DataFrame())

    def tune_and_train_logistic_regression(
        self,
        param_space,
        cv=5,
        n_initial=20,
        n_iterations=50,
        n_jobs=-1,
        stop_iter=10,
        uncertain_jump=5,
        use_pipeline=False,
    ):
        if not use_pipeline:
            self._ensure_features()
        checkpoint_dir = self.output_dir + "/optimization_checkpoints"
        trainer = self._create_trainer(n_jobs, use_pipeline=use_pipeline)
        self._fit_and_store_results(
            trainer,
            "fit_with_bayesian_optimization",
            param_space,
            n_initial=n_initial,
            n_iterations=n_iterations,
            stop_iter=stop_iter,
            uncertain_jump=uncertain_jump,
            cv=cv,
            checkpoint_dir=checkpoint_dir,
            parsimony_param="C",
        )

    def train_fixed_parametrer_logistic_regression(self, C=1.0, l1_ratio=0.5):
        self._ensure_features()
        trainer = self._create_trainer()
        self._fit_and_store_results(
            trainer,
            "fit_with_fixed_params",
            C=C,
            l1_ratio=l1_ratio,
        )
        results = trainer.get_results()
        self.model = results["pipeline"]
        self.model_parameters = results["params"]
        self.model_coefficients = results["coefficients"]
        self.cv_tuning_history = pd.DataFrame()

    def save_experiment(self):
        with open(self.output_dir + "/complete_experiment.pkl", "wb") as f:
            pickle.dump(self, f)
        print(f"Experiment saved to {self.output_dir}/complete_experiment.pkl")

    def show_train_test_genrefreq_comparison(self):
        plot_comparison_genre_distributions(
            self.corpus_train["genre"], self.corpus_test["genre"]
        )

    def show_random_baseline_evaluation(self):
        print(self.random_performance_baseline)

    def show_tuning_history(self):
        print("Selected model parameters:")
        for parameter, value in self.model_parameters.items():
            print(f"  {parameter}: {value:.3f}")
        print("=" * 60)
        print(self.cv_tuning_history)
        # plots of cv tuning history
        # chosen model parameters

    def show_model_evaluation(self, top_n_coefficients=10):
        print("Selected model parameters:")
        for parameter, value in self.model_parameters.items():
            print(f"  {parameter}: {value}")
        print("=" * 60)
        y_pred = self.model.predict(self.X_test)
        print(LyricsClassificationMetrics(self.y_test, y_pred))
        print("=" * 60)
        print(classification_report(self.y_test, y_pred))
        # metrics (make metrics per class available: P, R, F1)
        # confusion matrix

    def show_top_coefficients_per_genre(self, top_n=10):
        genres = self.model_coefficients.columns
        for genre in genres:
            print(f"Top {top_n} coefficients for genre: {genre.upper()}")
            top_coeffs = (
                self.model_coefficients[genre]
                .abs()
                .sort_values(ascending=False)
                .head(top_n)
            )
            for feature, value in top_coeffs.items():
                print(f"{feature} ({self.model_coefficients.at[feature, genre]:.3f})")
            print("\n")
