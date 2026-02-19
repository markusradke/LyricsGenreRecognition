import pandas as pd
import numpy as np
import pickle

from pandas import DataFrame, Series
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # for the time being

from helpers.split_group_stratified_and_join import (
    split_group_stratified_and_join,
    plot_comparison_genre_distributions,
)
from helpers.LyricsClassificationMetrics import LyricsClassificationMetrics
from helpers.NGramFeatureExctractorFS import NGramFeatureExtractorFS
from helpers.ExpressionExtractor import ExpressionExtractor
from helpers.TopicFeatureExtractor import TopicFeatureExtractor
from helpers.GenreClassifierTrainer import GenreClassifierTrainer


class LyricsClassificationExperiment:
    def __init__(
        self,
        corpus,
        genrecol,
        lyricscol,
        artistcol,
        yearcol,
        output_dir,
        test_size=0.2,
        random_state=42,
        subsample_debug=1.0,
    ):
        self.random_state = random_state
        self.output_dir = output_dir
        self.test_size = test_size
        self.subsample_debug = subsample_debug
        self.corpus_train, self.corpus_test = self._prepare_corpus(
            corpus, genrecol, lyricscol, artistcol, yearcol
        )
        self.corpus_train_replaced = (
            self.corpus_train
        )  # initilaize for direct topic modelling
        self.corpus_test_replced = self.corpus_test
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

    def _prepare_corpus(self, corpus, genrecol, lyricscol, artistcol, yearcol):
        selected = corpus[[genrecol, lyricscol, artistcol, yearcol]].rename(
            {
                genrecol: "genre",
                lyricscol: "lyrics",
                artistcol: "artist",
                yearcol: "releaseyear",
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

    def compute_fs_ngram_features(self, min_artists=50, top_n_per_genre_and_ngram=100):
        ngram_extractor = NGramFeatureExtractorFS(
            min_artists=min_artists,
            top_n=top_n_per_genre_and_ngram,
            random_state=self.random_state,
        )
        self.X_train = ngram_extractor.fit(self.corpus_train)
        self.ngram_extractor = ngram_extractor
        self.X_test = ngram_extractor.transform(self.corpus_test["lyrics"])
        self.feature_type = f"Fell-Spohrleder (2014) N-grams (top {top_n_per_genre_and_ngram} (per genre and ngram type), min. {min_artists} artists)"

    def compute_idiom_ngram_features(
        self,
        min_artists=50,
        min_tracks=100,
        llr_threshold=10,
        top_n_per_ngram_pergenre=300,
    ):
        ngram_extractor = ExpressionExtractor(
            min_artists=min_artists,
            min_tracks=min_tracks,
            llr_treshold=llr_threshold,
            top_vocab_per_genre=top_n_per_ngram_pergenre,
        )
        self.X_train, self.corpus_train_replaced = ngram_extractor.fit_transform(
            self.corpus_train
        )
        self.ngram_extractor = ngram_extractor
        self.X_test, self.corpus_test_replaced = ngram_extractor.transform(
            self.corpus_test
        )
        self.feature_type = f"Informed N-grams (top {top_n_per_ngram_pergenre} per genre, min. {min_artists} artists, min. {min_tracks})"

    def _ensure_features(self):
        if self.X_train is None:
            raise ValueError(
                "Features not computed. Call compute_fs_ngram_features() first."
            )

    def _create_trainer(self, n_jobs=None):
        if n_jobs is None:
            return GenreClassifierTrainer(self.X_train, self.y_train, self.random_state)
        return GenreClassifierTrainer(
            self.X_train, self.y_train, self.random_state, n_jobs
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
    ):
        self._ensure_features()
        checkpoint_dir = self.output_dir + "/optimization_checkpoints"
        trainer = self._create_trainer(n_jobs)
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

    def train_fixed_parametrer_logistic_regression(
        self, C=1.0, l1_ratio=0.5, target_ratio=3.0
    ):
        self._ensure_features()
        trainer = self._create_trainer()
        self._fit_and_store_results(
            trainer,
            "fit_with_fixed_params",
            C=C,
            l1_ratio=l1_ratio,
            target_ratio=target_ratio,
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
