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
from helpers.NGramFeatureExtractorFS import NGramFeatureExtractorFS
from helpers.training_pipeline import train_model_with_optimization


class LyricsClassificationExperiment:
    granularity: int
    output_dir: str
    test_size: float
    corpus_train: DataFrame
    corpus_test: DataFrame
    random_performance_baseline: LyricsClassificationMetrics
    X_train: DataFrame
    X_test: DataFrame
    y_train: Series
    y_test: Series
    feature_type: str
    model: object
    model_parameters: Dict[str, float]
    model_coefficients: DataFrame
    cv_tuning_history: DataFrame
    random_state: int
    subsample_debug: float

    def __init__(
        self,
        corpus,
        genrecol,
        lyricscol,
        artistcol,
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

    def compute_fs_ngram_features(self, min_artists=50, top_n=100):
        ngram_extractor = NGramFeatureExtractorFS(
            min_artists=min_artists,
            top_n=top_n,
            random_state=self.random_state,
        )
        self.X_train = ngram_extractor.fit(self.corpus_train)
        self.X_test = ngram_extractor.transform(self.corpus_test["lyrics"])
        self.feature_type = (
            f"Fell-Spohrleder (2014) N-grams (top {top_n}, min. {min_artists} artists)"
        )

    def tune_and_train_logistic_regression(
        self, param_space, cv=5, n_initial=20, n_iterations=50, n_jobs=-1
    ):
        if self.X_train is None:
            raise ValueError(
                "Features not computed. Call compute_fs_ngram_features() first."
            )

        checkpoint_dir = self.output_dir + "/optimization_checkpoints"
        (
            self.model,
            self.model_parameters,
            self.model_coefficients,
            self.cv_tuning_history,
        ) = train_model_with_optimization(
            self.X_train,
            self.y_train,
            param_space,
            cv=cv,
            n_initial=n_initial,
            n_iterations=n_iterations,
            n_jobs=n_jobs,
            checkpoint_dir=checkpoint_dir,
            random_state=self.random_state,
        )

    def save_experiment(self):
        with open(self.output_dir + "/complete_experiment.pkl", "wb") as f:
            pickle.dump(self, f)

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
            print(f"  {parameter}: {value:.3f}")
        print("=" * 60)
        y_pred = self.model.predict(self.X_test)
        print(LyricsClassificationMetrics(self.y_test, y_pred))
        print("=" * 60)
        print(classification_report(self.y_test, y_pred))
        # metrics (make metrics per class available: P, R, F1)
        # confusion matrix
        # top n coefficients per genre
