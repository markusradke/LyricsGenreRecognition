import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # for the time being

from helpers.split_group_stratified_and_join import (
    plot_comparison_genre_distributions,
)
from helpers.LyricsClassificationMetrics import LyricsClassificationMetrics
from helpers.GenreClassifierTrainer import GenreClassifierTrainer


class LyricsClassificationExperiment:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        output_dir,
        subsample_debug=None,
        random_state=42,
    ):
        self.random_state = random_state
        self.output_dir = output_dir
        self.subsample_debug = subsample_debug if subsample_debug is not None else 1.0
        self.granularity = y_train.nunique()

        self.X_train, self.y_train = self._subsample_train_set(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

        self.random_performance_baseline = self._compute_random_baseline()
        self.model = None
        self.training_time = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __str__(self):
        printout = (
            f"LyricsClassificationExperiment with {self.granularity} genres\n"
            + "=" * 60
            + "\n"
            + f"Performed on {round(self.subsample_debug *100)}% of the data\n"
            + "\n"
            + f"Train size: {self.X_train.shape[0]} samples\n"
            + f"Test size: {self.X_test.shape[0]} samples\n"
        )
        printout += f"# of features: {self.X_train.shape[1]}\n"

        if self.model is None:
            printout += "Model not yet trained.\n"
        else:
            printout += (
                f"Training time: {self.training_time:.2f} min\n"
                if self.training_time
                else ""
            )
        printout += "=" * 60 + "\n" + f"Output directory: {self.output_dir}\n"

        return printout

    def _subsample_train_set(self, X_train, y_train):
        if self.subsample_debug < 1.0:
            X_train_sub, _, y_train_sub, _ = train_test_split(
                X_train,
                y_train,
                train_size=self.subsample_debug,
                stratify=y_train,
                random_state=self.random_state,
            )
            return X_train_sub, y_train_sub
        return X_train, y_train

    def _compute_random_baseline(self):
        p = self.y_train.value_counts(normalize=True)
        rng = np.random.default_rng(self.random_state)
        labels = p.index.to_numpy()
        weights = p.to_numpy()
        sampled = rng.choice(labels, size=self.y_test.shape[0], p=weights)
        y_pred = pd.Series(sampled, index=self.y_test.index, name="pred")
        return LyricsClassificationMetrics(self.y_test, y_pred)

    def _create_trainer(self):
        if isinstance(self.X_train, pd.DataFrame):
            feature_names = self.X_train.columns
        else:
            feature_names = None

        return GenreClassifierTrainer(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.random_state,
            feature_names=feature_names,
        )

    def _fit_and_store_results(self, trainer, method_name, *args, **kwargs):
        fit_fn = getattr(trainer, method_name)
        fit_fn(*args, **kwargs)
        results = trainer.get_results()
        self.model = results.get("pipeline", None)
        self.model_parameters = results.get("params", {})
        self.cv_tuning_history = results.get("tuning_history", pd.DataFrame())
        try:
            self.model_coefficients = results.get("coefficients", pd.DataFrame())
        except:
            self.model_coefficients = None
        try:
            self.holdout_permutation_importance = results.get(
                "holdout_permutation_importance", None
            )
        except:
            self.holdout_permutation_importance = None

    def tune_and_train_classifier(
        self,
        param_space,
        parsimony_param,
        parsimony_ascending=True,
        cv=5,
        n_initial=25,
        n_iterations=100,
        stop_iter=20,
        uncertain_jump=5,
    ):
        starttime = pd.Timestamp.now()
        checkpoint_dir = self.output_dir + "/optimization_checkpoints"
        trainer = self._create_trainer()
        self._fit_and_store_results(
            trainer,
            "fit_with_bayesian_optimization",
            param_space=param_space,
            parsimony_param=parsimony_param,
            parsimony_ascending=parsimony_ascending,
            n_initial=n_initial,
            n_iterations=n_iterations,
            stop_iter=stop_iter,
            uncertain_jump=uncertain_jump,
            cv=cv,
            checkpoint_dir=checkpoint_dir,
        )
        endtime = pd.Timestamp.now()
        self.training_time = (endtime - starttime).total_seconds() / 60

    def train_fixed_parametrer_logistic_regression(self, C=1.0, l1_ratio=0.5):
        starttime = pd.Timestamp.now()
        trainer = self._create_trainer()
        self._fit_and_store_results(
            trainer,
            "fit_logistic_regression_with_fixed_params",
            C=C,
            l1_ratio=l1_ratio,
        )
        results = trainer.get_results()
        self.model = results["pipeline"]
        self.model_parameters = results["params"]
        self.model_coefficients = results["coefficients"]
        self.cv_tuning_history = pd.DataFrame()
        endtime = pd.Timestamp.now()
        self.training_time = (endtime - starttime).total_seconds() / 60

    def save_experiment(self):
        """Save experiment state."""
        state = self.__dict__.copy()
        with open(self.output_dir + "/complete_experiment.pkl", "wb") as f:
            pickle.dump(state, f)
        print(f"Experiment saved to {self.output_dir}/complete_experiment.pkl")

    def show_train_test_genrefreq_comparison(self):
        plot_comparison_genre_distributions(self.y_train["genre"], self.y_test["genre"])

    def _get_random_baseline_evaluation(self):
        printout = "=" * 60 + "\n"
        printout += (
            "Random baseline performance (genre distribution weighted sampling):\n"
        )
        printout += str(self.random_performance_baseline)
        return printout

    def _get_tuning_history(self):
        printout = "=" * 60 + "\n"
        printout += "Selected model parameters:\n"
        printout += "\n".join(
            f"  {parameter}: {value:.3f}"
            for parameter, value in self.model_parameters.items()
        )
        printout += "\n" + "-" * 60 + "\n"
        printout += "Cross-validation tuning history:\n"
        printout += str(self.cv_tuning_history)
        return printout
        # plots of cv tuning history
        # chosen model parameters

    def _get_model_evaluation(self):
        y_pred = self.model.predict(self.X_test)

        printout = "=" * 60 + "\n"
        printout += "Model evaluation on test set:\n"
        printout += str(LyricsClassificationMetrics(self.y_test, y_pred)) + "\n"
        printout += "-" * 60 + "\n"
        printout += "Classification report:\n"
        printout += classification_report(self.y_test, y_pred)
        return printout
        # confusion matrix

    def _get_top_coefficients_per_genre(self, top_n=20):
        if self.model_coefficients is None:
            return None
        printout = "=" * 60 + "\n"
        printout += f"Top {top_n} coefficients per genre:\n"

        genres = self.model_coefficients.columns
        for genre in genres:
            printout += f"\n{genre.upper()}\n"
            top_coeffs = (
                self.model_coefficients[genre]
                .abs()
                .sort_values(ascending=False)
                .head(top_n)
            )
            for feature, value in top_coeffs.items():
                printout += (
                    f"  {feature} ({self.model_coefficients.at[feature, genre]:.3f})\n"
                )
        return printout

    def _get_holdout_permutation_importance(self):
        if self.holdout_permutation_importance is None:
            return None
        printout = "=" * 60 + "\n"
        printout += "Holdout set permutation importance:\n"
        importance_df = pd.DataFrame(
            {
                "feature": self.X_test.columns,
                "importance_mean": self.holdout_permutation_importance.importances_mean,
                "importance_std": self.holdout_permutation_importance.importances_std,
            }
        ).sort_values(by="importance_mean", ascending=False)
        for _, row in importance_df.iterrows():
            printout += f"  {row['feature']}: {row['importance_mean']:.3f} +- {row['importance_std']:.3f}\n"
        return printout

    def show_random_baseline_evaluation(self):
        eval = self._get_random_baseline_evaluation()
        print(eval)

    def show_tuning_history(self):
        eval = self._get_tuning_history()
        print(eval)

    def show_model_evaluation(self):
        eval = self._get_model_evaluation()
        print(eval)

    def show_top_coefficients_per_genre(self, top_n=20):
        eval = self._get_top_coefficients_per_genre(top_n)
        print(eval)

    def show_holdout_permutation_importance(self):
        eval = self._get_holdout_permutation_importance()
        print(eval)

    def save_model_evaluation_txt(self):
        if self.model is None:
            print("Model not yet fitted, cannot save evaluation")
            return None

        evaluation_str = self.__str__() + "\n"
        evaluation_str += (
            self._get_random_baseline_evaluation() + "\n" + self._get_model_evaluation()
        )
        if self.model_coefficients is not None:
            evaluation_str += "\n" + self._get_top_coefficients_per_genre()
        if self.holdout_permutation_importance is not None:
            evaluation_str += "\n" + self._get_holdout_permutation_importance()
        if self._get_tuning_history() is not None:
            evaluation_str += "\n" + self._get_tuning_history()

        with open(self.output_dir + "/model_evaluation.txt", "w") as f:
            f.write(evaluation_str)
        print(f"Model evaluation saved to {self.output_dir}/model_evaluation.txt")
        return None
