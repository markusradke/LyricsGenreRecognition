import pandas as pd
import numpy as np

from scipy import sparse

from helpers.LyricsClassficationExperiment import LyricsClassificationExperiment

SUBSAMPLE_DEBUG = 1.0

C = 1.0
L1_RATIO = 0.5

N_JOBS = 1
N_INITAL = 20
N_ITERATIONS = 50
CV = 5
N_POINTS = 1
STOP_ITER = 15
UNCERTAIN_JUMP = 5

train_metadata = pd.read_csv("data/X_train_metadata.csv")
test_metadata = pd.read_csv("data/X_test_metadata.csv")


#######################################################################################
# ATTENTION: After rerunning the corpus cleaning, please remove these steps and
# corresponding code in the load_classification_features function
#######################################################################################
# remove index from metadata and training set
train_metadata_classical = train_metadata[train_metadata["cat25"] == "classical"].index
print(train_metadata_classical)
train_metadata = train_metadata.drop(train_metadata_classical).reset_index(drop=True)


def load_classification_features(type, granularity):
    if type == "fs":
        X_train = sparse.load_npz(f"data/X_train_fs_G{granularity}.npz")
        X_test = sparse.load_npz(f"data/X_test_fs_G{granularity}.npz")
        # TODO: remove when rerunning after cleaned corpus
        # remove classical tracks from training set, because they are mostly non-english and would distort the classification results
        X_train = sparse.vstack(
            [
                X_train[: train_metadata_classical[0]],
                X_train[train_metadata_classical[-1] + 1 :],
            ]
        )
    elif type == "topic":
        X_train = pd.read_csv(f"data/X_train_topics_G{granularity}.csv")
        X_train.columns = [f"topic_{col}" for col in X_train.columns]
        X_test = pd.read_csv(
            f"data/X_test_topics_G{granularity}.csv",
        )
        X_test.columns = [f"topic_{col}" for col in X_test.columns]
        # TODO: remove when rerunning after cleaned corpus
        # remove classical tracks from training set, because they are mostly non-english and would distort the classification results
        X_train = X_train.drop(
            index=train_metadata_classical,
        ).reset_index(drop=True)
    elif type == "style":
        X_train = pd.read_csv(f"data/X_train_style_G{granularity}.csv")
        X_train.columns = [f"style_{col}" for col in X_train.columns]
        X_test = pd.read_csv(f"data/X_test_style_G{granularity}.csv")
        X_test.columns = [f"style_{col}" for col in X_test.columns]
        # TODO: remove when rerunning after cleaned corpus
        # remove classical tracks from training set, because they are mostly non-english and would distort the classification results
        X_train = X_train.drop(index=train_metadata_classical).reset_index(drop=True)
    else:
        raise ValueError("Invalid type. Must be one of 'fs', 'topic', or 'style'.")
    return X_train, X_test


def get_rf_param_space(X_train):
    return {
        "max_features": (1 / X_train.shape[1], 1.0),
        # log scale for min_samples_leaf from 1 to number of samples in training set
        "min_samples_leaf": (np.log(1 / X_train.shape[0]), 0),
    }


def run_experiment(X_train, X_test, y_train, y_test, model_path, mode="lr"):
    experiment = LyricsClassificationExperiment(
        X_train, y_train, X_test, y_test, model_path, subsample_debug=SUBSAMPLE_DEBUG
    )
    if mode == "lr":
        experiment.train_fixed_parametrer_logistic_regression()
    elif mode == "rf":
        param_space = get_rf_param_space(X_train)
        experiment.tune_and_train_classifier(
            param_space,
            parsimony_param="min_samples_leaf",
            parsimony_ascending=False,  # select highest for parsimony
            cv=CV,
            n_initial=N_INITAL,
            n_iterations=N_ITERATIONS,
            n_jobs=N_JOBS,
            n_points=N_POINTS,
            stop_iter=STOP_ITER,
            uncertain_jump=UNCERTAIN_JUMP,
        )
    experiment.save_model_evaluation_txt()
    experiment.save_experiment()


def load_combined_features(granularity):
    X_train_topic, X_test_topic = load_classification_features("topic", granularity)
    X_train_style, X_test_style = load_classification_features("style", granularity)
    return (
        pd.concat([X_train_topic, X_train_style], axis=1),
        pd.concat([X_test_topic, X_test_style], axis=1),
    )


# (feature_type, model_type, model_name_prefix)
EXPERIMENTS = [
    # ("fs", "lr", "classificator_fs_lr"),
    # ("topic", "lr", "classificator_topic_lr"),
    # ("style", "lr", "classificator_style_lr"),
    # ("combined", "lr", "classificator_topicstyle_lr"),
    # ("topic", "rf", "classificator_topic_rf"),
    ("style", "rf", "classificator_style_rf"),
    ("combined", "rf", "classificator_topicstyle_rf"),
]

for feature_type, mode, name_prefix in EXPERIMENTS:
    for granularity in [5, 12, 25]:
        if feature_type == "combined":
            X_train, X_test = load_combined_features(granularity)
        else:
            X_train, X_test = load_classification_features(feature_type, granularity)
        run_experiment(
            X_train,
            X_test,
            train_metadata[f"cat{granularity}"],
            test_metadata[f"cat{granularity}"],
            f"models/{name_prefix}_G{granularity}",
            mode=mode,
        )
