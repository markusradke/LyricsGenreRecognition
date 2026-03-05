import pandas as pd

from scipy import sparse

from helpers.LyricsClassficationExperiment import LyricsClassificationExperiment

train_metadata = pd.read_csv("data/X_train_metadata.csv")
test_metadata = pd.read_csv("data/X_test_metadata.csv")


def load_classification_features(type, granularity):
    if type == "fs":
        X_train = sparse.load_npz(f"data/X_train_fs_G{granularity}.npz")
        X_test = sparse.load_npz(f"data/X_test_fs_G{granularity}.npz")
    elif type == "topic":
        X_train = pd.read_csv(f"data/X_train_topics_G{granularity}.csv", index_col=0)
        X_test = pd.read_csv(f"data/X_test_topics_G{granularity}.csv", index_col=0)
    elif type == "style":
        X_train = pd.read_csv(f"data/X_train_style_G{granularity}.csv", index_col=0)
        X_test = pd.read_csv(f"data/X_test_style_G{granularity}.csv", index_col=0)
    else:
        raise ValueError("Invalid type. Must be one of 'fs', 'topic', or 'style'.")
    return X_train, X_test


for granularity in [5, 12, 25]:
    X_train, X_test = load_classification_features("fs", granularity)
    experiment = LyricsClassificationExperiment(
        X_train,
        train_metadata[f"cat{granularity}"],
        X_test,
        test_metadata[f"cat{granularity}"],
        f"models/classificator_fs_lr_G{granularity}",
        subsample_debug=1.0,
    )
    experiment.train_fixed_parametrer_logistic_regression()
    experiment.save_model_evaluation_txt()
    experiment.save_experiment()
