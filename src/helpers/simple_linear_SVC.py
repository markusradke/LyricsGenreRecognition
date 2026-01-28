import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, ConfusionMatrixDisplay


def perform_linear_svc(
    features: pd.DataFrame,
    labels_and_artists: pd.DataFrame,
    granularity: int,
    subsample: float = 1.0,
    output_dir: str = "../models/fell_spohrleder_svm",
    seed: int = 42,
) -> tuple[OneVsOneClassifier, pd.Series, pd.Series]:
    """
    Train One-vs-One Linear SVM for genre classification.

    Uses artist-stratified train/test split to prevent artist bias.
    Features are standardized before training.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    labels_and_artists : pd.DataFrame
        DataFrame with 'cat{granularity}' and
        'track.s.firstartist.name' columns.
    granularity : int
        Genre granularity level for model naming.
    subsample : float, optional
        Fraction of data to use (default 1.0).
    output_dir : str, optional
        Directory for saving model artifacts.
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    tuple[OneVsOneClassifier, pd.Series, pd.Series]
        Trained model, predictions, and test labels.
    """
    artists_train, artists_test = _split_by_artist(
        labels_and_artists, granularity, seed
    )
    train_mask, test_mask = _create_train_test_masks(
        labels_and_artists, artists_train, artists_test
    )

    X_train, X_test, y_train, y_test = _split_features_labels(
        features, labels_and_artists, granularity, train_mask, test_mask
    )

    X_train, X_test, y_train, y_test = _subsample_data(
        X_train, X_test, y_train, y_test, granularity, subsample, seed
    )

    _plot_genre_distribution(y_train, y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ovo_clf = OneVsOneClassifier(
        LinearSVC(C=1.0, random_state=seed, max_iter=10000), jobs=-3  # all but 2 cores
    )
    ovo_clf.fit(X_train_scaled, y_train)
    y_pred = pd.Series(ovo_clf.predict(X_test_scaled))

    _save_model_artifacts(y_test, y_pred, ovo_clf, granularity, output_dir)

    macro_f1 = f1_score(y_test, y_pred, average="macro")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        normalize="true",
        values_format=".2f",
        cmap=plt.cm.Greens,
        xticks_rotation="vertical",
    )

    return ovo_clf, y_pred, y_test, macro_f1


def _get_artist_genres(labels_and_artists, granularity):
    return labels_and_artists.groupby("track.s.firstartist.name")[
        f"cat{granularity}"
    ].agg(lambda x: x.value_counts().index[0])


def _split_by_artist(labels_and_artists, granularity, seed):
    artist_genres = _get_artist_genres(labels_and_artists, granularity)
    artists_train, artists_test = train_test_split(
        artist_genres.index,
        test_size=0.2,
        stratify=artist_genres.values,
        random_state=seed,
    )
    return artists_train, artists_test


def _create_train_test_masks(labels_and_artists, artists_train, artists_test):
    train_mask = labels_and_artists["track.s.firstartist.name"].isin(artists_train)
    test_mask = labels_and_artists["track.s.firstartist.name"].isin(artists_test)
    return train_mask, test_mask


def _split_features_labels(
    features, labels_and_artists, granularity, train_mask, test_mask
):
    X_train = features[train_mask].reset_index(drop=True)
    X_test = features[test_mask].reset_index(drop=True)
    y_train = labels_and_artists.loc[train_mask, f"cat{granularity}"].reset_index(
        drop=True
    )
    y_test = labels_and_artists.loc[test_mask, f"cat{granularity}"].reset_index(
        drop=True
    )
    return X_train, X_test, y_train, y_test


def _subsample_data(X_train, X_test, y_train, y_test, granularity, subsample, seed):
    if subsample >= 1.0:
        return X_train, X_test, y_train, y_test

    Xy_train = pd.concat([X_train, y_train], axis=1)
    Xy_test = pd.concat([X_test, y_test], axis=1)

    Xy_train_sub = (
        Xy_train.groupby(f"cat{granularity}")
        .sample(frac=subsample, random_state=seed)
        .reset_index(drop=True)
    )
    Xy_test_sub = (
        Xy_test.groupby(f"cat{granularity}")
        .sample(frac=subsample, random_state=seed)
        .reset_index(drop=True)
    )

    return (
        Xy_train_sub.drop(columns=[f"cat{granularity}"]),
        Xy_test_sub.drop(columns=[f"cat{granularity}"]),
        Xy_train_sub[f"cat{granularity}"],
        Xy_test_sub[f"cat{granularity}"],
    )


def _plot_genre_distribution(y_train, y_test):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    y_train.value_counts().plot(
        kind="bar", ax=ax[0], title="Training set genre distribution"
    )
    y_test.value_counts().plot(
        kind="bar", ax=ax[1], title="Test set genre distribution"
    )


def _save_model_artifacts(y_test, y_pred, ovo_clf, granularity, output_dir):
    y_test.to_csv(
        f"{output_dir}/english_cat{granularity}_test_labels.csv",
        index=False,
    )
    y_pred.to_csv(
        f"{output_dir}/english_cat{granularity}_predictions.csv",
        index=False,
    )
    with open(f"{output_dir}/english_cat{granularity}_model.pkl", "wb") as f:
        pickle.dump(ovo_clf, f)
