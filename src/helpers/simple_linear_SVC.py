import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay


def perform_linear_SVC(
    features: pd.DataFrame, labels_and_artists: pd.DataFrame, granularity: int
) -> tuple[OneVsOneClassifier, pd.Series, pd.Series]:
    """
    Train a One-vs-One Linear SVM classifier for genre classification
    with penalty C = 1.0 following the methodology of Fell & Sporleder (2014)
    without aggregating over multiple models.

    Additionally implements artist-stratified train/test splitting to prevent artist bias.
    Features are
    standardized before training.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    labels_and_artists : pd.DataFrame
        DataFrame containing at least two columns:
        - 'cat{granularity}': genre labels for classification
        - 'track.s.firstartist.name': artist identifiers for stratification
    granularity : int
        Genre granularity level (used for model and output file naming).

    Returns
    -------
    tuple[OneVsOneClassifier, pd.Series, pd.Series]
        Trained model, test predictions, and test labels.

    Notes
    -----
    - Ensures no artist overlap between train and test sets
    - Saves model, predictions, and test labels to disk
    - Displays genre distribution histograms and confusion matrix
    """
    X = features
    y_full = labels_and_artists

    # Create artist-level splits to avoid artist effects, stratified by most common artist genre
    artist_genres = y_full.groupby("track.s.firstartist.name")[f"cat{granularity}"].agg(
        lambda x: x.value_counts().index[0]
    )
    artists_train, artists_test = train_test_split(
        artist_genres.index,
        test_size=0.2,
        stratify=artist_genres.values,
        random_state=42,
    )

    train_mask = y_full["track.s.firstartist.name"].isin(artists_train)
    test_mask = y_full["track.s.firstartist.name"].isin(artists_test)

    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y_full.loc[train_mask, "cat12"].reset_index(drop=True)
    y_test = y_full.loc[test_mask, "cat12"].reset_index(drop=True)

    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])
    print(
        "Number of unique artists in train:",
        y_full.loc[train_mask, "track.s.firstartist.name"].nunique(),
    )
    print(
        "Number of unique artists in test:",
        y_full.loc[test_mask, "track.s.firstartist.name"].nunique(),
    )
    print(
        "Artist overlap (should be 0):",
        len(
            set(y_full.loc[train_mask, "track.s.firstartist.name"]).intersection(
                set(y_full.loc[test_mask, "track.s.firstartist.name"])
            )
        ),
    )

    # Plot histogram of genre distribution in train and test
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    y_train.value_counts().plot(
        kind="bar", ax=ax[0], title="Training set genre distribution"
    )
    y_test.value_counts().plot(
        kind="bar", ax=ax[1], title="Test set genre distribution"
    )

    # Normalize features (-mean, scale to unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fell + Sporleder 2014 also use a linear SVM with OVO classification, takes 4-5 min to train on 20% subsample
    # Fell + Sporleder repeat 100-1000 times with different random train/test splits, we do it once here
    ovo_clf = OneVsOneClassifier(LinearSVC(C=1.0, random_state=42, max_iter=10000))
    ovo_clf.fit(X_train_scaled, y_train)
    y_pred = ovo_clf.predict(X_test_scaled)
    y_pred = pd.Series(y_pred)

    # Save model artifacts
    y_test.to_csv(
        f"../models/fell_spohrleder_svm/english_cat{granularity}_svm_ovo_test_labels.csv",
        index=False,
    )
    y_pred.to_csv(
        f"../models/fell_spohrleder_svm/english_cat{granularity}_svm_ovo_predictions.csv",
        index=False,
    )
    with open(
        f"../models/fell_spohrleder_svm/english_cat{granularity}_svm_ovo_model.pkl",
        "wb",
    ) as f:
        pickle.dump(ovo_clf, f)

    # Evaluate and display results
    print(
        f"Macro F1 for cat{granularity}: {f1_score(y_test, y_pred, average='macro'):.3f}"
    )
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        normalize="true",
        values_format=".2f",
        cmap=plt.cm.Greens,
        xticks_rotation="vertical",
    )

    return ovo_clf, y_pred, y_test
