"""Run this after any data pipeline changes."""

import pandas as pd
from helpers.split_group_stratified_and_join import split_group_stratified_and_join


def validate_artist_split_on_real_data():
    corpus = pd.read_csv("data/poptrag_lyrics_genres_corpus_filtered_english.csv")
    corpus_sample = corpus.sample(1000, random_state=42)

    labels_and_group = corpus_sample[["track.s.firstartist.name", "cat12"]].rename(
        columns={"track.s.firstartist.name": "group", "cat12": "label"}
    )
    train, test, _, _ = split_group_stratified_and_join(labels_and_group, corpus_sample)

    train_artists = set(train["track.s.firstartist.name"])
    test_artists = set(test["track.s.firstartist.name"])
    overlap = train_artists & test_artists

    assert len(overlap) == 0, f"FAILED: {len(overlap)} artists overlap!"
    print(
        f"PASSED: {len(train_artists)} train, {len(test_artists)} test artists (disjoint)"
    )


def validate_feature_cache():
    """Test that feature caching works correctly."""
    from helpers.FeatureCache import FeatureCache
    from helpers.NGramFeatureExctractorFS import NGramFeatureExtractorFS
    import time
    import shutil

    test_cache_dir = "tests/test_cache"
    cache = FeatureCache(cache_dir=test_cache_dir)

    corpus = pd.DataFrame(
        {
            "lyrics": ["test song one", "test song two"],
            "artist": ["artist1", "artist2"],
            "genre": ["genre1", "genre2"],
        }
    )

    start = time.time()
    result1 = cache.cache_features(NGramFeatureExtractorFS, corpus, min_artists=1)
    time1 = time.time() - start

    start = time.time()
    result2 = cache.cache_features(NGramFeatureExtractorFS, corpus, min_artists=1)
    time2 = time.time() - start

    assert result1["features"].equals(
        result2["features"]
    ), "FAILED: Cached features differ!"
    assert time2 < time1, "FAILED: Cache not faster than first computation!"

    shutil.rmtree(test_cache_dir)

    print(f"PASSED: Feature caching (first: {time1:.3f}s, cached: {time2:.4f}s)")


def validate_fs_extractor():
    """Test FSExtractor sklearn API compliance and caching."""
    from helpers.FSExtractor import FSExtractor
    from scipy.sparse import issparse
    import numpy as np

    # Create test corpus with multiple genres and artists
    train_corpus = pd.DataFrame(
        {
            "lyrics": [
                "love you baby love",
                "rock and roll music",
                "baby love you so",
                "rock music forever",
                "dance all night long",
                "dance music baby",
            ],
            "genre": ["pop", "rock", "pop", "rock", "dance", "dance"],
            "artist": [
                "artist1",
                "artist2",
                "artist3",
                "artist2",
                "artist4",
                "artist5",
            ],
        }
    )

    test_corpus = pd.DataFrame(
        {
            "lyrics": ["love rock baby", "dance forever"],
            "genre": ["pop", "dance"],
            "artist": ["artist6", "artist7"],
        }
    )

    # Test 1: Basic fit/transform
    extractor = FSExtractor(min_artists=1, top_vocab_per_genre=5, random_state=42)

    # Fit on training data
    extractor.fit(
        train_corpus["lyrics"], train_corpus["genre"], artist=train_corpus["artist"]
    )

    assert hasattr(extractor, "vocabulary_"), "FAILED: No vocabulary_ after fit!"
    assert hasattr(extractor, "vectorizer_"), "FAILED: No vectorizer_ after fit!"
    assert extractor._is_fitted, "FAILED: _is_fitted not set!"
    assert len(extractor.vocabulary_) > 0, "FAILED: Empty vocabulary!"

    # Transform training data
    X_train = extractor.transform(train_corpus["lyrics"])
    assert issparse(X_train), "FAILED: Output not sparse matrix!"
    assert X_train.shape[0] == len(train_corpus), "FAILED: Wrong number of rows!"
    assert X_train.shape[1] == len(
        extractor.vocabulary_
    ), "FAILED: Wrong number of features!"

    # Transform test data
    X_test = extractor.transform(test_corpus["lyrics"])
    assert X_test.shape[0] == len(test_corpus), "FAILED: Wrong test shape!"
    assert X_test.shape[1] == X_train.shape[1], "FAILED: Test/train feature mismatch!"

    # Test 2: get_feature_names_out
    feature_names = extractor.get_feature_names_out()
    assert len(feature_names) == len(
        extractor.vocabulary_
    ), "FAILED: Feature names length mismatch!"

    # Test 3: Caching behavior - refit with same data should reuse vocabulary
    extractor2 = FSExtractor(min_artists=1, top_vocab_per_genre=5, random_state=42)
    extractor2.fit(
        train_corpus["lyrics"], train_corpus["genre"], artist=train_corpus["artist"]
    )

    assert (
        extractor2.vocabulary_ == extractor.vocabulary_
    ), "FAILED: Cached vocabulary differs!"

    # Test 4: Transform without fit should raise error
    extractor3 = FSExtractor()
    try:
        extractor3.transform(test_corpus["lyrics"])
        assert False, "FAILED: Should raise ValueError before fit!"
    except ValueError as e:
        assert "Must call fit()" in str(e), "FAILED: Wrong error message!"

    print(
        f"PASSED: FSExtractor ({len(extractor.vocabulary_)} features, {X_train.shape[0]} train, {X_test.shape[0]} test)"
    )


def validate_monroe_extractor():
    """Test MonroeExtractor sklearn API and Monroe scoring."""
    from helpers.MonroeExtractor import MonroeExtractor
    from scipy.sparse import issparse

    # Create test corpus with genre-specific phrases
    train_corpus = pd.DataFrame(
        {
            "lyrics": [
                "love you baby love love baby",
                "rock hard rock guitar heavy metal",
                "baby love you forever baby",
                "rock guitar heavy rock",
                "dance all night dance floor music",
                "dance music dance beat",
            ],
            "genre": ["pop", "rock", "pop", "rock", "dance", "dance"],
            "artist": [
                "artist1",
                "artist2",
                "artist3",
                "artist2",
                "artist4",
                "artist5",
            ],
        }
    )

    test_corpus = pd.DataFrame(
        {
            "lyrics": ["love baby rock", "dance forever"],
            "genre": ["pop", "dance"],
            "artist": ["artist6", "artist7"],
        }
    )

    # Test with low thresholds to ensure some features pass
    extractor = MonroeExtractor(
        min_artists=1,
        alpha_unigram=0.1,
        alpha_bigram=0.1,
        alpha_trigram=0.1,
        alpha_quadgram=0.1,
        z_threshold=1.0,  # Permissive for test
        apply_fdr=False,  # Disable FDR for simpler test
        random_state=42,
    )

    # Fit
    extractor.fit(
        train_corpus["lyrics"], train_corpus["genre"], artist=train_corpus["artist"]
    )

    assert hasattr(extractor, "vocabulary_"), "FAILED: No vocabulary_!"
    assert extractor._is_fitted, "FAILED: Not fitted!"
    assert len(extractor.vocabulary_) > 0, "FAILED: Empty vocabulary!"

    # Transform
    X_train = extractor.transform(train_corpus["lyrics"])
    assert issparse(X_train), "FAILED: Output not sparse!"
    assert X_train.shape[0] == len(train_corpus), "FAILED: Wrong train rows!"

    X_test = extractor.transform(test_corpus["lyrics"])
    assert X_test.shape[1] == X_train.shape[1], "FAILED: Feature mismatch!"

    # Feature names
    feature_names = extractor.get_feature_names_out()
    assert len(feature_names) == len(
        extractor.vocabulary_
    ), "FAILED: Feature names mismatch!"

    print(
        f"PASSED: MonroeExtractor ({len(extractor.vocabulary_)} features, FDR={'on' if extractor.apply_fdr else 'off'})"
    )


def validate_fdr_correction():
    """Test Benjamini-Hochberg FDR correction reduces false positives."""
    from helpers import monroe_logodds
    import numpy as np
    from scipy.stats import norm

    # Simulate multiple testing scenario
    np.random.seed(42)
    n_ngrams = 1000
    n_genres = 5

    # Create z-scores: 90% null (random noise), 10% true signals
    z_scores = np.zeros((n_ngrams, n_genres))

    # True signals: high z-scores in specific (ngram, genre) pairs
    n_true_signals = int(0.1 * n_ngrams * n_genres)
    true_signal_indices = np.random.choice(
        n_ngrams * n_genres, size=n_true_signals, replace=False
    )

    for idx in true_signal_indices:
        i, j = divmod(idx, n_genres)
        z_scores[i, j] = np.random.uniform(3.0, 6.0)  # Strong signals

    # Null hypothesis: random noise around 0
    null_mask = np.ones_like(z_scores, dtype=bool)
    for idx in true_signal_indices:
        i, j = divmod(idx, n_genres)
        null_mask[i, j] = False
    z_scores[null_mask] = np.random.normal(0, 1, size=null_mask.sum())

    # Test 1: Uncorrected threshold
    uncorrected_selected = (z_scores > 2.326).sum()

    # Test 2: FDR correction
    fdr_mask = monroe_logodds.apply_fdr_correction(z_scores, fdr_level=0.01)
    fdr_selected = fdr_mask.sum()

    # Expected false positives without correction: ~1% * (90% * total tests)
    n_null_tests = null_mask.sum()
    expected_fp_uncorrected = 0.01 * n_null_tests

    # Compute actual false positives
    uncorrected_fp = ((z_scores > 2.326) & null_mask).sum()
    fdr_fp = (fdr_mask & null_mask).sum()

    # FDR should select fewer features overall
    assert (
        fdr_selected <= uncorrected_selected
    ), f"FAILED: FDR selected more ({fdr_selected}) than uncorrected ({uncorrected_selected})!"

    # FDR should have lower false positive rate
    uncorrected_fdr_rate = uncorrected_fp / max(uncorrected_selected, 1)
    actual_fdr_rate = fdr_fp / max(fdr_selected, 1)

    # Allow some tolerance due to randomness, but FDR should be better
    assert (
        actual_fdr_rate <= uncorrected_fdr_rate + 0.05
    ), f"FAILED: FDR worse ({actual_fdr_rate:.3f}) than uncorrected ({uncorrected_fdr_rate:.3f})!"

    # FDR should control false discovery rate near target (0.01)
    # Due to finite sample effects, allow up to 0.05
    assert (
        actual_fdr_rate <= 0.05
    ), f"FAILED: FDR rate ({actual_fdr_rate:.3f}) exceeds 0.05!"

    print(
        f"PASSED: FDR correction "
        f"(uncorrected: {uncorrected_selected} selected, {uncorrected_fdr_rate:.3f} FDR; "
        f"corrected: {fdr_selected} selected, {actual_fdr_rate:.3f} FDR)"
    )


if __name__ == "__main__":
    validate_artist_split_on_real_data()
    validate_feature_cache()
    validate_fs_extractor()
    validate_monroe_extractor()
    validate_fdr_correction()
