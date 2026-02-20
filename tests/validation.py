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


if __name__ == "__main__":
    validate_artist_split_on_real_data()
    validate_feature_cache()
