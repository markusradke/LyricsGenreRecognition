import pandas as pd
from scipy import sparse

from helpers.STMTopicModeler import STMTopicModeler
from helpers.aggregate_artist_dtm import aggregate_dtm_by_artist

K_RANGE = (2, 20)

X_train_metadata = pd.read_csv("data/X_train_metadata.csv")
artists = X_train_metadata["track.s.firstartist.name"]

topics_vocab = pd.read_csv("data/monroe_topic_vocabulary.csv").to_numpy().flatten()
style_vocab = pd.read_csv("data/monroe_style_vocabulary.csv").to_numpy().flatten()

X_train_topics_monroe_full = sparse.load_npz("data/X_train_topics_monroe_full.npz")
X_test_topics_monroe_full = sparse.load_npz("data/X_test_topics_monroe_full.npz")

X_train_style_monroe_full = sparse.load_npz("data/X_train_style_monroe_full.npz")
X_test_style_monroe_full = sparse.load_npz("data/X_test_style_monroe_full.npz")

for granularity in [5, 12, 25]:
    print(f"FITTING STM TOPIC MODEL WITH GENRE GRANULARITY {granularity}...")
    genres = X_train_metadata[f"cat{granularity}"]

    X_train_topics_artist_agg, genres_topics_agg = aggregate_dtm_by_artist(
        X_train_topics_monroe_full, artists, genres
    )
    topic_modeler = STMTopicModeler(
        k_range=K_RANGE,
        use_genre_prevalence=True,
        random_state=42,
        model_dir=f"models/stm_topics_G{granularity}",
    )
    topic_modeler.tune_and_fit(
        X_train_topics_artist_agg, genres_topics_agg, topics_vocab
    )
    X_train_topics = topic_modeler.transform(X_train_topics_monroe_full, topics_vocab)
    X_test_topics = topic_modeler.transform(X_test_topics_monroe_full, topics_vocab)
    pd.DataFrame(X_train_topics).to_csv(
        f"data/X_train_topics_G{granularity}.csv", index=False
    )
    pd.DataFrame(X_test_topics).to_csv(
        f"data/X_test_topics_G{granularity}.csv", index=False
    )

    print(f"FITTING STM STYLE MODEL WITH GENRE GRANULARITY {granularity}...")
    X_train_style_monroe_full_artist_agg, genres_style_agg = aggregate_dtm_by_artist(
        X_train_style_monroe_full, artists, genres
    )
    style_modeler = STMTopicModeler(
        k_range=K_RANGE,
        use_genre_prevalence=True,
        random_state=42,
        model_dir=f"models/stm_style_G{granularity}",
    )
    style_modeler.tune_and_fit(
        X_train_style_monroe_full_artist_agg, genres_style_agg, style_vocab
    )
    X_train_style = style_modeler.transform(X_train_style_monroe_full, style_vocab)
    X_test_style = style_modeler.transform(X_test_style_monroe_full, style_vocab)
    pd.DataFrame(X_train_style).to_csv(
        f"data/X_train_style_G{granularity}.csv", index=False
    )
    pd.DataFrame(X_test_style).to_csv(
        f"data/X_test_style_G{granularity}.csv", index=False
    )
