import pandas as pd

from scipy import sparse

from helpers.FSExtractor import FSExtractor
from helpers.MonroeExtractor import MonroeExtractor
from helpers.split_group_stratified_and_join import split_group_stratified_and_join


data = pd.read_csv("data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv")

labels_and_groups = data[["cat32", "track.s.firstartist.name"]].rename(
    columns={"cat32": "label", "track.s.firstartist.name": "group"}
)
X_train, X_test, y_train, y_test = split_group_stratified_and_join(
    labels_and_groups,
    data,
    test_size=0.2,
    random_state=42,
)


X_train_metadata = X_train[
    ["track.s.firstartist.name", "cat32", "cat25", "cat12", "cat5"]
]
X_train_metadata.to_csv("data/X_train_metadata.csv", index=False)
X_test_metadata = X_test[
    ["track.s.firstartist.name", "cat32", "cat25", "cat12", "cat5"]
]
X_test_metadata.to_csv("data/X_test_metadata.csv", index=False)

# fs_extractor = FSExtractor(
#     min_artists=20,
#     use_stopword_filter=False,
#     top_vocab_per_genre=100,
#     random_state=42,
#     checkpoint_dir="data/checkpoints/fs_extractor",
# )

# for granularity in [5, 12, 25, 32]:
#     fs_extractor.fit(
#         X_train["full_lyrics"],
#         X_train_metadata[f"cat{granularity}"],
#         X_train["track.s.firstartist.name"],
#     )
#     X_train_fs = fs_extractor.transform(X_train["full_lyrics"])
#     X_test_fs = fs_extractor.transform(X_test["full_lyrics"])
#     sparse.save_npz(f"data/X_train_fs_G{granularity}.npz", X_train_fs)
#     sparse.save_npz(f"data/X_test_fs_G{granularity}.npz", X_test_fs)

# topic_extractor = MonroeExtractor(
#     min_artists=20,
#     p_value=0.001,
#     prior_concentration=1.0,
#     use_stopword_filter=True,
#     ngram_types=(1,),
#     random_state=42,
#     checkpoint_dir="data/checkpoints/monroe_extractor_topic",
# )
# topic_extractor.fit(
#     X_train["lyrics_lemmatized"], y_train, X_train["track.s.firstartist.name"]
# )
# topic_vocab = pd.Series(topic_extractor.vocabulary_)
# topic_vocab.to_csv("data/monroe_topic_vocabulary.csv", index=False)

# X_train_topics_monroe_full = topic_extractor.transform(X_train["lyrics_lemmatized"])
# X_test_topics_monroe_full = topic_extractor.transform(X_test["lyrics_lemmatized"])

# sparse.save_npz("data/X_train_topics_monroe_full.npz", X_train_topics_monroe_full)
# sparse.save_npz("data/X_test_topics_monroe_full.npz", X_test_topics_monroe_full)

# style_extractor = MonroeExtractor(
#     min_artists=20,
#     p_value=0.001,
#     prior_concentration=1.0,
#     use_stopword_filter=True,
#     use_bigram_boundary_filter=True,
#     ngram_types=(1, 2, 3, 4),
#     random_state=42,
#     checkpoint_dir="data/checkpoints/monroe_extractor_style",
# )
# style_extractor.fit(
#     X_train["full_lyrics"], y_train, X_train["track.s.firstartist.name"]
# )
# style_vocab = pd.Series(style_extractor.vocabulary_)
# style_vocab.to_csv("data/monroe_style_vocabulary.csv", index=False)
# X_train_style_monroe_full = style_extractor.transform(X_train["full_lyrics"])
# X_test_style_monroe_full = style_extractor.transform(X_test["full_lyrics"])
# sparse.save_npz("data/X_train_style_monroe_full.npz", X_train_style_monroe_full)
# sparse.save_npz("data/X_test_style_monroe_full.npz", X_test_style_monroe_full)
