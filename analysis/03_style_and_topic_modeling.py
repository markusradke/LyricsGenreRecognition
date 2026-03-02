import pandas as pd
from scipy import sparse

from helpers.STMTopicModeler import STMTopicModeler
from helpers.aggregate_artist_dtm import aggregate_artist_dtm


fulldata = pd.read_csv(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)

artists = fulldata["track.s.firstartist.name"]

X_train_topics_monroe_full = sparse.load_npz("data/X_train_topics_monroe_full.npz")
X_test_topics_monroe_full = sparse.load_npz("data/X_test_topics_monroe_full.npz")

X_train_style_monroe_full = sparse.load_npz("data/X_train_style_monroe_full.npz")
X_test_style_monroe_full = sparse.load_npz("data/X_test_style_monroe_full.npz")

# for each of the three genres compute
