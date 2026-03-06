import pandas as pd


from helpers.CorpusLemmatizer import CorpusLemmatizer
from helpers.ensure_english_lyrics import (
    get_english_confidence,
    get_english_vocab_ratio,
)
from helpers.expand_contractions import expand_contractions, strip_apostrophe_with_s

print("READ IN FULL CORPUS CSV FILE...")
corpus = pd.read_csv(
    "data-raw/poptrag_lyrics_genres_corpus_20260118.csv", delimiter=","
)
cols_to_string = [
    c for c in corpus.columns if not (c.startswith("pmax") or c.startswith("nmax"))
]
corpus[cols_to_string] = corpus[cols_to_string].astype("string")

print(
    "FILTER FOR ENGLISH TRACKS WITH LYRICS AND GENRE INFORMATION AND SAVE TO CSV IN DATA FOLDER..."
)
english = corpus[
    (corpus["cat5"].notna())
    & (corpus["full_lyrics"].notna())
    & (corpus["track.language"].isin(["English"]))
    & (
        corpus["album.s.title"] != "No Grave but the Sea (Deluxe Edition)"
    )  # contains only "woof woof"
    & (corpus["cat32"] != "schlager")  # German Genre
    & (corpus["cat25"] != "schlager")  # German Genre
    & (corpus["cat12"] != "schlager")  # German Genre
    & (corpus["cat32"] != "classical")  # mostly non-english lyrics
    & (corpus["cat25"] != "classical")  # mostly non-english lyrics
]

english["english_conf"] = english["full_lyrics"].apply(get_english_confidence)
english["english_vocab_ratio"] = english["full_lyrics"].apply(get_english_vocab_ratio)
# thresholds were determined by manually checking tracks with low confidence and vocab ratio
filtered_english = english.query("english_conf > 0.75 and english_vocab_ratio > 0.75")

print(
    "EXPAND CONTRACTIONS, LEMMATIZE AND SAVE LEMMATIZED CORPUS TO CSV IN DATA FOLDER..."
)
filtered_english["lyrics_expanded"] = filtered_english["full_lyrics"].map(
    expand_contractions
)
filtered_english["lyrics_expanded"] = filtered_english["lyrics_expanded"].map(
    strip_apostrophe_with_s
)  # strip 's

processor = CorpusLemmatizer(filtered_english, lyrics_column="lyrics_expanded")
processor.lemmatize()
processor.save_lemmatized(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)

print("DONE.")
