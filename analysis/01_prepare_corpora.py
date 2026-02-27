import pandas as pd
from helpers.generate_report import generate_report
from helpers.CorpusProcessor import CorpusLemmatizer

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
    & (corpus["cat12"] != "schlager")  # German Genre
]
english.to_csv("data/poptrag_lyrics_genres_corpus_filtered_english.csv", index=True)


processor = CorpusLemmatizer(english, lyrics_column="full_lyrics")
processor.lemmatize()
processor.save_lemmatized(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)


print("GENERATE REPORT...")
generate_report(
    notebook_path="notebooks/reporting/01_prepare_corpora.ipynb", output_path="reports"
)

print("DONE.")
