import pandas as pd
from helpers.n_gram_features import build_ngram_features

MIN_ARTISTS = 50  # minimum number of artists required to include an ngram
TOP_N = 100  # number of top unigrams, bigrams, and trigrams to keep per genre
GRANULARITIES = [5, 12, 25, 32]  # n-gram granularities to process

english = pd.read_csv("data/poptrag_lyrics_genres_corpus_filtered_english.csv")

for granularity in GRANULARITIES:
    print(f"BUILD N-GRAM FEATURES FOR GRANULARITY {granularity}...")
    print("=" * 60)
    build_ngram_features(
        corpus=english, granularity=granularity, min_artists=MIN_ARTISTS, top_n=TOP_N
    )
