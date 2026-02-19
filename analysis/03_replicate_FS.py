import pandas as pd
import numpy as np
from helpers.LyricsClassficationExperiment import LyricsClassificationExperiment

MIN_ARTISTS = 50  # minimum number of artists required to include an ngram
TOP_N = 100  # number of top unigrams, bigrams, and trigrams to keep per genre
GRANULARITIES = [5, 12, 25, 32]  # n-gram granularities to process
PARAM_SPACE = {
    "C": [np.log10(0.001), np.log10(100.0)],
    "l1_ratio": [0.0, 1.0],
    # "target_ratio": [np.log10(1), np.log10(5)], # adaptive sampling will only be included if in parameter space; always class weighted loss
}

english = pd.read_csv("data/poptrag_lyrics_genres_corpus_filtered_english.csv")


def run_experiment(corpus, granularity):
    exp = LyricsClassificationExperiment(
        corpus=corpus,
        genrecol=f"cat{granularity}",
        lyricscol="full_lyrics",
        artistcol="track.s.firstartist.name",
        output_dir=f"models/cat{granularity}_FS_only_ngram_experiment",
        test_size=0.2,
        random_state=42,
    )
    exp.compute_fs_ngram_features(min_artists=MIN_ARTISTS, top_n=TOP_N)
    exp.tune_and_train_logistic_regression(
        param_space=PARAM_SPACE, cv=5, n_initial=10, n_iterations=25, n_jobs=50
    )
    exp.save_experiment()


for granularity in GRANULARITIES:
    run_experiment(english, granularity)
