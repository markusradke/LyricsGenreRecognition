import pandas as pd
import numpy as np
from helpers.LyricsClassficationExperiment import LyricsClassificationExperiment

from helpers.config import (
    MIN_ARTISTS, 
    GENRE_CATEGORIES,
    N_BAYESIAN_ITER,
    N_BAYESIAN_INITIAL,
    STOP_ITER,
    UNCERTAIN_JUMP,
    CV_FOLDS,
    N_JOBS,
    MONROE_PARAM_SPACE_3D,
    MONROE_PARAM_SPACE_6D,
    get_genre_column,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.helpers.config import N_JOBS


print("Loading English corpus...")
english = pd.read_csv(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)

print("Running informed n-gram experiments...")


def run_experiment(corpus, granularity):
    exp = LyricsClassificationExperiment(
        corpus=corpus,
        genrecol=get_genre_column(granularity),
        lyricscol="lyrics_lemmatized",
        artistcol="track.s.firstartist.name",
        output_dir=f"models/cat{granularity}_informed_ngram_experiment",
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    exp.compute_informed_ngram_features(min_artists=MIN_ARTISTS)
    exp.tune_and_train_logistic_regression(
        param_space=MONROE_PARAM_SPACE_3D, 
        cv=CV_FOLDS,
        n_initial=N_BAYESIAN_INITIAL,
        n_iterations=N_BAYESIAN_ITER,
        stop_iter=STOP_ITER,
        uncertain_jump=UNCERTAIN_JUMP,
        n_jobs=N_JOBS
    )
    exp.save_experiment()


for granularity in GENRE_CATEGORIES:
    run_experiment(english, granularity)
