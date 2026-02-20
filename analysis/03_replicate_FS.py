from scipy.constants import year
import pandas as pd
import numpy as np
from helpers.LyricsClassficationExperiment import LyricsClassificationExperiment
from helpers.config import (
    MIN_ARTISTS,
    TOP_VOCAB_PER_GENRE,
    GENRE_CATEGORIES,
    N_BAYESIAN_ITER,
    N_BAYESIAN_INITIAL,
    STOP_ITER,
    UNCERTAIN_JUMP,
    CV_FOLDS,
    N_JOBS,
    FS_PARAM_SPACE,
    get_genre_column,
    RANDOM_SEED,
    TEST_SIZE,
)

english = pd.read_csv("data/poptrag_lyrics_genres_corpus_filtered_english.csv")


def run_experiment(corpus, granularity):
    exp = LyricsClassificationExperiment(
        corpus=corpus,
        genrecol=get_genre_column(granularity),
        lyricscol="full_lyrics",
        artistcol="track.s.firstartist.name",
        yearcol="album.s.releaseyear",
        output_dir=f"models/cat{granularity}_FS_only_ngram_experiment",
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    exp.compute_fs_ngram_features_pipeline(
        min_artists=MIN_ARTISTS, top_vocab_per_genre=TOP_VOCAB_PER_GENRE
    )
    exp.tune_and_train_logistic_regression(
        param_space=FS_PARAM_SPACE,
        cv=CV_FOLDS,
        n_initial=N_BAYESIAN_INITIAL,
        n_iterations=N_BAYESIAN_ITER,
        stop_iter=STOP_ITER,
        uncertain_jump=UNCERTAIN_JUMP,
        n_jobs=N_JOBS,
        use_pipeline=True,
    )
    exp.save_experiment()


for granularity in GENRE_CATEGORIES:
    run_experiment(english, granularity)
