import pandas as pd
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
    get_genre_column,
    RANDOM_SEED,
    TEST_SIZE,
)

english = pd.read_csv(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)


def run_experiment(corpus, granularity):
    exp = LyricsClassificationExperiment(
        corpus=corpus,
        genrecol=get_genre_column(granularity),
        lyricscol="lyrics_lemmatized",
        artistcol="track.s.firstartist.name",
        yearcol="album.s.releaseyear",
        output_dir=f"models/cat{granularity}_monroe_3d_experiment",
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    exp.compute_monroe_ngram_features(min_artists=MIN_ARTISTS, mode="3D")
    exp.tune_and_train_logistic_regression(
        param_space=MONROE_PARAM_SPACE_3D,
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
