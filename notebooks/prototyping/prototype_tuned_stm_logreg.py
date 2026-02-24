import pandas as pd
import copy
from helpers.LyricsClassficationExperiment import LyricsClassificationExperiment
from helpers.config import CLASSIFIER_PARAM_SPACE

english = pd.read_csv(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)

experiment_monroe = LyricsClassificationExperiment(
    corpus=english,
    genrecol="cat12",
    lyricscol="lyrics_lemmatized",
    artistcol="track.s.firstartist.name",
    random_state=42,
    subsample_debug=1.0,
    output_dir="notebooks/prototyping/experiment_outputs/Monroe_Extractor_Test",
)
experiment_monroe.compute_monroe_ngram_features(
    use_stopword_filter=True,
    use_bigram_boundary_filter=True,
    include_unigrams=True,
    prior_concentration=1.0,
    p_value=0.001,
)

X_train_new = pd.read_csv(
    "notebooks/prototyping/experiment_outputs/STM_Test/X_train_stm.csv"
)
X_test_new = pd.read_csv(
    "notebooks/prototyping/experiment_outputs/STM_Test/X_test_stm.csv"
)


experiment_stm = copy.deepcopy(experiment_monroe)
experiment_stm.output_dir = "notebooks/prototyping/experiment_outputs/STM_Test"
experiment_stm.X_train = X_train_new
experiment_stm.X_test = X_test_new
experiment_stm.tune_and_train_logistic_regression(
    param_space=CLASSIFIER_PARAM_SPACE,
    n_initial=2,
    n_iterations=6,
    n_points=2,
    n_jobs=-1,
    stop_iter=2,
    uncertain_jump=1,
)
experiment_stm.show_model_evaluation()
experiment_stm.show_top_coefficients_per_genre()
