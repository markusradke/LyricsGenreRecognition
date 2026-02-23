import pandas as pd
import copy
from helpers.LyricsClassficationExperiment import LyricsClassificationExperiment

english = pd.read_csv(
    "../../data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)

experiment_fs = LyricsClassificationExperiment(
    corpus=english,
    genrecol="cat12",
    lyricscol="full_lyrics",
    artistcol="track.s.firstartist.name",
    random_state=42,
    subsample_debug=0.005,
    output_dir="data/experiment_outputs/FS_Extractor_Test",
)
experiment_fs.compute_fs_ngram_features()
experiment_fs.train_fixed_parametrer_logistic_regression()
experiment_fs.save_experiment()
# experiment_fs.show_random_baseline_evaluation()
# experiment_fs.show_model_evaluation()
# experiment_fs.show_top_coefficients_per_genre()

experiment_monroe = LyricsClassificationExperiment(
    corpus=english,
    genrecol="cat12",
    lyricscol="lyrics_lemmatized",
    artistcol="track.s.firstartist.name",
    random_state=42,
    subsample_debug=0.02,
    output_dir="data/experiment_outputs/Monroe_Extractor_Test",
)
experiment_monroe.compute_monroe_ngram_features(
    use_stopword_filter=True,
    use_bigram_boundary_filter=True,
    include_unigrams=True,
    prior_concentration=1.0,
    p_value=0.001,
)

experiment_monroe.train_fixed_parametrer_logistic_regression()
experiment_monroe.save_experiment()
# experiment_monroe.show_random_baseline_evaluation()
# experiment_monroe.show_model_evaluation()
# experiment_monroe.show_top_coefficients_per_genre()

experiment_stm = copy.deepcopy(experiment_monroe)
experiment_stm.compute_stm_topic_features(k_range=(2, 3))

experiment_stm.train_fixed_parametrer_logistic_regression()
experiment_stm.save_experiment()
# experiment_stm.show_random_baseline_evaluation()
# experiment_stm.show_model_evaluation()
# experiment_stm.show_top_coefficients_per_genre()
