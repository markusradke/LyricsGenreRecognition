import pandas as pd
import copy
from helpers.LyricsClassficationExperiment import LyricsClassificationExperiment

english = pd.read_csv(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)


def prototype_granularity_experiment(
    granularity, sample_size=1.0, train_raw_monroe=False
):
    if granularity not in [5, 12, 25, 32]:
        raise ValueError("Granularity must be in 5, 12, 25 or 32.")

    experiment_fs = LyricsClassificationExperiment(
        corpus=english,
        genrecol=f"cat{granularity}",
        lyricscol="full_lyrics",
        artistcol="track.s.firstartist.name",
        random_state=42,
        subsample_debug=sample_size,
        output_dir=f"notebooks/prototyping/experiment_outputs/FS_Extractor_Test_G{granularity}",
    )
    experiment_fs.compute_fs_ngram_features()
    experiment_fs.train_fixed_parametrer_logistic_regression()
    experiment_fs.save_model_evaluation_txt()
    experiment_fs.save_experiment()

    experiment_monroe = LyricsClassificationExperiment(
        corpus=english,
        genrecol=f"cat{granularity}",
        lyricscol="lyrics_lemmatized",
        artistcol="track.s.firstartist.name",
        random_state=42,
        subsample_debug=sample_size,
        output_dir=f"notebooks/prototyping/experiment_outputs/Monroe_Extractor_Test_G{granularity}",
    )
    experiment_monroe.compute_monroe_ngram_features(
        use_stopword_filter=True,
        use_bigram_boundary_filter=True,
        include_unigrams=True,
        prior_concentration=1.0,
        p_value=0.001,
    )
    if train_raw_monroe:
        experiment_monroe.train_fixed_parametrer_logistic_regression()
        experiment_monroe.save_model_evaluation_txt()
        experiment_monroe.save_experiment()

    experiment_stm = copy.deepcopy(experiment_monroe)
    experiment_stm.output_dir = (
        f"notebooks/prototyping/experiment_outputs/STM_Test_G{granularity}"
    )
    experiment_stm.compute_stm_topic_features(k_range=(2, 3))  # CHANGE TO MORE

    experiment_stm.train_fixed_parametrer_logistic_regression()
    experiment_stm.save_model_evaluation_txt()
    experiment_stm.save_experiment()


if __name__ == "__main__":
    prototype_granularity_experiment(
        granularity=5, sample_size=1.0, train_raw_monroe=False
    )
    # prototype_granularity_experiment(
    #     granularity=12, sample_size=0.02, train_raw_monroe=False
    # )
    prototype_granularity_experiment(
        granularity=25, sample_size=1.0, train_raw_monroe=False
    )
