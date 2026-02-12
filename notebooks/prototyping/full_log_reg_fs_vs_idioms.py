import pandas as pd
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from helpers.LyricsClassficationExperiment import (
    LyricsClassificationExperiment,
)


def load_corpus() -> pd.DataFrame:
    return pd.read_csv(
        "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
    )


def run_fs_experiment(full: pd.DataFrame) -> LyricsClassificationExperiment:
    exp = LyricsClassificationExperiment(
        corpus=full,
        genrecol="cat12",
        lyricscol="full_lyrics",
        artistcol="track.s.firstartist.name",
        output_dir="notebooks/prototyping/cat12_experiment_fs",
        test_size=0.2,
        random_state=42,
    )
    exp.compute_fs_ngram_features(min_artists=20, top_n_per_genre_and_ngram=100)
    exp.train_fixed_parametrer_logistic_regression()
    exp.save_experiment()
    return exp


def run_idiom_experiment_strict(
    full: pd.DataFrame,
) -> LyricsClassificationExperiment:
    exp = LyricsClassificationExperiment(
        corpus=full,
        genrecol="cat12",
        lyricscol="lyrics_lemmatized",
        artistcol="track.s.firstartist.name",
        output_dir="notebooks/prototyping/cat12_experiment_idioms_strict",
        test_size=0.2,
        random_state=42,
    )
    exp.compute_idiom_ngram_features(
        min_artists=50, min_tracks=100, llr_threshold=10, top_n_per_ngram_pergenre=100
    )
    exp.train_fixed_parametrer_logistic_regression()
    exp.save_experiment()
    return exp


def run_idiom_experiment_relaxed(
    full: pd.DataFrame,
) -> LyricsClassificationExperiment:
    exp = LyricsClassificationExperiment(
        corpus=full,
        genrecol="cat12",
        lyricscol="lyrics_lemmatized",
        artistcol="track.s.firstartist.name",
        output_dir="notebooks/prototyping/cat12_experiment_idioms_relaxed",
        test_size=0.2,
        random_state=42,
    )
    exp.compute_idiom_ngram_features(
        min_artists=50, min_tracks=0, llr_threshold=10, top_n_per_ngram_pergenre=100
    )
    exp.train_fixed_parametrer_logistic_regression()
    exp.save_experiment()
    return exp


def run_all_experiments(full: pd.DataFrame) -> dict:
    with ProcessPoolExecutor(max_workers=3) as executor:
        fs_future = executor.submit(run_fs_experiment, full)
        idiom_strict_future = executor.submit(run_idiom_experiment_strict, full)
        idiom_relaxed_future = executor.submit(run_idiom_experiment_relaxed, full)

        return {
            "fs": fs_future.result(),
            "idiom_strict": idiom_strict_future.result(),
            "idiom_relaxed": idiom_relaxed_future.result(),
        }


def make_output_directories():
    os.makedirs("notebooks/prototyping/cat12_experiment_fs", exist_ok=True)
    os.makedirs("notebooks/prototyping/cat12_experiment_idioms_strict", exist_ok=True)
    os.makedirs("notebooks/prototyping/cat12_experiment_idioms_relaxed", exist_ok=True)


def main() -> None:
    make_output_directories()

    full = load_corpus()
    results = run_all_experiments(full)

    for name, exp in [
        ("FS", results["fs"]),
        ("Idioms (Strict)", results["idiom_strict"]),
        ("Idioms (Relaxed)", results["idiom_relaxed"]),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Results: {name}")
        print("=" * 60)
        exp.show_model_evaluation()


def read_models_from_files_and_display_results():
    for name, dir in [
        ("FS", "notebooks/prototyping/cat12_experiment_fs"),
        ("Idioms (Strict)", "notebooks/prototyping/cat12_experiment_idioms_strict"),
        (
            "Idioms (Relaxed)",
            "notebooks/prototyping/cat12_experiment_idioms_relaxed",
        ),
    ]:
        with open(os.path.join(dir, "model.pkl"), "rb") as f:
            exp = pickle.load(f)
            print(f"\n{'=' * 60}")
            print(f"Results: {name}")
            print("=" * 60)
            exp.show_model_evaluation()


if __name__ == "__main__":
    main()
