import pandas as pd
from helpers.interrater_agreement import monte_carlo_interrater_agreement_f1
from helpers.generate_report import generate_report

genre_counts = pd.read_csv("data-raw/poptrag_lyrics_genre_counts.csv")
english = pd.read_csv(
    "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)
id_in_english = set(english["track.s.id"])
genre_counts = genre_counts[genre_counts["track.s.id"].isin(id_in_english)]
genre_counts = genre_counts.loc[
    :, ~genre_counts.columns.str.contains("classical|schlager", case=False)
]


N_MC = 10000  # Number of Monte Carlo simulations
SEED = 42
GRANULARITIES = [5, 12, 25, 32]

for granularity in GRANULARITIES:
    print(f"ESTIMATING INTERRATER AGREEMENT (IRA) FOR {granularity} GENRES...")
    ira_result = monte_carlo_interrater_agreement_f1(
        genre_counts, granularity=granularity, n_mc=N_MC, seed=SEED
    )
    ira_result.to_csv(f"models/interrater_agreement/ira_cat{granularity}.csv")

print("DONE.")
