import pandas as pd
import pickle
from helpers.interrater_agreement import monte_carlo_interrater_agreement_f1
from helpers.generate_report import generate_report

genre_counts = pd.read_csv("data-raw/poptrag_lyrics_genre_counts.csv")

N_MC = 10000
SEED = 42
GRANULARITIES = [5, 12, 25, 32]

for granularity in GRANULARITIES:
    print(f"ESTIMATING INTERRATER AGREEMENT (IRA) FOR {granularity} GENRES...")
    ira_result = monte_carlo_interrater_agreement_f1(
        genre_counts, granularity=granularity, n_mc=N_MC, seed=SEED
    )
    with open(f"models/interrater_agreement/ira_cat{granularity}.pkl", mode="wb") as f:
        pickle.dump(ira_result, f)

print("GENERATE REPORT...")
generate_report(
    notebook_path="notebooks/reporting/02_estimate_interrater_agreement.ipynb",
    output_path="reports",
)

print("DONE.")
