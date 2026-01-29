import pandas as pd
import pickle
from helpers.interrater_agreement import monte_carlo_interrater_agreement_f1
from helpers.generate_report import generate_report

genre_counts = pd.read_csv("data-raw/poptrag_lyrics_genre_counts.csv")

N_MC = 10000
SEED = 42

print("ESTIMATING INTERRATER AGREEMENT (IRA) FOR 5 GENRES...")
cat5_mc = monte_carlo_interrater_agreement_f1(
    genre_counts, granularity=5, n_mc=N_MC, seed=SEED
)
with open("models/interrater_agreement/ira_cat5.pkl", mode="wb") as f:
    pickle.dump(cat5_mc, f)

print("ESTIMATING IRA FOR 12 GENRES...")
cat12_mc = monte_carlo_interrater_agreement_f1(
    genre_counts, granularity=12, n_mc=N_MC, seed=SEED
)
with open("models/interrater_agreement/ira_cat12.pkl", mode="wb") as f:
    pickle.dump(cat12_mc, f)


print("ESTIMATING IRA FOR 25 GENRES...")
cat25_mc = monte_carlo_interrater_agreement_f1(
    genre_counts, granularity=25, n_mc=N_MC, seed=SEED
)
with open("models/interrater_agreement/ira_cat25.pkl", mode="wb") as f:
    pickle.dump(cat25_mc, f)

print("ESTIMATING IRA FOR 32 GENRES...")
cat32_mc = monte_carlo_interrater_agreement_f1(
    genre_counts, granularity=32, n_mc=N_MC, seed=SEED
)
with open("models/interrater_agreement/ira_cat32.pkl", mode="wb") as f:
    pickle.dump(cat32_mc, f)
print("GENERATE REPORT...")


generate_report(
    notebook_path="notebooks/reporting/02_estimate_interrater_agreement.ipynb",
    output_path="reports",
)

print("DONE.")
