RANDOM_SEED = 42 # Reproducibility for all random operations
TEST_SIZE = 0.2 # Data split ratio for train/test sets

MIN_ARTISTS = 20 # Min. # of artists for an ngram / phrase
TOP_VOCAB_PER_GENRE = 100 # FS Baseline

GENRE_CATEGORIES = [5, 12, 25, 32]

def get_genre_column(granularity: int) -> str:
    """Returns 'cat5', 'cat12', etc."""
    return f"cat{granularity}"

# Bayesian optimization defaults
N_BAYESIAN_ITER = 50
N_BAYESIAN_INITIAL = 20
STOP_ITER = 10
UNCERTAIN_JUMP = 5
CV_FOLDS = 5
N_JOBS = 30

DEFAULT_PARAM_SPACE = {
    "C": (0.001, 100.0),
    "l1_ratio": (0.0, 1.0),
}

# Monroe-specific param spaces (3D/6D modes)
MONROE_PARAM_SPACE_3D = {
    "alpha_global": (0.01, 10.0),
    **DEFAULT_PARAM_SPACE,
}

MONROE_PARAM_SPACE_6D = {
    "alpha_unigram": (0.01, 10.0),
    "alpha_bigram": (0.01, 10.0),
    "alpha_trigram": (0.01, 10.0),
    "alpha_quadgram": (0.01, 10.0),
    **DEFAULT_PARAM_SPACE,
}