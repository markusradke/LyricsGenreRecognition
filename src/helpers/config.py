RANDOM_SEED = 42  # Reproducibility for all random operations
TEST_SIZE = 0.2  # Data split ratio for train/test sets

MIN_ARTISTS = 20  # Min. # of artists for an ngram / phrase
TOP_VOCAB_PER_GENRE = 100  # FS Baseline

# Monroe method settings
MONROE_Z_THRESHOLD = 2.326  # One-sided test at alpha=0.01
EXTRACT_WITHIN_LINES = True  # Extract n-grams within line boundaries

# Feature extraction toggles
ENABLE_STOPWORD_FILTER = False  # Set False to disable stopword filtering
INCLUDE_UNIGRAMS = True  # Set False for phrase-only experiments (bigrams+)

# Sampling
DEBUG_SAMPLE_SIZE = None  # If set (e.g., 0.01 for 1%), override subsample_debug

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

# Classifier-only parameter space (for FS baseline)
CLASSIFIER_PARAM_SPACE = {
    "C": (-3.0, 2.0),  # Log scale: 10^-3 to 10^2
    "l1_ratio": (0.0, 1.0),
}

# FS baseline uses classifier params only
FS_PARAM_SPACE = CLASSIFIER_PARAM_SPACE

# Monroe-specific param spaces (3D/6D modes)
# Log scale for alpha: 10^-2 to 10^1
MONROE_PARAM_SPACE_3D = {
    "alpha_unigram": (-2.0, 1.0),
    "alpha_bigram": (-2.0, 1.0),
    "alpha_trigram": (-2.0, 1.0),
    "alpha_quadgram": (-2.0, 1.0),
    **CLASSIFIER_PARAM_SPACE,
}

MONROE_PARAM_SPACE_6D = {
    "alpha_unigram": (-2.0, 1.0),
    "alpha_bigram": (-2.0, 1.0),
    "alpha_trigram": (-2.0, 1.0),
    "alpha_quadgram": (-2.0, 1.0),
    **CLASSIFIER_PARAM_SPACE,
}
