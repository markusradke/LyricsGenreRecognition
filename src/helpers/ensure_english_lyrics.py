from lingua import Language, LanguageDetectorBuilder
from nltk.corpus import words

DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.GERMAN
).build()

# buiding a set of English words for vocabulary check
ENGLISH_VOCAB = set(w.lower() for w in words.words())


def get_english_confidence(lyrics: str) -> float:
    """Returns lingua's confidence score for English."""
    results = DETECTOR.compute_language_confidence_values(lyrics)
    for r in results:
        if r.language == Language.ENGLISH:
            return r.value
    return 0.0


def get_english_vocab_ratio(lyrics: str) -> bool:
    """Checks if the ratio of English words in the lyrics exceeds the given threshold."""
    words = [w.lower() for w in lyrics.split() if w.lower().isalpha()]
    if len(words) == 0:
        return False
    ratio = sum(1 for w in words if w in ENGLISH_VOCAB) / len(words)
    return ratio
