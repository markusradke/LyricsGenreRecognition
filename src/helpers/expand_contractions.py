import re

CONTRACTIONS: dict[str, str] = {
    # be
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "there'll": "there will",
    "this'll": "this will",
    # negations
    "can't": "cannot",
    "won't": "will not",
    "ain't": "is not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    # wh-contractions
    "what's": "what is",
    "what're": "what are",
    "what've": "what have",
    "where's": "where is",
    "who's": "who is",
    "who'd": "who would",
    "who'll": "who will",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",
    "here's": "here is",
    # informal
    "might've": "might have",
    "d'you": "do you",
    "how'd": "how did",
    "ev'ry": "every",
    "should've": "should have",
    "would've": "would have",
    "could've": "could have",
    "party's": "party",
    "pressure's": "pressure",
    "s'posed": "supposed",
    "satan's": "satan",
    "santa's": "santa",
    "everybody's": "everybody is",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "outta": "out of",
    "lotta": "lot of",
    "inna": "in a",
    "gimmie": "give me",
    "hella": "hell of a",
    "imma": "i am going to",
    "ima": "i am going to",
    "i'ma": "i am going to",
    "i'mma": "i am going to",
    "o'er": "over",
    "why'd": "why would",
    "y'all": "you all",
    "y'know": "you know",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
    "musta": "must have",
    "lemme": "let me",
    "gimme": "give me",
    "tryna": "trying to",
    "finna": "fixing to",
    "bout": "about",
    "'cause": "because",
    "cuz": "because",
    "'em": "them",
    "ya": "you",
    "til": "until",
    "'til": "until",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b",
    flags=re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    def _replace(match: re.Match) -> str:
        token = match.group(0)
        replacement = CONTRACTIONS[token.lower()]
        # Preserve capitalisation of first letter
        return replacement.capitalize() if token[0].isupper() else replacement

    return _CONTRACTION_RE.sub(_replace, text)


def strip_apostrophe_with_s(text: str) -> str:
    """Strips 's from words, which is often used in lyrics to indicate possession or contractions."""
    return re.sub(r"\b(\w+)'s\b", r"\1", text)
