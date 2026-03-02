import re
from helpers.STOPWORDS_ENGLISH import STOPWORDS_ENGLISH

_NUM_RE = re.compile(r"^\d+\S*$")


class StopwordFilter:
    """Filter n-grams that consist entirely of stopwords."""

    def __init__(self):
        """Initialize filter with stopword set."""
        self.stopwords = STOPWORDS_ENGLISH

    def _is_stopword(self, token: str) -> bool:
        """Check if a single token is a stopword or numeric."""
        return token in self.stopwords or bool(_NUM_RE.match(token))

    def is_stopword_only(self, ngram: str) -> bool:
        """Check if n-gram consists only of stopwords.

        Args:
            ngram: The n-gram string to check.

        Returns:
            True if all tokens in n-gram are stopwords, False otherwise.
        """
        tokens = ngram.lower().split()
        return all(self._is_stopword(token) for token in tokens)

    def filter_ngrams(self, ngrams: set[str]) -> set[str]:
        """Remove n-grams that consist entirely of stopwords.

        Args:
            ngrams: Set of n-gram strings to filter.

        Returns:
            Set of n-grams with stopword-only entries removed.
        """
        return {ngram for ngram in ngrams if not self.is_stopword_only(ngram)}
