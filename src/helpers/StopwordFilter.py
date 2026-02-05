from helpers.STOPWORDS_ENGLISH import STOPWORDS_ENGLISH


class StopwordFilter:
    """Filter n-grams that consist entirely of stopwords."""

    def __init__(self):
        """Initialize filter with stopword set."""
        self.stopwords = STOPWORDS_ENGLISH

    def is_stopword_only(self, ngram: str) -> bool:
        """Check if n-gram consists only of stopwords.

        Args:
            ngram: The n-gram string to check.

        Returns:
            True if all tokens in n-gram are stopwords, False otherwise.
        """
        tokens = ngram.lower().split()
        return all(token in self.stopwords for token in tokens)

    def filter_ngrams(self, ngrams: set[str]) -> set[str]:
        """Remove n-grams that consist entirely of stopwords.

        Args:
            ngrams: Set of n-gram strings to filter.

        Returns:
            Set of n-grams with stopword-only entries removed.
        """
        return {ngram for ngram in ngrams if not self.is_stopword_only(ngram)}
