from __future__ import annotations


class BoundaryStripper:
    """Conservatively strip boundary tokens from whitespace-tokenized n-grams.

    This is intended to reduce near-duplicate n-grams like 'the dark' vs 'dark'
    without changing interior tokens.

    Rules:
    - strip at most one token on the left and at most one token on the right
    """

    def __init__(self, strip_left=True, strip_right=True):
        self.boundary_words = frozenset(
            {
                "a",
                "an",
                "the",
                "to",
            }
        )
        self.strip_left = strip_left
        self.strip_right = strip_right

    def strip_boundaries(self, ngram: str) -> str:
        """Strip boundary words from ngram and return the result."""
        tokens = ngram.split()
        tokens = self._strip_one_side(tokens, from_left=True)
        tokens = self._strip_one_side(tokens, from_left=False)
        return " ".join(tokens)

    def _strip_one_side(self, tokens: list[str], from_left: bool) -> list[str]:
        if not tokens:
            return tokens

        idx = 0 if from_left else -1
        if tokens[idx] in self.boundary_words:
            return tokens[1:] if from_left else tokens[:-1]

        return tokens
