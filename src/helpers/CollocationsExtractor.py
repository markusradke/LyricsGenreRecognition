import pandas as pd

from collections import defaultdict
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

from helpers.StopwordFilter import StopwordFilter


class CollocationsExtractor:
    """Extract collocations using LLR-scored n-grams."""

    def __init__(
        self,
        min_artists: int = 50,
        top_bigrams: int = 1000,
        top_trigrams: int = 1000,
        random_state: int = 42,
    ):
        """Initialize extractor.

        Args:
            min_artists: Minimum artists for n-gram to be retained.
            top_bigrams: Number of top bigrams to select.
            top_trigrams: Number of top trigrams to select.
            random_state: Random seed for reproducibility.
        """
        self.min_artists = min_artists
        self.top_bigrams = top_bigrams
        self.top_trigrams = top_trigrams
        self.random_state = random_state
        self.stopword_filter = StopwordFilter()
        self.selected_bigrams = []
        self.selected_trigrams = []
        self.vocabulary = []
        self._tokens_by_doc = []
        self._trigram_tuples = {}
        self._bigram_tuples = {}

    def fit(self, corpus: pd.DataFrame) -> "CollocationsExtractor":
        """Fit extractor on corpus: extract, filter, score, and rank n-grams.

        Args:
            corpus: DataFrame with "lyrics" and "artist" columns.

        Returns:
            Self for method chaining.
        """
        self._tokens_by_doc = [self._tokenize(text) for text in corpus["lyrics"]]

        print("Extracting bigrams and trigrams...")
        bigrams, trigrams = self._extract_ngrams_from_tokens(self._tokens_by_doc)

        print("Filtering by artist diversity and stopwords...")
        artists = corpus["artist"].values
        bigrams = self._filter_ngrams(bigrams, artists, self._tokens_by_doc)
        trigrams = self._filter_ngrams(trigrams, artists, self._tokens_by_doc)

        print("Scoring bigrams with LLR...")
        bigram_scores = self._score_ngrams(bigrams, corpus, order=2)

        print("Scoring trigrams with LLR...")
        trigram_scores = self._score_ngrams(trigrams, corpus, order=3)

        print("Selecting top n-grams...")
        self.selected_bigrams = self._select_top_k(bigram_scores, self.top_bigrams)
        self.selected_trigrams = self._select_top_k(trigram_scores, self.top_trigrams)

        self._trigram_tuples = {
            tuple(ng.split("_")): ng for ng in self.selected_trigrams
        }
        self._bigram_tuples = {tuple(ng.split("_")): ng for ng in self.selected_bigrams}

        print(
            f"Selected {len(self.selected_bigrams)} bigrams and {len(self.selected_trigrams)} trigrams"
        )

        print("Replacing n-grams in corpus...")
        replaced_texts = self._replace_ngrams_in_corpus(
            tokens_by_doc=self._tokens_by_doc
        )

        print("Building vocabulary from replaced corpus...")
        self.vocabulary = self._build_vocabulary(replaced_texts, corpus)

        print(f"Final vocabulary size: {len(self.vocabulary)}")
        return self

    def _extract_ngrams_from_tokens(
        self, tokens_by_doc: list[list[str]]
    ) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
        """Extract all bigrams and trigrams from tokens.

        Args:
            tokens_by_doc: Token lists per document.

        Returns:
            Tuple of (bigrams, trigrams) as lists of tuples.
        """
        all_bigrams = []
        all_trigrams = []

        for tokens in tokens_by_doc:
            finder_bi = BigramCollocationFinder.from_words(tokens)
            finder_tri = TrigramCollocationFinder.from_words(tokens)

            all_bigrams.extend(finder_bi.ngram_fd.keys())
            all_trigrams.extend(finder_tri.ngram_fd.keys())

        print(
            f"Extracted {len(set(all_bigrams))} unique bigrams, {len(set(all_trigrams))} unique trigrams"
        )
        return list(set(all_bigrams)), list(set(all_trigrams))

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words.

        Args:
            text: Input text.

        Returns:
            List of lowercase tokens.
        """
        return [word.lower() for word in text.split() if word.isalpha()]

    def _filter_ngrams(
        self,
        ngrams: list[tuple[str, ...]],
        artists: list[str],
        tokens_by_doc: list[list[str]],
    ) -> list[tuple[str, ...]]:
        """Filter n-grams by artist diversity and stopwords.

        Args:
            ngrams: List of n-gram tuples.
            artists: Artist labels per document.
            tokens_by_doc: Token lists per document.

        Returns:
            Filtered list of n-gram tuples.
        """
        artist_counts = self._count_artists_per_ngram(ngrams, artists, tokens_by_doc)
        filtered = [
            ng
            for ng in ngrams
            if artist_counts.get(ng, 0) >= self.min_artists
            and not self.stopword_filter.is_stopword_only(" ".join(ng))
        ]
        print(
            f"Filtered to {len(filtered)} n-grams (>= {self.min_artists} artists, no stopword-only)"
        )
        return filtered

    def _count_artists_per_ngram(
        self,
        ngrams: list[tuple[str, ...]],
        artists: list[str],
        tokens_by_doc: list[list[str]],
    ) -> dict[tuple[str, ...], int]:
        """Count unique artists per n-gram.

        Args:
            ngrams: List of n-gram tuples.
            artists: Artist labels per document.
            tokens_by_doc: Token lists per document.

        Returns:
            Dict mapping n-gram to unique artist count.
        """
        if not ngrams:
            return {}

        ngram_artists = defaultdict(set)
        n_len = len(ngrams[0])
        ngram_set = set(ngrams)

        for tokens, artist in zip(tokens_by_doc, artists):
            text_ngrams = set(
                tuple(tokens[i : i + n_len]) for i in range(len(tokens) - n_len + 1)
            )
            for ng in text_ngrams:
                if ng in ngram_set:
                    ngram_artists[ng].add(artist)

        return {ng: len(artist_set) for ng, artist_set in ngram_artists.items()}

    def _score_ngrams(
        self, ngrams: list[tuple[str, ...]], corpus: pd.DataFrame, order: int
    ) -> pd.DataFrame:
        """Score n-grams with LLR.

        Args:
            ngrams: List of n-gram tuples.
            corpus: DataFrame with "lyrics" column.
            order: N-gram order (2 for bigrams, 3 for trigrams).

        Returns:
            DataFrame with columns: ngram, llr_score.
        """
        llr_scores = self._calculate_llr(ngrams, corpus, order)

        scores_df = pd.DataFrame(
            {
                "ngram": list(llr_scores.keys()),
                "llr_score": list(llr_scores.values()),
            }
        )

        return scores_df.sort_values("llr_score", ascending=False)

    def _calculate_llr(
        self, ngrams: list[tuple[str, ...]], corpus: pd.DataFrame, order: int
    ) -> dict[tuple[str, ...], float]:
        """Calculate log-likelihood ratio for n-grams using NLTK.

        Args:
            ngrams: List of n-gram tuples.
            corpus: DataFrame with "lyrics" column.
            order: N-gram order (2 or 3).

        Returns:
            Dict mapping n-gram to LLR score.
        """
        all_tokens = []
        for text in corpus["lyrics"]:
            all_tokens.extend(self._tokenize(text))

        if order == 2:
            finder = BigramCollocationFinder.from_words(all_tokens)
            llr_measure = BigramAssocMeasures.likelihood_ratio
        else:
            finder = TrigramCollocationFinder.from_words(all_tokens)
            llr_measure = TrigramAssocMeasures.likelihood_ratio

        llr_scores = {}
        for ng in ngrams:
            try:
                llr_scores[ng] = finder.score_ngram(llr_measure, *ng)
            except (KeyError, ZeroDivisionError):
                llr_scores[ng] = 0.0

        return llr_scores

    def _select_top_k(self, scores_df: pd.DataFrame, k: int) -> list[str]:
        """Select top k n-grams by LLR score.

        Args:
            scores_df: DataFrame with ngram and llr_score columns.
            k: Number of n-grams to select.

        Returns:
            List of n-gram strings (underscore-separated).
        """
        top_ngrams = scores_df.head(k)["ngram"].tolist()
        return ["_".join(ng) for ng in top_ngrams]

    def _replace_ngrams_in_corpus(
        self,
        texts: pd.Series | None = None,
        tokens_by_doc: list[list[str]] | None = None,
    ) -> list[list[str]]:
        """Replace n-grams in corpus using greedy longest-first matching.

        Args:
            texts: Series of lyrics.
            tokens_by_doc: Token lists per document.

        Returns:
            List of token lists (one per text) with n-grams replaced.
        """
        if not self._trigram_tuples or not self._bigram_tuples:
            self._trigram_tuples = {
                tuple(ng.split("_")): ng for ng in self.selected_trigrams
            }
            self._bigram_tuples = {
                tuple(ng.split("_")): ng for ng in self.selected_bigrams
            }

        if tokens_by_doc is None:
            tokens_by_doc = [self._tokenize(text) for text in texts]

        replaced_corpus = []
        for tokens in tokens_by_doc:
            replaced_tokens = self._replace_ngrams_in_tokens(
                tokens, self._trigram_tuples, self._bigram_tuples
            )
            replaced_corpus.append(replaced_tokens)

        return replaced_corpus

    def _replace_ngrams_in_tokens(
        self,
        tokens: list[str],
        trigrams: dict[tuple[str, ...], str],
        bigrams: dict[tuple[str, ...], str],
    ) -> list[str]:
        """Replace n-grams in token list using greedy longest-first.

        Args:
            tokens: List of tokens.
            trigrams: Dict mapping trigram tuples to underscore strings.
            bigrams: Dict mapping bigram tuples to underscore strings.

        Returns:
            List of tokens with n-grams replaced.
        """
        result = []
        i = 0

        while i < len(tokens):
            matched = False

            if i + 2 < len(tokens):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                if trigram in trigrams:
                    result.append(trigrams[trigram])
                    i += 3
                    matched = True

            if not matched and i + 1 < len(tokens):
                bigram = (tokens[i], tokens[i + 1])
                if bigram in bigrams:
                    result.append(bigrams[bigram])
                    i += 2
                    matched = True

            if not matched:
                result.append(tokens[i])
                i += 1

        return result

    def _build_vocabulary(
        self, replaced_texts: list[list[str]], corpus: pd.DataFrame
    ) -> list[str]:
        """Build vocabulary from replaced corpus with filtering.

        Args:
            replaced_texts: List of token lists from replaced corpus.
            corpus: DataFrame with "artist" column.

        Returns:
            Sorted list of vocabulary tokens.
        """
        token_artists = defaultdict(set)

        for tokens, artist in zip(replaced_texts, corpus["artist"]):
            for token in tokens:
                token_artists[token].add(artist)

        vocabulary = [
            token
            for token, artists in token_artists.items()
            if len(artists) >= self.min_artists
            and not self.stopword_filter.is_stopword_only(token.replace("_", " "))
        ]

        return sorted(vocabulary)

    def transform(self, texts: pd.Series) -> list[list[str]]:
        """Transform texts by replacing n-grams and filtering vocabulary.

        Args:
            texts: Series of lyrics to transform.

        Returns:
            List of token lists with n-grams replaced and filtered.

        Raises:
            ValueError: If transform called before fit.
        """
        if not self.vocabulary:
            raise ValueError("Must call fit() before transform()")

        replaced_texts = self._replace_ngrams_in_corpus(texts)
        vocab_set = set(self.vocabulary)

        filtered_texts = [
            [token for token in tokens if token in vocab_set]
            for tokens in replaced_texts
        ]

        return filtered_texts
