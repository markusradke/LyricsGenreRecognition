import pandas as pd
import numpy as np

from collections import defaultdict
from nltk.collocations import (
    BigramAssocMeasures,
    TrigramAssocMeasures,
    QuadgramAssocMeasures,
)
from nltk.collocations import (
    BigramCollocationFinder,
    TrigramCollocationFinder,
    QuadgramCollocationFinder,
)
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from helpers.StopwordFilter import StopwordFilter


class IdiomExtractor:
    """Extract idioms using LLR-scored n-grams and TF-IDF ranking."""

    def __init__(
        self,
        min_artists: int = 50,
        min_tracks: int = 100,
        llr_treshold: int = 1000,
        top_vocab_per_genre: int = 300,
        random_state: int = 42,
    ):
        self.min_artists = min_artists
        self.min_tracks = min_tracks
        self.llr_treshold = llr_treshold
        self.top_vocab_per_genre = top_vocab_per_genre
        self.random_state = random_state
        self.stopword_filter = StopwordFilter()
        self.vocabulary = []
        self.selected_ngrams = set()
        self._bigram_map = {}
        self._trigram_map = {}
        self._quadgram_map = {}

    def fit_transform(
        self,
        corpus: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the extractor to a lyrics corpus: identify genre-specific n-grams, replace them
        in the corpus, extract unigrams from the modified text, and build a final ranked
        vocabulary via genre TF–IDF.
        Args:
            corpus (pd.DataFrame): Input dataframe containing at least a "lyrics" and a "genre" column.
        Returns:
          Tuple[pd.DataFrame, pd.DataFrame]: (ngram_counts, replaced_corpus)
                - ngram_counts: DataFrame with counts of detected n-grams from corpus["lyrics"].
                - replaced_corpus: Corpus DataFrame with matched n-grams replaced according to the fitted vocabulary.
        Side effects:
        - Sets self.selected_ngrams to the top n-grams found per genre (output of
          _extract_top_ngrams_per_genre).
        - Sets self.vocabulary to the final ranked token set (output of
          _rank_tokens_via_tfidf).

        """
        print("Extracting and scoring n-grams per genre...")
        bi_ngrams, tri_ngrams, quad_ngrams = self._extract_top_ngrams_per_genre(corpus)

        print("\nReplacing n-grams in corpus...")
        replaced_corpus = self._replace_ngrams_in_corpus(corpus)

        print("\nRanking tokens via TF-IDF per n-gram type...")
        self.vocabulary = self._rank_tokens_via_tfidf_per_type(
            replaced_corpus, corpus, bi_ngrams, tri_ngrams, quad_ngrams
        )
        print(f"\nFinal vocabulary size: {len(self.vocabulary)}")

        dtm = self._count_ngrams_in_corpus(corpus["lyrics"])
        replaced_corpus = self._tidy_replaced_corpus(replaced_corpus, corpus["genre"])

        return dtm, replaced_corpus

    def _extract_top_ngrams_per_genre(
        self, corpus: pd.DataFrame
    ) -> tuple[set[str], set[str], set[str]]:
        tokens_by_doc = [self._tokenize(text) for text in corpus["lyrics"]]

        all_bigrams, all_trigrams, all_quadgrams = self._extract_all_ngrams(
            tokens_by_doc
        )

        stripped_bi = self._strip_left_boundary(all_bigrams)

        filtered_bi = self._filter_ngrams(
            stripped_bi, corpus["artist"].values, tokens_by_doc
        )
        filtered_tri = self._filter_ngrams(
            all_trigrams, corpus["artist"].values, tokens_by_doc
        )
        filtered_quad = self._filter_ngrams(
            all_quadgrams, corpus["artist"].values, tokens_by_doc
        )

        bi_selected = set()
        tri_selected = set()
        quad_selected = set()

        for genre in corpus["genre"].unique():
            genre_mask = corpus["genre"] == genre
            genre_tokens = [
                tokens_by_doc[i] for i in genre_mask.to_numpy().nonzero()[0]
            ]

            bi_scores = (
                self._score_ngrams_llr(filtered_bi, genre_tokens, 2)
                if filtered_bi
                else pd.DataFrame()
            )
            tri_scores = (
                self._score_ngrams_llr(filtered_tri, genre_tokens, 3)
                if filtered_tri
                else pd.DataFrame()
            )
            quad_scores = (
                self._score_ngrams_llr(filtered_quad, genre_tokens, 4)
                if filtered_quad
                else pd.DataFrame()
            )

            top_bi = (
                bi_scores.query(f"llr_score >= {self.llr_treshold}")["ngram"].tolist()
                if not bi_scores.empty
                else []
            )
            top_tri = (
                tri_scores.query(f"llr_score >= {self.llr_treshold}")["ngram"].tolist()
                if not tri_scores.empty
                else []
            )
            top_quad = (
                quad_scores.query(f"llr_score >= {self.llr_treshold}")["ngram"].tolist()
                if not quad_scores.empty
                else []
            )

            bi_selected.update(["_".join(ng) for ng in top_bi])
            tri_selected.update(["_".join(ng) for ng in top_tri])
            quad_selected.update(["_".join(ng) for ng in top_quad])

        self.selected_ngrams = bi_selected | tri_selected | quad_selected
        self._build_ngram_maps()
        print(
            f"Selected {len(bi_selected)} bigrams, {len(tri_selected)} "
            f"trigrams, and {len(quad_selected)} quadgrams across genres"
        )
        return bi_selected, tri_selected, quad_selected

    def _extract_all_ngrams(
        self, tokens_by_doc: list[list[str]]
    ) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], list[tuple[str, ...]]]:
        all_bigrams = set()
        all_trigrams = set()
        all_quadgrams = set()

        for tokens in tokens_by_doc:
            finder_bi = BigramCollocationFinder.from_words(tokens)
            finder_tri = TrigramCollocationFinder.from_words(tokens)
            finder_quad = QuadgramCollocationFinder.from_words(tokens)
            all_bigrams.update(finder_bi.ngram_fd.keys())
            all_trigrams.update(finder_tri.ngram_fd.keys())
            all_quadgrams.update(finder_quad.ngram_fd.keys())

        print(
            f"Extracted {len(all_bigrams)} unique bigrams, "
            f"{len(all_trigrams)} unique trigrams, "
            f"{len(all_quadgrams)} unique quadgrams"
        )
        return list(all_bigrams), list(all_trigrams), list(all_quadgrams)

    def _strip_left_boundary(
        self, ngrams: list[tuple[str, ...]]
    ) -> list[tuple[str, ...]]:
        """Remove n-grams starting with articles or infinitive markers."""
        banned = {"a", "an", "the", "to"}
        return [ng for ng in ngrams if ng and ng[0].lower() not in banned]

    def _filter_ngrams(
        self,
        ngrams: list[tuple[str, ...]],
        artists: np.ndarray,
        tokens_by_doc: list[list[str]],
    ) -> list[tuple[str, ...]]:
        counts = self._count_ngrams_per_artist_and_track(ngrams, artists, tokens_by_doc)
        filtered = [
            ng
            for ng in ngrams
            if counts[ng]["artists"] >= self.min_artists
            and counts[ng]["tracks"] >= self.min_tracks
            and not self.stopword_filter.is_stopword_only(" ".join(ng))
        ]
        print(
            f"Filtered to {len(filtered)} n-grams "
            f"(>= {self.min_artists} artists, "
            f">= {self.min_tracks} tracks, no stopwords)"
        )
        return filtered

    def _count_ngrams_per_artist_and_track(
        self,
        ngrams: list[tuple[str, ...]],
        artists: np.ndarray,
        tokens_by_doc: list[list[str]],
    ) -> dict[tuple[str, ...], dict[str, int]]:
        if not ngrams:
            return {}

        ngram_artists = defaultdict(set)
        ngram_tracks = defaultdict(int)
        n_len = len(ngrams[0])
        ngram_set = set(ngrams)

        for tokens, artist in zip(tokens_by_doc, artists):
            text_ngrams = set(
                tuple(tokens[i : i + n_len]) for i in range(len(tokens) - n_len + 1)
            )
            for ng in text_ngrams & ngram_set:
                ngram_artists[ng].add(artist)
                ngram_tracks[ng] += 1

        return {
            ng: {"artists": len(ngram_artists[ng]), "tracks": ngram_tracks[ng]}
            for ng in ngrams
        }

    def _score_ngrams_llr(
        self, ngrams: list[tuple[str, ...]], tokens_by_doc: list[list[str]], order: int
    ) -> pd.DataFrame:
        all_tokens = [token for tokens in tokens_by_doc for token in tokens]

        if order == 2:
            finder = BigramCollocationFinder.from_words(all_tokens)
            llr_measure = BigramAssocMeasures.likelihood_ratio
        elif order == 3:
            finder = TrigramCollocationFinder.from_words(all_tokens)
            llr_measure = TrigramAssocMeasures.likelihood_ratio
        else:
            finder = QuadgramCollocationFinder.from_words(all_tokens)
            llr_measure = QuadgramAssocMeasures.likelihood_ratio

        scores = []
        for ng in ngrams:
            try:
                score = finder.score_ngram(llr_measure, *ng)
            except (KeyError, ZeroDivisionError):
                score = 0.0
            scores.append({"ngram": ng, "llr_score": score})

        return pd.DataFrame(scores)

    def _build_ngram_maps(self) -> None:
        self._bigram_map = {}
        self._trigram_map = {}
        self._quadgram_map = {}

        for ng_str in self.selected_ngrams:
            parts = ng_str.split("_")
            if len(parts) == 2:
                self._bigram_map[tuple(parts)] = ng_str
            elif len(parts) == 3:
                self._trigram_map[tuple(parts)] = ng_str
            elif len(parts) == 4:
                self._quadgram_map[tuple(parts)] = ng_str

    def _replace_ngrams_in_corpus(self, corpus: pd.DataFrame) -> list[list[str]]:
        replaced_corpus = []
        for text in corpus["lyrics"]:
            tokens = self._tokenize(text)
            replaced = self._replace_ngrams_in_tokens(tokens)
            replaced_corpus.append(replaced)
        return replaced_corpus

    def _replace_ngrams_in_tokens(self, tokens: list[str]) -> list[str]:
        result = []
        i = 0
        n = len(tokens)

        while i < n:
            if i <= n - 4:
                quadgram = (
                    tokens[i],
                    tokens[i + 1],
                    tokens[i + 2],
                    tokens[i + 3],
                )
                replacement = self._quadgram_map.get(quadgram)
                if replacement:
                    result.append(replacement)
                    i += 4
                    continue

            if i <= n - 3:
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                replacement = self._trigram_map.get(trigram)
                if replacement:
                    result.append(replacement)
                    i += 3
                    continue

            if i <= n - 2:
                bigram = (tokens[i], tokens[i + 1])
                replacement = self._bigram_map.get(bigram)
                if replacement:
                    result.append(replacement)
                    i += 2
                    continue

            result.append(tokens[i])
            i += 1

        return result

    def _rank_tokens_via_tfidf_per_type(
        self,
        replaced_corpus: list[list[str]],
        corpus: pd.DataFrame,
        bi_ngrams: set[str],
        tri_ngrams: set[str],
        quad_ngrams: set[str],
    ) -> list[str]:
        corpus_texts = [" ".join(tokens) for tokens in replaced_corpus]
        vocabulary = set()

        for ngram_set in [bi_ngrams, tri_ngrams, quad_ngrams]:
            if not ngram_set:
                continue

            vectorizer = CountVectorizer(
                vocabulary=list(ngram_set),
                token_pattern=r"\b[\w'_]+\b",
                lowercase=True,
            )
            matrix = vectorizer.fit_transform(corpus_texts)

            if matrix.nnz == 0:
                continue

            features = vectorizer.get_feature_names_out()
            tfidf_df = self._calculate_genre_tfidf(corpus, matrix, features)

            if tfidf_df.empty:
                continue

            top_per_genre = (
                tfidf_df.groupby("genre")
                .head(self.top_vocab_per_genre)["ngram"]
                .unique()
            )
            vocabulary.update(set(top_per_genre))

        return sorted(list(vocabulary))

    def _calculate_genre_tfidf(
        self, corpus: pd.DataFrame, matrix: csr_matrix, features: np.ndarray
    ) -> pd.DataFrame:
        binary_matrix = (matrix > 0).astype(int).tocoo()
        genres_array = corpus["genre"].values
        unique_genres = corpus["genre"].unique()
        num_genres = len(unique_genres)

        genre_ngram_counts = defaultdict(lambda: defaultdict(int))

        for track_idx, ngram_idx in zip(binary_matrix.row, binary_matrix.col):
            genre = genres_array[track_idx]
            ngram = features[ngram_idx]
            genre_ngram_counts[genre][ngram] += 1

        # Compute IDF once for all ngrams
        all_ngrams = set()
        for genre_dict in genre_ngram_counts.values():
            all_ngrams.update(genre_dict.keys())

        ngram_idf = {}
        for ngram in all_ngrams:
            genres_with_ngram = sum(
                1 for g_dict in genre_ngram_counts.values() if ngram in g_dict
            )
            ngram_idf[ngram] = np.log((num_genres + 1) / (genres_with_ngram + 1)) + 1

        results = []
        for genre, ngram_dict in genre_ngram_counts.items():
            total = sum(ngram_dict.values())
            for ngram, count in ngram_dict.items():
                tf = count / total
                tfidf = tf * ngram_idf[ngram]
                results.append({"genre": genre, "ngram": ngram, "tfidf": tfidf})

        return pd.DataFrame(results).sort_values(
            ["genre", "tfidf"], ascending=[True, False]
        )

    def _tokenize(self, text: str) -> list[str]:
        return [word.lower() for word in text.split() if word.isalpha()]

    def _tidy_replaced_corpus(self, replaced: list, genres=None) -> pd.DataFrame:
        """Convert corpus to a DataFrame and remove all words / tokens that are not part of the fitted vocabulary."""
        documents = list()
        for document in replaced:
            vocab_only = [token for token in document if token in self.vocabulary]
            documents.append(" ".join(vocab_only))
        tidied = pd.DataFrame(documents, columns=["lyrics"])
        if genres is not None:
            tidied["genre"] = genres
        return tidied

    def transform(self, corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Transform the corpus by replacing known n-grams and computing n-gram counts.
        Args:
            corpus (pd.DataFrame): Input dataframe containing at least a "lyrics" column.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (ngram_counts, replaced_corpus)
                - ngram_counts: DataFrame with counts of detected n-grams from corpus["lyrics"].
                - replaced_corpus: Corpus DataFrame with matched n-grams replaced according to the fitted vocabulary.
        """
        if not self.vocabulary:
            raise ValueError("Must call fit() before transform()")

        replaced_corpus = self._replace_ngrams_in_corpus(corpus)
        dtm = self._count_ngrams_in_corpus(corpus["lyrics"])
        replaced_corpus = self._tidy_replaced_corpus(replaced_corpus)

        return dtm, replaced_corpus

    def _count_ngrams_in_corpus(self, lyrics: pd.Series) -> pd.DataFrame:
        vocab_set = set(self.vocabulary)
        replaced_corpus = []

        for text in lyrics:
            tokens = self._tokenize(text)
            replaced = self._replace_ngrams_in_tokens(tokens)
            filtered = " ".join(token for token in replaced if token in vocab_set)
            replaced_corpus.append(filtered)

        vectorizer = CountVectorizer(
            vocabulary=list(self.vocabulary),
            token_pattern=r"\b[\w'_]+\b",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(replaced_corpus)
        return pd.DataFrame(
            matrix.toarray(), columns=vectorizer.get_feature_names_out()
        )


if __name__ == "__main__":
    english = pd.read_csv(
        "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
    )
    english.rename(
        columns={
            "lyrics_lemmatized": "lyrics",
            "track.s.firstartist.name": "artist",
            "cat12": "genre",
        },
        inplace=True,
    )
    english = english.sample(1000, random_state=42)

    extractor = IdiomExtractor(
        min_artists=10,
        min_tracks=20,
        llr_treshold=10,
        top_vocab_per_genre=100,
    )
    dtm, replaced = extractor.fit_transform(english)

    print(f"\nVocabulary for {english}:")
    print(extractor.vocabulary)
