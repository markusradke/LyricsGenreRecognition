import pandas as pd
import numpy as np

from collections import defaultdict
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from helpers.StopwordFilter import StopwordFilter


class IdiomExtractor:
    """Extract idioms using LLR-scored n-grams and TF-IDF ranking."""

    def __init__(
        self,
        min_artists: int = 50,
        llr_treshold: int = 1000,
        top_vocab_per_genre: int = 300,
        random_state: int = 42,
    ):
        self.min_artists = min_artists
        self.llr_treshold = llr_treshold
        self.top_vocab_per_genre = top_vocab_per_genre
        self.random_state = random_state
        self.stopword_filter = StopwordFilter()
        self.vocabulary = []
        self._bigram_map = {}
        self._trigram_map = {}

    def fit(self, corpus: pd.DataFrame) -> pd.DataFrame:
        print("Step 1: Extracting and scoring n-grams per genre...")
        selected_ngrams = self._extract_top_ngrams_per_genre(corpus)

        print("\nStep 2: Replacing n-grams in corpus...")
        replaced_corpus = self._replace_ngrams_in_corpus(corpus, selected_ngrams)

        print("\nStep 3: Extracting unigrams from replaced corpus...")
        unigrams = self._extract_unigrams(replaced_corpus, corpus)

        print("\nStep 4: Ranking all tokens via TF-IDF...")
        all_tokens = selected_ngrams | unigrams
        self.vocabulary = self._rank_tokens_via_tfidf(
            replaced_corpus, corpus, all_tokens
        )

        print(f"\nFinal vocabulary size: {len(self.vocabulary)}")
        return self._count_ngrams_in_corpus(corpus["lyrics"])

    def _extract_top_ngrams_per_genre(self, corpus: pd.DataFrame) -> set[str]:
        tokens_by_doc = [self._tokenize(text) for text in corpus["lyrics"]]

        all_bigrams, all_trigrams = self._extract_all_ngrams(tokens_by_doc)

        filtered_bi = self._filter_ngrams(
            all_bigrams, corpus["artist"].values, tokens_by_doc
        )
        filtered_tri = self._filter_ngrams(
            all_trigrams, corpus["artist"].values, tokens_by_doc
        )

        selected = set()
        for genre in corpus["genre"].unique():
            genre_mask = corpus["genre"] == genre
            genre_corpus = corpus[genre_mask].reset_index(drop=True)

            bi_scores = self._score_ngrams_llr(filtered_bi, genre_corpus, 2)
            tri_scores = self._score_ngrams_llr(filtered_tri, genre_corpus, 3)

            top_bi = bi_scores.query(f"llr_score >= {self.llr_treshold}")[
                "ngram"
            ].tolist()
            top_tri = tri_scores.query(f"llr_score >= {self.llr_treshold}")[
                "ngram"
            ].tolist()

            selected.update(["_".join(ng) for ng in top_bi])
            selected.update(["_".join(ng) for ng in top_tri])

        self._build_ngram_maps(selected)
        print(
            f"Selected {len(top_bi)} bigrams and {len(top_tri)} trigrams across genres"
        )
        return selected

    def _extract_all_ngrams(
        self, tokens_by_doc: list[list[str]]
    ) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
        all_bigrams = set()
        all_trigrams = set()

        for tokens in tokens_by_doc:
            finder_bi = BigramCollocationFinder.from_words(tokens)
            finder_tri = TrigramCollocationFinder.from_words(tokens)
            all_bigrams.update(finder_bi.ngram_fd.keys())
            all_trigrams.update(finder_tri.ngram_fd.keys())

        print(
            f"Extracted {len(all_bigrams)} unique bigrams, "
            f"{len(all_trigrams)} unique trigrams"
        )
        return list(all_bigrams), list(all_trigrams)

    def _filter_ngrams(
        self,
        ngrams: list[tuple[str, ...]],
        artists: np.ndarray,
        tokens_by_doc: list[list[str]],
    ) -> list[tuple[str, ...]]:
        artist_counts = self._count_artists_per_ngram(ngrams, artists, tokens_by_doc)
        filtered = [
            ng
            for ng in ngrams
            if artist_counts.get(ng, 0) >= self.min_artists
            and not self.stopword_filter.is_stopword_only(" ".join(ng))
        ]
        print(
            f"Filtered to {len(filtered)} n-grams "
            f"(>= {self.min_artists} artists, no stopwords)"
        )
        return filtered

    def _count_artists_per_ngram(
        self,
        ngrams: list[tuple[str, ...]],
        artists: np.ndarray,
        tokens_by_doc: list[list[str]],
    ) -> dict[tuple[str, ...], int]:
        if not ngrams:
            return {}

        ngram_artists = defaultdict(set)
        n_len = len(ngrams[0])
        ngram_set = set(ngrams)

        for tokens, artist in zip(tokens_by_doc, artists):
            text_ngrams = set(
                tuple(tokens[i : i + n_len]) for i in range(len(tokens) - n_len + 1)
            )
            for ng in text_ngrams & ngram_set:
                ngram_artists[ng].add(artist)

        return {ng: len(artists) for ng, artists in ngram_artists.items()}

    def _score_ngrams_llr(
        self, ngrams: list[tuple[str, ...]], corpus: pd.DataFrame, order: int
    ) -> pd.DataFrame:
        all_tokens = []
        for text in corpus["lyrics"]:
            all_tokens.extend(self._tokenize(text))

        if order == 2:
            finder = BigramCollocationFinder.from_words(all_tokens)
            llr_measure = BigramAssocMeasures.likelihood_ratio
        else:
            finder = TrigramCollocationFinder.from_words(all_tokens)
            llr_measure = TrigramAssocMeasures.likelihood_ratio

        scores = []
        for ng in ngrams:
            try:
                score = finder.score_ngram(llr_measure, *ng)
            except (KeyError, ZeroDivisionError):
                score = 0.0
            scores.append({"ngram": ng, "llr_score": score})

        return pd.DataFrame(scores)

    def _build_ngram_maps(self, selected_ngrams: set[str]) -> None:
        self._bigram_map = {}
        self._trigram_map = {}

        for ng_str in selected_ngrams:
            parts = ng_str.split("_")
            if len(parts) == 2:
                self._bigram_map[tuple(parts)] = ng_str
            elif len(parts) == 3:
                self._trigram_map[tuple(parts)] = ng_str

    def _replace_ngrams_in_corpus(
        self, corpus: pd.DataFrame, selected_ngrams: set[str]
    ) -> list[list[str]]:
        replaced_corpus = []
        for text in corpus["lyrics"]:
            tokens = self._tokenize(text)
            replaced = self._replace_ngrams_in_tokens(tokens)
            replaced_corpus.append(replaced)
        return replaced_corpus

    def _replace_ngrams_in_tokens(self, tokens: list[str]) -> list[str]:
        result = []
        i = 0

        while i < len(tokens):
            matched = False

            if i + 2 < len(tokens):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                if trigram in self._trigram_map:
                    result.append(self._trigram_map[trigram])
                    i += 3
                    matched = True

            if not matched and i + 1 < len(tokens):
                bigram = (tokens[i], tokens[i + 1])
                if bigram in self._bigram_map:
                    result.append(self._bigram_map[bigram])
                    i += 2
                    matched = True

            if not matched:
                result.append(tokens[i])
                i += 1

        return result

    def _extract_unigrams(
        self, replaced_corpus: list[list[str]], corpus: pd.DataFrame
    ) -> set[str]:
        token_artists = defaultdict(set)

        for tokens, artist in zip(replaced_corpus, corpus["artist"]):
            for token in tokens:
                if "_" not in token:
                    token_artists[token].add(artist)

        unigrams = {
            token
            for token, artists in token_artists.items()
            if len(artists) >= self.min_artists
            and not self.stopword_filter.is_stopword_only(token)
        }

        print(f"Extracted {len(unigrams)} unigrams meeting criteria")
        return unigrams

    def _rank_tokens_via_tfidf(
        self,
        replaced_corpus: list[list[str]],
        corpus: pd.DataFrame,
        all_tokens: set[str],
    ) -> list[str]:
        corpus_texts = [" ".join(tokens) for tokens in replaced_corpus]

        vectorizer = CountVectorizer(
            vocabulary=list(all_tokens),
            token_pattern=r"\b[\w']+\b",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(corpus_texts)
        features = vectorizer.get_feature_names_out()

        tfidf_df = self._calculate_genre_tfidf(corpus, matrix, features)

        top_per_genre = (
            tfidf_df.groupby("genre").head(self.top_vocab_per_genre)["ngram"].unique()
        )

        return sorted(list(set(top_per_genre)))

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

        ngram_idf = {}
        for genre_dict in genre_ngram_counts.values():
            for ngram in genre_dict.keys():
                if ngram not in ngram_idf:
                    genres_with_ngram = sum(
                        1
                        for g in genre_ngram_counts.keys()
                        if ngram in genre_ngram_counts[g]
                    )
                    ngram_idf[ngram] = (
                        np.log((num_genres + 1) / (genres_with_ngram + 1)) + 1
                    )

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

    def transform(self, corpus: pd.DataFrame) -> pd.DataFrame:
        if not self.vocabulary:
            raise ValueError("Must call fit() before transform()")

        return self._count_ngrams_in_corpus(corpus["lyrics"])

    def _count_ngrams_in_corpus(self, lyrics: pd.Series) -> pd.DataFrame:
        vocab_set = set(self.vocabulary)
        replaced_corpus = []

        for text in lyrics:
            tokens = self._tokenize(text)
            replaced = self._replace_ngrams_in_tokens(tokens)
            filtered = [token for token in replaced if token in vocab_set]
            replaced_corpus.append(" ".join(filtered))

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
        llr_treshold=10,
        top_vocab_per_genre=300,
    )
    extractor.fit(english)

    print(f"\nVocabulary for {english}:")
    print(extractor.vocabulary)
