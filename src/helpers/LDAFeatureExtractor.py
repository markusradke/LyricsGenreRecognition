from __future__ import annotations

from typing import Iterable

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

from helpers.IdiomExtractor import IdiomExtractor


class LDAFeatureExtractor:
    """Unsupervised LDA topic modeling over collocation-aware tokens. Aggregate Artist texts for building the lda model, but transform each song separately to get song-level topic distributions."""

    def __init__(
        self,
        range_of_topics: tuple[int, int] = (5, 50),
        topics_step: int = 5,
        min_artists: int = 50,
        top_bigrams: int = 1000,
        top_trigrams: int = 1000,
        random_state=42,
    ) -> None:
        self.range_no_topics = range_of_topics
        self.topics_step = topics_step
        self.model: LdaModel | None = None
        self.dictionary: Dictionary | None = None
        self.coherence_scores: dict[int, float] = {}
        self.collocations_extractor: IdiomExtractor | None = None
        self.min_artists = min_artists
        self.top_bigrams = top_bigrams
        self.top_trigrams = top_trigrams
        self.random_state = random_state

    def fit(self, corpus: pd.DataFrame) -> "LDAFeatureExtractor":
        """Fit collocations and select best LDA model by NPMI coherence.

        Args:
            corpus: DataFrame with "lyrics" and "artist" columns.

        Returns:
            Self for method chaining.
        """
        self.collocations_extractor = IdiomExtractor(
            min_artists=self.min_artists,
            top_bigrams=self.top_bigrams,
            top_trigrams=self.top_trigrams,
            random_state=self.random_state,
        )
        self.collocations_extractor.fit(corpus)
        tokens = self.collocations_extractor.transform(corpus, aggregate_artists=True)

        self.dictionary = Dictionary(tokens)
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in tokens]

        topic_nums = range(
            self.range_no_topics[0],
            self.range_no_topics[1] + 1,
            self.topics_step,
        )

        best_model = None
        best_score = float("-inf")

        print("Fitting LDA models and evaluating coherence...")
        for n_topics in topic_nums:
            print(
                f"Evaluating {n_topics} topics (model {topic_nums.index(n_topics) + 1}/{len(topic_nums)})..."
            )
            model = LdaModel(
                corpus=corpus_bow,
                id2word=self.dictionary,
                num_topics=n_topics,
                random_state=42,
            )
            coherence = CoherenceModel(
                model=model,
                texts=tokens,
                dictionary=self.dictionary,
                coherence="c_npmi",
                processes=1,
            ).get_coherence()
            self.coherence_scores[n_topics] = coherence
            if coherence > best_score:
                best_score = coherence
                best_model = model

        self.model = best_model
        return self

    def transform(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Transform corpus into topic proportion vectors.

        Args:
            corpus: DataFrame with "lyrics" column.

        Returns:
            DataFrame with topic probabilities in colums.

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None or self.dictionary is None:
            raise ValueError("Model has not been fitted yet.")
        if self.collocations_extractor is None:
            raise ValueError("Model has not been fitted yet.")

        tokens = self.collocations_extractor.transform(corpus, aggregate_artists=False)
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in tokens]

        res = [
            self._topic_vector(
                self.model.get_document_topics(bow, minimum_probability=0.0)
            )
            for bow in corpus_bow
        ]

        topic_cols = [f"topic_{i}" for i in range(self.model.num_topics)]
        return pd.DataFrame(res, columns=topic_cols)

    def show_coherence_scores(self) -> dict[int, float]:
        """Return coherence scores for all evaluated topic counts.

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return dict(self.coherence_scores)

    def top_tokens_by_topic(self, n_tokens: int = 10) -> pd.DataFrame:
        """Return top tokens per topic.

        Args:
            n_tokens: Number of top tokens per topic.

        Returns:
            DataFrame with columns: topic_id, token, weight.

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        rows = []
        for topic_id in range(self.model.num_topics):
            for token, weight in self.model.show_topic(topic_id, topn=n_tokens):
                rows.append({"topic_id": topic_id, "token": token, "weight": weight})
        return pd.DataFrame(rows)

    def _topic_vector(self, topics: Iterable[tuple[int, float]]) -> list[float]:
        """Convert topic distribution into a dense vector.

        Args:
            topics: Iterable of (topic_id, weight) pairs.

        Returns:
            Dense topic probability vector.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        vector = [0.0] * self.model.num_topics
        for topic_id, weight in topics:
            vector[topic_id] = weight
        return vector
