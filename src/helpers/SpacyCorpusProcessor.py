from __future__ import annotations

from typing import Final

import spacy
from pandas import DataFrame
from spacy.tokens import Doc


class SpacyCorpusProcessor:
    """Process lyrics corpus using spaCy for lemmatization, POS tagging, and ngram extraction."""

    _DOMAIN_LEXICON: Final[dict[str, str]] = {
        "bitches": "bitch",
        "hoes": "hoe",
        "niggas": "nigga",
        "niggaz": "nigga",
        "coz": "cuz",
        "imma": "i'ma",
        "ima": "i'ma",
    }

    _CONTENT_POS: Final[set[str]] = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}
    _BOUNDARY_POS: Final[set[str]] = {"ADP", "DET", "AUX", "CCONJ", "SCONJ", "PART"}

    def __init__(
        self,
        corpus: DataFrame,
        lyrics_column: str = "full_lyrics",
        model: str = "en_core_web_sm",
        batch_size: int = 50,
    ) -> None:
        self.corpus = corpus
        self.lyrics_column = lyrics_column
        self.batch_size = batch_size
        self.processed_corpus = None

        self.nlp = spacy.load(model, disable=["parser", "ner"])
        self.nlp.max_length = 2000000

    def process(self) -> DataFrame:
        """Process corpus: lemmatize, tag POS, extract valid ngrams.

        Returns:
            Copy of corpus with added columns:
            - lyrics_lemmatized: lemmatized text
            - pos_tags: POS tags as pipe-separated string (for inspection)
        """
        results = []
        texts = self.corpus[self.lyrics_column].astype(str)

        print(f"Processing {len(texts):,} lyrics with spaCy...")
        for doc in self.nlp.pipe(texts, batch_size=self.batch_size):
            lemmatized = self._get_lemmatized_text(doc)
            pos_tags = self._get_pos_tags(doc)

            results.append(
                {"lyrics_lemmatized": lemmatized, "pos_tags_inspection": pos_tags}
            )

        out = self.corpus.copy()
        result_df = DataFrame(results)
        out["lyrics_lemmatized"] = result_df["lyrics_lemmatized"]
        out["pos_tags_inspection"] = result_df["pos_tags_inspection"]

        self.processed_corpus = out
        print("Processing complete.")
        return out

    def _get_lemmatized_text(self, doc: Doc) -> str:
        """Extract lemmatized text preserving line structure."""
        lines = []
        current_line = []

        for token in doc:
            if token.text == "\n":
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = []
                lines.append("")
            else:
                lemma = self._apply_domain_lexicon(token.lemma_.lower())
                current_line.append(lemma)

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def _get_pos_tags(self, doc: Doc) -> str:
        """Extract POS tags as pipe-separated string for inspection.

        Format: token|POS|token|POS|...
        Newlines preserved as \\n markers.
        """
        tags = []
        for token in doc:
            if token.text == "\n":
                tags.append("\\n")
            else:
                tags.append(f"{token.text}|{token.pos_}")

        return "|".join(tags)

    def _apply_domain_lexicon(self, lemma: str) -> str:
        """Apply domain-specific lexicon corrections."""
        return self._DOMAIN_LEXICON.get(lemma, lemma)

    def save_processed(self, path: str) -> None:
        """Save processed corpus to CSV."""
        if self.processed_corpus is None:
            raise ValueError("Processed corpus not available. Call process() first.")
        self.processed_corpus.to_csv(path, index=True)


class SpacyNGramValidator:
    """Validate ngrams based on POS patterns and content requirements."""

    _CONTENT_POS: Final[set[str]] = {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}
    _BOUNDARY_POS: Final[set[str]] = {"ADP", "DET", "AUX", "CCONJ", "SCONJ", "PART"}
    _STOPWORDS: Final[set[str]] = set(spacy.load("en_core_web_sm").Defaults.stop_words)

    @classmethod
    def is_valid_ngram(cls, tokens: list) -> bool:
        """Check if ngram meets validation criteria.

        Rules:
        - At least one content word (NOUN/VERB/ADJ/PROPN/ADV)
        - Not all stopwords
        - No boundary POS at start/end for ngrams > 1
        """
        if not tokens:
            return False

        has_content = any(t.pos_ in cls._CONTENT_POS for t in tokens)
        not_all_stop = any(t.lemma_.lower() not in cls._STOPWORDS for t in tokens)

        if len(tokens) > 1:
            no_boundary_edges = (
                tokens[0].pos_ not in cls._BOUNDARY_POS
                and tokens[-1].pos_ not in cls._BOUNDARY_POS
            )
            return has_content and not_all_stop and no_boundary_edges

        return has_content and not_all_stop

    @classmethod
    def extract_valid_ngrams(
        cls, doc: Doc, n_range: tuple[int, int] = (1, 4)
    ) -> list[str]:
        """Extract valid ngrams from spaCy Doc.

        Args:
            doc: Processed spaCy document
            n_range: Min and max ngram length (inclusive)

        Returns:
            List of valid ngrams as underscore-joined strings
        """
        valid = []

        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(doc) - n + 1):
                ngram_tokens = doc[i : i + n]

                if any(t.is_space for t in ngram_tokens):
                    continue

                if cls.is_valid_ngram(ngram_tokens):
                    ngram_str = "_".join(t.lemma_.lower() for t in ngram_tokens)
                    valid.append(ngram_str)

        return valid
