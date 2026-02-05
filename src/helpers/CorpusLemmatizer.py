from __future__ import annotations

import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from typing import Final

from pandas import DataFrame


class CorpusLemmatizer:
    _TOKEN_PATTERN: Final = re.compile(r"\b[\w']+\b")
    _TOKEN_CLEAN_RE: Final = re.compile(r"^\W+$")
    _DOMAIN_LEXICON: Final[dict[str, str]] = {
        "bitches": "bitch",
        "hoes": "hoe",
        "niggas": "nigga",
        "niggaz": "nigga",
        "coz": "cuz",
        "imma": "i'ma",
        "ima": "i'ma",
    }

    def __init__(self, corpus: DataFrame, lyrics_column: str = "lyrics") -> None:
        self.corpus = corpus
        self.lemmatized_corpus = None
        self.lyrics_column = lyrics_column

    def lemmatize(self) -> DataFrame:
        """Lemmatize the corpus lyrics line-by-line.

        The operation is applied per text independently.

        Returns:
            A copy of the corpus with an added column "lyrics_lemmatized".
        """
        lemmatized = (
            self.corpus[self.lyrics_column].astype(str).map(self._lemmatize_text)
        )
        out = self.corpus.copy()
        out["lyrics_lemmatized"] = lemmatized
        self.lemmatized_corpus = out
        return out

    def _lemmatize_text(self, text: str) -> str:

        lemmatizer = WordNetLemmatizer()

        lines = text.split("\n")
        lemmatized_lines: list[str] = []

        for line in lines:
            tokens = self._tokenize_line(line)
            tagged = pos_tag(tokens)
            lemmas: list[str] = []

            for tok, pos in tagged:
                tok_norm = self._apply_domain_lexicon(tok)
                wn_pos = self._wordnet_pos(pos)
                if wn_pos is None:
                    lemma = lemmatizer.lemmatize(tok_norm)
                else:
                    lemma = lemmatizer.lemmatize(tok_norm, pos=wn_pos)
                lemmas.append(lemma)

            lemmatized_lines.append(self._join_tokens(lemmas))

        return "\n".join(lemmatized_lines)

    def _tokenize_line(self, line: str) -> list[str]:
        """Tokenize a line using the same pattern as the ngram feature extractors.

        Pattern r"\b[\w']+\b" preserves apostrophes and extracts word boundaries.
        """
        return self._TOKEN_PATTERN.findall(line.lower())

    def _apply_domain_lexicon(self, token: str) -> str:
        return self._DOMAIN_LEXICON.get(token, token)

    @staticmethod
    def _wordnet_pos(treebank_pos: str) -> str | None:
        if not treebank_pos:
            return None

        tag = treebank_pos[0].upper()
        if tag == "J":
            return wordnet.ADJ
        if tag == "V":
            return wordnet.VERB
        if tag == "N":
            return wordnet.NOUN
        if tag == "R":
            return wordnet.ADV
        return None

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        """Join tokens back into a readable string.

        NLTK's tokenizer returns punctuation as separate tokens; this keeps common
        punctuation tight while leaving token spacing readable.
        """
        if not tokens:
            return ""

        no_space_before = {".", ",", ":", ";", "!", "?", ")", "]", "}", "'"}
        no_space_after = {"(", "[", "{"}

        out: list[str] = []
        for tok in tokens:
            if not out:
                out.append(tok)
                continue

            if tok in no_space_before:
                out[-1] = f"{out[-1]}{tok}"
            elif out[-1] in no_space_after:
                out[-1] = f"{out[-1]}{tok}"
            else:
                out.append(f" {tok}")

        return "".join(out)

    def save_lemmatized(self, path: str) -> None:
        if self.lemmatized_corpus is None:
            raise ValueError(
                "Lemmatized corpus is not available. Please run lemmatize() first."
            )
        self.lemmatized_corpus.to_csv(path, index=False)
