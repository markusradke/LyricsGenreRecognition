"""Shared utilities for n-gram extraction and filtering."""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Set
from nltk import bigrams, trigrams
from nltk.util import ngrams as nltk_ngrams

from .StopwordFilter import StopwordFilter


def tokenize(text: str, preserve_lines: bool = False) -> List[str]:
    """
    Tokenize text to words, optionally preserving line boundaries.

    Parameters
    ----------
    text : str
        Raw lyrics text.
    preserve_lines : bool, default=False
        If True, keep '\n' as special token for line-aware extraction.

    Returns
    -------
    tokens : List[str]
        Lowercase alphabetic tokens (and '\n' if preserve_lines=True).
    """
    if preserve_lines:
        tokens = []
        for token in text.split():
            if token == "\n":
                tokens.append("\n")
            elif token.isalpha():
                tokens.append(token.lower())
        return tokens
    else:
        return [t.lower() for t in text.split() if t.isalpha()]


def extract_ngrams_by_order(
    tokens: List[str],
    orders: List[int] = [1, 2, 3, 4],
    extract_within_lines: bool = True,
) -> dict:
    """
    Extract n-grams of specified orders from tokens.

    Parameters
    ----------
    tokens : List[str]
        Tokenized text (may contain '\n' markers).
    orders : List[int], default=[1, 2, 3, 4]
        N-gram orders to extract (1=unigram, 2=bigram, etc.).
    extract_within_lines : bool, default=True
        If True and tokens contain '\n', extract n-grams within line
        boundaries only (prevents cross-line phrases).

    Returns
    -------
    ngrams_by_order : dict
        Maps order to list of n-gram tuples.
        Example: {1: [('love',), ('you',)], 2: [('love', 'you')]}
    """
    if extract_within_lines and "\n" in tokens:
        # Split by line markers
        lines = []
        current_line = []
        for token in tokens:
            if token == "\n":
                if current_line:
                    lines.append(current_line)
                    current_line = []
            else:
                current_line.append(token)
        if current_line:
            lines.append(current_line)

        # Extract n-grams per line and union
        ngrams_by_order = {order: set() for order in orders}
        for line in lines:
            if len(line) == 0:
                continue
            for order in orders:
                if order == 1:
                    ngrams_by_order[order].update((t,) for t in line)
                elif len(line) >= order:
                    ngrams_by_order[order].update(nltk_ngrams(line, order))

        return {k: list(v) for k, v in ngrams_by_order.items()}
    else:
        # Extract from full token sequence
        ngrams_by_order = {}
        for order in orders:
            if order == 1:
                ngrams_by_order[order] = [(t,) for t in tokens if t != "\n"]
            elif len(tokens) >= order:
                ngrams_by_order[order] = list(
                    nltk_ngrams([t for t in tokens if t != "\n"], order)
                )
            else:
                ngrams_by_order[order] = []
        return ngrams_by_order


def strip_boundary_ngrams(ngrams: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    """Remove n-grams starting with articles or infinitive markers."""
    banned_starts = {"a", "an", "the", "to"}
    return [ng for ng in ngrams if ng and ng[0].lower() not in banned_starts]


def filter_stopword_only(
    ngrams: List[Tuple[str, ...]], stopword_filter: StopwordFilter
) -> List[Tuple[str, ...]]:
    """Remove n-grams containing only stopwords."""
    return [ng for ng in ngrams if not stopword_filter.is_stopword_only(" ".join(ng))]


def count_artists_per_ngram(
    ngrams: Set[Tuple[str, ...]],
    corpus: pd.Series,
    artists: pd.Series,
    extract_within_lines: bool = True,
    tokens_cache: list = None,
) -> dict:
    """
    Count how many unique artists use each n-gram.

    Parameters
    ----------
    ngrams : Set[Tuple[str, ...]]
        N-grams to count.
    corpus : pd.Series
        Lyrics text per track.
    artists : pd.Series
        Artist name per track.
    extract_within_lines : bool
        Must match extraction mode used for ngrams.
    tokens_cache : list, optional
        Pre-tokenized corpus (list of token lists). If provided, skips
        tokenization step for performance.

    Returns
    -------
    artist_counts : dict
        Maps n-gram tuple to number of unique artists.
    """
    ngram_to_artists = defaultdict(set)

    if tokens_cache is not None:
        # Use cached tokens (avoid re-tokenization)
        iterator = zip(tokens_cache, artists)
    else:
        # Tokenize on-the-fly (fallback for non-cached usage)
        iterator = zip(
            [tokenize(text, preserve_lines=extract_within_lines) for text in corpus],
            artists,
        )

    order = len(next(iter(ngrams)))  # Infer order from first ngram

    for tokens, artist in iterator:
        doc_ngrams = extract_ngrams_by_order(
            tokens, orders=[order], extract_within_lines=extract_within_lines
        )[order]

        # Check which ngrams appear in this doc
        doc_ngrams_set = set(doc_ngrams)
        for ng in ngrams:
            if ng in doc_ngrams_set:
                ngram_to_artists[ng].add(artist)

    return {ng: len(artists) for ng, artists in ngram_to_artists.items()}
