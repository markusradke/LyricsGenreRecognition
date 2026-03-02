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
        "amazin": "amazing",
        "bitches": "bitch",
        "ballin": "ball",
        "bangin": "bang",
        "beatin": "beat",
        "becomes": "become",
        "beefin": "beef",
        "beginnin": "begin",
        "behaviour": "behavior",
        "being": "be",
        "believin": "believe",
        "betta": "better",
        "achin": "ache",
        "bitchin": "bitch",
        "bitin": "bite",
        "blastin": "blast",
        "bleedin": "bleed",
        "bleeds": "bleed",
        "blessin": "bless",
        "blowin": "blow",
        "bouncnin": "bounce",
        "braggin": "brag",
        "breakin": "break",
        "breathin": "breathe",
        "bringin": "bring",
        "brings": "bring",
        "buggin": "bug",
        "buildin": "build",
        "bumpin": "bump",
        "burnin": "burn",
        "bustin": "bust",
        "buyin": "buy",
        "buzzin": "buzz",
        "callin": "call",
        "carryin": "carry",
        "catchin": "catch",
        "causin": "cause",
        "ceilin": "ceil",
        "chasin": "chase",
        "cheatin": "cheat",
        "checkin": "check",
        "chokin": "choke",
        "clappin": "clap",
        "cliché": "cliche",
        "climbin": "climb",
        "clutchin": "clutch",
        "comin": "come",
        "controllin": "control",
        "cookin": "cook",
        "coolin": "cool",
        "countin": "count",
        "crackin": "crack",
        "crashin": "crash",
        "crawlin": "crawl",
        "creepin": "creep",
        "cruisin": "cruise",
        "cryin": "cry",
        "cuttin": "cut",
        "dancin": "dance",
        "darlin": "darling",
        "dealin": "deal",
        "diggin": "dig",
        "dippin": "dip",
        "dissin": "diss",
        "doin": "do",
        "draggin": "drag",
        "dreamin": "dream",
        "drinkin": "drink",
        "drippin": "drip",
        "dropin": "drop",
        "duckin": "duck",
        "eatin": "eat",
        "evenin": "evening",
        "eyes": "eye",
        "facin": "face",
        "fakin": "fake",
        "fallin": "fall",
        "feedin": "feed",
        "fiendin": "fiend",
        "fishin": "fish",
        "flashin": "flash",
        "flippin": "flip",
        "floatin": "float",
        "flossin": "floss",
        "flowin": "flow",
        "flyin": "fly",
        "foolin": "fool",
        "freakin": "freak",
        "friends": "friend",
        "fronin": "front",
        "gettin": "get",
        "girlies": "girlie",
        "givin": "give",
        "goin": "go",
        "grabbin": "grab",
        "grindin": "grind",
        "grippin": "grip",
        "groovin": "groove",
        "growin": "grow",
        "gunnin": "gun",
        "hangin": "hang",
        "happenin": "happen",
        "hatin": "hate",
        "havin": "have",
        "headin": "head",
        "healin": "heal",
        "hearin": "hear",
        "hidin": "hide",
        "hmm": "hm",
        "hmmm": "hm",
        "holdin": "hold",
        "homies": "homie",
        "hoppin": "hop",
        "howlin": "howl",
        "hummin": "humm",
        "hunin": "hunt",
        "hurtin": "hurt",
        "hustlin": "hustle",
        "jamin": "jam",
        "judgement": "judgment",
        "jumpin": "jump",
        "keepin": "keep",
        "kickin": "kick",
        "kiddin": "kid",
        "kissin": "kiss",
        "knockin": "knock",
        "knowin": "know",
        "knew": "know",
        "laughin": "laugh",
        "layin": "lay",
        "leanin": "lean",
        "leavin": "leave",
        "lettin": "let",
        "lickin": "lick",
        "lightnin": "lightning",
        "listenin": "listen",
        "lookin": "look",
        "losin": "loose",
        "lyin": "lie",
        "marchin": "march",
        "needin": "need",
        "nothin": "nothing",
        "packin": "pack",
        "passin": "pass",
        "pimpin": "pimp",
        "poppin": "pop",
        "pourin": "pour",
        "prayin": "pray",
        "preachin": "preach",
        "pretendin": "pretend",
        "puffin": "puff",
        "pullin": "pull",
        "pummpin": "pump",
        "puttin": "put",
        "racin": "race",
        "rainin": "rain",
        "rappin": "rap",
        "rhymin": "rhyme",
        "ridin": "ride",
        "risin": "rise",
        "robbin": "rob",
        "rushin": "rush",
        "schemin": "scheme",
        "screamin": "scream",
        "searchin": "search",
        "settin": "set",
        "shinin": "shine",
        "shootin": "shoot",
        "sippin": "sip",
        "slammin": "slam",
        "slippin": "slip",
        "sniffin": "sniff",
        "speedin": "speed",
        "spillin": "spill",
        "suckin": "suck",
        "swingin": "swing",
        "taking": "take",
        "tickin": "tick",
        "trippin": "trip",
        "tumblin": "tumble",
        "wantin": "want",
        "whippin": "whip",
        "wishin": "wish",
        "wonderin": "wonder",
        "workin": "work",
        "worryin": "worry",
        "writin": "write",
        "yellin": "yell",
        "seein": "see",
        "sellin": "sell",
        "plannin": "plan",
        "timin": "time",
        "totin": "tot",
        "driftin": "drift",
        "disappears": "disappear",
        "thuggin": "thug",
        "touchin": "touch",
        "pushin": "push",
        "pumpin": "pump",
        "plottin": "plot",
        "aight": "alright",
        "lil": "little",
        "missin": "miss",
        "mixin": "mix",
        "mornin": "morning",
        "movin": "move",
        "walkin": "walk",
        "runnin": "run",
        "talkin": "talk",
        "tryin": "try",
        "livin": "live",
        "dyin": "die",
        "beggin": "beg",
        "killa": "killer",
        "killas": "killer",
        "killin": "kill",
        "drivin": "drive",
        "makin": "make",
        "hittin": "hit",
        "sittin": "sit",
        "chillin": "chill",
        "smokin": "smoke",
        "playin": "play",
        "praying": "pray",
        "ramblin": "ramble",
        "sailin": "sail",
        "savin": "sav",
        "shootin ": "shoot",
        "shoppin": "shop",
        "sinkin": "sink",
        "skippin": "skip",
        "slidin": "slide",
        "sleepin": "sleep",
        "slowin": "slow",
        "smilin": "smile",
        "sneakin": "sneak",
        "soakin": "soak",
        "somethin": "something",
        "lovin": "love",
        "speakin": "speak",
        "spendin": "spend",
        "spreadin": "spread",
        "stackin": "stack",
        "standin": "stand",
        "starin": "stare",
        "stayin": "stay",
        "swimmin": "swim",
        "stealin": "steal",
        "sweatin": "sweat",
        "steppin": "step",
        "stickin": "stick",
        "stinkin": "stink",
        "stompin": "stomp",
        "stoppin": "stop",
        "strugglin": "struggle",
        "stumblin": "stumble",
        "stuntin": "stunt",
        "stylin": "style",
        "suckas": "sucker",
        "switchin": "switch",
        "tellin": "tell",
        "thinkin": "think",
        "feelin": "feel",
        "winnin": "win",
        "sayin": "say",
        "wreckin": "wreck",
    }

    _GERUND_RE = re.compile(r"^(.{3,}?)(?:ing)$")

    _FILLER_TOKENS: Final[frozenset[str]] = frozenset(
        {
            "ah",
            "ahh",
            "ahhh",
            "aaaah",
            "aaah",
            "aah",
            "oh",
            "ohh",
            "ohhh",
            "ohhhh",
            "uh",
            "uhh",
            "mm",
            "mmh",
            "mmm",
            "mmmm",
            "hm",
            "hmm",
            "hmmm",
            "oo",
            "ooh",
            "oooh",
            "ooooh",
            "ay",
            "ayy",
            "ayyyy",
            "yeah",
            "yea",
            "yeh",
            "yo",
            "yuh",
            "huh",
            "hah",
            "haha",
            "hahah",
            "hahahaha",
            "hmm",
            "aw",
            "aww",
        }
    )

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
                tok_norm = self._normalize_gerund(tok_norm)
                wn_pos = self._wordnet_pos(pos)
                if wn_pos is None:
                    lemma = lemmatizer.lemmatize(tok_norm)
                else:
                    lemma = lemmatizer.lemmatize(tok_norm, pos=wn_pos)
                lemmas.append(lemma)
                lemmas = [
                    lemma
                    for lemma in lemmas
                    if lemma not in self._FILLER_TOKENS
                    and len(lemma) > 1
                    and not any(c.isdigit() for c in lemma)
                ]

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

    def _normalize_gerund(self, token: str) -> str:
        """Strip -ing suffix as a fallback for uncovered gerunds/progressives."""
        m = self._GERUND_RE.match(token)
        if m:
            stem = m.group(1)
            # Restore doubled consonant: running -> run, swimming -> swim, use filer to avoid over-trimming: sing -> sing, king -> king
            if len(stem) >= 3 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        return token

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
        self.lemmatized_corpus.to_csv(path, index=True)
