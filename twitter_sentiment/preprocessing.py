"""Tweet text cleaning, tokenisation and stemming.

All helpers are pure functions on strings so they are trivially testable and
safe to reuse from the scraper, the model pipeline and the CLI.
"""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache
from typing import Iterable, Sequence

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[A-Za-z0-9_]{1,15}")
HASHTAG_RE = re.compile(r"#(\w+)")
RETWEET_RE = re.compile(r"^\s*RT\s+@?[A-Za-z0-9_]*\s*:?\s*", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^a-z0-9'\s]")
WHITESPACE_RE = re.compile(r"\s+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")

# Words that carry no sentiment on their own but appear in nearly every tweet in
# a hashtag-driven dataset. Used on top of the general stop word list.
DOMAIN_STOPWORDS = frozenset(
    {"rt", "amp", "via", "http", "https", "co", "twitter", "tweet", "retweet"}
)

# Offline fallback so the package never *requires* an NLTK download to run.
FALLBACK_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can't cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he he'd he'll he's her here here's hers herself him himself his how
    how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our ours ourselves out
    over own same shan't she she'd she'll she's should shouldn't so some such than that
    that's the their theirs them themselves then there there's these they they'd they'll
    they're they've this those through to too under until up very was wasn't we we'd we'll
    we're we've were weren't what what's when when's where where's which while who who's
    whom why why's with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves
    """.split()
)


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """Return English stop words, preferring NLTK's list when it is downloaded.

    Falls back to a bundled list so nothing breaks offline. Negation words are
    deliberately kept: ``not good`` must not collapse to ``good``.
    """
    words: set[str] = set(FALLBACK_STOPWORDS)
    try:
        from nltk.corpus import stopwords as nltk_stopwords

        words.update(nltk_stopwords.words("english"))
    except Exception:  # pragma: no cover - depends on local NLTK data
        pass
    words |= set(DOMAIN_STOPWORDS)
    words -= set(NEGATION_WORDS)
    return frozenset(words)


NEGATION_WORDS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nowhere",
        "cannot",
        "cant",
        "can't",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "isnt",
        "isn't",
        "wasnt",
        "wasn't",
        "arent",
        "aren't",
        "werent",
        "weren't",
        "wont",
        "won't",
        "wouldnt",
        "wouldn't",
        "shouldnt",
        "shouldn't",
        "couldnt",
        "couldn't",
        "hardly",
        "barely",
        "rarely",
        "without",
        "lack",
        "lacks",
    }
)


def split_hashtag(tag: str) -> str:
    """Split ``#StockMarketCrash`` into ``Stock Market Crash``.

    Digits are separated too, so ``Budget2025`` becomes ``Budget 2025``.
    """
    tag = tag.lstrip("#")
    spaced = CAMEL_RE.sub(" ", tag)
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", spaced)
    spaced = spaced.replace("_", " ")
    return WHITESPACE_RE.sub(" ", spaced).strip()


def normalise_unicode(text: str) -> str:
    """Unescape HTML entities and normalise curly quotes and dashes."""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", " ")
        .replace("–", " ")
    )


def clean_text(
    text: object,
    *,
    expand_hashtags: bool = True,
    drop_mentions: bool = True,
    lowercase: bool = True,
) -> str:
    """Strip the boilerplate that carries no sentiment out of a tweet.

    Removes URLs, the leading ``RT @user:`` marker and (optionally) @mentions,
    expands hashtags into their component words, collapses stretched characters
    (``sooooo`` -> ``soo``) and squashes whitespace.
    """
    if text is None:
        return ""
    raw = normalise_unicode(text)
    if not raw.strip():
        return ""

    raw = RETWEET_RE.sub(" ", raw)
    raw = URL_RE.sub(" ", raw)
    if drop_mentions:
        raw = MENTION_RE.sub(" ", raw)
    if expand_hashtags:
        raw = HASHTAG_RE.sub(lambda m: " " + split_hashtag(m.group(1)) + " ", raw)
    else:
        raw = HASHTAG_RE.sub(" ", raw)

    raw = REPEATED_CHAR_RE.sub(r"\1\1", raw)
    if lowercase:
        raw = raw.lower()
    return WHITESPACE_RE.sub(" ", raw).strip()


def tokenize(text: object) -> list[str]:
    """Lowercase word tokens, punctuation dropped, contractions preserved."""
    cleaned = clean_text(text)
    cleaned = NON_WORD_RE.sub(" ", cleaned)
    return [token for token in cleaned.split() if token and token != "'"]


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    stops = get_stopwords()
    return [token for token in tokens if token not in stops and len(token) > 1]


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem.porter import PorterStemmer

    return PorterStemmer()


def stem_tokens(tokens: Iterable[str]) -> list[str]:
    """Porter-stem tokens, degrading gracefully if NLTK is unavailable."""
    try:
        stemmer = _stemmer()
    except Exception:  # pragma: no cover - NLTK is a hard requirement in practice
        return list(tokens)
    return [stemmer.stem(token) for token in tokens]


def preprocess(text: object, *, stem: bool = True, drop_stopwords: bool = True) -> str:
    """Full cleaning chain returning a space-joined string for vectorisers."""
    tokens = tokenize(text)
    if drop_stopwords:
        tokens = remove_stopwords(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


def preprocess_many(texts: Sequence[object], **kwargs: object) -> list[str]:
    return [preprocess(text, **kwargs) for text in texts]  # type: ignore[arg-type]


def extract_hashtags(text: object) -> list[str]:
    """Return the lowercased hashtags found in a tweet, in order, deduplicated."""
    if text is None:
        return []
    seen: dict[str, None] = {}
    for match in HASHTAG_RE.finditer(normalise_unicode(text)):
        seen.setdefault(match.group(1).lower(), None)
    return list(seen)


def extract_mentions(text: object) -> list[str]:
    """Return the lowercased @handles found in a tweet, in order, deduplicated."""
    if text is None:
        return []
    seen: dict[str, None] = {}
    for match in MENTION_RE.finditer(normalise_unicode(text)):
        seen.setdefault(match.group(0)[1:].lower(), None)
    return list(seen)
