"""An offline valence lexicon for scoring tweets.

Scraped tweets arrive unlabelled, so something has to produce the ground truth
the classifiers learn from. This module is a self-contained, deterministic
rule-based scorer in the spirit of VADER: a word-level valence table plus
handling for negation, intensifiers, capitalisation, punctuation and emoji.

It ships with the package on purpose -- no corpus download, no network call --
so labelling, training and the test suite all run offline. If NLTK's VADER
lexicon happens to be installed you can opt into it with
``score_text(text, scorer="vader")``.

Scores are normalised to ``[-1.0, 1.0]``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Sequence

from twitter_sentiment.preprocessing import (
    HASHTAG_RE,
    MENTION_RE,
    NEGATION_WORDS,
    REPEATED_CHAR_RE,
    URL_RE,
    WHITESPACE_RE,
    normalise_unicode,
    split_hashtag,
)

# --------------------------------------------------------------------------- #
# Valence table
# --------------------------------------------------------------------------- #

POSITIVE_WORDS: dict[str, float] = {
    # general praise
    "good": 1.9, "great": 2.6, "excellent": 3.0, "amazing": 2.8, "awesome": 3.0,
    "fantastic": 2.9, "wonderful": 2.8, "superb": 2.9, "brilliant": 2.7,
    "outstanding": 2.9, "perfect": 2.8, "best": 2.6, "better": 1.6, "nice": 1.7,
    "love": 2.8, "loved": 2.6, "loves": 2.4, "like": 1.3, "liked": 1.3,
    "happy": 2.4, "glad": 1.9, "joy": 2.4, "delight": 2.3, "delighted": 2.4,
    "pleased": 2.0, "proud": 2.2, "grateful": 2.3, "thankful": 2.3,
    "thanks": 1.8, "thank": 1.8, "appreciate": 2.0, "appreciated": 2.0,
    "congrats": 2.3, "congratulations": 2.4, "kudos": 2.2, "bravo": 2.3,
    "impressive": 2.4, "impressed": 2.2, "remarkable": 2.3, "exceptional": 2.6,
    "solid": 1.5, "strong": 1.6, "smart": 1.7, "clever": 1.7, "useful": 1.6,
    "helpful": 1.9, "beneficial": 1.9, "valuable": 1.9, "worthy": 1.6,
    "positive": 1.8, "optimistic": 2.0, "hopeful": 1.8, "hope": 1.4,
    "confident": 1.8, "reliable": 1.8, "trust": 1.8, "trusted": 1.8,
    "safe": 1.4, "secure": 1.5, "stable": 1.4, "steady": 1.2, "healthy": 1.7,
    "clear": 1.0, "clean": 1.2, "fair": 1.3, "honest": 1.9, "genuine": 1.7,
    "transparent": 1.6, "bold": 1.2, "historic": 1.3, "landmark": 1.5,
    "welcome": 1.7, "welcomed": 1.7, "support": 1.4, "supports": 1.4,
    "supported": 1.4, "supporting": 1.4, "praise": 2.0, "praised": 2.0,
    "win": 2.2, "wins": 2.2, "winner": 2.3, "winning": 2.2, "won": 2.0,
    "success": 2.4, "successful": 2.3, "achieve": 1.9, "achieved": 1.9,
    "achievement": 2.1, "progress": 1.7, "improve": 1.7, "improved": 1.9,
    "improvement": 1.9, "breakthrough": 2.4, "milestone": 1.8, "boost": 1.8,
    "boosted": 1.8, "reward": 1.8, "rewarding": 2.0, "opportunity": 1.7,
    "opportunities": 1.7, "promising": 2.0, "potential": 1.0, "growth": 1.7,
    "recovery": 1.5, "relief": 1.7, "benefit": 1.8, "benefits": 1.8,
    "advantage": 1.6, "efficient": 1.7, "effective": 1.7, "productive": 1.7,
    "innovative": 2.0, "innovation": 1.9, "quality": 1.4, "premium": 1.3,
    "affordable": 1.6, "cheap": 0.6, "discount": 1.2, "free": 1.0,
    "easy": 1.3, "simple": 1.0, "smooth": 1.5, "fast": 1.0, "quick": 0.9,
    "excited": 2.2, "exciting": 2.2, "enjoy": 2.1, "enjoyed": 2.1,
    "fun": 1.9, "beautiful": 2.4, "cool": 1.6, "fine": 1.1, "fantastically": 2.6,
    "recommend": 1.9, "recommended": 1.9, "favourite": 2.0, "favorite": 2.0,
    "satisfied": 2.0, "satisfying": 2.0, "satisfaction": 2.0, "comfort": 1.6,
    "peace": 1.9, "peaceful": 2.0, "friendly": 1.8, "kind": 1.6, "generous": 2.0,
    "inspiring": 2.2, "inspired": 2.0, "motivated": 1.9, "encouraging": 2.0,
    # market / policy specific
    "bullish": 2.2, "rally": 1.9, "rallied": 1.9, "surge": 1.9, "surged": 1.9,
    "soar": 2.1, "soared": 2.1, "soaring": 2.1, "jump": 1.2, "jumped": 1.2,
    "gain": 1.7, "gains": 1.7, "gained": 1.7, "profit": 1.8, "profits": 1.8,
    "profitable": 2.0, "upside": 1.5, "outperform": 1.9, "outperformed": 1.9,
    "rebound": 1.6, "rebounded": 1.6, "record": 1.0, "high": 0.6, "up": 0.5,
    "upgrade": 1.6, "upgraded": 1.6, "dividend": 1.0, "bonus": 1.4,
    "exemption": 1.4, "exempt": 1.3, "rebate": 1.4, "reform": 1.2,
    "incentive": 1.4, "subsidy": 1.0, "savings": 1.5, "cheer": 2.0,
    "cheers": 2.0, "boom": 1.8, "booming": 1.9, "thrive": 2.0, "thriving": 2.1,
    "prosperity": 2.2, "prosperous": 2.2, "resilient": 1.8, "robust": 1.8,
}

NEGATIVE_WORDS: dict[str, float] = {
    # general criticism
    "bad": -2.0, "worse": -2.2, "worst": -2.8, "terrible": -2.8, "awful": -2.8,
    "horrible": -2.9, "poor": -2.0, "pathetic": -2.6, "useless": -2.5,
    "worthless": -2.7, "garbage": -2.6, "trash": -2.5, "rubbish": -2.4,
    "nonsense": -2.2, "stupid": -2.5, "dumb": -2.3, "idiotic": -2.6,
    "ridiculous": -2.2, "absurd": -2.1, "shameful": -2.6, "shame": -2.2,
    "disgusting": -2.9, "disgusted": -2.7, "hate": -2.9, "hated": -2.7,
    "hates": -2.7, "dislike": -1.9, "angry": -2.4, "anger": -2.3, "mad": -2.0,
    "furious": -2.7, "outrage": -2.6, "outrageous": -2.5, "annoyed": -2.0,
    "annoying": -2.1, "frustrated": -2.3, "frustrating": -2.3, "upset": -2.1,
    "sad": -2.2, "unhappy": -2.2, "miserable": -2.6, "depressing": -2.5,
    "depressed": -2.5, "disappointed": -2.5, "disappointing": -2.4,
    "disappointment": -2.4, "regret": -2.0, "sorry": -1.2, "cry": -1.8,
    "crying": -1.8, "pain": -2.0, "painful": -2.2, "suffer": -2.3,
    "suffering": -2.4, "struggle": -1.9, "struggling": -2.0, "hurt": -2.1,
    "damage": -2.1, "damaged": -2.1, "destroy": -2.6, "destroyed": -2.6,
    "ruin": -2.5, "ruined": -2.5, "broken": -2.1, "broke": -1.8, "fail": -2.4,
    "failed": -2.4, "failing": -2.4, "failure": -2.5, "flop": -2.2,
    "problem": -1.8, "problems": -1.8, "issue": -1.2, "issues": -1.2,
    "trouble": -1.9, "difficult": -1.5, "hard": -1.0, "tough": -1.3,
    "concern": -1.3, "concerned": -1.5, "concerns": -1.3, "worry": -1.9,
    "worried": -2.0, "worrying": -2.0, "fear": -2.2, "fearful": -2.3,
    "afraid": -2.0, "panic": -2.5, "scared": -2.2, "scary": -2.1,
    "risk": -1.2, "risky": -1.7, "danger": -2.3, "dangerous": -2.4,
    "threat": -2.1, "crisis": -2.6, "disaster": -2.8, "disastrous": -2.8,
    "catastrophe": -2.9, "chaos": -2.4, "mess": -2.0, "messy": -1.7,
    "corrupt": -2.8, "corruption": -2.8, "fraud": -2.9, "scam": -2.9,
    "cheat": -2.6, "cheated": -2.6, "lie": -2.3, "lies": -2.3, "liar": -2.7,
    "fake": -2.3, "misleading": -2.2, "unfair": -2.3, "injustice": -2.6,
    "biased": -2.0, "ignored": -1.8, "ignore": -1.6, "neglect": -2.2,
    "neglected": -2.2, "insult": -2.5, "insulting": -2.5, "abuse": -2.8,
    "attack": -2.2, "attacked": -2.2, "violence": -2.8, "war": -2.2,
    "kill": -2.8, "killed": -2.8, "death": -2.6, "dead": -2.4, "died": -2.4,
    "wrong": -1.9, "mistake": -1.9, "error": -1.7, "flaw": -1.8, "flawed": -2.0,
    "weak": -1.8, "weakness": -1.8, "slow": -1.1, "delay": -1.5,
    "delayed": -1.5, "expensive": -1.5, "costly": -1.6, "waste": -2.2,
    "wasted": -2.2, "boring": -1.9, "bored": -1.8, "confusing": -1.6,
    "confused": -1.5, "complicated": -1.3, "doubt": -1.4, "doubtful": -1.6,
    "unfortunately": -1.7, "unfortunate": -1.8, "sick": -1.8, "tired": -1.4,
    "hopeless": -2.6, "helpless": -2.3, "desperate": -2.2, "shocking": -2.0,
    "shocked": -1.9, "protest": -1.4, "oppose": -1.5, "opposed": -1.5,
    "reject": -1.9, "rejected": -2.0, "criticise": -1.9, "criticised": -1.9,
    "criticize": -1.9, "criticized": -1.9, "criticism": -1.8, "blame": -2.0,
    "blamed": -2.0, "condemn": -2.4, "condemned": -2.4,
    # market / policy specific
    "bearish": -2.0, "crash": -2.7, "crashed": -2.7, "crashing": -2.7,
    "plunge": -2.3, "plunged": -2.3, "plummet": -2.4, "plummeted": -2.4,
    "slump": -2.2, "slumped": -2.2, "tumble": -2.0, "tumbled": -2.0,
    "fall": -1.5, "falls": -1.5, "falling": -1.6, "fell": -1.5, "drop": -1.5,
    "dropped": -1.5, "dropping": -1.5, "decline": -1.7, "declined": -1.7,
    "sink": -1.8, "sinking": -1.9, "loss": -2.1, "losses": -2.2, "lose": -2.0,
    "losing": -2.1, "lost": -2.0, "downgrade": -2.0, "downgraded": -2.0,
    "downside": -1.6, "selloff": -2.1, "sell": -0.8, "correction": -1.2,
    "recession": -2.6, "inflation": -1.6, "slowdown": -2.0, "stagnation": -2.1,
    "deficit": -1.8, "debt": -1.5, "burden": -1.9, "hike": -1.2, "tax": -0.6,
    "taxes": -0.6, "taxed": -1.0, "cut": -0.5, "unemployment": -2.4,
    "layoff": -2.5, "layoffs": -2.5, "bankrupt": -2.9, "bankruptcy": -2.9,
    "volatile": -1.4, "volatility": -1.3, "uncertain": -1.7,
    "uncertainty": -1.8, "bubble": -1.5, "overvalued": -1.6, "underperform": -1.9,
    "redflag": -2.0, "bloodbath": -2.7, "meltdown": -2.7,
}

VALENCE: dict[str, float] = {**POSITIVE_WORDS, **NEGATIVE_WORDS}

# Multiplicative modifiers applied to the *following* sentiment word.
BOOSTERS: dict[str, float] = {
    "absolutely": 0.35, "completely": 0.30, "extremely": 0.40, "very": 0.28,
    "really": 0.28, "so": 0.22, "too": 0.20, "totally": 0.30, "utterly": 0.35,
    "highly": 0.30, "incredibly": 0.40, "exceptionally": 0.40, "super": 0.32,
    "insanely": 0.40, "hugely": 0.35, "massively": 0.35, "seriously": 0.28,
    "truly": 0.28, "deeply": 0.30, "particularly": 0.22, "especially": 0.25,
    "quite": 0.15, "most": 0.20, "more": 0.18, "much": 0.18, "far": 0.20,
    "damn": 0.30, "freaking": 0.35, "fucking": 0.40,
    # dampeners
    "slightly": -0.25, "somewhat": -0.22, "kinda": -0.25, "kind": -0.15,
    "sort": -0.15, "little": -0.20, "bit": -0.22, "marginally": -0.25,
    "barely": -0.30, "hardly": -0.30, "partly": -0.20, "almost": -0.15,
    "less": -0.25, "least": -0.25, "occasionally": -0.20,
}

# Idioms whose sentiment is not the sum of their parts.
IDIOMS: dict[str, float] = {
    "not good": -1.6, "not bad": 1.2, "no good": -1.8, "no problem": 1.2,
    "well done": 2.4, "thank you": 2.0, "the best": 2.8, "the worst": -3.0,
    "big deal": 1.0, "no thanks": -1.0, "hats off": 2.2, "game changer": 2.4,
    "waste of time": -2.6, "waste of money": -2.7, "rip off": -2.6,
    "cutting edge": 2.0, "state of the art": 2.2, "red flag": -2.0,
    "wake up call": -1.5, "good luck": 1.6, "worth it": 1.9,
    "not worth": -2.0, "no issue": 1.2, "no issues": 1.2,
}

EMOJI_VALENCE: dict[str, float] = {
    "😀": 2.0, "😃": 2.2, "😄": 2.3, "😁": 2.2, "😆": 2.0, "😊": 2.3, "🙂": 1.4,
    "😍": 2.8, "🥰": 2.8, "😘": 2.4, "🤩": 2.6, "😎": 1.8, "👍": 2.0, "👏": 2.1,
    "🙌": 2.2, "💪": 1.9, "🔥": 1.6, "✨": 1.5, "🎉": 2.4, "🥳": 2.5, "❤": 2.6,
    "❤️": 2.6, "💚": 2.2, "💯": 2.2, "🚀": 2.0, "📈": 1.5, "🏆": 2.3, "😂": 1.6,
    "🤣": 1.8, "🙏": 1.2, "😢": -2.0, "😭": -2.3, "😞": -2.1, "😔": -2.0,
    "😡": -2.7, "😠": -2.4, "🤬": -2.9, "👎": -2.2, "💔": -2.5, "😰": -2.1,
    "😱": -2.2, "😨": -2.0, "🤮": -2.7, "🤢": -2.4, "📉": -1.6, "⚠": -1.2,
    "⚠️": -1.2, "🚨": -1.3, "😐": -0.3, "😕": -1.2, "🙄": -1.5, "😤": -1.6,
}

EMOTICON_VALENCE: dict[str, float] = {
    ":)": 1.8, ":-)": 1.8, ":D": 2.3, ":-D": 2.3, "=)": 1.7, ":]": 1.6,
    ";)": 1.4, ";-)": 1.4, "<3": 2.4, ":p": 1.2, ":P": 1.2,
    ":(": -1.9, ":-(": -1.9, ":'(": -2.3, ":/": -1.0, ":-/": -1.0,
    ":|": -0.5, ">:(": -2.4, "</3": -2.2, ":o": 0.2,
}

NEGATION_SCALE = -0.74
NEGATION_WINDOW = 3
CAPS_BOOST = 0.733
EXCLAMATION_BOOST = 0.292
QUESTION_BOOST = -0.18
MAX_PUNCTUATION = 4
NORMALISATION_ALPHA = 15.0

_EMOTICON_RE = re.compile(
    "|".join(re.escape(token) for token in sorted(EMOTICON_VALENCE, key=len, reverse=True))
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*|\d+(?:\.\d+)?")
_PUNCT_STRIP = "\"'`.,;:!?()[]{}<>*_~-"


def _normalise(score: float) -> float:
    """Squash an unbounded valence sum into ``[-1, 1]`` (VADER's formula)."""
    normalised = score / ((score * score + NORMALISATION_ALPHA) ** 0.5)
    return max(-1.0, min(1.0, normalised))


def _strip_noise(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(lambda m: " " + split_hashtag(m.group(1)) + " ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _is_shouted(token: str) -> bool:
    return len(token) > 2 and token.isupper() and token.isalpha()


def _punctuation_amplifier(text: str) -> float:
    exclamations = min(text.count("!"), MAX_PUNCTUATION)
    questions = min(text.count("?"), MAX_PUNCTUATION)
    return exclamations * EXCLAMATION_BOOST + questions * QUESTION_BOOST


def _emoji_score(text: str) -> float:
    total = 0.0
    for symbol, valence in EMOJI_VALENCE.items():
        count = text.count(symbol)
        if count:
            total += valence * min(count, 3)
    for match in _EMOTICON_RE.finditer(text):
        total += EMOTICON_VALENCE[match.group(0)]
    return total


def _idiom_score(lowered: str) -> float:
    total = 0.0
    for phrase, valence in IDIOMS.items():
        if phrase in lowered:
            total += valence
    return total


def _word_scores(tokens: Sequence[str]) -> float:
    total = 0.0
    lowered = [token.lower().strip(_PUNCT_STRIP) for token in tokens]

    for index, word in enumerate(lowered):
        if word not in VALENCE:
            continue
        valence = VALENCE[word]

        if _is_shouted(tokens[index]):
            valence += CAPS_BOOST if valence > 0 else -CAPS_BOOST

        # Intensifiers immediately before the sentiment word, decayed by distance.
        for distance in range(1, 3):
            position = index - distance
            if position < 0:
                break
            modifier = BOOSTERS.get(lowered[position])
            if modifier is None:
                continue
            scaled = modifier * (1.0 - 0.05 * (distance - 1))
            valence *= 1.0 + scaled

        # Negation anywhere in the preceding window flips and dampens.
        window = lowered[max(0, index - NEGATION_WINDOW) : index]
        if any(token in NEGATION_WORDS for token in window):
            valence *= NEGATION_SCALE

        total += valence
    return total


def _prepare(text: object) -> tuple[str, list[str]]:
    raw = normalise_unicode(text if text is not None else "")
    stripped = _strip_noise(raw)
    stripped = REPEATED_CHAR_RE.sub(r"\1\1", stripped)
    tokens = _TOKEN_RE.findall(stripped)
    return stripped, tokens


def score_text(text: object, *, scorer: str = "builtin") -> float:
    """Return a sentiment score in ``[-1.0, 1.0]`` for a single tweet.

    ``scorer="builtin"`` (default) uses the bundled lexicon and always works
    offline. ``scorer="vader"`` uses NLTK's VADER lexicon and raises a helpful
    error if that corpus has not been downloaded.
    """
    if scorer == "vader":
        return _vader_analyzer().polarity_scores(str(text or ""))["compound"]
    if scorer != "builtin":
        raise ValueError(f"Unknown scorer {scorer!r}; use 'builtin' or 'vader'.")

    if text is None:
        return 0.0
    stripped, tokens = _prepare(text)
    if not stripped:
        return 0.0

    total = _word_scores(tokens)
    total += _idiom_score(stripped.lower())
    total += _emoji_score(stripped)

    if total != 0.0:
        amplifier = _punctuation_amplifier(stripped)
        total += amplifier if total > 0 else -amplifier

    return round(_normalise(total), 4)


def score_texts(texts: Iterable[object], *, scorer: str = "builtin") -> list[float]:
    return [score_text(text, scorer=scorer) for text in texts]


@lru_cache(maxsize=1)
def _vader_analyzer():  # pragma: no cover - requires an optional NLTK download
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "The 'vader' scorer needs NLTK. Install it with `pip install nltk`."
        ) from exc
    try:
        return SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise RuntimeError(
            "The VADER lexicon is not downloaded. Run:\n"
            "  python -c \"import nltk; nltk.download('vader_lexicon')\"\n"
            "or use the default offline scorer."
        ) from exc


def explain(text: object) -> dict[str, object]:
    """Break a score into its contributing parts -- handy when tuning labels."""
    stripped, tokens = _prepare(text)
    lowered = [token.lower().strip(_PUNCT_STRIP) for token in tokens]
    matched = {word: VALENCE[word] for word in lowered if word in VALENCE}
    return {
        "cleaned": stripped,
        "matched_words": matched,
        "word_score": round(_word_scores(tokens), 4),
        "idiom_score": round(_idiom_score(stripped.lower()), 4),
        "emoji_score": round(_emoji_score(stripped), 4),
        "punctuation_amplifier": round(_punctuation_amplifier(stripped), 4),
        "score": score_text(text),
    }
