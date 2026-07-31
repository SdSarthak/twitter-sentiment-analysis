"""Turn continuous sentiment scores into the three classes the README promises."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from twitter_sentiment.lexicon import score_texts

POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"

LABELS: tuple[str, str, str] = (NEGATIVE, NEUTRAL, POSITIVE)

#: Scores within +/- this band are treated as neutral. 0.05 is the conventional
#: VADER cut-off and keeps the three classes reasonably balanced on tweets.
DEFAULT_THRESHOLD = 0.05


def label_for_score(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Map a score in ``[-1, 1]`` onto ``negative``/``neutral``/``positive``."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if score >= threshold:
        return POSITIVE
    if score <= -threshold:
        return NEGATIVE
    return NEUTRAL


def label_texts(
    texts: Iterable[object],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    scorer: str = "builtin",
) -> tuple[list[float], list[str]]:
    """Score and label an iterable of tweets, returning ``(scores, labels)``."""
    scores = score_texts(texts, scorer=scorer)
    return scores, [label_for_score(score, threshold) for score in scores]


def label_dataframe(
    frame: pd.DataFrame,
    *,
    text_column: str = "Tweet",
    score_column: str = "sentiment_score",
    label_column: str = "sentiment",
    threshold: float = DEFAULT_THRESHOLD,
    scorer: str = "builtin",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with score and label columns attached."""
    if text_column not in frame.columns:
        raise KeyError(
            f"Column {text_column!r} not found; available: {list(frame.columns)}"
        )
    labelled = frame.copy()
    scores, labels = label_texts(
        labelled[text_column].tolist(), threshold=threshold, scorer=scorer
    )
    labelled[score_column] = scores
    labelled[label_column] = labels
    return labelled


def label_distribution(labels: Sequence[str]) -> pd.Series:
    """Counts per class, always covering all three labels (zeros included)."""
    counts = pd.Series(list(labels), dtype="object").value_counts()
    return counts.reindex(LABELS, fill_value=0).astype(int)


def distribution_summary(labels: Sequence[str]) -> pd.DataFrame:
    """Counts and percentages per class, ordered negative -> neutral -> positive."""
    counts = label_distribution(labels)
    total = int(counts.sum())
    share = (counts / total * 100).round(2) if total else counts.astype(float)
    return pd.DataFrame({"count": counts, "percentage": share})
