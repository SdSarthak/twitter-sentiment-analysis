"""Loading and normalising scraped tweet CSVs.

The two CSVs this project produces have slightly different shapes (the raw
Selenium dump is text only, the enriched one carries timestamp/likes/retweets),
and hand-edited exports show up with all sorts of column spellings. Everything
downstream expects the canonical schema defined here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from twitter_sentiment.preprocessing import clean_text, preprocess

TEXT_COLUMN = "Tweet"
TIMESTAMP_COLUMN = "Timestamp"
LIKES_COLUMN = "Likes"
RETWEETS_COLUMN = "Retweets"

CANONICAL_COLUMNS = (TEXT_COLUMN, TIMESTAMP_COLUMN, LIKES_COLUMN, RETWEETS_COLUMN)

#: Lowercased aliases seen in the wild -> canonical column name.
COLUMN_ALIASES: dict[str, str] = {
    "tweet": TEXT_COLUMN,
    "tweets": TEXT_COLUMN,
    "text": TEXT_COLUMN,
    "content": TEXT_COLUMN,
    "body": TEXT_COLUMN,
    "full_text": TEXT_COLUMN,
    "tweet_text": TEXT_COLUMN,
    "timestamp": TIMESTAMP_COLUMN,
    "time": TIMESTAMP_COLUMN,
    "date": TIMESTAMP_COLUMN,
    "datetime": TIMESTAMP_COLUMN,
    "created_at": TIMESTAMP_COLUMN,
    "likes": LIKES_COLUMN,
    "like_count": LIKES_COLUMN,
    "favorites": LIKES_COLUMN,
    "favourite_count": LIKES_COLUMN,
    "retweets": RETWEETS_COLUMN,
    "retweet_count": RETWEETS_COLUMN,
    "reposts": RETWEETS_COLUMN,
}


class EmptyDatasetError(ValueError):
    """Raised when a CSV contains no usable tweets."""


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to the canonical schema, leaving extras untouched."""
    renamed: dict[str, str] = {}
    taken = set()
    for column in frame.columns:
        canonical = COLUMN_ALIASES.get(str(column).strip().lower())
        if canonical and canonical not in taken and canonical != column:
            renamed[column] = canonical
            taken.add(canonical)
        elif canonical:
            taken.add(canonical)
    return frame.rename(columns=renamed)


def _coerce_counts(series: pd.Series) -> pd.Series:
    """Parse engagement counts, tolerating ``1.2K`` / ``3M`` style shorthand."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype("int64")

    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    multiplier = pd.Series(1.0, index=series.index)
    for suffix, factor in (("k", 1_000.0), ("m", 1_000_000.0), ("b", 1_000_000_000.0)):
        mask = text.str.lower().str.endswith(suffix).fillna(False)
        multiplier[mask] = factor
        text = text.mask(mask, text.str.slice(stop=-1))
    numbers = pd.to_numeric(text, errors="coerce").fillna(0.0)
    return (numbers * multiplier).round().astype("int64")


def load_tweets(
    path: str | Path,
    *,
    text_column: str | None = None,
    drop_duplicates: bool = True,
    min_length: int = 3,
) -> pd.DataFrame:
    """Read a tweet CSV and return a clean, canonical frame.

    Adds a ``clean_text`` column, drops rows whose cleaned text is empty or
    shorter than ``min_length`` characters, and de-duplicates on the cleaned
    text (scrolling scrapers re-capture the same tweet repeatedly).
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No tweet CSV at {csv_path}. Run `python Main.py scrape --query ...` "
            "first, or point --input at an existing export."
        )

    frame = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    if text_column:
        if text_column not in frame.columns:
            raise KeyError(
                f"Column {text_column!r} not found; available: {list(frame.columns)}"
            )
        frame = frame.rename(columns={text_column: TEXT_COLUMN})
    frame = normalise_columns(frame)

    if TEXT_COLUMN not in frame.columns:
        raise KeyError(
            f"{csv_path} has no recognisable tweet text column "
            f"(looked for {sorted(set(COLUMN_ALIASES))}); found {list(frame.columns)}."
        )

    return prepare_frame(
        frame, drop_duplicates=drop_duplicates, min_length=min_length, source=str(csv_path)
    )


def prepare_frame(
    frame: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    min_length: int = 3,
    source: str = "<dataframe>",
) -> pd.DataFrame:
    """Apply the canonical cleaning steps to an already-loaded frame."""
    prepared = frame.copy()
    prepared[TEXT_COLUMN] = prepared[TEXT_COLUMN].astype("string").fillna("")
    prepared["clean_text"] = [clean_text(value) for value in prepared[TEXT_COLUMN]]

    prepared = prepared[prepared["clean_text"].str.len() >= min_length]
    if drop_duplicates:
        prepared = prepared.drop_duplicates(subset="clean_text", keep="first")

    if TIMESTAMP_COLUMN in prepared.columns:
        prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
            prepared[TIMESTAMP_COLUMN], errors="coerce", utc=True, format="mixed"
        )
    for column in (LIKES_COLUMN, RETWEETS_COLUMN):
        if column in prepared.columns:
            prepared[column] = _coerce_counts(prepared[column])

    prepared = prepared.reset_index(drop=True)
    if prepared.empty:
        raise EmptyDatasetError(
            f"{source} contained no usable tweets after cleaning "
            f"(min_length={min_length})."
        )
    return prepared


def add_model_input(
    frame: pd.DataFrame,
    *,
    column: str = "model_input",
    stem: bool = True,
) -> pd.DataFrame:
    """Attach the stemmed, stop-word-free string the vectorisers consume."""
    enriched = frame.copy()
    enriched[column] = [preprocess(value, stem=stem) for value in enriched[TEXT_COLUMN]]
    return enriched


def save_tweets(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a frame to CSV, creating the parent directory if needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def frame_from_texts(texts: Iterable[object]) -> pd.DataFrame:
    """Build a canonical frame from bare strings (used by ``predict``)."""
    return pd.DataFrame({TEXT_COLUMN: [str(text) for text in texts]})


def engagement_summary(frame: pd.DataFrame, label_column: str = "sentiment") -> pd.DataFrame:
    """Mean likes/retweets per sentiment class, when those columns are present."""
    available = [c for c in (LIKES_COLUMN, RETWEETS_COLUMN) if c in frame.columns]
    if not available or label_column not in frame.columns:
        return pd.DataFrame()
    grouped = frame.groupby(label_column)[available].mean().round(2)
    grouped["tweets"] = frame.groupby(label_column).size()
    return grouped


def top_terms(texts: Sequence[object], limit: int = 15) -> pd.Series:
    """Most frequent content words -- a cheap stand-in for a word cloud."""
    from collections import Counter

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(preprocess(text, stem=False).split())
    return pd.Series(dict(counter.most_common(limit)), dtype="int64")
