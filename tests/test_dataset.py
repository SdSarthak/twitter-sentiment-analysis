from __future__ import annotations

import pandas as pd
import pytest

from twitter_sentiment.dataset import (
    EmptyDatasetError,
    LIKES_COLUMN,
    RETWEETS_COLUMN,
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    add_model_input,
    engagement_summary,
    frame_from_texts,
    load_tweets,
    normalise_columns,
    prepare_frame,
    save_tweets,
    top_terms,
)


def test_load_tweets_returns_canonical_columns(sample_frame):
    for column in (TEXT_COLUMN, TIMESTAMP_COLUMN, LIKES_COLUMN, RETWEETS_COLUMN):
        assert column in sample_frame.columns
    assert "clean_text" in sample_frame.columns


def test_load_tweets_drops_exact_duplicates(sample_csv):
    raw = pd.read_csv(sample_csv)
    frame = load_tweets(sample_csv)
    assert len(frame) < len(raw)
    assert frame["clean_text"].duplicated().sum() == 0


def test_load_tweets_parses_timestamps_as_utc(sample_frame):
    assert str(sample_frame[TIMESTAMP_COLUMN].dtype) == "datetime64[ns, UTC]"
    assert sample_frame[TIMESTAMP_COLUMN].notna().all()


def test_load_tweets_counts_are_integers(sample_frame):
    assert sample_frame[LIKES_COLUMN].dtype.kind == "i"
    assert sample_frame[RETWEETS_COLUMN].dtype.kind == "i"


def test_load_tweets_missing_file():
    with pytest.raises(FileNotFoundError):
        load_tweets("does/not/exist.csv")


def test_load_tweets_rejects_unknown_text_column(sample_csv):
    with pytest.raises(KeyError):
        load_tweets(sample_csv, text_column="nope")


def test_normalise_columns_maps_aliases():
    frame = pd.DataFrame(
        {"text": ["a"], "created_at": ["2025-01-01"], "like_count": [1], "reposts": [2]}
    )
    renamed = normalise_columns(frame)
    assert list(renamed.columns) == [
        TEXT_COLUMN,
        TIMESTAMP_COLUMN,
        LIKES_COLUMN,
        RETWEETS_COLUMN,
    ]


def test_prepare_frame_parses_shorthand_counts():
    frame = pd.DataFrame(
        {
            TEXT_COLUMN: ["a great budget", "a terrible budget", "a neutral note"],
            LIKES_COLUMN: ["1.2K", "3M", "45"],
            RETWEETS_COLUMN: ["1,200", "", "7"],
        }
    )
    prepared = prepare_frame(frame)
    assert prepared[LIKES_COLUMN].tolist() == [1200, 3_000_000, 45]
    assert prepared[RETWEETS_COLUMN].tolist() == [1200, 0, 7]


def test_prepare_frame_drops_rows_that_clean_to_nothing():
    frame = pd.DataFrame({TEXT_COLUMN: ["real tweet here", "https://t.co/x", "  "]})
    assert len(prepare_frame(frame)) == 1


def test_prepare_frame_raises_when_nothing_survives():
    frame = pd.DataFrame({TEXT_COLUMN: ["https://t.co/x", "   "]})
    with pytest.raises(EmptyDatasetError):
        prepare_frame(frame)


def test_prepare_frame_tolerates_missing_optional_columns():
    prepared = prepare_frame(pd.DataFrame({TEXT_COLUMN: ["a great budget"]}))
    assert TIMESTAMP_COLUMN not in prepared.columns
    assert len(prepared) == 1


def test_prepare_frame_keeps_bad_timestamps_as_nat():
    frame = pd.DataFrame(
        {TEXT_COLUMN: ["a great budget"], TIMESTAMP_COLUMN: ["not a date"]}
    )
    assert prepare_frame(frame)[TIMESTAMP_COLUMN].isna().all()


def test_add_model_input(sample_frame):
    enriched = add_model_input(sample_frame)
    assert "model_input" in enriched.columns
    assert enriched["model_input"].str.len().gt(0).any()


def test_frame_from_texts():
    frame = frame_from_texts(["one", "two"])
    assert frame[TEXT_COLUMN].tolist() == ["one", "two"]


def test_save_tweets_creates_parent_directory(tmp_path, sample_frame):
    target = tmp_path / "nested" / "out.csv"
    written = save_tweets(sample_frame, target)
    assert written.is_file()
    assert len(pd.read_csv(written)) == len(sample_frame)


def test_engagement_summary(labelled_frame):
    summary = engagement_summary(labelled_frame)
    assert not summary.empty
    assert "tweets" in summary.columns
    assert summary["tweets"].sum() == len(labelled_frame)


def test_engagement_summary_without_metrics_is_empty():
    frame = pd.DataFrame({TEXT_COLUMN: ["a"], "sentiment": ["neutral"]})
    assert engagement_summary(frame).empty


def test_top_terms_counts_content_words():
    terms = top_terms(["budget budget market", "budget rally"], limit=2)
    assert terms.index[0] == "budget"
    assert terms.iloc[0] == 3
    assert len(terms) == 2
