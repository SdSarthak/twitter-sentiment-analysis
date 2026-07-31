"""Shared fixtures. Everything here is synthetic -- no network, no real data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURE_DIR / "sample_tweets.csv"


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    return SAMPLE_CSV


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    from twitter_sentiment.dataset import load_tweets

    return load_tweets(SAMPLE_CSV)


@pytest.fixture()
def labelled_frame(sample_frame: pd.DataFrame) -> pd.DataFrame:
    from twitter_sentiment.labeling import label_dataframe

    return label_dataframe(sample_frame)
