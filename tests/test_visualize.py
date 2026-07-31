from __future__ import annotations

import pandas as pd

from twitter_sentiment.dataset import TEXT_COLUMN, TIMESTAMP_COLUMN
from twitter_sentiment.visualize import (
    plot_confusion_matrix,
    plot_score_histogram,
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    plot_sentiment_pie,
    plot_top_terms,
    render_all,
)

LABELS = ["positive", "positive", "negative", "neutral"]


def test_distribution_chart_is_written(tmp_path):
    path = plot_sentiment_distribution(LABELS, tmp_path / "dist.png")
    assert path.is_file() and path.stat().st_size > 0


def test_pie_chart_is_written(tmp_path):
    assert plot_sentiment_pie(LABELS, tmp_path / "pie.png").is_file()


def test_pie_chart_handles_a_single_class(tmp_path):
    assert plot_sentiment_pie(["positive"] * 3, tmp_path / "pie.png").is_file()


def test_score_histogram_is_written(tmp_path):
    scores = [-0.8, -0.1, 0.0, 0.02, 0.4, 0.9]
    assert plot_score_histogram(scores, tmp_path / "hist.png").is_file()


def test_confusion_matrix_is_written(tmp_path):
    matrix = [[3, 1, 0], [0, 2, 1], [1, 0, 4]]
    path = plot_confusion_matrix(matrix, ["negative", "neutral", "positive"], tmp_path / "cm.png")
    assert path.is_file()


def test_top_terms_chart_skips_empty_input(tmp_path):
    assert plot_top_terms(pd.Series(dtype="int64"), tmp_path / "terms.png") is None


def test_trend_chart_is_written_when_timestamps_exist(labelled_frame, tmp_path):
    path = plot_sentiment_over_time(labelled_frame, tmp_path / "trend.png")
    assert path is not None and path.is_file()


def test_trend_chart_skipped_without_timestamps(tmp_path):
    frame = pd.DataFrame({TEXT_COLUMN: ["a"], "sentiment": ["positive"]})
    assert plot_sentiment_over_time(frame, tmp_path / "trend.png") is None


def test_trend_chart_skipped_when_all_timestamps_are_invalid(tmp_path):
    frame = pd.DataFrame(
        {
            TEXT_COLUMN: ["a", "b"],
            TIMESTAMP_COLUMN: [pd.NaT, pd.NaT],
            "sentiment": ["positive", "negative"],
        }
    )
    assert plot_sentiment_over_time(frame, tmp_path / "trend.png") is None


def test_render_all_writes_every_supported_chart(labelled_frame, tmp_path):
    written = render_all(labelled_frame, tmp_path)
    assert len(written) == 5
    assert all(path.is_file() for path in written)
    assert {path.name for path in written} == {
        "sentiment_distribution.png",
        "sentiment_share.png",
        "score_histogram.png",
        "top_terms.png",
        "sentiment_over_time.png",
    }


def test_render_all_creates_the_output_directory(labelled_frame, tmp_path):
    target = tmp_path / "nested" / "charts"
    written = render_all(labelled_frame, target)
    assert target.is_dir() and written
