"""Charts for the sentiment report.

Matplotlib runs on the headless ``Agg`` backend so these work from a terminal,
a cron job or CI. Every function writes a PNG and returns its path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from twitter_sentiment.dataset import TIMESTAMP_COLUMN  # noqa: E402
from twitter_sentiment.labeling import (  # noqa: E402
    LABELS,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    label_distribution,
)

PALETTE: dict[str, str] = {
    NEGATIVE: "#d1495b",
    NEUTRAL: "#8d99ae",
    POSITIVE: "#2a9d8f",
}

sns.set_theme(style="whitegrid")


def _prepare_axes(title: str, figsize: tuple[float, float] = (8.0, 5.0)):
    figure, axes = plt.subplots(figsize=figsize)
    axes.set_title(title, fontsize=13, weight="bold")
    return figure, axes


def _save(figure, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return out_path


def plot_sentiment_distribution(
    labels: Sequence[str], path: str | Path, *, title: str = "Sentiment distribution"
) -> Path:
    """Bar chart of tweets per sentiment class, annotated with percentages."""
    counts = label_distribution(labels)
    total = int(counts.sum()) or 1
    figure, axes = _prepare_axes(title)
    bars = axes.bar(
        list(counts.index),
        list(counts.values),
        color=[PALETTE.get(label, "#4c566a") for label in counts.index],
    )
    for bar, value in zip(bars, counts.values):
        axes.annotate(
            f"{value} ({value / total * 100:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes.set_xlabel("sentiment")
    axes.set_ylabel("tweets")
    axes.set_ylim(0, max(counts.max() * 1.18, 1))
    return _save(figure, path)


def plot_sentiment_pie(
    labels: Sequence[str], path: str | Path, *, title: str = "Sentiment share"
) -> Path:
    """Pie chart of the same distribution, dropping empty classes."""
    counts = label_distribution(labels)
    counts = counts[counts > 0]
    figure, axes = _prepare_axes(title, figsize=(6.0, 6.0))
    axes.pie(
        list(counts.values),
        labels=list(counts.index),
        autopct="%1.1f%%",
        colors=[PALETTE.get(label, "#4c566a") for label in counts.index],
        startangle=120,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes.axis("equal")
    axes.grid(False)
    return _save(figure, path)


def plot_sentiment_over_time(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    label_column: str = "sentiment",
    timestamp_column: str = TIMESTAMP_COLUMN,
    freq: str = "h",
    title: str = "Sentiment over time",
) -> Path | None:
    """Stacked area of sentiment volume per time bucket.

    Returns ``None`` when the frame has no usable timestamps, so callers can
    skip the chart instead of crashing on the raw text-only export.
    """
    if timestamp_column not in frame.columns or label_column not in frame.columns:
        return None
    timed = frame[[timestamp_column, label_column]].dropna()
    timed = timed[pd.to_datetime(timed[timestamp_column], errors="coerce").notna()]
    if timed.empty:
        return None

    timed[timestamp_column] = pd.to_datetime(timed[timestamp_column], errors="coerce")
    grouped = (
        timed.groupby([pd.Grouper(key=timestamp_column, freq=freq), label_column])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[label for label in LABELS], fill_value=0)
    )
    if grouped.empty:
        return None

    figure, axes = _prepare_axes(title, figsize=(9.0, 5.0))
    axes.stackplot(
        grouped.index,
        [grouped[label] for label in grouped.columns],
        labels=list(grouped.columns),
        colors=[PALETTE.get(label, "#4c566a") for label in grouped.columns],
        alpha=0.85,
    )
    axes.set_xlabel("time")
    axes.set_ylabel("tweets")
    axes.legend(loc="upper left")
    figure.autofmt_xdate()
    return _save(figure, path)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    path: str | Path,
    *,
    title: str = "Confusion matrix",
) -> Path:
    """Heatmap of true vs predicted classes."""
    frame = pd.DataFrame(matrix, index=list(labels), columns=list(labels))
    figure, axes = _prepare_axes(title, figsize=(6.5, 5.5))
    sns.heatmap(
        frame,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=axes,
    )
    axes.set_xlabel("predicted")
    axes.set_ylabel("actual")
    return _save(figure, path)


def plot_top_terms(
    terms: pd.Series, path: str | Path, *, title: str = "Most frequent terms"
) -> Path | None:
    """Horizontal bar chart of the most common content words."""
    if terms is None or len(terms) == 0:
        return None
    ordered = terms.sort_values(ascending=True)
    figure, axes = _prepare_axes(title, figsize=(8.0, max(4.0, 0.35 * len(ordered))))
    axes.barh(list(ordered.index), list(ordered.values), color="#457b9d")
    axes.set_xlabel("occurrences")
    return _save(figure, path)


def plot_score_histogram(
    scores: Sequence[float],
    path: str | Path,
    *,
    threshold: float = 0.05,
    title: str = "Sentiment score distribution",
) -> Path:
    """Histogram of raw lexicon scores with the neutral band marked."""
    figure, axes = _prepare_axes(title)
    axes.hist(list(scores), bins=30, color="#457b9d", edgecolor="white")
    axes.axvspan(-threshold, threshold, color="#8d99ae", alpha=0.25, label="neutral band")
    axes.set_xlabel("sentiment score")
    axes.set_ylabel("tweets")
    axes.legend(loc="upper right")
    return _save(figure, path)


def render_all(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    label_column: str = "sentiment",
    score_column: str = "sentiment_score",
    threshold: float = 0.05,
) -> list[Path]:
    """Render the standard report charts, skipping any the data cannot support."""
    from twitter_sentiment.dataset import TEXT_COLUMN, top_terms

    out_dir = Path(output_dir)
    written: list[Path] = []
    labels = frame[label_column].tolist()

    written.append(plot_sentiment_distribution(labels, out_dir / "sentiment_distribution.png"))
    written.append(plot_sentiment_pie(labels, out_dir / "sentiment_share.png"))
    if score_column in frame.columns:
        written.append(
            plot_score_histogram(
                frame[score_column].tolist(),
                out_dir / "score_histogram.png",
                threshold=threshold,
            )
        )
    if TEXT_COLUMN in frame.columns:
        terms_chart = plot_top_terms(
            top_terms(frame[TEXT_COLUMN].tolist(), limit=20), out_dir / "top_terms.png"
        )
        if terms_chart is not None:
            written.append(terms_chart)
    trend = plot_sentiment_over_time(
        frame, out_dir / "sentiment_over_time.png", label_column=label_column
    )
    if trend is not None:
        written.append(trend)
    return written


__all__ = [
    "PALETTE",
    "plot_confusion_matrix",
    "plot_score_histogram",
    "plot_sentiment_distribution",
    "plot_sentiment_over_time",
    "plot_sentiment_pie",
    "plot_top_terms",
    "render_all",
]
