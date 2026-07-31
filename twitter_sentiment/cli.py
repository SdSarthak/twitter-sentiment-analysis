"""Command line entry point: ``scrape``, ``analyze``, ``train``, ``predict``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from twitter_sentiment.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    MissingCredentialError,
    ScraperConfig,
    load_dotenv_if_available,
)
from twitter_sentiment.dataset import (
    EmptyDatasetError,
    TEXT_COLUMN,
    engagement_summary,
    frame_from_texts,
    load_tweets,
    save_tweets,
    top_terms,
)
from twitter_sentiment.labeling import (
    DEFAULT_THRESHOLD,
    distribution_summary,
    label_dataframe,
)
from twitter_sentiment.lexicon import explain, score_text
from twitter_sentiment.model import (
    DEFAULT_MODEL,
    MODEL_CHOICES,
    NotEnoughDataError,
    load_model,
    predict_with_confidence,
    save_model,
    train,
)

LOGGER = logging.getLogger("twitter_sentiment")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="neutral band half-width for lexicon labels (default: %(default)s)",
    )
    parser.add_argument(
        "--scorer",
        choices=("builtin", "vader"),
        default="builtin",
        help="lexicon used to label tweets (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twitter-sentiment",
        description="Collect, label, model and report on tweet sentiment.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape", help="collect tweets with Selenium")
    scrape.add_argument("--query", help="search query, e.g. '#Budget2025'")
    scrape.add_argument(
        "--output",
        default="data/tweets_scraped.csv",
        help="destination CSV (default: %(default)s)",
    )
    scrape.add_argument("--max-scrolls", type=int, help="how far to scroll the timeline")
    scrape.add_argument("--headless", action="store_true", help="run Chrome headless")

    analyze = subparsers.add_parser(
        "analyze", help="label a CSV with the lexicon and write a report"
    )
    analyze.add_argument("--input", required=True, help="tweet CSV to analyse")
    analyze.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="where charts and the labelled CSV are written (default: %(default)s)",
    )
    analyze.add_argument("--text-column", help="override the tweet text column name")
    analyze.add_argument("--no-charts", action="store_true", help="skip PNG rendering")
    _add_common(analyze)

    train_parser = subparsers.add_parser(
        "train", help="train a classifier on lexicon-labelled tweets"
    )
    train_parser.add_argument("--input", required=True, help="tweet CSV to train on")
    train_parser.add_argument(
        "--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL,
        help="estimator to fit (default: %(default)s)",
    )
    train_parser.add_argument(
        "--vectorizer", choices=("tfidf", "count"), default="tfidf",
        help="feature representation (default: %(default)s)",
    )
    train_parser.add_argument(
        "--label-column",
        help="use existing labels from this column instead of the lexicon",
    )
    train_parser.add_argument("--text-column", help="override the tweet text column name")
    train_parser.add_argument(
        "--test-size", type=float, default=0.25, help="held-out fraction (default: %(default)s)"
    )
    train_parser.add_argument(
        "--output", default=str(DEFAULT_MODEL_PATH),
        help="where to save the fitted pipeline (default: %(default)s)",
    )
    train_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="where metrics and the confusion matrix are written (default: %(default)s)",
    )
    train_parser.add_argument("--no-charts", action="store_true", help="skip PNG rendering")
    _add_common(train_parser)

    predict_parser = subparsers.add_parser(
        "predict", help="classify tweets with a saved model or the lexicon"
    )
    predict_parser.add_argument("text", nargs="*", help="tweet text to classify")
    predict_parser.add_argument("--input", help="CSV of tweets to classify instead")
    predict_parser.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH),
        help="saved pipeline to use (default: %(default)s)",
    )
    predict_parser.add_argument(
        "--lexicon", action="store_true",
        help="score with the built-in lexicon instead of a trained model",
    )
    predict_parser.add_argument("--output", help="write predictions to this CSV")
    predict_parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of a table"
    )
    _add_common(predict_parser)

    explain_parser = subparsers.add_parser(
        "explain", help="show why the lexicon scored a tweet the way it did"
    )
    explain_parser.add_argument("text", nargs="+", help="tweet text to break down")

    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _cmd_scrape(args: argparse.Namespace) -> int:
    from twitter_sentiment.scraper import ScraperError, scrape_query

    config = ScraperConfig.from_env()
    overrides: dict[str, object] = {}
    if args.query:
        overrides["query"] = args.query
    if args.max_scrolls:
        overrides["max_scrolls"] = args.max_scrolls
    if args.headless:
        overrides["headless"] = True
    if overrides:
        config = dataclasses.replace(config, **overrides)

    try:
        frame = scrape_query(config.query, config=config, output=args.output)
    except (MissingCredentialError, ScraperError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Scraped {len(frame)} tweets -> {args.output}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    frame = load_tweets(args.input, text_column=args.text_column)
    labelled = label_dataframe(
        frame, threshold=args.threshold, scorer=args.scorer
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labelled_path = output_dir / (Path(args.input).stem + "_labelled.csv")
    save_tweets(labelled, labelled_path)

    summary = distribution_summary(labelled["sentiment"])
    print(f"Analysed {len(labelled)} tweets from {args.input}\n")
    print(summary.to_string())
    print(f"\nmean sentiment score : {labelled['sentiment_score'].mean():.4f}")

    engagement = engagement_summary(labelled)
    if not engagement.empty:
        print("\nEngagement by sentiment:")
        print(engagement.to_string())

    terms = top_terms(labelled[TEXT_COLUMN].tolist(), limit=10)
    if len(terms):
        print("\nMost frequent terms:")
        print(terms.to_string())

    written = [labelled_path]
    if not args.no_charts:
        from twitter_sentiment.visualize import render_all

        written += render_all(labelled, output_dir, threshold=args.threshold)

    print("\nWrote:")
    for path in written:
        print(f"  {path}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    frame = load_tweets(args.input, text_column=args.text_column)
    if args.label_column:
        if args.label_column not in frame.columns:
            print(
                f"error: label column {args.label_column!r} not in {args.input}",
                file=sys.stderr,
            )
            return 2
        labelled = frame.rename(columns={args.label_column: "sentiment"})
    else:
        labelled = label_dataframe(frame, threshold=args.threshold, scorer=args.scorer)

    labelled = labelled[labelled["sentiment"].notna()]
    try:
        pipeline, result = train(
            labelled[TEXT_COLUMN].tolist(),
            labelled["sentiment"].astype(str).tolist(),
            model=args.model,
            vectorizer=args.vectorizer,
            test_size=args.test_size,
        )
    except NotEnoughDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model_path = save_model(pipeline, args.output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )

    print(result.summary())
    written = [model_path, metrics_path]
    if not args.no_charts:
        from twitter_sentiment.visualize import plot_confusion_matrix

        written.append(
            plot_confusion_matrix(
                result.confusion_matrix,
                result.labels,
                output_dir / "confusion_matrix.png",
                title=f"Confusion matrix ({result.model})",
            )
        )

    print("\nWrote:")
    for path in written:
        print(f"  {path}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    if args.input:
        frame = load_tweets(args.input)
    elif args.text:
        frame = frame_from_texts(args.text)
    else:
        print("error: pass tweet text or --input <csv>", file=sys.stderr)
        return 2

    texts = frame[TEXT_COLUMN].tolist()
    if args.lexicon:
        labelled = label_dataframe(
            frame, threshold=args.threshold, scorer=args.scorer
        )
        rows = [
            {
                "text": text,
                "sentiment": label,
                "score": float(score),
                "confidence": None,
            }
            for text, label, score in zip(
                texts, labelled["sentiment"], labelled["sentiment_score"]
            )
        ]
    else:
        try:
            pipeline = load_model(args.model)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        predictions = predict_with_confidence(pipeline, texts)
        rows = [
            {
                "text": text,
                "sentiment": label,
                "score": score_text(text),
                "confidence": confidence,
            }
            for text, (label, confidence) in zip(texts, predictions)
        ]

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            confidence = (
                f"  p={row['confidence']:.3f}" if row["confidence"] is not None else ""
            )
            snippet = " ".join(str(row["text"]).split())[:90]
            print(f"{row['sentiment']:<8} score={row['score']:+.3f}{confidence}  {snippet}")

    if args.output:
        import pandas as pd

        save_tweets(pd.DataFrame(rows), args.output)
        print(f"\nWrote {args.output}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    for text in args.text:
        print(json.dumps(explain(text), indent=2, ensure_ascii=False))
    return 0


COMMANDS = {
    "scrape": _cmd_scrape,
    "analyze": _cmd_analyze,
    "train": _cmd_train,
    "predict": _cmd_predict,
    "explain": _cmd_explain,
}


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_available()
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    handler = COMMANDS[args.command]
    try:
        return handler(args)
    except (FileNotFoundError, KeyError, EmptyDatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
