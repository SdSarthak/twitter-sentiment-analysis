from __future__ import annotations

import json

import pandas as pd
import pytest

from twitter_sentiment.cli import main


def test_analyze_writes_labelled_csv_and_charts(sample_csv, tmp_path, capsys):
    exit_code = main(
        ["analyze", "--input", str(sample_csv), "--output-dir", str(tmp_path)]
    )
    assert exit_code == 0

    labelled = tmp_path / "sample_tweets_labelled.csv"
    assert labelled.is_file()
    frame = pd.read_csv(labelled)
    assert {"sentiment", "sentiment_score"} <= set(frame.columns)

    for chart in ("sentiment_distribution.png", "sentiment_share.png", "top_terms.png"):
        assert (tmp_path / chart).is_file()

    output = capsys.readouterr().out
    assert "positive" in output and "negative" in output


def test_analyze_can_skip_charts(sample_csv, tmp_path):
    assert (
        main(
            [
                "analyze",
                "--input",
                str(sample_csv),
                "--output-dir",
                str(tmp_path),
                "--no-charts",
            ]
        )
        == 0
    )
    assert not list(tmp_path.glob("*.png"))


def test_analyze_missing_input_returns_error(tmp_path, capsys):
    assert main(["analyze", "--input", str(tmp_path / "nope.csv")]) == 2
    assert "error" in capsys.readouterr().err


def test_train_writes_model_metrics_and_confusion_matrix(sample_csv, tmp_path, capsys):
    model_path = tmp_path / "model.joblib"
    exit_code = main(
        [
            "train",
            "--input",
            str(sample_csv),
            "--model",
            "linear_svm",
            "--output",
            str(model_path),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert model_path.is_file()
    assert (tmp_path / "confusion_matrix.png").is_file()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model"] == "linear_svm"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == len(metrics["labels"])
    assert "accuracy" in capsys.readouterr().out


def test_train_rejects_unknown_label_column(sample_csv, tmp_path, capsys):
    exit_code = main(
        [
            "train",
            "--input",
            str(sample_csv),
            "--label-column",
            "not_there",
            "--output",
            str(tmp_path / "m.joblib"),
            "--output-dir",
            str(tmp_path),
            "--no-charts",
        ]
    )
    assert exit_code == 2
    assert "not_there" in capsys.readouterr().err


def test_predict_with_lexicon(capsys):
    assert main(["predict", "--lexicon", "this budget is excellent"]) == 0
    assert "positive" in capsys.readouterr().out


def test_predict_with_trained_model(sample_csv, tmp_path, capsys):
    model_path = tmp_path / "model.joblib"
    main(
        [
            "train",
            "--input",
            str(sample_csv),
            "--output",
            str(model_path),
            "--output-dir",
            str(tmp_path),
            "--no-charts",
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "predict",
            "--model",
            str(model_path),
            "--json",
            "excellent and wonderful budget",
        ]
    )
    assert exit_code == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["sentiment"] in {"positive", "negative", "neutral"}
    assert "score" in rows[0]


def test_predict_writes_output_csv(tmp_path, capsys):
    target = tmp_path / "predictions.csv"
    exit_code = main(
        ["predict", "--lexicon", "--output", str(target), "great news", "awful news"]
    )
    assert exit_code == 0
    frame = pd.read_csv(target)
    assert len(frame) == 2
    assert frame["sentiment"].tolist() == ["positive", "negative"]


def test_predict_without_text_or_input_errors(capsys):
    assert main(["predict", "--lexicon"]) == 2
    assert "error" in capsys.readouterr().err


def test_predict_with_missing_model_errors(tmp_path, capsys):
    assert main(["predict", "--model", str(tmp_path / "absent.joblib"), "hello"]) == 2
    assert "error" in capsys.readouterr().err


def test_explain_emits_json(capsys):
    assert main(["explain", "not good at all"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["score"] < 0


def test_scrape_without_credentials_fails_cleanly(monkeypatch, tmp_path, capsys):
    for variable in ("TWITTER_USERNAME", "TWITTER_EMAIL", "TWITTER_PASSWORD"):
        monkeypatch.delenv(variable, raising=False)
    # Ignore any .env the developer happens to have locally.
    monkeypatch.setattr(
        "twitter_sentiment.config.load_dotenv_if_available", lambda *a, **k: False
    )
    monkeypatch.setattr(
        "twitter_sentiment.cli.load_dotenv_if_available", lambda *a, **k: False
    )

    exit_code = main(
        ["scrape", "--query", "#test", "--output", str(tmp_path / "out.csv")]
    )
    assert exit_code == 2
    assert "TWITTER_USERNAME" in capsys.readouterr().err


def test_no_command_exits_with_usage_error():
    with pytest.raises(SystemExit):
        main([])
