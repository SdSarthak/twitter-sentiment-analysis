"""Bag-of-words sentiment classifiers: build, train, evaluate, persist.

The original script reached for ``GaussianNB`` on a dense count matrix, which is
the wrong distributional assumption for word counts (and blows up on memory for
any real vocabulary). Sparse-friendly estimators are used instead, with
``MultinomialNB`` as the Naive Bayes option.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from twitter_sentiment.labeling import LABELS
from twitter_sentiment.preprocessing import preprocess

MODEL_CHOICES = ("naive_bayes", "logistic_regression", "linear_svm", "random_forest")
DEFAULT_MODEL = "logistic_regression"
DEFAULT_RANDOM_STATE = 42


class NotEnoughDataError(ValueError):
    """Raised when a dataset cannot support a train/test split."""


def _build_estimator(name: str, random_state: int):
    if name == "naive_bayes":
        return MultinomialNB(alpha=0.3)
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, C=4.0, class_weight="balanced", random_state=random_state
        )
    if name == "linear_svm":
        return LinearSVC(C=0.5, class_weight="balanced", random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_CHOICES}.")


def build_pipeline(
    model: str = DEFAULT_MODEL,
    *,
    vectorizer: str = "tfidf",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int | None = 20_000,
    min_df: int = 1,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Assemble ``preprocess -> vectorise -> classify`` as one sklearn pipeline.

    Preprocessing lives inside the pipeline (via the vectoriser's
    ``preprocessor`` hook) so a persisted model can be handed raw tweet text and
    still behave exactly as it did during training.
    """
    common: dict[str, Any] = {
        "preprocessor": preprocess,
        "ngram_range": ngram_range,
        "max_features": max_features,
        "min_df": min_df,
        "lowercase": False,  # preprocess() already lowercases
        "token_pattern": r"(?u)\b\w[\w']+\b",
    }
    if vectorizer == "tfidf":
        vec = TfidfVectorizer(sublinear_tf=True, **common)
    elif vectorizer == "count":
        vec = CountVectorizer(**common)
    else:
        raise ValueError(f"Unknown vectorizer {vectorizer!r}; use 'tfidf' or 'count'.")

    return Pipeline(
        [("vectorizer", vec), ("classifier", _build_estimator(model, random_state))]
    )


@dataclass
class EvaluationResult:
    """Everything the CLI and the charts need to describe model quality."""

    model: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    labels: list[str]
    confusion_matrix: list[list[int]]
    report: str = ""
    train_size: int = 0
    test_size: int = 0
    cross_val_mean: float | None = None
    cross_val_std: float | None = None
    class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"model              : {self.model}",
            f"train / test size  : {self.train_size} / {self.test_size}",
            f"accuracy           : {self.accuracy:.4f}",
            f"precision (macro)  : {self.precision_macro:.4f}",
            f"recall (macro)     : {self.recall_macro:.4f}",
            f"f1 (macro)         : {self.f1_macro:.4f}",
        ]
        if self.cross_val_mean is not None:
            lines.append(
                f"cross-val accuracy : {self.cross_val_mean:.4f} "
                f"(+/- {self.cross_val_std or 0.0:.4f})"
            )
        lines.append("")
        lines.append(self.report.rstrip())
        return "\n".join(lines)


def _present_labels(y: Sequence[str]) -> list[str]:
    present = set(map(str, y))
    ordered = [label for label in LABELS if label in present]
    ordered += sorted(present - set(ordered))
    return ordered


def evaluate(
    pipeline: Pipeline,
    x_test: Sequence[str],
    y_test: Sequence[str],
    *,
    model_name: str = DEFAULT_MODEL,
    train_size: int = 0,
) -> EvaluationResult:
    """Score a fitted pipeline on held-out data."""
    predictions = pipeline.predict(list(x_test))
    labels = _present_labels(list(y_test) + list(predictions))
    counts = {label: int(np.sum(np.asarray(y_test) == label)) for label in labels}
    return EvaluationResult(
        model=model_name,
        accuracy=float(accuracy_score(y_test, predictions)),
        precision_macro=float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        recall_macro=float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        f1_macro=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        labels=labels,
        confusion_matrix=confusion_matrix(y_test, predictions, labels=labels).tolist(),
        report=classification_report(
            y_test, predictions, labels=labels, zero_division=0
        ),
        train_size=train_size,
        test_size=len(y_test),
        class_counts=counts,
    )


def _can_stratify(y: Sequence[str], test_size: float) -> bool:
    values, counts = np.unique(np.asarray(y, dtype=object), return_counts=True)
    if len(values) < 2 or counts.min() < 2:
        return False
    # Every class needs at least one row on each side of the split.
    return int(round(len(y) * test_size)) >= len(values)


def train(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    vectorizer: str = "tfidf",
    test_size: float = 0.25,
    random_state: int = DEFAULT_RANDOM_STATE,
    cross_validate: bool = True,
    cv_folds: int = 5,
) -> tuple[Pipeline, EvaluationResult]:
    """Fit a classifier and report held-out performance.

    Returns the pipeline refitted on *all* the data (so the shipped model uses
    every labelled tweet) along with the metrics measured on the held-out split.
    """
    x = [str(text) for text in texts]
    y = [str(label) for label in labels]
    if len(x) != len(y):
        raise ValueError(f"texts and labels differ in length: {len(x)} vs {len(y)}")
    if len(x) < 4:
        raise NotEnoughDataError(
            f"Need at least 4 labelled tweets to train, got {len(x)}."
        )
    if len(set(y)) < 2:
        raise NotEnoughDataError(
            "Need at least two distinct sentiment classes to train, "
            f"got only {set(y)!r}."
        )

    stratify = y if _can_stratify(y, test_size) else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = build_pipeline(model, vectorizer=vectorizer, random_state=random_state)
    pipeline.fit(x_train, y_train)
    result = evaluate(
        pipeline, x_test, y_test, model_name=model, train_size=len(x_train)
    )

    if cross_validate:
        folds = min(cv_folds, min(np.unique(y, return_counts=True)[1]))
        if folds >= 2:
            scores = cross_val_score(
                build_pipeline(model, vectorizer=vectorizer, random_state=random_state),
                x,
                y,
                cv=folds,
                scoring="accuracy",
            )
            result.cross_val_mean = float(scores.mean())
            result.cross_val_std = float(scores.std())

    # Refit on everything for the artefact that gets saved.
    final = build_pipeline(model, vectorizer=vectorizer, random_state=random_state)
    final.fit(x, y)
    return final, result


def predict(pipeline: Pipeline, texts: Sequence[str]) -> list[str]:
    """Predict labels for raw tweet strings."""
    return [str(label) for label in pipeline.predict([str(text) for text in texts])]


def predict_with_confidence(
    pipeline: Pipeline, texts: Sequence[str]
) -> list[tuple[str, float | None]]:
    """Predict labels plus a probability, where the estimator exposes one.

    ``LinearSVC`` has no ``predict_proba``; its confidence comes back ``None``
    rather than a fabricated number.
    """
    inputs = [str(text) for text in texts]
    labels = predict(pipeline, inputs)
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "predict_proba"):
        return [(label, None) for label in labels]
    probabilities = pipeline.predict_proba(inputs)
    return [
        (label, float(row.max())) for label, row in zip(labels, probabilities)
    ]


def top_features(pipeline: Pipeline, limit: int = 15) -> dict[str, list[tuple[str, float]]]:
    """Most influential n-grams per class for linear models (empty otherwise)."""
    classifier = pipeline.named_steps["classifier"]
    coefficients = getattr(classifier, "coef_", None)
    if coefficients is None:
        return {}
    names = pipeline.named_steps["vectorizer"].get_feature_names_out()
    classes = list(classifier.classes_)
    if coefficients.shape[0] == 1:  # binary problem
        classes = classes[-1:]
    output: dict[str, list[tuple[str, float]]] = {}
    for index, label in enumerate(classes):
        row = coefficients[index]
        ranked = np.argsort(row)[::-1][:limit]
        output[str(label)] = [(str(names[i]), float(row[i])) for i in ranked]
    return output


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted pipeline with joblib."""
    import joblib

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    return out_path


def load_model(path: str | Path) -> Pipeline:
    """Load a pipeline previously written by :func:`save_model`."""
    import joblib

    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No model at {model_path}. Train one first: "
            "`python Main.py train --input <tweets.csv>`."
        )
    return joblib.load(model_path)
