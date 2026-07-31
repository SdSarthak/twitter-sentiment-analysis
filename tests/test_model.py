from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from twitter_sentiment.dataset import TEXT_COLUMN
from twitter_sentiment.model import (
    MODEL_CHOICES,
    EvaluationResult,
    NotEnoughDataError,
    build_pipeline,
    evaluate,
    load_model,
    predict,
    predict_with_confidence,
    save_model,
    top_features,
    train,
)

POSITIVE_SAMPLES = [
    "excellent budget, very happy",
    "great reforms, wonderful news",
    "fantastic growth and strong profits",
    "love the new savings scheme",
    "impressive rally, markets surged",
    "brilliant work, congratulations",
]
NEGATIVE_SAMPLES = [
    "terrible budget, complete disaster",
    "awful policy, huge losses",
    "markets crashed badly today",
    "worst decision, painful and unfair",
    "shameful failure, deeply disappointing",
    "bearish selloff, nifty plunged",
]

NEUTRAL_SAMPLES = [
    "the session begins at 11 am tomorrow",
    "nifty closed at 23000 points today",
    "live coverage starts on the official channel",
    "trading volumes were reported for the january series",
    "the statement will be tabled next week",
    "follow this account for daily updates",
]

TEXTS = POSITIVE_SAMPLES + NEGATIVE_SAMPLES
LABELS = ["positive"] * len(POSITIVE_SAMPLES) + ["negative"] * len(NEGATIVE_SAMPLES)

THREE_CLASS_TEXTS = TEXTS + NEUTRAL_SAMPLES
THREE_CLASS_LABELS = LABELS + ["neutral"] * len(NEUTRAL_SAMPLES)


def test_build_pipeline_returns_two_stage_pipeline():
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["vectorizer", "classifier"]


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError):
        build_pipeline("nope")


def test_build_pipeline_rejects_unknown_vectorizer():
    with pytest.raises(ValueError):
        build_pipeline(vectorizer="nope")


@pytest.mark.parametrize("model", MODEL_CHOICES)
def test_every_model_trains_and_predicts(model):
    pipeline, result = train(TEXTS, LABELS, model=model, test_size=0.34)
    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.model == model
    predictions = predict(pipeline, TEXTS)
    assert set(predictions) <= {"positive", "negative"}
    # The returned pipeline is refit on every row, so it must fit its own data.
    assert predictions == LABELS


@pytest.mark.parametrize("model", ("naive_bayes", "logistic_regression", "linear_svm"))
def test_linear_models_generalise_to_unseen_wording(model):
    pipeline, _ = train(TEXTS, LABELS, model=model, cross_validate=False)
    assert predict(pipeline, ["wonderful and excellent news"]) == ["positive"]
    assert predict(pipeline, ["terrible and awful disaster"]) == ["negative"]


def test_training_is_reproducible():
    first, first_result = train(TEXTS, LABELS)
    second, second_result = train(TEXTS, LABELS)
    assert first_result.accuracy == second_result.accuracy
    assert predict(first, TEXTS) == predict(second, TEXTS)


def test_train_refits_on_all_data():
    pipeline, result = train(TEXTS, LABELS, test_size=0.25)
    assert result.train_size + result.test_size == len(TEXTS)
    # The returned pipeline saw every row, so it separates the training set.
    assert predict(pipeline, TEXTS) == LABELS


def test_train_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        train(TEXTS, LABELS[:-1])


def test_train_rejects_tiny_dataset():
    with pytest.raises(NotEnoughDataError):
        train(["a", "b"], ["positive", "negative"])


def test_train_rejects_single_class():
    with pytest.raises(NotEnoughDataError):
        train(TEXTS, ["positive"] * len(TEXTS))


def test_train_handles_class_too_small_to_stratify():
    texts = TEXTS + ["a neutral note about the schedule"]
    labels = LABELS + ["neutral"]
    _, result = train(texts, labels, test_size=0.25, cross_validate=False)
    assert result.test_size > 0


def test_evaluation_result_confusion_matrix_shape():
    pipeline, result = train(TEXTS, LABELS, test_size=0.34)
    assert len(result.confusion_matrix) == len(result.labels)
    assert all(len(row) == len(result.labels) for row in result.confusion_matrix)
    assert sum(sum(row) for row in result.confusion_matrix) == result.test_size


def test_evaluation_result_serialises_and_summarises():
    _, result = train(TEXTS, LABELS)
    payload = result.to_dict()
    assert payload["model"] == result.model
    assert "accuracy" in payload
    assert "accuracy" in result.summary()


def test_evaluate_on_perfectly_separable_holdout():
    pipeline = build_pipeline("linear_svm")
    pipeline.fit(TEXTS, LABELS)
    result = evaluate(pipeline, TEXTS, LABELS, model_name="linear_svm")
    assert result.accuracy == 1.0
    assert result.class_counts == {"negative": 6, "positive": 6}


def test_predict_with_confidence_returns_probability():
    pipeline, _ = train(TEXTS, LABELS, model="logistic_regression")
    label, confidence = predict_with_confidence(pipeline, ["great and wonderful"])[0]
    assert label == "positive"
    assert confidence is not None and 0.0 <= confidence <= 1.0


def test_predict_with_confidence_is_none_for_svm():
    pipeline, _ = train(TEXTS, LABELS, model="linear_svm")
    _, confidence = predict_with_confidence(pipeline, ["great and wonderful"])[0]
    assert confidence is None


def test_top_features_for_multiclass_linear_model():
    pipeline, _ = train(
        THREE_CLASS_TEXTS,
        THREE_CLASS_LABELS,
        model="logistic_regression",
        cross_validate=False,
    )
    features = top_features(pipeline, limit=3)
    assert set(features) == {"negative", "neutral", "positive"}
    assert all(len(rows) == 3 for rows in features.values())


def test_top_features_for_binary_model_reports_the_positive_class():
    # A binary logistic regression has a single coefficient row, which describes
    # the second class only -- reporting both would duplicate the same weights.
    pipeline, _ = train(TEXTS, LABELS, model="logistic_regression")
    assert set(top_features(pipeline, limit=3)) == {"positive"}


def test_top_features_empty_for_tree_model():
    pipeline, _ = train(TEXTS, LABELS, model="random_forest", cross_validate=False)
    assert top_features(pipeline) == {}


def test_save_and_load_round_trip(tmp_path):
    pipeline, _ = train(TEXTS, LABELS)
    path = save_model(pipeline, tmp_path / "models" / "sentiment.joblib")
    assert path.is_file()
    reloaded = load_model(path)
    assert predict(reloaded, TEXTS) == predict(pipeline, TEXTS)


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")


def test_pipeline_trains_on_the_bundled_fixture(labelled_frame):
    pipeline, result = train(
        labelled_frame[TEXT_COLUMN].tolist(),
        labelled_frame["sentiment"].tolist(),
        test_size=0.3,
    )
    assert result.accuracy >= 0.0
    assert len(predict(pipeline, labelled_frame[TEXT_COLUMN].tolist())) == len(
        labelled_frame
    )
