from __future__ import annotations

import pytest

from twitter_sentiment.labeling import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    distribution_summary,
    label_distribution,
    label_for_score,
    label_texts,
)
from twitter_sentiment.lexicon import explain, score_text, score_texts


@pytest.mark.parametrize(
    "text",
    [
        "This budget is excellent and very helpful",
        "Absolutely brilliant work, congratulations!",
        "Markets rallied and profits surged",
    ],
)
def test_positive_texts_score_above_zero(text):
    assert score_text(text) > 0.05


@pytest.mark.parametrize(
    "text",
    [
        "This is a terrible and shameful failure",
        "Markets crashed badly, huge losses",
        "Awful policy, deeply disappointing",
    ],
)
def test_negative_texts_score_below_zero(text):
    assert score_text(text) < -0.05


@pytest.mark.parametrize(
    "text",
    [
        "The session begins at 11 am on 1 February",
        "Nifty closed at 23000 points",
        "",
        None,
    ],
)
def test_neutral_texts_score_near_zero(text):
    assert abs(score_text(text)) < 0.05


def test_scores_are_bounded():
    extreme = "excellent " * 50
    assert -1.0 <= score_text(extreme) <= 1.0
    assert -1.0 <= score_text("terrible " * 50) <= 1.0


def test_negation_flips_polarity():
    assert score_text("this is good") > 0
    assert score_text("this is not good") < 0


def test_negation_is_dampened_not_mirrored():
    assert abs(score_text("this is not good")) < abs(score_text("this is good"))


def test_intensifier_amplifies():
    assert score_text("very good news") > score_text("good news")
    assert score_text("extremely bad news") < score_text("bad news")


def test_dampener_reduces_magnitude():
    assert score_text("slightly good news") < score_text("good news")


def test_shouting_amplifies():
    assert score_text("TERRIBLE budget") < score_text("terrible budget")


def test_exclamation_amplifies():
    assert score_text("great budget!!!") > score_text("great budget")


def test_emoji_carry_sentiment():
    assert score_text("the budget \U0001F621") < 0
    assert score_text("the budget \U0001F389") > 0


def test_emoticons_carry_sentiment():
    assert score_text("nice work :)") > score_text("nice work")
    assert score_text("the result :(") < 0


def test_idiom_overrides_component_words():
    assert score_text("not bad at all") > 0
    assert score_text("a complete waste of time") < 0


def test_hashtags_are_expanded_before_scoring():
    assert score_text("#StockMarketCrash") < 0


def test_urls_and_mentions_do_not_contribute():
    assert score_text("https://t.co/best @winner") == 0.0


def test_score_is_deterministic():
    text = "Really great budget, very happy #Budget2025"
    assert score_text(text) == score_text(text)


def test_score_texts_matches_elementwise():
    texts = ["great", "terrible", "table"]
    assert score_texts(texts) == [score_text(t) for t in texts]


def test_unknown_scorer_rejected():
    with pytest.raises(ValueError):
        score_text("hello", scorer="nope")


def test_explain_exposes_matched_words():
    breakdown = explain("this budget is terrible")
    assert "terrible" in breakdown["matched_words"]
    assert breakdown["word_score"] < 0
    assert breakdown["score"] < 0
    assert breakdown["cleaned"] == "this budget is terrible"


def test_explain_attributes_idioms_separately_from_words():
    breakdown = explain("this is not good")
    # The idiom consumes its component words, so nothing is double counted.
    assert breakdown["matched_words"] == {}
    assert breakdown["idiom_score"] < 0
    assert breakdown["word_score"] == 0.0


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.9, POSITIVE),
        (0.05, POSITIVE),
        (0.049, NEUTRAL),
        (0.0, NEUTRAL),
        (-0.049, NEUTRAL),
        (-0.05, NEGATIVE),
        (-0.9, NEGATIVE),
    ],
)
def test_label_for_score_boundaries(score, expected):
    assert label_for_score(score) == expected


def test_label_for_score_rejects_negative_threshold():
    with pytest.raises(ValueError):
        label_for_score(0.1, threshold=-0.1)


def test_label_texts_returns_aligned_scores_and_labels():
    scores, labels = label_texts(["excellent news", "terrible news", "a table"])
    assert len(scores) == len(labels) == 3
    assert labels == [POSITIVE, NEGATIVE, NEUTRAL]


def test_label_distribution_covers_all_classes():
    counts = label_distribution([POSITIVE, POSITIVE, NEGATIVE])
    assert counts[NEUTRAL] == 0
    assert counts[POSITIVE] == 2
    assert list(counts.index) == [NEGATIVE, NEUTRAL, POSITIVE]


def test_distribution_summary_percentages_sum_to_100():
    summary = distribution_summary([POSITIVE, NEGATIVE, NEUTRAL, POSITIVE])
    assert summary["count"].sum() == 4
    assert round(summary["percentage"].sum(), 2) == 100.0
