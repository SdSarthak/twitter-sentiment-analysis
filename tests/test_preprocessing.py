from __future__ import annotations

import pytest

from twitter_sentiment.preprocessing import (
    clean_text,
    extract_hashtags,
    extract_mentions,
    preprocess,
    remove_stopwords,
    split_hashtag,
    stem_tokens,
    tokenize,
)


def test_clean_text_strips_urls_mentions_and_retweet_marker():
    raw = "RT @finance_guy: Budget is great https://t.co/abc123 @someone"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "rt " not in cleaned
    assert "budget is great" in cleaned


def test_clean_text_expands_camelcase_hashtags():
    assert "stock market crash" in clean_text("Bad day #StockMarketCrash")


def test_clean_text_separates_digits_in_hashtags():
    assert "budget 2025" in clean_text("Waiting for #Budget2025")


def test_clean_text_collapses_stretched_characters():
    assert clean_text("this is sooooooo good") == "this is soo good"


def test_clean_text_unescapes_html_entities():
    assert clean_text("small caps &amp; mid caps") == "small caps & mid caps"


@pytest.mark.parametrize("value", [None, "", "   ", "https://t.co/only"])
def test_clean_text_handles_empty_input(value):
    assert clean_text(value) == ""


def test_clean_text_can_keep_mentions():
    assert "@handle" in clean_text("hello @handle", drop_mentions=False)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("StockMarketCrash", "Stock Market Crash"),
        ("Budget2025", "Budget 2025"),
        ("budget", "budget"),
        ("NSE_India", "NSE India"),
        ("#Hashtag", "Hashtag"),
    ],
)
def test_split_hashtag(tag, expected):
    assert split_hashtag(tag) == expected


def test_tokenize_drops_punctuation_but_keeps_contractions():
    assert tokenize("It isn't great, really!") == ["it", "isn't", "great", "really"]


def test_remove_stopwords_keeps_negations():
    tokens = ["this", "is", "not", "good", "for", "markets"]
    kept = remove_stopwords(tokens)
    assert "not" in kept
    assert "is" not in kept
    assert "good" in kept and "markets" in kept


def test_stem_tokens_reduces_inflections():
    assert stem_tokens(["crashing", "markets"]) == ["crash", "market"]


def test_preprocess_is_deterministic():
    text = "The markets are CRASHING badly today!! #StockMarketCrash"
    assert preprocess(text) == preprocess(text)
    assert "crash" in preprocess(text)


def test_preprocess_without_stemming_keeps_full_words():
    assert "crashing" in preprocess("markets are crashing", stem=False)


def test_extract_hashtags_deduplicates_case_insensitively():
    assert extract_hashtags("#Budget2025 rally #budget2025 #Nifty") == [
        "budget2025",
        "nifty",
    ]


def test_extract_mentions():
    assert extract_mentions("cc @Alice and @bob and @Alice") == ["alice", "bob"]
