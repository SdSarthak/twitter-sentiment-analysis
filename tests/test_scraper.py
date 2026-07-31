"""Scraper tests driven by fake Selenium objects -- no browser, no network."""

from __future__ import annotations

import pytest

from twitter_sentiment.config import MissingCredentialError, ScraperConfig
from twitter_sentiment.scraper import (
    LIKE_SELECTOR,
    RETWEET_SELECTOR,
    TEXT_SELECTOR,
    TWEET_SELECTOR,
    extract_tweet,
    scrape_tweets,
)

selenium = pytest.importorskip("selenium")
from selenium.common.exceptions import NoSuchElementException  # noqa: E402


class FakeElement:
    """Minimal stand-in for a Selenium ``WebElement``."""

    def __init__(self, text: str = "", attributes: dict[str, str] | None = None):
        self.text = text
        self._attributes = attributes or {}
        self.children: dict[str, "FakeElement"] = {}

    def get_attribute(self, name: str) -> str | None:
        return self._attributes.get(name)

    def find_element(self, by: str, selector: str) -> "FakeElement":
        if selector not in self.children:
            raise NoSuchElementException(selector)
        return self.children[selector]


def make_article(
    text: str = "a great budget",
    timestamp: str = "2025-01-27T06:00:00.000Z",
    likes: str = "12 Likes",
    retweets: str = "3 reposts",
) -> FakeElement:
    article = FakeElement(text=text)
    article.children[TEXT_SELECTOR] = FakeElement(text=text)
    article.children["time"] = FakeElement(attributes={"datetime": timestamp})
    article.children[LIKE_SELECTOR] = FakeElement(attributes={"aria-label": likes})
    article.children[RETWEET_SELECTOR] = FakeElement(attributes={"aria-label": retweets})
    return article


class FakeDriver:
    """Serves a fixed list of articles and reports a growing page height."""

    def __init__(self, articles: list[FakeElement], heights: list[int]):
        self._articles = articles
        self._heights = list(heights)
        self.scrolls = 0

    def find_elements(self, by: str, selector: str) -> list[FakeElement]:
        assert selector == TWEET_SELECTOR
        return self._articles

    def execute_script(self, script: str):
        if "scrollTo" in script:
            self.scrolls += 1
            return None
        if len(self._heights) > 1:
            return self._heights.pop(0)
        return self._heights[0]


@pytest.fixture()
def fast_config() -> ScraperConfig:
    return ScraperConfig(
        username="tester",
        password="secret",
        max_scrolls=6,
        scroll_pause_min=0.0,
        scroll_pause_max=0.0,
    )


def test_extract_tweet_reads_all_fields():
    record = extract_tweet(make_article())
    assert record["Tweet"] == "a great budget"
    assert record["Timestamp"] == "2025-01-27T06:00:00.000Z"
    assert record["Likes"] == "12"
    assert record["Retweets"] == "3"


def test_extract_tweet_falls_back_to_article_text():
    article = FakeElement(text="raw fallback text")
    assert extract_tweet(article)["Tweet"] == "raw fallback text"


def test_extract_tweet_defaults_missing_engagement_to_zero():
    article = FakeElement(text="hello")
    record = extract_tweet(article)
    assert record["Likes"] == "0"
    assert record["Retweets"] == "0"
    assert record["Timestamp"] == ""


def test_scrape_tweets_deduplicates_repeated_articles(fast_config):
    articles = [make_article("first tweet"), make_article("first tweet"), make_article("second tweet")]
    driver = FakeDriver(articles, heights=[100, 200, 300, 300, 300, 300])
    frame = scrape_tweets(driver, fast_config)
    assert frame["Tweet"].tolist() == ["first tweet", "second tweet"]
    assert list(frame.columns) == ["Tweet", "Timestamp", "Likes", "Retweets"]


def test_scrape_tweets_stops_when_the_page_stops_growing(fast_config):
    driver = FakeDriver([make_article()], heights=[500])
    scrape_tweets(driver, fast_config)
    assert driver.scrolls < fast_config.max_scrolls


def test_scrape_tweets_respects_max_scrolls(fast_config):
    driver = FakeDriver([make_article()], heights=[100, 200, 300, 400, 500, 600, 700])
    scrape_tweets(driver, fast_config)
    assert driver.scrolls == fast_config.max_scrolls


def test_scrape_tweets_skips_articles_without_text(fast_config):
    driver = FakeDriver([make_article(""), make_article("real tweet")], heights=[10])
    frame = scrape_tweets(driver, fast_config)
    assert frame["Tweet"].tolist() == ["real tweet"]


def test_require_login_reports_every_missing_variable():
    with pytest.raises(MissingCredentialError) as excinfo:
        ScraperConfig().require_login()
    message = str(excinfo.value)
    assert "TWITTER_USERNAME" in message and "TWITTER_PASSWORD" in message


def test_require_login_accepts_a_complete_config(fast_config):
    assert fast_config.require_login() == ("tester", None, "secret")


def test_config_from_env_reads_overrides(monkeypatch):
    monkeypatch.setattr(
        "twitter_sentiment.config.load_dotenv_if_available", lambda *a, **k: False
    )
    monkeypatch.setenv("TWITTER_USERNAME", "someone")
    monkeypatch.setenv("SCRAPER_MAX_SCROLLS", "7")
    monkeypatch.setenv("SCRAPER_HEADLESS", "true")
    config = ScraperConfig.from_env()
    assert config.username == "someone"
    assert config.max_scrolls == 7
    assert config.headless is True


def test_config_from_env_rejects_non_numeric_values(monkeypatch):
    monkeypatch.setattr(
        "twitter_sentiment.config.load_dotenv_if_available", lambda *a, **k: False
    )
    monkeypatch.setenv("SCRAPER_MAX_SCROLLS", "many")
    with pytest.raises(ValueError):
        ScraperConfig.from_env()
