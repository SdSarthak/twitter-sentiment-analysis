"""Selenium collector for public tweets.

Rewritten from the original top-level script so that importing this module
never launches a browser and never needs credentials. Selenium is imported
lazily inside the functions that use it, which keeps ``pip install -r
requirements.txt`` optional for anyone who only wants to analyse an existing
CSV.

The brittle absolute XPath in the original (``div[8]/div[1]/div/article``) broke
the moment the page layout shifted; tweets are located by their stable
``data-testid`` attributes instead.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

from twitter_sentiment.config import ScraperConfig

LOGGER = logging.getLogger(__name__)

LOGIN_URL = "https://x.com/i/flow/login"
SEARCH_URL = "https://x.com/search?q={query}&src=typed_query&f=live"

TWEET_SELECTOR = 'article[data-testid="tweet"]'
TEXT_SELECTOR = '[data-testid="tweetText"]'
LIKE_SELECTOR = '[data-testid="like"]'
RETWEET_SELECTOR = '[data-testid="retweet"]'


class ScraperError(RuntimeError):
    """Raised when the browser session cannot be established or driven."""


def build_driver(config: ScraperConfig) -> Any:
    """Start Chrome, preferring an explicit driver path then webdriver-manager."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ScraperError(
            "Selenium is not installed. Run `pip install -r requirements.txt` "
            "to use the scraper."
        ) from exc

    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=en-US")
    if config.headless:
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")

    if config.chromedriver_path:
        driver_path = Path(config.chromedriver_path).expanduser()
        if not driver_path.is_file():
            raise ScraperError(
                f"CHROMEDRIVER_PATH points at {driver_path}, which does not exist."
            )
        service = Service(str(driver_path))
    else:
        try:
            from webdriver_manager.chrome import ChromeDriverManager

            service = Service(ChromeDriverManager().install())
        except Exception as exc:  # pragma: no cover - network dependent
            raise ScraperError(
                "No CHROMEDRIVER_PATH set and webdriver-manager could not fetch a "
                "driver. Download chromedriver manually and set CHROMEDRIVER_PATH."
            ) from exc

    return webdriver.Chrome(service=service, options=options)


def login(driver: Any, config: ScraperConfig) -> None:
    """Walk the multi-step login flow, tolerating the optional verify step."""
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    username, email, password = config.require_login()
    wait = WebDriverWait(driver, config.page_timeout)
    driver.get(LOGIN_URL)

    try:
        username_field = wait.until(
            EC.presence_of_element_located((By.NAME, "text"))
        )
        username_field.send_keys(username)
        wait.until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
        ).click()
    except TimeoutException as exc:
        raise ScraperError("Login page did not present the username field.") from exc

    # X sometimes interposes an "unusual activity" check asking for the email
    # or phone number. It is optional, so a timeout here is not an error.
    try:
        verify_field = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.NAME, "text"))
        )
        if email:
            verify_field.send_keys(email)
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
            ).click()
    except TimeoutException:
        LOGGER.debug("No verification step presented, continuing to password.")

    try:
        password_field = wait.until(
            EC.presence_of_element_located((By.NAME, "password"))
        )
        password_field.send_keys(password)
        wait.until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']"))
        ).click()
    except TimeoutException as exc:
        raise ScraperError(
            "Could not complete login -- the password step never appeared. "
            "A 2FA challenge usually causes this; log in once manually first."
        ) from exc

    LOGGER.info("Logged in as %s", username)


def _text_of(element: Any, selector: str) -> str:
    from selenium.common.exceptions import NoSuchElementException
    from selenium.webdriver.common.by import By

    try:
        return element.find_element(By.CSS_SELECTOR, selector).text.strip()
    except NoSuchElementException:
        return ""


def _count_of(element: Any, selector: str) -> str:
    """Engagement counts live in the button's aria-label, not its text."""
    from selenium.common.exceptions import NoSuchElementException
    from selenium.webdriver.common.by import By

    try:
        button = element.find_element(By.CSS_SELECTOR, selector)
    except NoSuchElementException:
        return "0"
    label = (button.get_attribute("aria-label") or "").strip()
    digits = "".join(ch for ch in label.split(" ")[0] if ch.isdigit())
    if digits:
        return digits
    return button.text.strip() or "0"


def extract_tweet(article: Any) -> dict[str, str]:
    """Pull text, timestamp and engagement out of one ``<article>`` element."""
    from selenium.common.exceptions import NoSuchElementException
    from selenium.webdriver.common.by import By

    text = _text_of(article, TEXT_SELECTOR) or article.text.strip()
    try:
        timestamp = article.find_element(By.CSS_SELECTOR, "time").get_attribute(
            "datetime"
        )
    except NoSuchElementException:
        timestamp = ""
    return {
        "Tweet": text,
        "Timestamp": timestamp or "",
        "Likes": _count_of(article, LIKE_SELECTOR),
        "Retweets": _count_of(article, RETWEET_SELECTOR),
    }


def scrape_tweets(driver: Any, config: ScraperConfig) -> pd.DataFrame:
    """Scroll the current results page, collecting tweets until it stops growing.

    De-duplicates by tweet text as it goes -- the virtualised timeline re-renders
    the same articles many times during a scroll.
    """
    from selenium.webdriver.common.by import By

    collected: dict[str, dict[str, str]] = {}
    last_height = driver.execute_script("return document.body.scrollHeight")
    stagnant_rounds = 0

    for scroll in range(config.max_scrolls):
        for article in driver.find_elements(By.CSS_SELECTOR, TWEET_SELECTOR):
            try:
                record = extract_tweet(article)
            except Exception as exc:  # stale element mid-scroll is expected
                LOGGER.debug("Skipping unreadable tweet: %s", exc)
                continue
            if record["Tweet"]:
                collected.setdefault(record["Tweet"], record)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(
            round(random.uniform(config.scroll_pause_min, config.scroll_pause_max), 2)
        )

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            stagnant_rounds += 1
            if stagnant_rounds >= 3:
                LOGGER.info("Reached the end of the timeline after %d scrolls", scroll + 1)
                break
        else:
            stagnant_rounds = 0
        last_height = new_height
        LOGGER.info("scroll %d/%d - %d tweets", scroll + 1, config.max_scrolls, len(collected))

    return pd.DataFrame(list(collected.values()), columns=["Tweet", "Timestamp", "Likes", "Retweets"])


def scrape_query(
    query: str | None = None,
    *,
    config: ScraperConfig | None = None,
    output: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end collection: launch, log in, search, scroll, save."""
    from urllib.parse import quote

    settings = config or ScraperConfig.from_env()
    search_term = query or settings.query
    # Fail on missing credentials before paying the cost of launching Chrome.
    settings.require_login()
    driver = build_driver(settings)
    try:
        login(driver, settings)
        time.sleep(3)
        driver.get(SEARCH_URL.format(query=quote(search_term)))
        time.sleep(3)
        frame = scrape_tweets(driver, settings)
    finally:
        driver.quit()

    if output is not None:
        from twitter_sentiment.dataset import save_tweets

        save_tweets(frame, output)
        LOGGER.info("Saved %d tweets to %s", len(frame), output)
    return frame
