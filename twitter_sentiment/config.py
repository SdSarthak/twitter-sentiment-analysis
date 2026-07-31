"""Runtime configuration.

Nothing in this project hardcodes a credential or an absolute path. Everything
that varies between machines is read from the environment, optionally seeded
from a local ``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment.joblib"


def load_dotenv_if_available(path: str | os.PathLike[str] | None = None) -> bool:
    """Populate ``os.environ`` from a ``.env`` file when python-dotenv is present.

    Returns ``True`` when a file was actually loaded. Missing library or missing
    file is not an error -- the environment may already be configured.
    """
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(env_path, override=False))


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class MissingCredentialError(RuntimeError):
    """Raised when a required credential is not present in the environment."""


@dataclass(frozen=True)
class ScraperConfig:
    """Settings for the Selenium collector."""

    username: str | None = None
    email: str | None = None
    password: str | None = None
    chromedriver_path: str | None = None
    headless: bool = False
    query: str = "#Budget2025"
    max_scrolls: int = 40
    scroll_pause_min: float = 1.0
    scroll_pause_max: float = 3.0
    page_timeout: int = 20

    @classmethod
    def from_env(cls) -> "ScraperConfig":
        load_dotenv_if_available()
        return cls(
            username=_env_str("TWITTER_USERNAME"),
            email=_env_str("TWITTER_EMAIL"),
            password=_env_str("TWITTER_PASSWORD"),
            chromedriver_path=_env_str("CHROMEDRIVER_PATH"),
            headless=_env_bool("SCRAPER_HEADLESS", False),
            query=_env_str("SCRAPER_QUERY", "#Budget2025") or "#Budget2025",
            max_scrolls=_env_int("SCRAPER_MAX_SCROLLS", 40),
            scroll_pause_min=_env_float("SCRAPER_SCROLL_PAUSE_MIN", 1.0),
            scroll_pause_max=_env_float("SCRAPER_SCROLL_PAUSE_MAX", 3.0),
            page_timeout=_env_int("SCRAPER_PAGE_TIMEOUT", 20),
        )

    def require_login(self) -> tuple[str, str | None, str]:
        """Return ``(username, email, password)`` or explain what is missing."""
        missing = [
            name
            for name, value in (
                ("TWITTER_USERNAME", self.username),
                ("TWITTER_PASSWORD", self.password),
            )
            if not value
        ]
        if missing:
            raise MissingCredentialError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill it in."
            )
        assert self.username is not None and self.password is not None
        return self.username, self.email, self.password
