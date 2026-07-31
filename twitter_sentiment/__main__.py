"""Allow ``python -m twitter_sentiment ...``."""

from twitter_sentiment.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
