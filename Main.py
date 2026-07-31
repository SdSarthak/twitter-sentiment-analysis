"""Project entry point.

The real work lives in the ``twitter_sentiment`` package so it can be imported
and tested; this script just forwards to the CLI.

    python Main.py analyze --input tweets_v2.csv
    python Main.py train   --input tweets_v2.csv --model linear_svm
    python Main.py predict "the budget looks great" "markets crashed again"
    python Main.py scrape  --query "#Budget2025" --output data/tweets.csv

Run ``python Main.py --help`` for the full option list.
"""

from __future__ import annotations

import sys

from twitter_sentiment.cli import main

if __name__ == "__main__":
    # `python Main.py` with no arguments does the useful default: analyse the
    # bundled dataset if it is present, otherwise print the help text.
    argv = sys.argv[1:]
    if not argv:
        from pathlib import Path

        default_input = Path(__file__).with_name("tweets_v2.csv")
        if default_input.is_file():
            argv = ["analyze", "--input", str(default_input)]
        else:
            argv = ["--help"]
    raise SystemExit(main(argv))
