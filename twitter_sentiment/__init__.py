"""Twitter sentiment analysis toolkit.

The package is split into small, importable pieces so every stage of the
pipeline can be tested without a browser, a network connection or a database:

``config``         runtime settings sourced from environment variables
``preprocessing``  tweet text cleaning, tokenisation and stemming
``lexicon``        offline valence lexicon used to score raw tweets
``labeling``       turn continuous sentiment scores into class labels
``dataset``        load and normalise scraped tweet CSVs
``model``          bag-of-words classifiers, evaluation and persistence
``visualize``      charts for distribution, trend and confusion matrix
``scraper``        Selenium collector for the raw tweets
"""

from twitter_sentiment.labeling import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    label_dataframe,
    label_for_score,
    label_texts,
)
from twitter_sentiment.lexicon import score_text
from twitter_sentiment.preprocessing import clean_text, preprocess, tokenize

__all__ = [
    "NEGATIVE",
    "NEUTRAL",
    "POSITIVE",
    "clean_text",
    "label_dataframe",
    "label_for_score",
    "label_texts",
    "preprocess",
    "score_text",
    "tokenize",
]

__version__ = "1.0.0"
