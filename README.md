# Twitter Sentiment Analysis

An end-to-end sentiment pipeline for tweets: collect them with Selenium, label
them with a bundled valence lexicon, train a bag-of-words classifier on those
labels, and produce charts and metrics.

The original dataset in this repo is ~99 usable tweets about India's
`#Budget2025`, but nothing in the code is tied to that query or that dataset.

## What it actually does

| Stage | Where | Notes |
|---|---|---|
| Collect | `twitter_sentiment/scraper.py` | Selenium, logs in, scrolls a search page, de-duplicates as it goes |
| Clean | `twitter_sentiment/preprocessing.py` | URL/mention stripping, hashtag splitting, stop words, Porter stemming |
| Label | `twitter_sentiment/lexicon.py` | Offline valence lexicon with negation, intensifiers, emoji, idioms |
| Model | `twitter_sentiment/model.py` | TF-IDF or counts feeding Naive Bayes / logistic regression / linear SVM / random forest |
| Report | `twitter_sentiment/visualize.py` | Distribution, share, score histogram, top terms, trend, confusion matrix |
| Drive | `twitter_sentiment/cli.py` | `scrape`, `analyze`, `train`, `predict`, `explain` |

Scraped tweets arrive **unlabelled**, so the lexicon is what produces the ground
truth the classifiers learn from. That is the honest description of the setup:
the classifier learns to reproduce the lexicon's judgements from bag-of-words
features, generalising to wording the lexicon has no entry for. It is not
trained on human annotations.

## Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Everything except the `scrape` command works with no network access, no API key
and no NLTK corpus download. The bundled lexicon and a fallback stop-word list
ship with the package.

Optional, for slightly better stop-word coverage:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

All commands are reachable through `Main.py` or `python -m twitter_sentiment`.

### Analyse a CSV of tweets

```bash
python Main.py analyze --input tweets_v2.csv --output-dir outputs
```

Writes `outputs/tweets_v2_labelled.csv` plus five PNG charts, and prints the
class distribution, mean sentiment score, engagement per class and the most
frequent terms.

```
          count  percentage
negative     32       32.32
neutral      29       29.29
positive     38       38.38

mean sentiment score : 0.0398

Engagement by sentiment:
           Likes  Retweets  tweets
sentiment
negative    1.97      0.16      32
neutral     0.48      0.07      29
positive    2.11      0.29      38
```

### Train a classifier

```bash
python Main.py train --input tweets_v2.csv --model linear_svm --output models/sentiment.joblib
```

Prints accuracy, macro precision/recall/F1, a cross-validated accuracy and a
per-class report; writes the fitted pipeline, `outputs/metrics.json` and
`outputs/confusion_matrix.png`.

Choose the estimator with `--model {naive_bayes,logistic_regression,linear_svm,random_forest}`
and the features with `--vectorizer {tfidf,count}`. If you have your own
annotations, skip the lexicon with `--label-column <column>`.

### Classify new text

```bash
python Main.py predict --model models/sentiment.joblib "the budget looks great" "markets crashed again"
python Main.py predict --lexicon "absolutely wonderful reforms"
python Main.py predict --input new_tweets.csv --output predictions.csv --json
```

```
positive score=+0.557  p=0.460  the budget looks great
negative score=-0.572  p=0.464  markets crashed again
```

`score` is always the lexicon score; `p` is the model's confidence and is
omitted for `linear_svm`, which has no calibrated probability.

### Understand a score

```bash
python Main.py explain "not good at all"
```

```json
{
  "cleaned": "not good at all",
  "matched_words": {},
  "word_score": 0.0,
  "idiom_score": -1.6,
  "emoji_score": 0.0,
  "punctuation_amplifier": 0.0,
  "score": -0.3818
}
```

### Collect tweets

```bash
cp .env.example .env      # then fill in TWITTER_USERNAME / TWITTER_PASSWORD
python Main.py scrape --query "#Budget2025" --output data/tweets.csv --max-scrolls 40
```

## Getting the data

There is no public bulk export of tweets any more, and the free API tier does
not cover search. Three options, in order of how much friction they involve:

1. **Scrape it** with the `scrape` command above. You need a normal X account,
   Chrome, and a matching `chromedriver` (set `CHROMEDRIVER_PATH`, or leave it
   blank and let `webdriver-manager` fetch one). Two-factor authentication
   breaks the automated login, so log in manually once first. Scraping is
   rate-limited and against X's terms of service for commercial use -- keep it
   to small, personal, research-scale collections.
2. **Use the X API** (`api.x.com`, paid tiers) and export to a CSV with a
   `Tweet` column. Everything downstream will pick it up.
3. **Use any existing tweet dataset** -- Kaggle's Sentiment140, the
   `tweet_eval` collection, or your own archive. `load_tweets` accepts common
   column spellings (`text`, `content`, `full_text`, `created_at`,
   `like_count`, ...) and normalises them.

Only a `Tweet` text column is required. `Timestamp`, `Likes` and `Retweets` are
used when present and skipped when absent.

**No tweet data is committed to this repository.** `.gitignore` excludes `*.csv`
so nothing scraped ends up in git; the only exception is the 20-row synthetic
fixture under `tests/fixtures/`, which contains no real tweets.

## Configuration

Copy `.env.example` to `.env`. Every setting is read from the environment --
no credential or absolute path is hardcoded anywhere.

| Variable | Purpose |
|---|---|
| `TWITTER_USERNAME` / `TWITTER_EMAIL` / `TWITTER_PASSWORD` | login for the scraper |
| `CHROMEDRIVER_PATH` | chromedriver location; blank means auto-download |
| `SCRAPER_HEADLESS` | run Chrome without a window |
| `SCRAPER_QUERY` / `SCRAPER_MAX_SCROLLS` | collection defaults |
| `SCRAPER_SCROLL_PAUSE_MIN` / `_MAX` | randomised pause between scrolls |
| `SCRAPER_PAGE_TIMEOUT` | Selenium explicit-wait timeout |

## How the lexicon scores a tweet

1. URLs, @mentions and the `RT @user:` marker are dropped; hashtags are split
   into words (`#StockMarketCrash` -> `stock market crash`).
2. Multi-word idioms are matched first and *consume* their component words, so
   `not good` scores once, not twice.
3. Each remaining word contributes its valence, adjusted by nearby intensifiers
   (`very good` > `good`, `slightly good` < `good`) and flipped-and-dampened by
   a negation in the preceding three tokens.
4. ALL-CAPS words, emoji, emoticons and repeated `!`/`?` adjust the total.
5. The sum is squashed into `[-1, 1]`; scores outside `+/-0.05` become
   `positive` / `negative`, the rest `neutral`. Tune the band with
   `--threshold`.

If you have NLTK's VADER lexicon downloaded, `--scorer vader` swaps it in.

## Measured performance

On the bundled 99-tweet `#Budget2025` sample, 25% held out, labels from the
lexicon:

| Model | Accuracy | Macro F1 | 5-fold CV accuracy |
|---|---|---|---|
| Naive Bayes | 0.40 | 0.39 | 0.49 |
| Logistic regression | 0.40 | 0.37 | 0.48 |
| Linear SVM | **0.44** | **0.43** | 0.46 |
| Random forest | 0.36 | 0.32 | 0.42 |

These are low, and honestly so: 99 tweets across three classes is far too small
to learn a vocabulary from, and the held-out set is 25 rows. Reproduce them with
`python Main.py train --input tweets_v2.csv --model <name>`. Expect the numbers
to climb substantially with a few thousand tweets -- the pipeline is the
deliverable here, not the score.

## Project structure

```
Main.py                         entry point, forwards to the CLI
twitter_sentiment/
  cli.py                        argparse commands
  config.py                     environment-backed settings
  dataset.py                    CSV loading, column normalisation, cleaning
  labeling.py                   score -> positive / neutral / negative
  lexicon.py                    the offline valence lexicon
  model.py                      pipelines, evaluation, persistence
  preprocessing.py              text cleaning and tokenisation
  scraper.py                    Selenium collector
  visualize.py                  matplotlib / seaborn charts
tests/                          140 tests, all offline
  fixtures/sample_tweets.csv    20 synthetic tweets
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

140 tests, no network, no browser and no real data. The Selenium code is
exercised through fake driver and element objects.

## Known limitations

- The lexicon is English-only and roughly 400 entries; sarcasm and code-mixed
  Hindi-English tweets are scored badly.
- Labels come from the lexicon, so the classifier inherits its blind spots.
  Supply `--label-column` if you have human annotations.
- X changes its DOM regularly; the scraper targets `data-testid` attributes,
  which are more stable than XPaths but not permanent.
- Automated login fails when two-factor authentication is enabled.

## License

MIT
