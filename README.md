# Are LLMs Calibrated Forecasters?

A small, public study of probability calibration in LLM forecasting, using
resolved prediction markets as ground truth.

An LLM is asked to blindly estimate the probability that a Kalshi market
resolves YES - it sees only the question, the official resolution rules, and
the close date. It never sees the market's price or any trading data. We then
ask three questions:

1. RQ1 - Are raw LLM probabilities calibrated out of the box
   (reliability diagrams, ECE, Brier score)?
2. RQ2 - Do simple post-hoc corrections (temperature scaling, Platt,
   isotonic regression) close the gap? Fitted on a time-ordered training
   split, evaluated on a newer holdout. No leakage.
3. RQ3 - Where does a blind LLM land relative to market-implied
   probabilities, and in which market categories does it do best/worst?

## Pipeline

    python -m src.cli pull        # resolved Kalshi markets -> data/markets.parquet
    python -m src.cli enrich      # market price T-24h -> data/prices.jsonl (resumable)
    python -m src.cli forecast    # blind LLM P(yes) -> data/forecasts.jsonl (resumable)
    python -m src.cli evaluate    # calibrators + metrics + plots -> reports/

    cp .env.example .env          # provider config (zai / openai / gemini / custom)
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

## Design notes

- Blind elicitation: the forecast prompt contains no market prices,
  volumes, or liquidity. Any edge the LLM shows comes from the question text.
- Time-split evaluation: calibrators are fit on the oldest 75 percent of
  markets and evaluated on the newest 25 percent, mirroring deployment
  (you calibrate on the past, forecast the future).
- Metrics: Brier score, log-loss, expected calibration error (10 bins),
  AUROC, against market-price baselines and a constant-0.5 baseline.
- Fair market baseline: a settled market's last trade converges to 0/1, so
  the primary market baseline is the mid-quote 24 hours before close,
  from the public candlestick API.
- Honest results live in reports/summary.md, including the runs where
  post-hoc correction does NOT help and where the market stays ahead.

## Data

Pulled from the public Kalshi trade API (settled market metadata and
candlesticks are public; no auth, bounded pagination, rate-limit friendly).
The dataset is built per-series with server-side series_ticker filters:
crypto strike tiers (KXBTCD, KXETHD, KXDOGE, KXXRP), macro (KXFED, KXCPI),
weather (KXHIGHDEN, KXHIGHMIA, KXRAIN), and sports (KXNBAGAME, KXNBATOTAL,
KXNFLGAME, KXMLBGAME). Only traded markets (volume > 0), capped per series
for balance. The series universe is configurable in src/config.py. Data
files are not redistributed - the pull command regenerates them.

## Status

Dataset built: 723 resolved markets (crypto 240, sports 240, weather 180,
macro 63) with T-24h market prices. Forecast + evaluation runs are driven by
.env config; findings land in reports/summary.md once the first model has
been run.

## Attribution

Market data courtesy of Kalshi. This project is for research/educational
purposes and is not affiliated with Kalshi.
