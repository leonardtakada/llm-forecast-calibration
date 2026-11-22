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

## Results

### v1 (723 markets, global time split)

Raw glm-4.6 scored AUROC 0.688 with 0.912 correlation to market prices -
but the holdout clustered into a single settlement day dominated by
same-day strike ladders, which inflates apparent agreement (both model and
market order strikes monotonically by moneyness). Classic evaluation
artifact; fixed in v2.

### v2 (1,028 markets, stratified per-series split, full T-24h baseline)

- Raw glm-4.6: Brier 0.266, ECE 0.196, AUROC 0.539
- Best post-hoc (isotonic): Brier 0.226, ECE 0.048
- Market at T-24h: Brier 0.121, AUROC 0.900
- corr(LLM, market) = 0.265 overall; by category: macro 0.91, sports 0.37,
  crypto 0.11, weather -0.26

### Takeaways

1. Single-cluster holdouts can inflate correlation via question structure;
   stratified time splits matter.
2. Blind-LLM forecasting skill is category-dependent: strong where priors
   are informative (Fed/CPI), absent where real-time data is required
   (weather).
3. Post-hoc calibration corrects overconfidence (ECE 0.20 -> 0.05) but
   cannot create discrimination (AUROC ~0.54 across all post-hoc methods).

### glm-5.3 comparison (same 1,028 markets, same split)

- Raw: Brier 0.2644, ECE 0.1983, AUROC 0.5507 (glm-4.6: 0.2657 / 0.1960 / 0.5386)
- Best post-hoc: Platt 0.2240 (glm-4.6 isotonic: 0.2259)
- corr(LLM, market): 0.272 (glm-4.6: 0.265)

Verdict: a statistical tie. The newer, larger model brings no meaningful
gain on blind forecasting - the bottleneck is missing information, not
model capability.

## Attribution

Market data courtesy of Kalshi. This project is for research/educational
purposes and is not affiliated with Kalshi.
