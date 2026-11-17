"""Central configuration: paths, series universe, defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

MARKETS_PARQUET = DATA_DIR / "markets.parquet"
FORECASTS_JSONL = DATA_DIR / "forecasts.jsonl"
PRICES_JSONL = DATA_DIR / "prices.jsonl"

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Series universe (discovered via public API). Server-side filter:
#   GET /markets?status=settled&series_ticker=<S>
SERIES: dict = {
    # crypto daily strike tiers
    "KXBTCD":    "crypto",
    "KXETHD":    "crypto",
    "KXDOGE":    "crypto",
    "KXXRP":     "crypto",
    # macro
    "KXFED":     "macro",
    "KXCPI":     "macro",
    # weather
    "KXHIGHDEN": "weather",
    "KXHIGHMIA": "weather",
    "KXRAIN":    "weather",
    # sports
    "KXNBAGAME": "sports",
    "KXNBATOTAL": "sports",
    "KXNFLGAME": "sports",
    "KXMLBGAME": "sports",
}

DEFAULT_MONTHS = 12
DEFAULT_PER_SERIES = 60

# Time split for calibration fitting (older portion trains calibrators)
TRAIN_FRACTION = 0.75
