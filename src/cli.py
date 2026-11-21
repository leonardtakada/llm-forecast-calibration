"""Command-line entry point."""
from __future__ import annotations

import argparse
import json

from . import calibrate, data, forecast
from .config import (
    DEFAULT_MONTHS,
    DEFAULT_PER_SERIES,
    FORECASTS_JSONL,
    MARKETS_PARQUET,
    PRICES_JSONL,
)


def main():
    ap = argparse.ArgumentParser(prog="llm-forecast-calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="pull resolved Kalshi markets")
    p.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    p.add_argument("--per-series", type=int, default=DEFAULT_PER_SERIES)

    p = sub.add_parser("enrich", help="fetch market price N hours before close")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("forecast", help="run blind LLM forecasts")
    p.add_argument("--model", default=None, help="override model name")
    p.add_argument("--provider", default="zai", choices=["openai", "zai", "gemini", "custom"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=4.0)

    p = sub.add_parser("evaluate", help="fit calibrators and evaluate on holdout")
    p.add_argument("--model", default=None, help="filter to one model tag")

    sub.add_parser("status", help="show dataset + forecast counts")

    args = ap.parse_args()
    if args.cmd == "pull":
        data.pull(months=args.months, per_series=args.per_series)
    elif args.cmd == "enrich":
        data.enrich(hours=args.hours, limit=args.limit)
    elif args.cmd == "forecast":
        forecast.forecast(model=args.model, provider=args.provider, limit=args.limit, sleep_s=args.sleep)
    elif args.cmd == "evaluate":
        calibrate.evaluate(model=args.model)
    elif args.cmd == "status":
        if MARKETS_PARQUET.exists():
            import pandas as pd
            df = pd.read_parquet(MARKETS_PARQUET)
            print(f"markets: {len(df)}")
            print(df.groupby("category")["outcome"].agg(["count", "mean"]).round(3))
        else:
            print("markets: none (run pull)")
        if FORECASTS_JSONL.exists():
            rows = [json.loads(l) for l in open(FORECASTS_JSONL)]
            from collections import Counter
            print("forecasts:", dict(Counter(r["model"] for r in rows)))
        else:
            print("forecasts: none (run forecast)")
        if PRICES_JSONL.exists():
            n = sum(1 for l in open(PRICES_JSONL) if l.strip())
            print(f"enriched prices: {n}")
        else:
            print("enriched prices: none (run enrich)")


if __name__ == "__main__":
    main()
