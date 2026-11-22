"""Pull resolved Kalshi markets via the public API into a parquet dataset.

No auth required: settled market metadata and candlesticks are public.
Per-series server-side filtering (the bulk settled feed is dominated by
exotic MVE markets). Bounded pagination, polite rate limiting, honors
Retry-After on 429.
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import requests

from .config import (
    KALSHI_BASE,
    MARKETS_PARQUET,
    PRICES_JSONL,
    SERIES,
)


def _get(url: str, params: dict) -> dict:
    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            print(f"    rate limited, sleeping {wait:.0f}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Rate limited beyond retries")


def fetch_series(series: str, months: int, target: int, max_pages: int = 4) -> pd.DataFrame:
    """Fetch settled, traded markets for one series (newest first)."""
    min_close_ts = int(pd.Timestamp.now(tz="UTC").timestamp() - 30 * months * 86400)
    rows: list = []
    cursor = ""
    for page in range(max_pages):
        params = {
            "status": "settled",
            "series_ticker": series,
            "min_close_ts": min_close_ts,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        payload = _get(f"{KALSHI_BASE}/markets", params)
        markets = payload.get("markets", [])
        rows.extend(markets)
        cursor = payload.get("cursor") or ""
        traded = sum(float(m.get("volume_fp") or 0) > 0 for m in rows)
        if traded >= target or not cursor:
            break
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "volume_fp" in df.columns:
        vols = pd.to_numeric(df["volume_fp"], errors="coerce").fillna(0)
        df = df[vols > 0]
    df = df.drop_duplicates(subset="ticker")
    df["series_ticker"] = series
    return df


def pull(months: int, per_series: int) -> pd.DataFrame:
    print(f"Pulling settled Kalshi markets (last {months}mo, target {per_series}/series)...")
    frames = []
    for series, cat in SERIES.items():
        try:
            df = fetch_series(series, months=months, target=per_series)
        except Exception as e:  # noqa: BLE001
            print(f"  {series}: FAILED ({e})")
            continue
        if df.empty:
            print(f"  {series}: 0")
            continue
        df["category"] = cat
        frames.append(df)
        print(f"  {series}: {len(df)} traded settled markets")

    if not frames:
        print("No markets collected.")
        sys.exit(1)
    allm = pd.concat(frames, ignore_index=True)

    keep = {
        "ticker": "market_id",
        "series_ticker": "series_ticker",
        "event_ticker": "event_id",
        "title": "question",
        "subtitle": "strike_subtitle",
        "rules_primary": "rules",
        "close_time": "close_time",
        "settlement_ts": "settlement_ts",
        "result": "settlement",
        "last_price_dollars": "market_close_frac",
        "volume_fp": "volume",
        "open_interest_fp": "open_interest",
        "floor_strike": "floor_strike",
        "category": "category",
    }
    cols = [c for c in keep if c in allm.columns]
    allm = allm[cols].rename(columns=keep)

    allm["settlement"] = allm["settlement"].astype(str).str.lower()
    allm = allm[allm["settlement"].isin(["yes", "no"])]
    allm["outcome"] = (allm["settlement"] == "yes").astype(int)

    allm["close_time"] = pd.to_datetime(allm["close_time"], utc=True, errors="coerce")
    allm = allm[allm["close_time"].notna()]
    allm["market_close_frac"] = pd.to_numeric(allm["market_close_frac"], errors="coerce")

    # Cap per series to keep the dataset balanced
    allm = (
        allm.sort_values("close_time", ascending=False)
        .groupby("series_ticker", group_keys=False)
        .head(per_series)
        .reset_index(drop=True)
    )

    MARKETS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    allm.to_parquet(MARKETS_PARQUET, index=False)
    print(f"\nSaved {len(allm)} markets -> {MARKETS_PARQUET}")
    print(allm.groupby("category")["outcome"].agg(["count", "mean"]).round(3))
    return allm


# ---- candlestick enrichment: market-implied price N hours before close ----

def _mid_price(candle: dict):
    bid = float(candle.get("yes_bid", {}).get("close_dollars") or 0)
    ask = float(candle.get("yes_ask", {}).get("close_dollars") or 0)
    if 0 < bid <= ask < 1:
        return round((bid + ask) / 2, 4)
    for solo in (ask, bid):
        if 0 < solo < 1:
            return round(solo, 4)
    prev = candle.get("price", {}).get("previous_dollars")
    return float(prev) if prev not in (None, "") else None


def _row_price(r: dict):
    for k, v in r.items():
        if k.startswith("price_"):
            return v
    return None


def enrich(hours: int, limit, retry_nulls: bool = False) -> None:
    """Fetch market price N hours before close for each market (resumable)."""
    import json

    df = pd.read_parquet(MARKETS_PARQUET)
    cached: dict = {}
    if PRICES_JSONL.exists():
        for l in open(PRICES_JSONL):
            if l.strip():
                r = json.loads(l)
                cached[r["market_id"]] = _row_price(r)
    done = {k for k, v in cached.items() if v is not None or not retry_nulls}
    todo = df[~df["market_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"Enriching {len(todo)} markets with price {hours}h before close "
          f"({len(done)} cached)")

    PRICES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    with open(PRICES_JSONL, "a") as f:
        for _, row in todo.iterrows():
            close_ts = int(row["close_time"].timestamp())
            start_ts = close_ts - hours * 3600
            price = None
            try:
                payload = _get(
                    f"{KALSHI_BASE}/series/{row['series_ticker']}/markets/"
                    f"{row['market_id']}/candlesticks",
                    {"start_ts": start_ts, "end_ts": close_ts, "period_interval": 60},
                )
                candles = payload.get("candlesticks", [])
                price = None
                if candles:
                    tgt = close_ts - hours * 3600
                    for c in sorted(candles, key=lambda c: abs(int(c["end_period_ts"]) - tgt)):
                        price = _mid_price(c)
                        if price is not None:
                            break
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP {row['market_id']}: {e}")
            f.write(json.dumps({
                "market_id": row["market_id"],
                f"price_{hours}h": price,
            }) + "\n")
            f.flush()
            ok += 1
            if ok % 50 == 0:
                print(f"  {ok}/{len(todo)}")
            time.sleep(0.35)
    total = {k: v for k, v in cached.items()}
    nulls = sum(1 for v in total.values() if v is None)
    print(f"Enrichment complete: {ok} rows written | cache: {len(total)} markets, {nulls} still null")
