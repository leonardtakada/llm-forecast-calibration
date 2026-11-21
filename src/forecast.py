"""Blind LLM forecasting: P(yes) from question + rules only, no market price.

The forecaster never sees the market's price, volume, or any trading data —
that is the design. Any information edge must come from the question text.
Results append to a JSONL cache; reruns skip already-forecast pairs.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests
from dotenv import load_dotenv

from .config import FORECASTS_JSONL, MARKETS_PARQUET

load_dotenv()

PROVIDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "zai": ("https://api.z.ai/api/paas/v4", "glm-4.6"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
    ),
}

SYSTEM_PROMPT = (
    "You are a careful, well-calibrated probabilistic forecaster. "
    "You will be given a question and its official resolution rules. "
    "Estimate the probability that the question resolves YES. "
    "Reason briefly about base rates, historical frequencies, and "
    "time remaining. Output ONLY a JSON object: "
    '{"probability": <float 0..1>, "rationale": "<one sentence>"}'
)


def _provider_config(provider, model, base_url, api_key):
    default_base, default_model = PROVIDER_DEFAULTS.get(provider, ("", ""))
    base = base_url or default_base or os.environ.get("LLM_BASE_URL", "")
    if not base:
        raise SystemExit(f"Unknown provider '{provider}' and no LLM_BASE_URL set.")
    key = api_key or os.environ.get("LLM_API_KEY", "")
    if not key:
        raise SystemExit("No API key: set LLM_API_KEY in .env")
    return base.rstrip("/"), model or os.environ.get("LLM_MODEL") or default_model, key


def _ask_llm(base, model, key, question, rules, close_date):
    user = (
        f"Question: {question}\n"
        f"Resolution rules: {rules or 'See question.'}\n"
        f"Question closes: {close_date}\n\n"
        "What is the probability this resolves YES?"
    )
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=120,
            )
        except requests.exceptions.RequestException:
            time.sleep(15 * (attempt + 1))
            continue
        if resp.status_code in (429, 502, 503):
            wait = float(resp.headers.get("Retry-After", 20 * (attempt + 1)))
            time.sleep(min(wait, 180))
            continue
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        text = message.get("content") or ""
        if not text.strip():
            text = message.get("reasoning_content") or ""
        return {"probability": _parse_prob(text), "rationale": text[:300]}
    raise RuntimeError("rate limited or timed out beyond retries")


def _parse_prob(text: str) -> float:
    m = re.search(r'"probability"\s*:\s*([0-9.]+)', text)
    if m:
        return float(m.group(1))
    floats = [float(x) for x in re.findall(r"\b0?\.?\d+\b", text) if 0.0 <= float(x) <= 1.0]
    if floats:
        return floats[0]
    raise ValueError(f"Could not parse probability from: {text[:200]}")


def load_cache() -> dict:
    cache: dict = {}
    if FORECASTS_JSONL.exists():
        with open(FORECASTS_JSONL) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    cache[(row["market_id"], row["model"])] = row
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def forecast(model, provider, limit, sleep_s=4.0):
    import pandas as pd

    if not MARKETS_PARQUET.exists():
        raise SystemExit("No markets.parquet - run the pull command first.")
    df = pd.read_parquet(MARKETS_PARQUET)

    base, model_name, key = _provider_config(
        provider, model, os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
    )
    tag = f"{provider}:{model_name}"
    cache = load_cache()
    todo = [row for _, row in df.iterrows() if (row["market_id"], tag) not in cache]
    if limit:
        todo = todo[:limit]
    print(f"Model tag {tag}: {len(todo)} to forecast ({len(cache)} cached)")

    FORECASTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(FORECASTS_JSONL, "a") as f:
        for row in todo:
            try:
                result = _ask_llm(
                    base, model_name, key,
                    str(row["question"]),
                    str(row.get("rules") or ""),
                    str(row["close_time"].date()),
                )
            except Exception as e:  # log and continue: this is batch work
                print(f"  SKIP {row['market_id']}: {e}")
                continue
            record = {
                "market_id": row["market_id"],
                "model": tag,
                "probability": round(float(result["probability"]), 6),
                "rationale": result["rationale"],
                "forecasted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}")
            time.sleep(sleep_s)
    print(f"Forecasting complete: {done} new, total {len(load_cache())} records")
