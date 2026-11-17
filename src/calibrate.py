"""Calibration evaluation: are LLM forecasts calibrated, and can simple
post-hoc corrections close the gap to market-implied probabilities?

Protocol:
  1. Merge blind LLM forecasts with resolved markets.
  2. Time-split: older 75% fits calibrators, newest 25% is the holdout.
  3. Fit temperature scaling / Platt / isotonic on train logits.
  4. Compare against identity (raw) plus baselines: market close price, 0.5.
No leakage: calibrators only ever see train outcomes.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import (
    FORECASTS_JSONL,
    MARKETS_PARQUET,
    PRICES_JSONL,
    REPORTS_DIR,
    TRAIN_FRACTION,
)

EPS = 1e-4


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def nll_at_temperature(t: float, z: np.ndarray, y: np.ndarray) -> float:
    p = sigmoid(z / t)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(z_train, y_train) -> float:
    lo, hi = 0.2, 20.0
    for _ in range(200):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll_at_temperature(m1, z_train, y_train) < nll_at_temperature(m2, z_train, y_train):
            hi = m2
        else:
            lo = m1
    return round((lo + hi) / 2, 4)


def brier(p, y) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = len(p)
    e = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        e += (mask.sum() / total) * abs(y[mask].mean() - p[mask].mean())
    return float(e)


def auroc(p, y) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def evaluate(model: str | None) -> pd.DataFrame | None:
    if not MARKETS_PARQUET.exists() or not FORECASTS_JSONL.exists():
        print("Need markets.parquet and forecasts.jsonl first (run pull + forecast).")
        return None

    df = pd.read_parquet(MARKETS_PARQUET)
    rows = [json.loads(l) for l in open(FORECASTS_JSONL)]
    fc = pd.DataFrame(rows)
    if model:
        fc = fc[fc["model"] == model]
        if fc.empty:
            print(f"No forecasts for model tag '{model}'")
            return None
    tag = fc["model"].iloc[0]

    m = df.merge(fc[["market_id", "probability"]], on="market_id", how="inner")
    if PRICES_JSONL.exists():
        def _pre(r):
            for k, v in r.items():
                if k.startswith("price_"):
                    return v
            return None
        prows = [json.loads(l) for l in open(PRICES_JSONL) if l.strip()]
        pr = pd.DataFrame([
            {"market_id": r["market_id"], "market_pre": _pre(r)} for r in prows
        ]).dropna(subset=["market_id"]).drop_duplicates("market_id")
        m = m.merge(pr, on="market_id", how="left")
    m = m.sort_values("close_time").reset_index(drop=True)
    n = len(m)
    if n < 40:
        print(f"Only {n} merged forecasts - need more for a stable holdout.")
        return None

    split = int(n * TRAIN_FRACTION)
    tr, te = m.iloc[:split], m.iloc[split:]
    print(f"Merged: {n} | train {len(tr)} ({tr.close_time.min().date()} to {tr.close_time.max().date()})")
    print(f"holdout {len(te)} ({te.close_time.min().date()} to {te.close_time.max().date()})")

    z_tr, z_te = logit(tr["probability"].values), logit(te["probability"].values)
    y_tr, y_te = tr["outcome"].values, te["outcome"].values

    # Fit calibrators on TRAIN only
    T = fit_temperature(z_tr, y_tr)
    platt = LogisticRegression()
    platt.fit(z_tr.reshape(-1, 1), y_tr)
    isotonic = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
    isotonic.fit(z_tr, y_tr)

    methods = {
        "identity (raw LLM)": sigmoid(z_te),
        f"temperature (T={T})": sigmoid(z_te / T),
        "platt scaling": platt.predict_proba(z_te.reshape(-1, 1))[:, 1],
        "isotonic regression": isotonic.predict(z_te),
    }
    baselines = {}
    if "market_pre" in te.columns and te["market_pre"].notna().sum() >= 30:
        baselines["baseline: market T-24h"] = te["market_pre"].values
    baselines["baseline: market last price"] = te["market_close_frac"].values
    baselines["baseline: constant 0.5"] = np.full(len(te), 0.5)

    def metrics_row(name, p):
        p = np.asarray(list(p), dtype=float)
        y = np.asarray(y_te)
        mask = ~np.isnan(p)
        p, y = p[mask], y[mask]
        return {
            "method": name,
            "n": int(mask.sum()),
            "brier": round(brier(p, y), 4),
            "log_loss": round(log_loss(p, y), 4),
            "ece": round(ece(p, y), 4),
            "auroc": round(auroc(p, y), 4),
        }

    result_rows = [metrics_row(name, p) for name, p in {**methods, **baselines}.items()]
    res = pd.DataFrame(result_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Reliability plot
    fig, ax = plt.subplots(figsize=(6, 5.5))
    edges = np.linspace(0, 1, 11)
    plot_sets = dict(methods)
    mkt_key = next((k for k in baselines if k.startswith("baseline: market")), None)
    if mkt_key:
        plot_sets["market"] = baselines[mkt_key]
    for name, p in plot_sets.items():
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() >= 3:
                xs.append(p[mask].mean())
                ys.append(y_te[mask].mean())
        ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed YES frequency")
    ax.set_title(f"Reliability on holdout - {tag}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "reliability.png", dpi=140)
    plt.close(fig)

    # Brier by category
    fig, ax = plt.subplots(figsize=(7, 4))
    cat = te.assign(brier_raw=(te["probability"] - te["outcome"]) ** 2)
    cat["brier_mkt"] = (te["market_close_cents"] / 100 - te["outcome"]) ** 2
    g = cat.groupby("category")[["brier_raw", "brier_mkt"]].mean().round(3)
    g.plot.bar(ax=ax)
    ax.set_title(f"Brier by category (holdout) - {tag}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "brier_by_category.png", dpi=140)
    plt.close(fig)

    raw_best = res.iloc[0]
    market_rows = res[res["method"].str.startswith("baseline: market")]
    market_row = market_rows.sort_values("brier").iloc[0]
    cal_rows = res.iloc[1:4].sort_values("brier")
    best_cal = cal_rows.iloc[0]
    mkt_col = (
        te["market_pre"]
        if "market_pre" in te.columns and te["market_pre"].notna().sum() > 0
        else te["market_close_frac"]
    )
    valid = mkt_col.notna()
    corr = float(np.corrcoef(te.loc[valid, "probability"], mkt_col[valid])[0, 1])

    summary = [
        f"# LLM Forecast Calibration - {tag}",
        "",
        f"- Merged forecasts: {n} | train: {len(tr)} | holdout: {len(te)}",
        f"- Holdout window: {te.close_time.min().date()} to {te.close_time.max().date()}",
        f"- Base rate (PYES in holdout): {y_te.mean():.3f}",
        "",
        "## Holdout results",
        "",
        res.to_markdown(index=False),
        "",
        "## Findings",
        "",
        f"- Raw LLM Brier {raw_best['brier']:.4f} (ECE {raw_best['ece']:.4f})",
        f"- Best post-hoc method: **{best_cal['method']}**, Brier {best_cal['brier']:.4f} "
        f"(delta {raw_best['brier'] - best_cal['brier']:+.4f})",
        f"- Market close baseline Brier {market_row['brier']:.4f} "
        f"({'LLM beats market' if best_cal['brier'] < market_row['brier'] else 'market still ahead'})",
        f"- Agreement: corr(LLM, market) = {corr:.3f}",
        "",
        "See reliability.png and brier_by_category.png.",
    ]
    (REPORTS_DIR / "summary.md").write_text("\n".join(summary))
    print("\n" + "\n".join(summary))
    return res
