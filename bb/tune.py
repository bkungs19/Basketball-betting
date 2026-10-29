"""Learn from every line the model scored, not just the ones it picked.

Each run saves the lines it scored, with the model's and the books' chance of
the over, in data/scored/. Once those games are final, every prop type gets
two numbers fit on the results:

- shift: a log-odds correction for a model that runs high or low on that prop
  type (the early-season Under lean, say);
- weight: how much the pick probability leans on the model rather than the
  books.

Both stay near the defaults until a prop type has a few thousand graded lines,
and recent nights count more than old ones, so they keep adjusting through the
season. A player's first game of a season is left out: the model runs low there
and that doesn't recur until next season.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from .config import DATA, MARKET_LABELS, MARKETS, MODEL_WEIGHT

SCORED_DIR = DATA / "scored"
TUNING = DATA / "tuning.json"
SCORED_COLUMNS = ["date", "player_id", "player_name", "stat", "line", "season_games", "p_model_raw", "market_p_over", "n_books"]
PRIOR_LINES = 3000  # graded lines a prop type needs before its own fit counts as much as the defaults
HALF_LIFE_DAYS = 30
SHIFTS = np.round(np.arange(-1, 1.0001, 0.02), 2)
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)


def save_scored(day: dt.date, props: pd.DataFrame) -> None:
    """Keep each scored line once (books quoting the same line share one row)."""
    if props.empty:
        return
    s = props.drop_duplicates(["player_id", "stat", "line"]).assign(date=day.isoformat())
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    s[SCORED_COLUMNS].to_csv(SCORED_DIR / f"{day.isoformat()}.csv", index=False)


def graded_lines(box: pd.DataFrame, before: dt.date) -> pd.DataFrame:
    """Saved lines from days before `before`, with whether the over hit (pushes and DNPs dropped)."""
    files = [p for p in sorted(SCORED_DIR.glob("*.csv")) if p.stem < before.isoformat()]
    if not files:
        return pd.DataFrame(columns=SCORED_COLUMNS + ["over"])
    lines = pd.concat([pd.read_csv(p, dtype={"player_id": str}) for p in files], ignore_index=True)
    lines = lines[lines["season_games"] > 0]
    played = box[(box["dnp"] == 0) & (box["min"] > 0)].copy()
    played["date"] = pd.to_datetime(played["game_date"]).dt.strftime("%Y-%m-%d")
    played["player_id"] = played["player_id"].astype(str)
    out = []
    for stat, cols in MARKETS.items():
        actual = played[["date", "player_id"]].assign(actual=played[cols].sum(axis=1))
        out.append(lines[lines["stat"] == stat].merge(actual, on=["date", "player_id"]))
    g = pd.concat(out, ignore_index=True)
    g = g[(g["actual"] != g["line"]) & g["p_model_raw"].notna() & g["market_p_over"].notna()]
    return g.assign(over=(g["actual"] > g["line"]).astype(int))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _loglik(z, y, w):
    """Weighted log-likelihood of outcomes y for log-odds z (z may have leading grid axes)."""
    return (w * (y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z))).sum(axis=-1)


def fit(box: pd.DataFrame, today: dt.date) -> dict:
    """Fit and save the shift and weight for every prop type."""
    g = graded_lines(box, today)
    report = {"through": g["date"].max() if len(g) else None, "lines": int(len(g)), "markets": {}}
    for stat in MARKETS:
        m = g[g["stat"] == stat]
        entry = {"label": MARKET_LABELS[stat], "lines": int(len(m)), "shift": 0.0, "weight": MODEL_WEIGHT}
        if len(m):
            age = (pd.Timestamp(today) - pd.to_datetime(m["date"])).dt.days.values
            sw = 0.5 ** (age / HALF_LIFE_DAYS)
            y, zm, zk = m["over"].values, _logit(m["p_model_raw"]), _logit(m["market_p_over"])
            # 1) how far the model runs high or low, 2) how much to lean on the corrected model
            b_hat = SHIFTS[np.argmax(_loglik(zm[None, :] + SHIFTS[:, None], y, sw))]
            z = WEIGHTS[:, None] * (zm + b_hat)[None, :] + (1 - WEIGHTS[:, None]) * zk[None, :]
            w_hat = WEIGHTS[np.argmax(_loglik(z, y, sw))]
            k = sw.sum() / (sw.sum() + PRIOR_LINES)
            shift, weight = k * b_hat, MODEL_WEIGHT + k * (w_hat - MODEL_WEIGHT)
            p = 1 / (1 + np.exp(-(weight * (zm + shift) + (1 - weight) * zk)))
            entry.update({
                "shift": round(float(shift), 3), "weight": round(float(weight), 3),
                "over_rate": round(float(y.mean()), 3),
                "brier_model": round(float(np.mean((m["p_model_raw"] - y) ** 2)), 4),
                "brier_market": round(float(np.mean((m["market_p_over"] - y) ** 2)), 4),
                "brier_tuned": round(float(np.mean((p - y) ** 2)), 4),
            })
        report["markets"][stat] = entry
    TUNING.write_text(json.dumps(report, indent=1))
    return report


def load() -> dict:
    return json.loads(TUNING.read_text()) if TUNING.exists() else {"markets": {}}
