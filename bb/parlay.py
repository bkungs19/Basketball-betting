"""Pick the best 2 or 3 leg parlay from today's scored props.

Brandon bets on Fliff and Rebet, which the free odds feed doesn't carry, so
picks are priced at the typical (median) odds across the books it does carry,
and every leg and parlay carries the worst odds still worth taking.
"""
from __future__ import annotations

import itertools
import math

import pandas as pd

from .config import PARLAY_MAX_LEGS, PARLAY_MIN_HIT, PARLAY_MIN_LEGS
from .odds import decimal_to_american

LEG_FIELDS = ["player_name", "player_id", "team", "opp_abbr", "event_id", "start", "stat", "market", "side", "line",
              "odds", "decimal", "p_hit", "p_model_hit", "p_market_hit", "ev", "proj", "recent_avg", "book",
              "deeplink", "injury", "min_odds", "n_books", "kind", "label"]
POOL = 40  # most likely positive-EV legs to combine
MIN_BOOKS = 2  # lines offered by several books are the ones other apps are likely to have too


def consensus(props: pd.DataFrame) -> pd.DataFrame:
    """Each prop line once, priced at the median odds across books, with the break-even odds."""
    e = props[props["eligible"]]
    if e.empty:
        return e.assign(min_odds=[], n_books=[])
    keys = ["player_id", "stat", "side", "line"]
    g = e.groupby(keys)
    c = e.drop_duplicates(keys).set_index(keys)
    c["decimal"] = g["decimal"].median()
    c["n_books"] = g["book"].nunique()
    c = c.reset_index()
    c["odds"] = c["decimal"].map(decimal_to_american)
    c["ev"] = c["p_hit"] * c["decimal"] - 1
    c["min_odds"] = (1 / c["p_hit"]).map(decimal_to_american)
    c["book"] = "any"
    c["deeplink"] = ""
    return c[c["n_books"] >= MIN_BOOKS]


def candidate_legs(lines: pd.DataFrame) -> pd.DataFrame:
    """Positive-EV legs, most likely to hit first."""
    return lines[lines["ev"] > 0].sort_values("p_hit", ascending=False).head(POOL)


def _combos(legs: pd.DataFrame):
    """2 and 3 leg combinations with one leg per player and per game, so legs stay independent."""
    rows = list(legs.itertuples(index=False))
    for k in range(PARLAY_MIN_LEGS, PARLAY_MAX_LEGS + 1):
        for combo in itertools.combinations(rows, k):
            if len({r.player_id for r in combo}) == k and len({r.event_id for r in combo}) == k:
                p_hit = math.prod(r.p_hit for r in combo)
                decimal = math.prod(r.decimal for r in combo)
                yield p_hit * decimal - 1, p_hit, decimal, combo


def best_for_book(legs: pd.DataFrame):
    """Highest-EV combination that hits at least PARLAY_MIN_HIT of the time."""
    best = None
    for ev, p_hit, decimal, combo in _combos(legs):
        if p_hit >= PARLAY_MIN_HIT and (best is None or ev > best[0]):
            best = (ev, p_hit, decimal, combo)
    return best


def build(props: pd.DataFrame) -> list[dict]:
    """Today's parlay, priced at typical odds, as a one-item list (empty if none qualifies).

    It's the 2 or 3 leg combination of positive-EV legs with the highest
    expected value, prod(p_i * d_i) - 1, among those whose chance to hit,
    prod(p_i), is at least 30%.
    """
    if props.empty:
        return []
    legs = candidate_legs(consensus(props))
    best = best_for_book(legs)
    if best is None:
        return []
    ev, p_hit, decimal, combo = best
    chosen = pd.DataFrame(combo, columns=legs.columns)
    return [{
        "book": "any",
        "legs": chosen[[f for f in LEG_FIELDS if f in chosen.columns]].to_dict("records"),
        "decimal": round(decimal, 3),
        "american": decimal_to_american(decimal),
        "min_american": decimal_to_american(1 / p_hit),
        "p_hit": round(p_hit, 4),
        "ev": round(ev, 4),
    }]
