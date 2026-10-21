"""Pick the best 2 or 3 leg parlay from today's scored props."""
from __future__ import annotations

import itertools
import math

import pandas as pd

from .config import PARLAY_MAX_LEGS, PARLAY_MIN_HIT, PARLAY_MIN_LEGS
from .odds import decimal_to_american

LEG_FIELDS = ["player_name", "player_id", "team", "opp_abbr", "event_id", "start", "stat", "market", "side", "line",
              "odds", "decimal", "p_hit", "p_model_hit", "p_market_hit", "ev", "proj", "recent_avg", "book",
              "deeplink", "injury"]
POOL = 40  # most likely positive-EV legs per book to combine


def legs_for_book(props: pd.DataFrame, book: str) -> pd.DataFrame:
    """Positive-EV legs at one book, most likely to hit first."""
    c = props[(props["book"] == book) & props["eligible"] & (props["ev"] > 0)]
    return c.sort_values("p_hit", ascending=False).head(POOL)


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
    """One parlay per book, best first.

    Each book's parlay is the 2 or 3 leg combination of positive-EV legs with
    the highest expected value, prod(p_i * d_i) - 1, among those whose chance
    to hit, prod(p_i), is at least 30%.
    """
    parlays = []
    if props.empty:
        return parlays
    for book in props["book"].unique():
        legs = legs_for_book(props, book)
        best = best_for_book(legs)
        if best is None:
            continue
        ev, p_hit, decimal, combo = best
        chosen = pd.DataFrame(combo, columns=legs.columns)
        parlays.append({
            "book": book,
            "legs": chosen[[f for f in LEG_FIELDS if f in chosen.columns]].to_dict("records"),
            "decimal": round(decimal, 3),
            "american": decimal_to_american(decimal),
            "p_hit": round(p_hit, 4),
            "ev": round(ev, 4),
        })
    return sorted(parlays, key=lambda p: p["ev"], reverse=True)
