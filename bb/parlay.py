"""Pick the highest expected value parlay from today's scored props."""
from __future__ import annotations

import math

import pandas as pd

from .config import PARLAY_MAX_LEGS, PARLAY_MIN_LEGS
from .odds import decimal_to_american

LEG_FIELDS = ["player_name", "player_id", "team", "opp_abbr", "event_id", "start", "stat", "market", "side", "line",
              "odds", "decimal", "p_hit", "p_model_hit", "p_market_hit", "ev", "proj", "recent_avg", "book",
              "deeplink", "injury"]


def legs_for_book(props: pd.DataFrame, book: str) -> pd.DataFrame:
    """Best positive-EV leg per game at one book (one player per game keeps legs independent)."""
    c = props[(props["book"] == book) & props["eligible"] & (props["ev"] > 0)]
    c = c.sort_values("ev", ascending=False)
    c = c.drop_duplicates("player_id").drop_duplicates("event_id")
    return c


def build(props: pd.DataFrame) -> list[dict]:
    """One parlay per book, best first.

    Parlay EV is prod(p_i * d_i) - 1, so with independent legs the best k-leg
    parlay takes the k legs with the largest p_i * d_i. Every candidate leg has
    p_i * d_i > 1, so adding a 4th leg raises EV whenever one exists.
    """
    parlays = []
    if props.empty:
        return parlays
    for book in props["book"].unique():
        legs = legs_for_book(props, book)
        if len(legs) < PARLAY_MIN_LEGS:
            continue
        legs = legs.assign(score=legs["p_hit"] * legs["decimal"]).sort_values("score", ascending=False)
        legs = legs.head(PARLAY_MAX_LEGS)
        decimal = math.prod(legs["decimal"])
        p_hit = math.prod(legs["p_hit"])
        parlays.append({
            "book": book,
            "legs": legs[[f for f in LEG_FIELDS if f in legs.columns]].to_dict("records"),
            "decimal": round(decimal, 3),
            "american": decimal_to_american(decimal),
            "p_hit": round(p_hit, 4),
            "ev": round(p_hit * decimal - 1, 4),
        })
    return sorted(parlays, key=lambda p: p["ev"], reverse=True)
