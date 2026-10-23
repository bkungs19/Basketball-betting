"""Price today's picks for the apps Brandon uses.

Brandon bets on Fliff and Rebet, which the free odds feed doesn't carry, so
picks are priced at the typical (median) odds across the books it does carry,
and every pick carries the worst odds still worth taking (break-even).
"""
from __future__ import annotations

import pandas as pd

from .odds import decimal_to_american

LEG_FIELDS = ["player_name", "player_id", "team", "opp_abbr", "event_id", "start", "stat", "market", "side", "line",
              "odds", "decimal", "p_hit", "p_model_hit", "p_market_hit", "ev", "proj", "recent_avg", "book",
              "deeplink", "injury", "min_odds", "n_books", "kind", "label"]
MIN_BOOKS = 2  # lines offered by several books are the ones other apps are likely to have too


def consensus(props: pd.DataFrame) -> pd.DataFrame:
    """Each line once, priced at the median odds across books, with the worst odds still worth taking."""
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
