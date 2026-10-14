"""Turn box scores into model features.

Every feature for a game is built only from games played before it, so the
same code serves training rows and tonight's games.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .config import MARKETS

BASE_STATS = ["pts", "reb", "ast", "fg3m", "min", "fga", "fg3a", "fta", "oreb", "dreb", "tov", "stl", "blk"]
TARGETS = {market: "t_" + market.replace("+", "_") for market in MARKETS}
RATE_STATS = ["pts", "reb", "ast", "fg3m", "fga", "fg3a"]
TEAM_STATS = ["pace", "off_rtg", "def_rtg", "allowed_pts", "allowed_reb", "allowed_ast", "allowed_fg3m"]


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    for market, col in TARGETS.items():
        df[col] = df[MARKETS[market]].sum(axis=1, min_count=1)
    return df


def _possessions(t: pd.DataFrame) -> pd.Series:
    return t["fga"] + 0.44 * t["fta"] - t["oreb"] + t["tov"]


def team_games(box: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with pace, ratings and what the defense allowed."""
    played = box[box["dnp"] == 0]
    agg = played.groupby(["game_id", "game_date", "season", "team", "opp"], as_index=False)[
        ["pts", "reb", "oreb", "ast", "fg3m", "fga", "fta", "tov", "min"]
    ].sum()
    opp = agg.rename(columns={c: "o_" + c for c in ["pts", "reb", "oreb", "ast", "fg3m", "fga", "fta", "tov", "min"]})
    opp = opp.rename(columns={"team": "opp", "opp": "team"})
    t = agg.merge(opp[["game_id", "team"] + [c for c in opp.columns if c.startswith("o_")]], on=["game_id", "team"])
    own_poss = _possessions(t)
    opp_poss = t["o_fga"] + 0.44 * t["o_fta"] - t["o_oreb"] + t["o_tov"]
    poss = (own_poss + opp_poss) / 2
    team_min = t["min"].clip(lower=200)
    t["pace"] = poss * 240 / team_min
    t["off_rtg"] = 100 * t["pts"] / poss
    t["def_rtg"] = 100 * t["o_pts"] / poss
    t["allowed_pts"] = t["o_pts"] * 240 / team_min
    t["allowed_reb"] = t["o_reb"] * 240 / team_min
    t["allowed_ast"] = t["o_ast"] * 240 / team_min
    t["allowed_fg3m"] = t["o_fg3m"] * 240 / team_min
    t["usage_total"] = t["fga"] + 0.44 * t["fta"] + t["tov"]
    return t


def team_state(tg: pd.DataFrame) -> pd.DataFrame:
    """Each team's form after each game (exponentially weighted, recent games count more)."""
    tg = tg.sort_values(["team", "game_date"])
    out = tg[["team", "game_date"]].copy()
    g = tg.groupby("team")
    for c in TEAM_STATS:
        out["ts_" + c] = g[c].transform(lambda s: s.ewm(halflife=10).mean())
    return out


def player_state(played: pd.DataFrame) -> pd.DataFrame:
    """Each player's form after each game they played."""
    p = played.sort_values(["player_id", "game_date"]).copy()
    g = p.groupby("player_id")
    out = p[["player_id", "game_date", "season", "team"]].copy()
    out = out.rename(columns={"season": "state_season", "team": "last_team"})
    cols = BASE_STATS + list(TARGETS.values()) + ["usg", "min_share"]
    for c in cols:
        out[f"ewm5_{c}"] = g[c].transform(lambda s: s.ewm(halflife=5).mean())
        out[f"ewm15_{c}"] = g[c].transform(lambda s: s.ewm(halflife=15).mean())
    for c in ["min"] + list(TARGETS.values()):
        out[f"avg10_{c}"] = g[c].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out[f"std10_{c}"] = g[c].transform(lambda s: s.rolling(10, min_periods=3).std())
        out[f"max10_{c}"] = g[c].transform(lambda s: s.rolling(10, min_periods=1).max())
    min20 = g["min"].transform(lambda s: s.rolling(20, min_periods=1).sum()).clip(lower=1)
    for c in RATE_STATS:
        out[f"rate20_{c}"] = g[c].transform(lambda s: s.rolling(20, min_periods=1).sum()) / min20
    out["n_games"] = g.cumcount() + 1
    gs = p.groupby(["player_id", "season"])
    out["szn_games"] = gs.cumcount() + 1
    out["szn_min"] = gs["min"].transform(lambda s: s.expanding().mean())
    out["last_min"] = p["min"].values
    out["last_starter"] = p["starter"].values
    out["starter10"] = g["starter"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    out["last_date"] = p["game_date"].values
    return out


def prepare_box(box: pd.DataFrame) -> pd.DataFrame:
    box = box.copy()
    box["game_date"] = pd.to_datetime(box["game_date"])
    box["player_id"] = box["player_id"].astype(str)
    box = add_targets(box)
    tg = team_games(box)
    box = box.merge(tg[["game_id", "team", "usage_total", "min"]].rename(columns={"min": "team_min"}),
                    on=["game_id", "team"], how="left")
    box["usg"] = (box["fga"] + 0.44 * box["fta"] + box["tov"]) / box["usage_total"].clip(lower=1)
    box["min_share"] = box["min"] / box["team_min"].clip(lower=200) * 5
    return box


def build(box: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-game features to `rows` (player_id, team, opp, home, game_date, season).

    `box` must already be passed through prepare_box.
    """
    played = box[(box["dnp"] == 0) & (box["min"] > 0)]
    pstate = player_state(played).sort_values("game_date")
    tg = team_games(box)
    tstate = team_state(tg).sort_values("game_date")

    rows = rows.copy()
    rows["game_date"] = pd.to_datetime(rows["game_date"])
    rows["player_id"] = rows["player_id"].astype(str)
    rows["_order"] = np.arange(len(rows))
    rows = rows.sort_values("game_date")

    f = pd.merge_asof(rows, pstate, on="game_date", by="player_id", allow_exact_matches=False)
    opp_state = tstate.rename(columns={"team": "opp", **{f"ts_{c}": f"opp_{c}" for c in TEAM_STATS}})
    f = pd.merge_asof(f.sort_values("game_date"), opp_state.sort_values("game_date"),
                      on="game_date", by="opp", allow_exact_matches=False)
    own_state = tstate.rename(columns={f"ts_{c}": f"team_{c}" for c in TEAM_STATS})[["team", "game_date", "team_pace", "team_off_rtg"]]
    f = pd.merge_asof(f.sort_values("game_date"), own_state.sort_values("game_date"),
                      on="game_date", by="team", allow_exact_matches=False)

    f["rest_days"] = (f["game_date"] - pd.to_datetime(f["last_date"])).dt.days.clip(upper=10)
    f["b2b"] = (f["rest_days"] == 1).astype(int)
    new_season = f["state_season"] != f["season"]
    f.loc[new_season, "szn_games"] = 0
    f.loc[new_season, "szn_min"] = np.nan
    f["changed_team"] = (f["last_team"] != f["team"]).astype(int)
    f["season_year"] = f["season"].str[:4].astype(int)
    f["exp_pace"] = (f["team_pace"] + f["opp_pace"]) / 2
    return f.sort_values("_order").drop(columns=["_order"])


def feature_columns(f: pd.DataFrame) -> list[str]:
    prefixes = ("ewm5_", "ewm15_", "avg10_", "std10_", "max10_", "rate20_", "opp_", "team_pace", "team_off_rtg")
    cols = [c for c in f.columns if c.startswith(prefixes)]
    cols += ["n_games", "szn_games", "szn_min", "last_min", "last_starter", "starter10", "rest_days", "b2b",
             "home", "changed_team", "season_year", "exp_pace"]
    return cols


_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'`\-]", " ", s.lower())
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def training_rows(box: pd.DataFrame) -> pd.DataFrame:
    played = box[(box["dnp"] == 0) & (box["min"] > 0)]
    keep = ["game_id", "player_id", "player_name", "team", "opp", "home", "game_date", "season", "season_type",
            "min"] + list(TARGETS.values())
    rows = played[keep].copy()
    f = build(box, rows.drop(columns=["min"] + list(TARGETS.values())))
    for c in ["min"] + list(TARGETS.values()):
        f[c] = rows[c].values
    return f[f["n_games"] >= 5]
