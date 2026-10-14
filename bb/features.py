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
SEASON_STATS = ["pts", "reb", "ast", "fg3m", "fga", "fg3a"]
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
    tg = tg.sort_values(["team", "game_date"]).reset_index(drop=True)
    ewm = tg.groupby("team", sort=False)[TEAM_STATS].ewm(halflife=10).mean().reset_index(level=0, drop=True).sort_index()
    return pd.concat([tg[["team", "game_date"]], ewm.add_prefix("ts_")], axis=1)


def player_state(played: pd.DataFrame) -> pd.DataFrame:
    """Each player's form after each game they played."""
    p = played.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    g = p.groupby("player_id", sort=False)
    out = p[["player_id", "game_date", "season", "team"]].rename(columns={"season": "state_season", "team": "last_team"})

    def by_player(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.reset_index(level=0, drop=True).sort_index()

    cols = BASE_STATS + list(TARGETS.values()) + ["usg", "min_share"]
    parts = [out]
    for hl in (5, 15):
        parts.append(by_player(g[cols].ewm(halflife=hl).mean()).add_prefix(f"ewm{hl}_"))
    roll_cols = ["min"] + list(TARGETS.values())
    parts.append(by_player(g[roll_cols].rolling(10, min_periods=1).mean()).add_prefix("avg10_"))
    parts.append(by_player(g[roll_cols].rolling(10, min_periods=3).std()).add_prefix("std10_"))
    parts.append(by_player(g[roll_cols].rolling(10, min_periods=1).max()).add_prefix("max10_"))
    sums = by_player(g[RATE_STATS + ["min"]].rolling(20, min_periods=1).sum())
    parts.append(sums[RATE_STATS].div(sums["min"].clip(lower=1), axis=0).add_prefix("rate20_"))
    out = pd.concat(parts, axis=1)

    out["n_games"] = g.cumcount() + 1
    gs = p.groupby(["player_id", "season"], sort=False)
    out["szn_games"] = gs.cumcount() + 1
    out["szn_min"] = gs["min"].cumsum() / out["szn_games"]
    for c in SEASON_STATS:
        out[f"szn_{c}"] = gs[c].cumsum() / out["szn_games"]
    out["last_min"] = p["min"]
    out["last_starter"] = p["starter"]
    out["starter10"] = by_player(g["starter"].rolling(10, min_periods=1).mean())
    out["last_date"] = p["game_date"]
    return out


def prepare_box(box: pd.DataFrame) -> pd.DataFrame:
    box = box.copy()
    box["game_date"] = pd.to_datetime(box["game_date"]).astype("datetime64[ns]")
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
    rows["game_date"] = pd.to_datetime(rows["game_date"]).astype("datetime64[ns]")
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

    days_off = (f["game_date"] - pd.to_datetime(f["last_date"])).dt.days
    f["rest_days"] = days_off.clip(upper=10)
    f["days_off"] = days_off.clip(upper=200)
    f["team_games_missed"] = _games_missed(tg, f)
    f["b2b"] = (f["rest_days"] == 1).astype(int)
    new_season = f["state_season"] != f["season"]
    f.loc[new_season, "szn_games"] = 0
    f.loc[new_season, ["szn_min"] + [f"szn_{c}" for c in SEASON_STATS]] = np.nan
    f = f.merge(_prev_season(played), on=["player_id", "season"], how="left")
    f["changed_team"] = (f["last_team"] != f["team"]).astype(int)
    f["season_year"] = f["season"].str[:4].astype(int)
    f["exp_pace"] = (f["team_pace"] + f["opp_pace"]) / 2
    return f.sort_values("_order").drop(columns=["_order"])


def _prev_season(played: pd.DataFrame) -> pd.DataFrame:
    """Each player's previous regular-season averages, keyed by the season they feed into."""
    reg = played[played["season_type"] == "regular"]
    prev = reg.groupby(["player_id", "season"])[["min"] + SEASON_STATS].mean().add_prefix("prev_").reset_index()
    prev["prev_games"] = reg.groupby(["player_id", "season"]).size().values
    start = prev["season"].str[:4].astype(int) + 1
    prev["season"] = start.astype(str) + "-" + (start + 1).astype(str).str[2:]
    return prev


def _games_missed(tg: pd.DataFrame, f: pd.DataFrame) -> np.ndarray:
    """Games the player's current team played between his last game and this one (injury absences)."""
    sched = tg[["team", "game_date"]].drop_duplicates().sort_values(["team", "game_date"])
    sched["n"] = sched.groupby("team").cumcount() + 1
    sched = sched.sort_values("game_date")

    def count_before(dates: pd.Series) -> np.ndarray:
        q = pd.DataFrame({"team": f["team"].values, "game_date": pd.to_datetime(dates).astype("datetime64[ns]").values,
                          "_i": np.arange(len(f))}).sort_values("game_date")
        m = pd.merge_asof(q, sched, on="game_date", by="team", allow_exact_matches=False)
        return m.sort_values("_i")["n"].fillna(0).values

    missed = count_before(f["game_date"]) - count_before(f["last_date"].fillna(f["game_date"])) - 1
    # last_date lookups are strict, so a game on last_date itself isn't counted; it was his own game
    return np.clip(missed, 0, 82)


def feature_columns(f: pd.DataFrame) -> list[str]:
    prefixes = ("ewm5_", "ewm15_", "avg10_", "std10_", "max10_", "rate20_", "opp_", "team_pace", "team_off_rtg",
                "szn_", "prev_")
    cols = [c for c in f.columns if c.startswith(prefixes)]
    cols += ["days_off", "team_games_missed", "n_games", "last_min", "last_starter", "starter10", "rest_days", "b2b",
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
