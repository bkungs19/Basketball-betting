"""Score today's prop lines: model probability, market probability, expected value."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import features
from .config import EARLY_SEASON_GAMES, EARLY_SEASON_MODEL_WEIGHT, MARKET_LABELS, MODEL_WEIGHT
from .http import get_json
from .odds import american_to_decimal, blend, no_vig_pair

INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Legs where the model and the market disagree by more than this usually mean
# news the model can't see (a late scratch, a minutes limit), so they aren't picked.
MAX_DISAGREEMENT = 0.20


def injuries() -> dict[str, str]:
    """Player name (normalized) -> status, e.g. 'Out', 'Doubtful', 'Questionable', 'Day-To-Day'."""
    try:
        data = get_json(INJURIES)
    except Exception as e:
        print(f"injury report unavailable: {e!r}")
        return {}
    out = {}
    for team in data.get("injuries", []):
        for inj in team.get("injuries", []):
            name = inj.get("athlete", {}).get("displayName")
            if name:
                out[features.norm_name(name)] = inj.get("status", "")
    return out


def team_map(box: pd.DataFrame) -> dict[str, str]:
    """Team nickname ('Trail Blazers') -> ESPN abbreviation ('POR')."""
    latest = box.sort_values("game_date").drop_duplicates("team", keep="last")
    return dict(zip(latest["team_name"], latest["team"]))


def _abbr_for_nick(nick: str, names: dict[str, str]) -> str | None:
    for full, abbr in names.items():
        if isinstance(nick, str) and nick and full.endswith(nick):
            return abbr
    return None


def player_map(box: pd.DataFrame) -> dict[str, str]:
    latest = box.sort_values("game_date").drop_duplicates("player_id", keep="last")
    return {features.norm_name(n): pid for n, pid in zip(latest["player_name"], latest["player_id"])}


def score(day: dt.date, season: str, box: pd.DataFrame, model, lines: pd.DataFrame, tuning: dict | None = None) -> pd.DataFrame:
    """Every available line with its blended chance and expected value; `tuning` comes from tune.fit."""
    lines = lines[lines["available"].astype(bool)].copy()
    lines["deeplink"] = lines["deeplink"].fillna("")
    if lines.empty:
        return pd.DataFrame()

    names = team_map(box)
    pmap = player_map(box)
    lines["team"] = lines["team_nick"].map(lambda n: _abbr_for_nick(n, names))
    lines["opp_abbr"] = lines["opp_nick"].map(lambda n: _abbr_for_nick(n, names))
    lines["player_id"] = lines["player_name"].map(lambda n: pmap.get(features.norm_name(n)))
    unmatched = lines[lines["player_id"].isna()]["player_name"].dropna().unique()  # saved lines can have blank names
    if len(unmatched):
        print(f"no stats history for {len(unmatched)} players: {', '.join(sorted(unmatched)[:15])}")
    lines = lines.dropna(subset=["player_id", "team", "opp_abbr"])

    players = lines.drop_duplicates("player_id")[["player_id", "player_name", "team", "opp_abbr", "player_team", "home"]].copy()
    players["home"] = (players["player_team"] == players["home"]).astype(int)  # both are SGO codes
    rows = players.rename(columns={"opp_abbr": "opp"})[["player_id", "player_name", "team", "opp", "home"]]
    rows["game_date"] = pd.Timestamp(day)
    rows["season"] = season
    feats = features.build(box, rows).set_index("player_id")
    feats = feats[feats["n_games"].fillna(0) >= 5]

    lines = lines[lines["player_id"].isin(feats.index)].copy()
    if lines.empty:
        return pd.DataFrame()

    # Market view: devig each book's two-way price, then average books quoting the same line
    fair = [no_vig_pair(o, u)[0] for o, u in zip(lines["over_odds"], lines["under_odds"])]
    lines["book_fair_over"] = fair
    lines["market_p_over"] = lines.groupby(["player_id", "stat", "line"])["book_fair_over"].transform("mean")
    lines["n_books"] = lines.groupby(["player_id", "stat", "line"])["book"].transform("count")

    out = []
    for market, grp in lines.groupby("stat"):
        f = feats.loc[grp["player_id"]].reset_index()
        pred = model.predict(f, market, grp["line"].values)
        g = grp.reset_index(drop=True).join(pred.reset_index(drop=True))
        g["games_played"] = f["n_games"].values
        g["season_games"] = f["szn_games"].values
        g["recent_avg"] = f[f"ewm5_{features.TARGETS[market]}"].values
        out.append(g)
    props = pd.concat(out, ignore_index=True)

    # Per prop type, tune.py learns how far the model runs high or low and how much to trust it
    tuned = (tuning or {}).get("markets", {})
    shift = props["stat"].map(lambda s: tuned.get(s, {}).get("shift", 0.0)).values
    weight = props["stat"].map(lambda s: tuned.get(s, {}).get("weight", MODEL_WEIGHT)).values
    props["p_model_raw"] = props["p_over"]
    z = np.log(np.clip(props["p_model_raw"], 1e-4, 1 - 1e-4) / np.clip(1 - props["p_model_raw"], 1e-4, 1))
    props["p_model_over"] = 1 / (1 + np.exp(-(z + shift)))
    # Early in the season the model leans on last season's form, so trust the market more
    weight = np.where(props["season_games"] < EARLY_SEASON_GAMES, np.minimum(weight, EARLY_SEASON_MODEL_WEIGHT), weight)
    props["p_over"] = [blend(pm, pk, w) for pm, pk, w in zip(props["p_model_over"], props["market_p_over"], weight)]
    dec_over = props["over_odds"].map(american_to_decimal)
    dec_under = props["under_odds"].map(american_to_decimal)
    ev_over = props["p_over"] * dec_over - 1
    ev_under = (1 - props["p_over"]) * dec_under - 1
    over = ev_over >= ev_under
    props["side"] = np.where(over, "Over", "Under")
    props["odds"] = np.where(over, props["over_odds"], props["under_odds"])
    props["decimal"] = np.where(over, dec_over, dec_under)
    props["p_hit"] = np.where(over, props["p_over"], 1 - props["p_over"])
    props["p_model_hit"] = np.where(over, props["p_model_over"], 1 - props["p_model_over"])
    props["p_market_hit"] = np.where(over, props["market_p_over"], 1 - props["market_p_over"])
    props["ev"] = np.where(over, ev_over, ev_under)
    props["market"] = props["stat"].map(MARKET_LABELS)

    inj = injuries()
    props["injury"] = props["player_name"].map(lambda n: inj.get(features.norm_name(n), ""))
    props["disagree"] = (props["p_model_hit"] - props["p_market_hit"]).abs() > MAX_DISAGREEMENT
    props["eligible"] = (
        ~props["injury"].str.lower().isin(["out", "doubtful", "questionable"])
        & ~props["disagree"]
        & (pd.to_datetime(props["start"], utc=True) > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=10))
    )
    props = props[~props["injury"].str.lower().isin(["out"])]
    return props.sort_values("ev", ascending=False).reset_index(drop=True)

