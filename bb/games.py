"""Moneyline and spread picks: a game-margin model blended with the market.

The model predicts the home team's winning margin from each team's recent and
season-long net rating, pace, rest, and how much of its usual production is
missing (regulars who sat out, or are listed Out or Doubtful today). Margins
are treated as normal with the spread of past errors, which gives the chance
to win and to cover any spread. Game lines are the sharpest markets, so the
market's own odds get most of the weight.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features, predict
from .config import DATA
from .odds import american_to_decimal, blend, no_vig_pair

GAME_MODEL_WEIGHT = 0.3
MAX_DISAGREEMENT = 0.12  # bigger gaps usually mean news the model can't see
REGULAR_MIN = 15  # a regular averages at least this many minutes
REGULAR_GAP_DAYS = 20  # ...and has played within this many days of his team's last game
STAR_MIN = 28  # a Questionable player this big makes the game too uncertain to bet
HOLDOUT_MIN_GAMES = 600  # a full season has about 1,300 games
STATE = ["net", "off_rtg", "def_rtg", "pace"]
REPORT = DATA / "game_model_report.json"


def _next_season(s: str) -> str:
    y = int(s[:4]) + 1
    return f"{y}-{str(y + 1)[2:]}"


def team_table(box: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: ratings, home flag and the final margin."""
    tg = features.team_games(box)
    home = box.groupby(["game_id", "team"], as_index=False)["home"].max()
    tg = tg.merge(home, on=["game_id", "team"])
    tg["net"] = tg["off_rtg"] - tg["def_rtg"]
    tg["margin"] = tg["pts"] - tg["o_pts"]
    return tg.sort_values(["team", "game_date"]).reset_index(drop=True)


def team_state(tg: pd.DataFrame) -> pd.DataFrame:
    """Each team's form after each game."""
    st = tg[["team", "game_date", "season"]].rename(columns={"season": "state_season"}).copy()
    g = tg.groupby("team", sort=False)
    for hl in (10, 40):
        e = g[STATE].ewm(halflife=hl).mean().reset_index(level=0, drop=True).sort_index()
        st[[f"e{hl}_{c}" for c in STATE]] = e.values
    gs = tg.groupby(["team", "season"], sort=False)
    st["szn_games"] = gs.cumcount() + 1
    st["szn_net"] = gs["net"].cumsum() / st["szn_games"]
    st["last_game"] = tg["game_date"]
    return st


def _prev_net(tg: pd.DataFrame) -> pd.DataFrame:
    prev = tg.groupby(["team", "season"], as_index=False)["net"].mean().rename(columns={"net": "prev_net"})
    prev["season"] = prev["season"].map(_next_season)
    return prev


def _player_form(box: pd.DataFrame) -> pd.DataFrame:
    played = box[(box["dnp"] == 0) & (box["min"] > 0)]
    ps = features.player_state(played)
    ps["prod"] = ps["ewm15_pts"] + ps["ewm15_reb"] + ps["ewm15_ast"]
    return ps[["player_id", "game_date", "last_team", "last_date", "ewm15_min", "prod"]].sort_values("game_date")


def missing_history(box: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """For each past team-game: usual minutes and production of regulars who didn't play."""
    played = box[(box["dnp"] == 0) & (box["min"] > 0)]
    tgames = box[["game_id", "game_date", "season", "team"]].drop_duplicates()
    roster = played[["team", "season", "player_id"]].drop_duplicates()
    pairs = tgames.merge(roster, on=["team", "season"]).sort_values("game_date")
    pairs = pd.merge_asof(pairs, form, on="game_date", by="player_id", allow_exact_matches=False)
    recent = (pairs["game_date"] - pairs["last_date"]).dt.days <= REGULAR_GAP_DAYS
    reg = pairs[(pairs["last_team"] == pairs["team"]) & (pairs["ewm15_min"] >= REGULAR_MIN) & recent].copy()
    played_keys = set(zip(played["game_id"], played["player_id"]))
    reg["absent"] = [(g, p) not in played_keys for g, p in zip(reg["game_id"], reg["player_id"])]
    return _missing(reg)


def _missing(reg: pd.DataFrame) -> pd.DataFrame:
    reg = reg.assign(miss_min=reg["ewm15_min"] * reg["absent"], miss_prod=reg["prod"] * reg["absent"])
    out = reg.groupby(["game_id", "team"], as_index=False)[["miss_min", "miss_prod", "prod"]].sum()
    out["miss_share"] = out["miss_prod"] / out["prod"].clip(lower=1)
    return out.drop(columns=["prod"])


def missing_today(box: pd.DataFrame, form: pd.DataFrame, tg: pd.DataFrame, games: pd.DataFrame,
                  injuries: dict[str, str]) -> tuple[pd.DataFrame, set]:
    """Usual production of today's regulars listed Out or Doubtful, and teams with a big name Questionable."""
    latest = form.sort_values("game_date").drop_duplicates("player_id", keep="last")
    last_game = tg.groupby("team")["game_date"].max()
    pmap = predict.player_map(box)
    status = {pmap[n]: s.lower() for n, s in injuries.items() if n in pmap}
    rows, shaky = [], set()
    for _, g in games.iterrows():
        for team in (g["home_abbr"], g["away_abbr"]):
            cutoff = last_game.get(team, g["game_date"]) - pd.Timedelta(days=REGULAR_GAP_DAYS)
            reg = latest[(latest["last_team"] == team) & (latest["ewm15_min"] >= REGULAR_MIN) & (latest["last_date"] >= cutoff)].copy()
            st = reg["player_id"].map(status).fillna("")
            reg["absent"] = st.isin(["out", "doubtful"])
            if ((st == "questionable") & (reg["ewm15_min"] >= STAR_MIN)).any():
                shaky.add(team)
            reg["game_id"], reg["team"] = g["game_id"], team
            rows.append(reg)
    if not rows:
        return pd.DataFrame(columns=["game_id", "team", "miss_min", "miss_prod", "miss_share"]), shaky
    return _missing(pd.concat(rows, ignore_index=True)), shaky


def game_features(games: pd.DataFrame, tg: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    """Pre-game features for `games` (game_id, game_date, season, home_abbr, away_abbr)."""
    st = team_state(tg).sort_values("game_date")
    prev = _prev_net(tg)
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"]).astype("datetime64[ns]")
    g["_order"] = np.arange(len(g))
    for side in ("home", "away"):
        s = st.rename(columns={"team": f"{side}_abbr"})
        g = pd.merge_asof(g.sort_values("game_date"), s, on="game_date", by=f"{side}_abbr", allow_exact_matches=False)
        new = g["state_season"] != g["season"]
        g.loc[new, "szn_games"] = 0
        g.loc[new, "szn_net"] = np.nan
        g["rest"] = (g["game_date"] - g["last_game"]).dt.days.clip(upper=10)
        g["b2b"] = (g["rest"] == 1).astype(int)
        g = g.merge(prev.rename(columns={"team": f"{side}_abbr"}), on=[f"{side}_abbr", "season"], how="left")
        g = g.merge(missing.rename(columns={"team": f"{side}_abbr"}), on=["game_id", f"{side}_abbr"], how="left")
        g[["miss_min", "miss_prod", "miss_share"]] = g[["miss_min", "miss_prod", "miss_share"]].fillna(0)
        cols = [f"e{hl}_{c}" for hl in (10, 40) for c in STATE] + ["szn_games", "szn_net", "rest", "b2b", "prev_net",
                                                                  "miss_min", "miss_prod", "miss_share"]
        g = g.rename(columns={c: f"{side}_{c}" for c in cols}).drop(columns=["state_season", "last_game"])
    for c in ["e10_net", "e40_net", "szn_net", "prev_net", "rest", "miss_prod", "miss_share"]:
        g[f"d_{c}"] = g[f"home_{c}"] - g[f"away_{c}"]
    return g.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def training_games(box: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tg = team_table(box)
    form = _player_form(box)
    hist = tg[tg["home"] == 1][["game_id", "game_date", "season", "team", "opp", "margin"]].rename(
        columns={"team": "home_abbr", "opp": "away_abbr"})
    g = game_features(hist, tg, missing_history(box, form))
    return g[g["home_e10_net"].notna() & g["away_e10_net"].notna()], tg, form


def feature_list(g: pd.DataFrame) -> list[str]:
    return [c for c in g.columns if c.startswith(("home_", "away_", "d_")) and c not in ("home_abbr", "away_abbr")]


def _regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


class GameModel:
    def fit(self, g: pd.DataFrame) -> "GameModel":
        self.features = feature_list(g)
        sizes = g.groupby("season").size()
        full = sorted(sizes[sizes >= HOLDOUT_MIN_GAMES].index)
        hold_season = full[-1]
        train, hold = g[g["season"] < hold_season], g[g["season"] == hold_season]
        cut = hold["game_date"].sort_values().iloc[len(hold) // 2]
        cal, test = hold[hold["game_date"] < cut], hold[hold["game_date"] >= cut]

        m = _regressor().fit(train[self.features], train["margin"])
        mu_cal, mu_te = m.predict(cal[self.features]), m.predict(test[self.features])
        # A normal margin with the spread of past errors: on the test half its
        # win chances scored better than a fitted calibration curve
        self.sd = float(np.std(cal["margin"] - mu_cal))
        p_te = norm.cdf(mu_te / self.sd)
        won = (test["margin"] > 0).astype(float).values
        base = float(train["margin"].mean())
        self.report = {
            "holdout_season": hold_season, "train_games": int(len(train)), "test_games": int(len(test)),
            "margin_sd": round(self.sd, 2),
            "mae_model": round(float(np.mean(np.abs(test["margin"] - mu_te))), 2),
            "mae_home_edge_only": round(float(np.mean(np.abs(test["margin"] - base))), 2),
            "winner_accuracy": round(float(np.mean((p_te > 0.5) == (won == 1))), 4),
            "brier_model": round(float(np.mean((p_te - won) ** 2)), 4),
            "brier_home_rate": round(float(np.mean((won.mean() - won) ** 2)), 4),
        }
        print(f"game model: margin MAE {self.report['mae_model']} vs {self.report['mae_home_edge_only']} "
              f"(home edge only), winner accuracy {self.report['winner_accuracy']:.3f}")
        self.mean = _regressor().fit(g[self.features], g["margin"])
        return self

    def save_report(self):
        REPORT.write_text(json.dumps(self.report, indent=1))

    def p_home(self, f: pd.DataFrame, bet: str, home_line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Projected home margin and the home side's chance to win (moneyline) or cover (spread)."""
        mu = self.mean.predict(f[self.features])
        return mu, norm.cdf((mu + home_line) / self.sd)  # a moneyline is a spread of 0


def score(day: dt.date, season: str, box: pd.DataFrame, model: GameModel, tg: pd.DataFrame,
          form: pd.DataFrame, glines: pd.DataFrame, injuries: dict[str, str]) -> pd.DataFrame:
    """Score today's moneylines and spreads into legs shaped like player props."""
    gl = glines[glines["available"].astype(bool)].copy()
    if gl.empty:
        return pd.DataFrame()
    names = predict.team_map(box)
    gl["home_abbr"] = gl["team_nick"].map(lambda n: predict._abbr_for_nick(n, names))
    gl["away_abbr"] = gl["opp_nick"].map(lambda n: predict._abbr_for_nick(n, names))
    gl = gl.dropna(subset=["home_abbr", "away_abbr"])
    if gl.empty:
        return pd.DataFrame()
    gl["deeplink"] = gl["deeplink"].fillna("")

    games = gl.drop_duplicates("event_id")[["event_id", "home_abbr", "away_abbr"]].rename(columns={"event_id": "game_id"})
    games["game_date"], games["season"] = pd.Timestamp(day), season
    missing, shaky = missing_today(box, form, tg, games, injuries)
    f = game_features(games, tg, missing).set_index("game_id")

    # Market view: devig each book's price, then average books quoting the same line
    gl["market_p_home"] = [no_vig_pair(h, a)[0] for h, a in zip(gl["home_odds"], gl["away_odds"])]
    gl["market_p_home"] = gl.groupby(["event_id", "bet", "home_line"])["market_p_home"].transform("mean")
    out = []
    for bet, grp in gl.groupby("bet"):
        mu, p_model = model.p_home(f.loc[grp["event_id"]], bet, grp["home_line"].values)
        grp = grp.assign(proj_home=mu, p_model_home=p_model)
        out.append(grp)
    gl = pd.concat(out, ignore_index=True)
    gl["p_home"] = [blend(pm, pk, GAME_MODEL_WEIGHT) for pm, pk in zip(gl["p_model_home"], gl["market_p_home"])]

    legs = []
    for side in ("home", "away"):
        other = "away" if side == "home" else "home"
        sign = 1 if side == "home" else -1
        p = gl["p_home"] if side == "home" else 1 - gl["p_home"]
        pm = gl["p_model_home"] if side == "home" else 1 - gl["p_model_home"]
        pk = gl["market_p_home"] if side == "home" else 1 - gl["market_p_home"]
        dec = gl[f"{side}_odds"].map(american_to_decimal)
        line = np.where(gl["bet"] == "spread", sign * gl["home_line"], 0.0)
        nick = gl["team_nick"] if side == "home" else gl["opp_nick"]
        label = np.where(gl["bet"] == "spread", [f"{n} {l:+g}" for n, l in zip(nick, line)], nick + " to win")
        legs.append(pd.DataFrame({
            "kind": "game", "label": label, "player_name": nick, "player_id": "team:" + gl[f"{side}_abbr"],
            "team": gl[f"{side}_abbr"], "opp_abbr": gl[f"{other}_abbr"], "event_id": gl["event_id"],
            "start": gl["start"], "stat": gl["bet"], "market": gl["bet"].str.capitalize(), "side": "",
            "line": line, "odds": gl[f"{side}_odds"], "decimal": dec, "p_hit": p, "p_model_hit": pm,
            "p_market_hit": pk, "ev": p * dec - 1, "proj": sign * gl["proj_home"], "recent_avg": np.nan,
            "book": gl["book"], "deeplink": gl["deeplink"], "injury": "",
            "shaky": gl[f"{side}_abbr"].isin(shaky) | gl[f"{other}_abbr"].isin(shaky),
        }))
    legs = pd.concat(legs, ignore_index=True)
    legs["disagree"] = (legs["p_model_hit"] - legs["p_market_hit"]).abs() > MAX_DISAGREEMENT
    legs["eligible"] = (~legs["disagree"] & ~legs["shaky"]
                        & (pd.to_datetime(legs["start"], utc=True) > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=10)))
    return legs.sort_values("ev", ascending=False).reset_index(drop=True)
