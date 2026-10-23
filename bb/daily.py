"""The daily run: update stats, grade yesterday, fetch lines, train, pick, publish."""
from __future__ import annotations

import datetime as dt
import json
import math
import time

import pandas as pd

from . import features, games as game_lines, grade, lines, predict, pricing, site, stats
from .config import DATA, ET, HISTORY_SEASONS, MIN_EV
from .model import PropModel

TOP_PROPS = 25  # one per player, best expected value first; Brandon bets the top 5 or 10 as singles
TOP_GAME_PICKS = 10
LINES_LEAD = dt.timedelta(hours=10)  # props post the morning of; fetching earlier wastes the odds allowance
PREGAME_LEAD = dt.timedelta(hours=1)
LONGEST_WAIT = dt.timedelta(hours=5, minutes=30)  # GitHub ends a job after 6 hours


def _locked(path) -> bool:
    """Picks are locked once the pregame picks are made or any picked game has started."""
    if not path.exists():
        return False
    old = json.loads(path.read_text())
    if old.get("final"):
        return True
    legs = old.get("top_props", []) + old.get("game_picks", [])
    now = pd.Timestamp.now(tz="UTC")
    return any(pd.Timestamp(l["start"]) <= now for l in legs)


def espn_tip_times(todays: pd.DataFrame, games: list[dict]) -> pd.DataFrame:
    """Use ESPN's tip times; the odds feed's start time can be hours off."""
    def tip(nick, opp):
        for g in games:
            names = (g["home_name"], g["away_name"])
            if any(n.endswith(str(nick)) for n in names) and any(n.endswith(str(opp)) for n in names):
                return g["start"]
        return None
    events = todays.drop_duplicates("event_id")
    tips = {e: tip(n, o) for e, n, o in zip(events["event_id"], events["team_nick"], events["opp_nick"])}
    espn = todays["event_id"].map(tips)
    return todays.assign(start=espn.where(espn.notna(), todays["start"]))


def pregame_wait(lead=PREGAME_LEAD, longest=LONGEST_WAIT) -> bool:
    """Sleep until an hour before today's first game, when that's within reach.

    GitHub starts scheduled runs hours late, so instead of trusting the clock,
    the run that lands within a few hours of the first game waits here and then
    refreshes the picks with the latest lines and injury news. Returns whether
    that refresh is due.
    """
    now = dt.datetime.now(dt.timezone.utc)
    games = [g for g in stats.todays_games(now.astimezone(ET).date()) if g["season_type"] in stats.SEASON_TYPES]
    tips = [t for t in (dt.datetime.fromisoformat(g["start"]) for g in games) if t > now]
    if not tips:
        print("no games left today")
        return False
    first = min(tips)
    target = first - lead
    if not now < target <= now + longest:
        print(f"first game at {first.astimezone(ET):%-I:%M %p} ET; no pregame refresh from this run")
        return False
    print(f"first game at {first.astimezone(ET):%-I:%M %p} ET; refreshing at {target.astimezone(ET):%-I:%M %p} ET")
    time.sleep((target - now).total_seconds())
    return True


def game_legs(day: dt.date, season: str, box: pd.DataFrame, games: list[dict]) -> pd.DataFrame:
    """Today's moneyline and spread legs, or an empty frame if there are no game lines."""
    glines = lines.load_game_lines(day)
    if glines.empty:
        return pd.DataFrame()
    if games:
        glines = espn_tip_times(glines, games)
    hist, tg, form = game_lines.training_games(box)
    model = game_lines.GameModel().fit(hist)
    model.save_report()
    return game_lines.score(day, season, box, model, tg, form, glines, predict.injuries())


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def clean(x):
    """Make values JSON-safe: NaN becomes null, numpy scalars become Python numbers."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def run(day: dt.date | None = None, offline: bool = False, final: bool = False):
    day = day or dt.datetime.now(ET).date()
    season = stats.current_season(day)
    seasons = HISTORY_SEASONS + ([season] if season not in HISTORY_SEASONS else [])

    if not offline:
        stats.update_season(season)
    box = features.prepare_box(stats.load_all(seasons))
    grade.grade_all(box, day.isoformat())

    picks_path = grade.PICKS_DIR / f"{day.isoformat()}.json"
    games = [] if offline else [g for g in stats.todays_games(day) if g["season_type"] in stats.SEASON_TYPES]
    now = pd.Timestamp.now(tz="UTC")
    tips = sorted(t for t in (pd.Timestamp(g["start"]) for g in games) if t > now)
    skip = None
    if offline:
        todays = lines.load_lines(day)
    elif not games:
        todays = lines.load_lines(day).iloc[0:0]  # no regular-season or playoff games: save the odds allowance
    elif not tips or _locked(picks_path):
        skip = "the picks are final or the games have started; keeping the earlier picks"
    elif tips[0] - now > LINES_LEAD:
        skip = "the first game is more than 10 hours away; prop lines aren't up yet"
    else:
        todays = espn_tip_times(lines.update_lines(day), games)

    if skip:
        print(f"{day}: {skip}")
    elif len(todays) and not _locked(picks_path):
        f = features.training_rows(box)
        model = PropModel().fit(f)
        model.save_report()
        props = predict.score(day, season, box, model, todays)
        gl = game_legs(day, season, box, games)
        typical = pricing.consensus(props) if len(props) else props
        # one prop per player: a player's points, PRA and P+R props mostly win or lose together
        best = (typical[typical["ev"] > MIN_EV].sort_values("ev", ascending=False)
                .drop_duplicates("player_id").head(TOP_PROPS)) if len(typical) else typical
        typical_games = pricing.consensus(gl) if len(gl) else gl
        best_games = (typical_games[typical_games["ev"] > MIN_EV].sort_values("ev", ascending=False)
                      .drop_duplicates(["event_id", "stat"]).head(TOP_GAME_PICKS)) if len(typical_games) else typical_games
        picks = {
            "date": day.isoformat(),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "games": int(todays["event_id"].nunique()),
            "props_scored": int(len(props)),
            "top_props": _records(best[[c for c in pricing.LEG_FIELDS if c in best.columns]]) if len(best) else [],
            "game_picks": _records(best_games[[c for c in pricing.LEG_FIELDS if c in best_games.columns]]) if len(best_games) else [],
            "graded": False,
            "final": final,
        }
        grade.PICKS_DIR.mkdir(parents=True, exist_ok=True)
        picks_path.write_text(json.dumps(clean(picks), indent=1, default=str, allow_nan=False))
        print(f"{day}: {picks['games']} games, {len(props)} props scored, "
              f"{len(picks['top_props'])} props and {len(picks['game_picks'])} game picks over {MIN_EV:.0%} EV")
    elif not len(todays):
        print(f"{day}: no NBA games with lines today")
    else:
        print(f"{day}: picks locked (a game has started); keeping the earlier picks")

    site.build(day)
