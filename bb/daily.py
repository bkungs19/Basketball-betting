"""The daily run: update stats, grade yesterday, fetch lines, train, pick, publish."""
from __future__ import annotations

import datetime as dt
import json
import math

import pandas as pd

from . import features, grade, lines, parlay, predict, site, stats
from .config import DATA, ET, HISTORY_SEASONS
from .model import PropModel

TOP_PROPS = 25


def _locked(path) -> bool:
    """A parlay is locked once any of its games has started."""
    if not path.exists():
        return False
    old = json.loads(path.read_text())
    legs = (old.get("parlay") or {}).get("legs", [])
    now = pd.Timestamp.now(tz="UTC")
    return any(pd.Timestamp(l["start"]) <= now for l in legs)


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


def run(day: dt.date | None = None, offline: bool = False):
    day = day or dt.datetime.now(ET).date()
    season = stats.current_season(day)
    seasons = HISTORY_SEASONS + ([season] if season not in HISTORY_SEASONS else [])

    if not offline:
        stats.update_season(season)
    box = features.prepare_box(stats.load_all(seasons))
    grade.grade_all(box, day.isoformat())

    picks_path = grade.PICKS_DIR / f"{day.isoformat()}.json"
    games = [] if offline else [g for g in stats.todays_games(day) if g["season_type"] in stats.SEASON_TYPES]
    upcoming = [g for g in games if pd.Timestamp(g["start"]) > pd.Timestamp.now(tz="UTC")]
    if offline:
        todays = lines.load_lines(day)
    elif not games:
        todays = lines.load_lines(day).iloc[0:0]  # no regular-season or playoff games: save the odds allowance
    elif not upcoming or _locked(picks_path):
        todays = None  # nothing left to pick: keep the earlier picks and save the odds allowance
    else:
        todays = lines.update_lines(day)

    if todays is None:
        print(f"{day}: games have started; keeping the earlier picks")
    elif len(todays) and not _locked(picks_path):
        f = features.training_rows(box)
        model = PropModel().fit(f)
        model.save_report()
        props = predict.score(day, season, box, model, todays)
        parlays = parlay.build(props)
        best = (props[props["eligible"]].sort_values("ev", ascending=False)
                .drop_duplicates(["player_id", "stat"]).head(TOP_PROPS)) if len(props) else props
        picks = {
            "date": day.isoformat(),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "games": int(todays["event_id"].nunique()),
            "props_scored": int(len(props)),
            "parlay": parlays[0] if parlays else None,
            "other_parlays": parlays[1:4],
            "top_props": _records(best[[c for c in parlay.LEG_FIELDS if c in best.columns]]) if len(best) else [],
            "graded": False,
        }
        grade.PICKS_DIR.mkdir(parents=True, exist_ok=True)
        picks_path.write_text(json.dumps(clean(picks), indent=1, default=str, allow_nan=False))
        print(f"{day}: {picks['games']} games, {len(props)} props scored, "
              f"parlay EV {picks['parlay']['ev'] if picks['parlay'] else 'n/a'}")
    elif not len(todays):
        print(f"{day}: no NBA games with lines today")
    else:
        print(f"{day}: parlay locked (a game has started); keeping the earlier picks")

    site.build(day)
