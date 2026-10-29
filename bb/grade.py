"""Grade past picks against the final box scores."""
from __future__ import annotations

import json

import pandas as pd

from .config import DATA, MARKETS

PICKS_DIR = DATA / "picks"


def _actual(box_day: pd.DataFrame, player_id: str, stat: str):
    row = box_day[box_day["player_id"].astype(str) == str(player_id)]
    if row.empty or int(row.iloc[0]["dnp"]) == 1 or float(row.iloc[0]["min"]) == 0:
        return None
    return float(row.iloc[0][MARKETS[stat]].sum())


def grade_leg(leg: dict, box_day: pd.DataFrame) -> dict:
    actual = _actual(box_day, leg["player_id"], leg["stat"])
    leg = dict(leg)
    leg["actual"] = actual
    if actual is None:
        leg["result"] = "void"
    elif actual == leg["line"]:
        leg["result"] = "push"
    else:
        over = actual > leg["line"]
        leg["result"] = "win" if over == (leg["side"] == "Over") else "loss"
    return leg


def grade_day(path, box: pd.DataFrame) -> bool:
    picks = json.loads(path.read_text())
    if picks.get("graded"):
        return False
    box_day = box[box["game_date"].astype(str).str[:10] == picks["date"]]
    if box_day.empty:
        return False
    game_teams = set(box_day["team"])
    legs = picks.get("top_props", [])
    if any(l.get("team") not in game_teams for l in legs):
        return False  # some games aren't final yet
    picks["top_props"] = [grade_leg(l, box_day) for l in legs]
    picks["graded"] = True
    path.write_text(json.dumps(picks, indent=1, default=str))
    results = [l["result"] for l in picks["top_props"]]
    print(f"graded {picks['date']}: {results.count('win')}-{results.count('loss')}")
    return True


def grade_all(box: pd.DataFrame, today: str):
    if not PICKS_DIR.exists():
        return
    for path in sorted(PICKS_DIR.glob("*.json")):
        if path.stem < today:
            grade_day(path, box)


def history() -> list[dict]:
    out = []
    if not PICKS_DIR.exists():
        return out
    for path in sorted(PICKS_DIR.glob("*.json")):
        out.append(json.loads(path.read_text()))
    return out
