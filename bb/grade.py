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


def _margin(box_day: pd.DataFrame, team: str, opp: str):
    pts = box_day.groupby("team")["pts"].sum()
    if team not in pts or opp not in pts:
        return None
    return float(pts[team] - pts[opp])


def grade_game_leg(leg: dict, box_day: pd.DataFrame) -> dict:
    """Moneyline wins if the team won; a spread wins if margin + line > 0 (equal is a push)."""
    margin = _margin(box_day, leg["team"], leg["opp_abbr"])
    leg = dict(leg)
    leg["actual"] = margin
    if margin is None:
        leg["result"] = "void"
    elif margin + float(leg["line"] or 0) == 0:
        leg["result"] = "push"
    else:
        leg["result"] = "win" if margin + float(leg["line"] or 0) > 0 else "loss"
    return leg


def grade_leg(leg: dict, box_day: pd.DataFrame) -> dict:
    if leg.get("kind") == "game":
        return grade_game_leg(leg, box_day)
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
    legs = picks.get("top_props", []) + picks.get("game_picks", [])
    if any(l.get("team") not in game_teams for l in legs):
        return False  # some games aren't final yet
    picks["top_props"] = [grade_leg(l, box_day) for l in picks.get("top_props", [])]
    picks["game_picks"] = [grade_leg(l, box_day) for l in picks.get("game_picks", [])]
    picks["graded"] = True
    path.write_text(json.dumps(picks, indent=1, default=str))
    results = [l["result"] for l in picks["top_props"] + picks["game_picks"]]
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
