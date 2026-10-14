"""Player box scores from ESPN's public site API.

stats.nba.com and cdn.nba.com block GitHub's runners, so box scores come from
ESPN and the advanced numbers (usage, pace, opponent defense) are computed
from them in features.py.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .config import DATA, ET
from .http import get_json

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

BOX_DIR = DATA / "boxscores"

COLUMNS = [
    "season", "season_type", "game_id", "game_date", "team", "team_name", "opp", "opp_name", "home",
    "player_id", "player_name", "starter", "dnp", "min", "pts", "reb", "oreb", "dreb", "ast", "stl",
    "blk", "tov", "pf", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "plus_minus",
]

# ESPN labels and keys -> our columns. "a-b" entries hold made-attempted pairs.
_STAT_ALIASES = {
    "MIN": "min", "minutes": "min",
    "FG": "fgm-fga", "fieldGoalsMade-fieldGoalsAttempted": "fgm-fga",
    "3PT": "fg3m-fg3a", "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "fg3m-fg3a",
    "FT": "ftm-fta", "freeThrowsMade-freeThrowsAttempted": "ftm-fta",
    "OREB": "oreb", "offensiveRebounds": "oreb",
    "DREB": "dreb", "defensiveRebounds": "dreb",
    "REB": "reb", "rebounds": "reb",
    "AST": "ast", "assists": "ast",
    "STL": "stl", "steals": "stl",
    "BLK": "blk", "blocks": "blk",
    "TO": "tov", "turnovers": "tov",
    "PF": "pf", "fouls": "pf",
    "+/-": "plus_minus", "plusMinus": "plus_minus",
    "PTS": "pts", "points": "pts",
}

SEASON_TYPES = {2: "regular", 3: "playoffs", 5: "play-in"}


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[2:]}"


def season_end_year(label: str) -> int:
    return int(label[:4]) + 1


def current_season(today: dt.date | None = None) -> str:
    today = today or dt.datetime.now(ET).date()
    end_year = today.year + 1 if today.month >= 8 else today.year
    return season_label(end_year)


def _num(x) -> float:
    try:
        return float(str(x).replace("+", ""))
    except ValueError:
        return 0.0


def scoreboard(day: dt.date) -> list[dict]:
    data = get_json(SCOREBOARD, params={"dates": day.strftime("%Y%m%d"), "limit": 100})
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c["team"] for c in comp["competitors"]}
        if any(int(t["id"]) > 30 for t in teams.values()):  # All-Star and exhibition teams
            continue
        start = dt.datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(ET)
        games.append({
            "game_id": ev["id"],
            "game_date": start.date().isoformat(),
            "start": start.isoformat(),
            "season_year": ev.get("season", {}).get("year"),
            "season_type": ev.get("season", {}).get("type"),
            "completed": bool(ev["status"]["type"].get("completed")),
            "home": teams["home"]["abbreviation"],
            "home_name": teams["home"]["displayName"],
            "away": teams["away"]["abbreviation"],
            "away_name": teams["away"]["displayName"],
        })
    return games


def boxscore(game: dict) -> list[dict]:
    data = get_json(SUMMARY, params={"event": game["game_id"]})
    rows = []
    for team_block in data.get("boxscore", {}).get("players", []):
        abbr = team_block["team"]["abbreviation"]
        home = abbr == game["home"]
        opp = game["away"] if home else game["home"]
        for block in team_block.get("statistics", []):
            names = block.get("keys") or block.get("labels") or []
            cols = [_STAT_ALIASES.get(n) for n in names]
            for a in block.get("athletes", []):
                if not a.get("athlete", {}).get("id"):
                    continue
                row = {
                    "season": season_label(game["season_year"]),
                    "season_type": SEASON_TYPES.get(game["season_type"], str(game["season_type"])),
                    "game_id": game["game_id"],
                    "game_date": game["game_date"],
                    "team": abbr,
                    "team_name": game["home_name"] if home else game["away_name"],
                    "opp": opp,
                    "opp_name": game["away_name"] if home else game["home_name"],
                    "home": int(home),
                    "player_id": a["athlete"]["id"],
                    "player_name": a["athlete"]["displayName"],
                    "starter": int(bool(a.get("starter"))),
                    "dnp": int(bool(a.get("didNotPlay")) or not a.get("stats")),
                }
                for c in COLUMNS[13:]:
                    row[c] = 0.0
                for col, val in zip(cols, a.get("stats") or []):
                    if col is None:
                        continue
                    if "-" in col:
                        made, att = col.split("-")
                        parts = str(val).split("-")
                        if len(parts) == 2:
                            row[made], row[att] = _num(parts[0]), _num(parts[1])
                    else:
                        row[col] = _num(val)
                rows.append(row)
    return rows


def season_dates(label: str, through: dt.date) -> list[dt.date]:
    end_year = season_end_year(label)
    start, end = dt.date(end_year - 1, 10, 1), min(dt.date(end_year, 6, 30), through)
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def season_path(label: str):
    return BOX_DIR / f"{label}.csv"


def load_season(label: str) -> pd.DataFrame:
    p = season_path(label)
    if p.exists():
        return pd.read_csv(p, dtype={"game_id": str, "player_id": str})
    return pd.DataFrame(columns=COLUMNS)


def update_season(label: str, through: dt.date | None = None, workers: int = 8) -> pd.DataFrame:
    """Fetch any completed games missing from data/boxscores/<season>.csv."""
    through = through or (dt.datetime.now(ET).date() - dt.timedelta(days=1))
    existing = load_season(label)
    have = set(existing["game_id"].astype(str))
    dates = season_dates(label, through)
    if have:
        last = dt.date.fromisoformat(existing["game_date"].max())
        dates = [d for d in dates if d >= last - dt.timedelta(days=3)]
    if not dates:
        return existing

    with ThreadPoolExecutor(workers) as pool:
        boards = list(pool.map(scoreboard, dates))
    games = [
        g for day in boards for g in day
        if g["completed"] and g["game_id"] not in have and g["season_type"] in SEASON_TYPES
    ]
    if not games:
        return existing
    def safe_box(game):
        try:
            return boxscore(game)
        except Exception as e:  # one bad game shouldn't sink the whole update
            print(f"skipped game {game['game_id']} on {game['game_date']}: {e!r}")
            return []

    with ThreadPoolExecutor(workers) as pool:
        new_rows = [r for rows in pool.map(safe_box, games) for r in rows]
    new = pd.DataFrame(new_rows, columns=COLUMNS)
    out = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    out = out.drop_duplicates(["game_id", "player_id"], keep="last")
    out = out.sort_values(["game_date", "game_id", "team", "starter", "min"], ascending=[True, True, True, False, False])
    BOX_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(season_path(label), index=False)
    print(f"{label}: +{len(games)} games, {len(out)} player rows")
    return out


def todays_games(day: dt.date | None = None) -> list[dict]:
    return scoreboard(day or dt.datetime.now(ET).date())


def load_all(seasons: list[str]) -> pd.DataFrame:
    frames = [load_season(s) for s in seasons]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df
