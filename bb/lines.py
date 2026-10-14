"""Player prop lines from Sports Game Odds (free Amateur tier).

Each event returned counts as one "object" against the 2,500/month allowance,
whatever the number of markets, so we ask only for today's games.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd

from .config import DATA, ET, MARKETS
from .http import get_json

API = "https://api.sportsgameodds.com/v2"
LINES_DIR = DATA / "lines"

LINE_COLUMNS = [
    "date", "event_id", "start", "home", "away", "sgo_player_id", "player_name", "player_team", "opp",
    "stat", "book", "line", "over_odds", "under_odds", "available", "deeplink",
    "fetched_at",
]


def _key() -> str:
    key = os.environ.get("SGO_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SGO_API_KEY is not set (add it under the repo's Actions secrets)")
    return key


def fetch_events(day: dt.date) -> list[dict]:
    start = dt.datetime.combine(day, dt.time(6, 0), ET)  # 6am ET through 6am ET next day
    end = start + dt.timedelta(days=1)
    params = {
        "leagueID": "NBA",
        "startsAfter": start.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "startsBefore": end.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "limit": 50,
    }
    events, cursor = [], None
    while True:
        if cursor:
            params["cursor"] = cursor
        data = get_json(f"{API}/events", params=params, headers={"x-api-key": _key()}, timeout=60)
        events.extend(data.get("data", []))
        cursor = data.get("nextCursor")
        if not cursor or not data.get("data"):
            break
        time.sleep(7)  # 10 requests/minute on the free tier
    return events


def parse_event(ev: dict, day: dt.date, fetched_at: str) -> list[dict]:
    teams = ev.get("teams", {})
    short = {side: t.get("names", {}).get("short") or t.get("teamID") for side, t in teams.items()}
    team_by_id = {t.get("teamID"): short[side] for side, t in teams.items()}
    players = ev.get("players", {})
    odds = ev.get("odds", {})
    start = ev.get("status", {}).get("startsAt")
    rows = []
    for oid, o in odds.items():
        if (o.get("periodID") != "game" or o.get("betTypeID") != "ou" or o.get("sideID") != "over"
                or o.get("statID") not in MARKETS):
            continue
        pid = o.get("playerID") or o.get("statEntityID")
        if pid not in players:
            continue
        under = odds.get(o.get("opposingOddID") or oid.replace("-over", "-under"), {})
        p = players[pid]
        team = team_by_id.get(p.get("teamID"))
        opp = short["away"] if team == short.get("home") else short.get("home")
        for book, b in (o.get("byBookmaker") or {}).items():
            u = (under.get("byBookmaker") or {}).get(book)
            if not u or u.get("overUnder") != b.get("overUnder") or b.get("overUnder") is None:
                continue
            rows.append({
                "date": day.isoformat(),
                "event_id": ev["eventID"],
                "start": start,
                "home": short.get("home"),
                "away": short.get("away"),
                "sgo_player_id": pid,
                "player_name": p.get("name") or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                "player_team": team,
                "opp": opp,
                "stat": o["statID"],
                "book": book,
                "line": float(b["overUnder"]),
                "over_odds": int(float(b["odds"])),
                "under_odds": int(float(u["odds"])),
                "available": bool(b.get("available")) and bool(u.get("available")),
                "deeplink": b.get("deeplink", ""),
                "fetched_at": fetched_at,
            })
    return rows


def lines_path(day: dt.date):
    return LINES_DIR / f"{day.isoformat()}.csv"


def update_lines(day: dt.date | None = None) -> pd.DataFrame:
    """Fetch today's lines and merge with any earlier snapshot from today.

    Props drop off the feed once a game starts, so rows from an earlier run
    are kept for games that are no longer returned.
    """
    day = day or dt.datetime.now(ET).date()
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    events = fetch_events(day)
    rows = [r for ev in events for r in parse_event(ev, day, fetched_at)]
    new = pd.DataFrame(rows, columns=LINE_COLUMNS)
    path = lines_path(day)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["event_id"].isin(new["event_id"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    LINES_DIR.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    print(f"lines {day}: {len(events)} games, {len(new)} book lines")
    return new


def load_lines(day: dt.date) -> pd.DataFrame:
    p = lines_path(day)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=LINE_COLUMNS)
