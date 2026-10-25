"""Email Brandon the final picks.

GitHub emails people about issues they're assigned to, so the pregame run
opens one issue a day with the picks, assigned to the repo owner, and closes
the day before's. No email account or password is needed.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess

import pandas as pd

from .config import ET, MIN_EV
from .grade import PICKS_DIR
from .site import ranked_props

PAGE = "https://bkungs19.github.io/Basketball-betting/"
LABEL = "final-picks"
TOP = 10


def _odds(x) -> str:
    return f"{x:+d}" if isinstance(x, int) else str(x)


def _time(iso: str) -> str:
    return pd.Timestamp(iso).tz_convert(ET).strftime("%-I:%M %p ET")


def message(picks: dict) -> tuple[str, str]:
    """Issue title and body (Markdown) for a day's final picks."""
    day = dt.date.fromisoformat(picks["date"])
    title = f"Final picks for {day:%a, %b %-d}"
    props = ranked_props(picks)[:TOP]
    games = [g for g in picks.get("game_picks", []) if g["ev"] > MIN_EV]
    starts = [p["start"] for p in props + games if p.get("start")]
    lines = [f"Final picks are out. [Open the page]({PAGE})", ""]
    if starts:
        lines += [f"First pick starts at {_time(min(starts))}.", ""]
    if props:
        lines += [f"**Top {len(props)} player props** (best first)", ""]
        better = {"Under": "higher", "Over": "lower"}
        lines += [f"{i}. {p['player_name']} {p['side']} {p['line']} {p['market']}: typical {_odds(p['odds'])}, "
                  f"take {_odds(p['min_odds'])} or better (line {p['line']} or {better.get(p['side'], 'better')})"
                  for i, p in enumerate(props, 1)]
        lines.append("")
    if games:
        lines += ["**Moneyline and spread**", ""]
        lines += [f"- {g['label']}: typical {_odds(g['odds'])}, take {_odds(g['min_odds'])} or better" for g in games]
        lines.append("")
    if not props and not games:
        lines += [f"No picks over {MIN_EV:.0%} expected value today.", ""]
    return title, "\n".join(lines)


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def send(day: dt.date | None = None, dry_run: bool = False) -> None:
    """Open today's picks issue (once) and close earlier ones. Needs GH_TOKEN with issues: write."""
    day = day or dt.datetime.now(ET).date()
    path = PICKS_DIR / f"{day.isoformat()}.json"
    picks = json.loads(path.read_text()) if path.exists() else {}
    if not picks.get("final"):
        print(f"{day}: no final picks, so no email")
        return
    title, body = message(picks)
    owner = os.environ.get("NOTIFY_USER", "bkungs19")
    body += f"\n@{owner}"
    if dry_run:
        print(title, body, sep="\n\n")
        return
    _gh("label", "create", LABEL, "--color", "1f6feb", "--description", "Daily final picks email", "--force")
    open_issues = json.loads(_gh("issue", "list", "--label", LABEL, "--state", "open", "--json", "number,title"))
    if any(i["title"] == title for i in open_issues):
        print(f"{day}: picks email already sent")
        return
    for i in open_issues:
        _gh("issue", "close", str(i["number"]))
    _gh("issue", "create", "--title", title, "--body", body, "--label", LABEL, "--assignee", owner)
    print(f"{day}: emailed the final picks to @{owner}")
