"""Command line entry point: python -m bb <command>."""
import argparse
import datetime as dt
import os

from . import daily, lines, stats
from .config import ET, HISTORY_SEASONS


def main():
    ap = argparse.ArgumentParser(prog="bb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="download box scores for past seasons")
    b.add_argument("--seasons", nargs="*", default=HISTORY_SEASONS)
    sub.add_parser("stats", help="update the current season's box scores")
    ln = sub.add_parser("lines", help="fetch today's prop lines")
    ln.add_argument("--date", type=dt.date.fromisoformat, default=None)
    d = sub.add_parser("daily", help="the full daily run")
    d.add_argument("--date", type=dt.date.fromisoformat, default=None)
    d.add_argument("--offline", action="store_true", help="use saved data only (no downloads)")
    sub.add_parser("wait", help="sleep until an hour before today's first game, if it's within 5 hours")
    args = ap.parse_args()

    if args.cmd == "backfill":
        for s in args.seasons:
            stats.update_season(s)
    elif args.cmd == "stats":
        stats.update_season(stats.current_season())
    elif args.cmd == "daily":
        daily.run(args.date, offline=args.offline)
    elif args.cmd == "wait":
        due = daily.pregame_wait()
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"refresh={'true' if due else 'false'}\n")
    elif args.cmd == "lines":
        lines.update_lines(args.date or dt.datetime.now(ET).date())


if __name__ == "__main__":
    main()
