"""Command line entry point: python -m bb <command>."""
import argparse
import datetime as dt

from . import lines, stats
from .config import ET, HISTORY_SEASONS


def main():
    ap = argparse.ArgumentParser(prog="bb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="download box scores for past seasons")
    b.add_argument("--seasons", nargs="*", default=HISTORY_SEASONS)
    sub.add_parser("stats", help="update the current season's box scores")
    ln = sub.add_parser("lines", help="fetch today's prop lines")
    ln.add_argument("--date", type=dt.date.fromisoformat, default=None)
    args = ap.parse_args()

    if args.cmd == "backfill":
        for s in args.seasons:
            stats.update_season(s)
    elif args.cmd == "stats":
        stats.update_season(stats.current_season())
    elif args.cmd == "lines":
        lines.update_lines(args.date or dt.datetime.now(ET).date())


if __name__ == "__main__":
    main()
