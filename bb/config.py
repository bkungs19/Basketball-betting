"""Shared settings for the pipeline."""
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
SITE = ROOT / "site"

ET = ZoneInfo("America/New_York")

# Seasons used to train the model. The current season is appended at run time.
HISTORY_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# Prop markets we model: Sports Game Odds statID -> box score columns summed.
MARKETS = {
    "points": ["pts"],
    "rebounds": ["reb"],
    "assists": ["ast"],
    "threePointersMade": ["fg3m"],
    "points+rebounds+assists": ["pts", "reb", "ast"],
    "points+rebounds": ["pts", "reb"],
    "points+assists": ["pts", "ast"],
    "rebounds+assists": ["reb", "ast"],
}

MARKET_LABELS = {
    "points": "Points",
    "rebounds": "Rebounds",
    "assists": "Assists",
    "threePointersMade": "3-Pointers",
    "points+rebounds+assists": "Pts+Reb+Ast",
    "points+rebounds": "Pts+Reb",
    "points+assists": "Pts+Ast",
    "rebounds+assists": "Reb+Ast",
}

# Weight on the model vs. the no-vig market price when blending probabilities.
# Markets are sharp; leaning partly on them keeps the model from chasing noise.
MODEL_WEIGHT = 0.5
EARLY_SEASON_GAMES = 5
EARLY_SEASON_MODEL_WEIGHT = 0.3

PARLAY_MIN_LEGS = 2
PARLAY_MAX_LEGS = 3
PARLAY_MIN_HIT = 0.30  # Brandon wants parlays that hit about 30% of the time
MIN_EV = 0.05  # only picks with expected value above 5% are shown or used in parlays (his request)
