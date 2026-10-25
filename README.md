# Basketball Betting

Daily NBA player prop picks. Several times a day before games, GitHub runs the pipeline: it updates player stats, grades yesterday's picks, pulls today's prop lines, retrains the model, scores moneylines and spreads with a separate game model, ranks the best single bets and publishes the dashboard. Runs start every two hours from early morning because GitHub often starts them late. The first run that lands within 5.5 hours of the day's first game waits and makes the final picks an hour before tip-off, with the latest lines and injury news; later runs leave the final picks alone. Once the final picks are made, the run opens a GitHub issue with them, assigned to the repo owner, which GitHub emails out (`bb/notify.py`); the previous day's issue is closed.

**Dashboard:** https://bkungs19.github.io/Basketball-betting/ (raw data for Game Plan at `/picks.json`)

## How it works

1. **Stats** (`bb/stats.py`): player box scores from ESPN's public API, four past seasons plus the current one, saved in `data/boxscores/`. Advanced numbers (usage share, pace, offensive and defensive rating, what each defense allows) are computed from them in `bb/features.py`.
2. **Lines** (`bb/lines.py`): points, rebounds, assists, threes and the combo props from Sports Game Odds' free plan (FanDuel, DraftKings, ESPN BET and other books), saved in `data/lines/`. The free plan allows 2,500 games a month; the job only asks on days with regular-season or playoff games.
3. **Model** (`bb/model.py`): gradient-boosted trees predict each player's expected stat line, a second model predicts how spread out it is, and together they give the chance of going over any line. It is checked on a season it never trained on (`data/model_report.json`).
4. **Picks** (`bb/predict.py`, `bb/pricing.py`): the model's chance is blended with the books' own no-vig odds, because sportsbook lines are sharp. Each line is priced at the typical (median) odds across the books that offer it. Only picks with expected value above 4% (`MIN_EV` in `bb/config.py`) are shown, best first and one prop per player, meant as single bets; each pick's "take" price is its break-even odds. Players listed out, doubtful or questionable on ESPN's injury report, and lines where the model and the market disagree wildly (usually news the model can't see), are left out.
5. **Grading** (`bb/grade.py`): each morning yesterday's picks are graded against the box scores. A player who doesn't play voids the pick. The dashboard shows the record of each day's top 5 and top 10 props, by prop type and by player or team, and each of today's picks shows that player's or team's record on past picks.

Picks are estimates, not guarantees.

## Running it

The schedule lives in `.github/workflows/daily.yml`. To run it now, open the Actions tab, pick **daily picks** and click **Run workflow**. The Sports Game Odds key is the `SGO_API_KEY` repository secret.

Locally: `pip install -r requirements.txt`, then `python -m bb daily` (needs `SGO_API_KEY`), or `python -m bb daily --date 2026-10-20 --offline` to rebuild from saved data.
