"""Build the static dashboard in site/ (published to GitHub Pages).

Also writes site/picks.json and site/history.json, which Game Plan reads.
"""
from __future__ import annotations

import datetime as dt
import json

from .config import DATA, MIN_EV, SITE
from .grade import history

BOOKS = {
    "fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM", "caesars": "Caesars",
    "espnbet": "ESPN BET", "bovada": "Bovada", "unibet": "Unibet", "pointsbet": "PointsBet",
    "williamhill": "William Hill", "fanatics": "Fanatics", "betrivers": "BetRivers", "bet365": "bet365",
    "any": "typical odds",
}


def ranked_props(day: dict) -> list[dict]:
    """A day's top props as the page lists them: over MIN_EV, one per player, best expected value first."""
    seen, out = set(), []
    for p in sorted(day.get("top_props", []), key=lambda p: -p["ev"]):
        if p["ev"] > MIN_EV and p["player_id"] not in seen:
            seen.add(p["player_id"])
            out.append(p)
    return out


def _graded(picks: list[dict]) -> list[dict]:
    return [p for p in picks if p.get("result") in ("win", "loss")]


def record(picks: list[dict]) -> dict:
    """Win-loss record and units, 1 unit a pick at the odds shown that day."""
    wins = sum(p["result"] == "win" for p in picks)
    units = sum((p["decimal"] - 1) if p["result"] == "win" else -1 for p in picks)
    return {"wins": wins, "losses": len(picks) - wins, "units": round(units, 2)}


def records_by(picks: list[dict], key) -> dict:
    groups: dict = {}
    for p in picks:
        groups.setdefault(key(p), []).append(p)
    return {k: {"name": grp[-1].get("player_name"), **record(grp)} for k, grp in groups.items()}


def summarize(days: list[dict]) -> dict:
    """Records for picks as the page shows them now, so they match how Brandon bets."""
    graded = sorted((d for d in days if d.get("graded")), key=lambda d: d["date"], reverse=True)
    ranked = {d["date"]: ranked_props(d) for d in graded}
    top = lambda n: [p for r in ranked.values() for p in _graded(r[:n])]  # noqa: E731
    props = top(None)
    recent = [{
        "date": d["date"],
        "top5": record(_graded(ranked[d["date"]][:5])),
        "top10": record(_graded(ranked[d["date"]][:10])),
        "props": record(_graded(ranked[d["date"]])),
    } for d in graded[:30]]
    return {
        "top5": record(top(5)),
        "top10": record(top(10)),
        "props": record(props),
        "recent": recent,
        # by prop type (Points, Rebounds, ...) and by player, keyed by player_id
        "by_market": records_by(props, lambda p: p.get("market")),
        "by_pick": records_by(props, lambda p: str(p.get("player_id"))),
    }


def build(day: dt.date):
    SITE.mkdir(exist_ok=True)
    days = history()
    today = next((d for d in days if d["date"] == day.isoformat()), None)
    report_path = DATA / "model_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    payload = {
        "date": day.isoformat(),
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "today": today,
        "summary": summarize(days),
        "model": report,
        "books": BOOKS,
        "min_ev": MIN_EV,
    }
    (SITE / "picks.json").write_text(json.dumps(payload, default=str))
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, default=str).replace("</", "<\\/"))
    (SITE / "index.html").write_text(html)
    (SITE / ".nojekyll").write_text("")
    print(f"site built for {day}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA Picks</title>
<style>
:root {
  --bg: #f6f7f9; --card: #ffffff; --text: #14171c; --muted: #5d6673; --line: #e3e6eb;
  --accent: #1f6feb; --win: #1a7f37; --loss: #cf222e; --warn: #9a6700; --chip: #eef1f5;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --card: #161b22; --text: #e6edf3; --muted: #8b949e; --line: #2a313c;
          --accent: #4493f8; --win: #3fb950; --loss: #f85149; --warn: #d29922; --chip: #21262d; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 20px 16px 48px; }
header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; }
h1 { font-size: 24px; margin: 0; }
h2 { font-size: 17px; margin: 28px 0 10px; }
.muted { color: var(--muted); }
.small { font-size: 13px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
.legs { list-style: none; padding: 0; margin: 14px 0 0; }
.legs li { border-top: 1px solid var(--line); padding: 10px 0; display: grid;
  grid-template-columns: 1fr auto; gap: 2px 12px; }
.pick { font-weight: 600; }
.chip { display: inline-block; background: var(--chip); border-radius: 999px; padding: 1px 8px; font-size: 12px; color: var(--muted); }
.pos { color: var(--win); } .neg { color: var(--loss); } .warn { color: var(--warn); }
a { color: var(--accent); text-decoration: none; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: .4px; }
td.num, th.num { text-align: right; }
.scroll { overflow-x: auto; }
.filters { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
.filters button { border: 1px solid var(--line); background: var(--card); color: var(--text); border-radius: 999px;
  padding: 4px 10px; font-size: 13px; cursor: pointer; }
.filters button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
.record { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
details summary { cursor: pointer; color: var(--accent); }
.empty { text-align: center; padding: 28px 16px; }
footer { margin-top: 36px; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<main>
  <header>
    <h1>NBA Picks</h1>
    <span class="muted small" id="updated"></span>
  </header>
  <section id="props"></section>
  <section id="record"></section>
  <section id="model"></section>
  <footer>Picks are model estimates, not guarantees. Odds move; check the price at your book before betting. Bet only what you can afford to lose.</footer>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const book = b => (D.books && D.books[b]) || b;
const pct = x => x == null ? '' : (x * 100).toFixed(0) + '%';
const sgn = x => (x > 0 ? '+' : '') + x;
const evTxt = x => (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const time = iso => new Date(iso).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
const fmtDate = s => new Date(s + 'T12:00:00').toLocaleDateString([], {weekday: 'short', month: 'short', day: 'numeric'});

const stamp = iso => new Date(iso).toLocaleString([], {month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'});
const made = D.today && D.today.generated_at;
const minEv = D.min_ev ?? 0;  // picks at or below this expected value aren't shown
const evFloor = (minEv * 100).toFixed(0) + '%';
document.getElementById('updated').textContent =
  fmtDate(D.date) + (made ? (D.today.final ? ' · final picks made ' : ' · early picks made ') + stamp(made) : ' · checked ' + stamp(D.built_at));

const pastRec = l => {
  const r = (D.summary.by_pick || {})[String(l.player_id)];
  return r ? `past picks ${r.wins}-${r.losses}` : 'no past picks';
};
// Fliff and Rebet often post a slightly different line. A line that's better for the
// pick (higher for an Under, lower for an Over) only raises the chance it hits, so
// it's fine at the same take price.
const lineOk = l => `, line ${l.line} or ${l.side === 'Under' ? 'higher' : 'lower'}`;
const takeText = l => l.min_odds != null ? ` · take ${sgn(l.min_odds)} or better${lineOk(l)}` : '';

// Today's props ranked the way they're meant to be bet: over the EV floor, one per player, best first
function rankedProps() {
  const seen = new Set();
  return ((D.today && D.today.top_props) || []).filter(r => r.ev > minEv).sort((a, b) => b.ev - a.ev)
    .filter(r => !seen.has(r.player_id) && seen.add(r.player_id)).map((r, i) => ({...r, rank: i + 1}));
}

let marketFilter = 'All';
function renderProps() {
  const rows = rankedProps();
  const el = document.getElementById('props');
  if (!D.today) {
    el.innerHTML = `<h2>Today's top player props</h2><div class="card empty muted">No NBA games with prop lines today. Picks appear here on game days, usually by about noon Eastern.</div>`;
    return;
  }
  if (!rows.length) { el.innerHTML = `<h2>Today's top player props</h2><div class="card empty muted">No player props over ${evFloor} expected value today.</div>`; return; }
  const markets = ['All', ...new Set(rows.map(r => r.market))];
  const shown = rows.filter(r => marketFilter === 'All' || r.market === marketFilter);
  const res = r => D.today.graded ? ` · <span class="${r.result === 'win' ? 'pos' : r.result === 'loss' ? 'neg' : 'muted'}">${esc(r.result)}${r.actual != null ? ' (' + r.actual + ')' : ''}</span>` : '';
  el.innerHTML = `<h2>Today's top player props</h2>
  <p class="muted small" style="margin:-4px 0 10px">Single bets, best first, one per player. Odds shown are typical prices at the big sportsbooks. On Fliff or Rebet, take a pick if the odds are at least its "take" price, the break-even odds. If their line is different, a better line for the pick (higher for an Under, lower for an Over) is fine; a worse one isn't.</p>
  <div class="filters">${markets.map(m => `<button class="${m === marketFilter ? 'on' : ''}" data-m="${esc(m)}">${esc(m)}</button>`).join('')}</div>
  <div class="card" style="padding:0 16px"><ul class="legs" style="margin:0">${shown.map((r, i) => `<li style="${i === 0 ? 'border-top:0' : ''}">
    <div><div class="pick"><span class="muted">${r.rank}.</span> ${esc(r.player_name)} ${esc(r.side)} ${r.line} ${esc(r.market)}${r.injury ? ` <span class="warn small">${esc(r.injury)}</span>` : ''}</div>
      <div class="muted small">${esc(r.team)} vs ${esc(r.opp_abbr)} · ${time(r.start)} · projects ${(+r.proj).toFixed(1)}${r.recent_avg != null ? ', recent avg ' + (+r.recent_avg).toFixed(1) : ''} · hits ${pct(r.p_hit)}${takeText(r)} · ${pastRec(r)}${res(r)}</div></div>
    <div style="text-align:right"><div class="pick">${r.deeplink ? `<a href="${esc(r.deeplink)}" target="_blank" rel="noopener">${sgn(r.odds)}</a>` : sgn(r.odds)}</div>
      <div class="small"><span class="muted">${esc(book(r.book))}</span> <span class="${r.ev >= 0 ? 'pos' : 'neg'}">${evTxt(r.ev)}</span></div></div>
  </li>`).join('')}</ul></div>`;
  el.querySelectorAll('button[data-m]').forEach(b => b.onclick = () => { marketFilter = b.dataset.m; renderProps(); });
}

function renderRecord() {
  const s = D.summary;
  const el = document.getElementById('record');
  const n = r => r.wins + r.losses;
  const wl = r => `${r.wins}-${r.losses}`;
  const units = r => `<span class="${r.units >= 0 ? 'pos' : 'neg'}">${sgn(r.units)}u</span>`;
  const card = (title, r) => `<div class="card"><span class="muted small">${title}</span><b style="display:block;font-size:20px">${wl(r)}</b>
      <span class="${r.units >= 0 ? 'pos' : 'neg'} small">${sgn(r.units)} units at 1 unit each</span></div>`;
  const recTable = (title, rows) => rows.length ? `<div class="card scroll" style="padding:4px 12px;margin-top:10px"><table>
    <thead><tr><th>${title}</th><th class="num">Record</th><th class="num">Win %</th><th class="num">Units</th></tr></thead>
    <tbody>${rows.map(([name, r]) => `<tr><td>${esc(name)}</td><td class="num">${wl(r)}</td>
      <td class="num">${pct(r.wins / n(r))}</td><td class="num ${r.units >= 0 ? 'pos' : 'neg'}">${sgn(r.units)}</td></tr>`).join('')}</tbody>
  </table></div>` : '';
  const cats = Object.entries(s.by_market || {}).sort((a, b) => n(b[1]) - n(a[1]));
  const picks = Object.values(s.by_pick || {}).map(r => [r.name, r]).sort((a, b) => n(b[1]) - n(a[1]) || b[1].units - a[1].units);
  const days = s.recent.filter(d => n(d.props));
  el.innerHTML = `<h2>Track record</h2>
  <div class="record">
    ${card('Top 5 props each day', s.top5)}
    ${card('Top 10 props each day', s.top10)}
    ${card('All top props', s.props)}
  </div>
  ${recTable('By category', cats)}
  ${picks.length ? `<details style="margin-top:12px"><summary>By player (${picks.length})</summary>${recTable('Player', picks)}</details>` : ''}
  ${days.length ? `<div class="card scroll" style="padding:4px 12px;margin-top:10px"><table>
    <thead><tr><th>Day</th><th class="num">Top 5</th><th class="num">Top 10</th><th class="num">All props</th></tr></thead>
    <tbody>${days.map(d => `<tr><td>${fmtDate(d.date)}</td>
      <td class="num">${wl(d.top5)} ${units(d.top5)}</td><td class="num">${wl(d.top10)} ${units(d.top10)}</td>
      <td class="num">${wl(d.props)} ${units(d.props)}</td></tr>`).join('')}</tbody>
  </table></div>` : ''}
  <p class="muted small" style="margin:8px 0 0">${days.length ? `Records count picks the way the page lists them now (over ${evFloor} expected value, one prop per player), 1 unit each at the odds shown that day.` : 'Results show up the morning after each game day.'}</p>`;
}

function renderModel() {
  const m = D.model;
  if (!m) return;
  const rows = Object.entries(m.markets);
  document.getElementById('model').innerHTML = `<h2>How the model is doing</h2>
  <details><summary>Backtest on the ${esc(m.holdout_season)} season</summary>
  <p class="small muted">Trained on ${m.train_rows.toLocaleString()} earlier player games, then tested on ${m.holdout_rows.toLocaleString()} games it never saw.
  Lines in this test are set at each player's recent average, the way books start; real book lines are sharper, which is why the app also blends in the market's own odds.</p>
  <div class="card scroll" style="padding:4px 12px"><table>
    <thead><tr><th>Market</th><th class="num">Avg miss</th><th class="num">Recent-avg miss</th><th class="num">Picks right</th><th class="num">Confident picks right</th></tr></thead>
    <tbody>${rows.map(([k, r]) => `<tr><td>${esc(k.replace(/\+/g, ' + ').replace('threePointersMade', '3-pointers'))}</td>
      <td class="num">${r.mae_model}</td><td class="num">${r.mae_recent_average}</td>
      <td class="num">${pct(r.pick_accuracy)}</td><td class="num">${r.strong_pick_accuracy == null ? '' : pct(r.strong_pick_accuracy)}</td></tr>`).join('')}</tbody>
  </table></div></details>`;
}

renderProps(); renderRecord(); renderModel();
</script>
</body>
</html>
"""
