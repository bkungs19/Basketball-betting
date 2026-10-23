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


def summarize(days: list[dict]) -> dict:
    parlays = [d["parlay"] for d in days if d.get("graded") and d.get("parlay") and d["parlay"].get("result") in ("win", "loss")]
    props = [p for d in days if d.get("graded") for p in d.get("top_props", []) if p.get("result") in ("win", "loss")]
    wins = sum(p["result"] == "win" for p in parlays)
    profit = sum(p.get("profit", 0) for p in parlays)
    prop_wins = sum(p["result"] == "win" for p in props)
    prop_profit = sum((p["decimal"] - 1) if p["result"] == "win" else -1 for p in props)
    gpicks = [p for d in days if d.get("graded") for p in d.get("game_picks", []) if p.get("result") in ("win", "loss")]
    game_wins = sum(p["result"] == "win" for p in gpicks)
    game_profit = sum((p["decimal"] - 1) if p["result"] == "win" else -1 for p in gpicks)
    recent = []
    for d in sorted(days, key=lambda d: d["date"], reverse=True)[:30]:
        p = d.get("parlay")
        recent.append({
            "date": d["date"],
            "result": (p or {}).get("result", "pending" if p else "none"),
            "american": (p or {}).get("american"),
            "profit": (p or {}).get("profit"),
            "legs": [
                {k: l.get(k) for k in ("player_name", "market", "side", "line", "actual", "result", "label")}
                for l in (p or {}).get("legs", [])
            ],
        })
    return {
        "parlays": {"wins": wins, "losses": len(parlays) - wins, "units": round(profit, 2)},
        "props": {"wins": prop_wins, "losses": len(props) - prop_wins, "units": round(prop_profit, 2)},
        "games": {"wins": game_wins, "losses": len(gpicks) - game_wins, "units": round(game_profit, 2)},
        "recent": recent,
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
.parlay-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }
.odds { font-size: 34px; font-weight: 700; letter-spacing: -0.5px; }
.stats { display: flex; gap: 20px; flex-wrap: wrap; }
.stat b { display: block; font-size: 18px; }
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
  <section id="parlay"></section>
  <section id="others"></section>
  <section id="games"></section>
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

const legName = l => l.kind === 'game' ? `${l.label} (${l.market.toLowerCase()})` : `${l.player_name} ${l.side} ${l.line} ${l.market}`;
const projText = l => l.proj >= 0 ? `projects a ${(+l.proj).toFixed(1)}-point win` : `projects a ${(-l.proj).toFixed(1)}-point loss`;

function legRow(l, showBook) {
  const link = l.deeplink ? ` <a href="${esc(l.deeplink)}" target="_blank" rel="noopener">Bet</a>` : '';
  const inj = l.injury ? ` <span class="warn small">${esc(l.injury)}</span>` : '';
  const meta = l.kind === 'game' ? projText(l) : `projects ${(+l.proj).toFixed(1)}, recent avg ${(+l.recent_avg).toFixed(1)}`;
  return `<li>
    <div><div class="pick">${esc(legName(l))}${inj}</div>
      <div class="muted small">${esc(l.team)} vs ${esc(l.opp_abbr)} · ${time(l.start)} · ${meta}${showBook ? ' · ' + esc(book(l.book)) : ''}</div></div>
    <div style="text-align:right"><div class="pick">${sgn(l.odds)}${link}</div>
      <div class="muted small">hits ${pct(l.p_hit)}${l.min_odds != null ? ' · take ' + sgn(l.min_odds) + ' or better' : ''}</div></div>
  </li>`;
}

function renderParlay() {
  const el = document.getElementById('parlay');
  const t = D.today;
  if (!t) {
    el.innerHTML = `<h2>Today's parlay</h2><div class="card empty muted">No NBA games with prop lines today. Picks appear here on game days, usually by about noon Eastern.</div>`;
    return;
  }
  const p = t.parlay;
  if (!p) {
    el.innerHTML = `<h2>Today's parlay</h2><div class="card empty muted">No parlay today: no 2 or 3 leg combination of picks over ${evFloor} expected value had at least a 30% chance to hit.</div>`;
    return;
  }
  el.innerHTML = `<h2>Today's parlay</h2>
  <div class="card">
    <div class="parlay-head">
      <div><div class="odds">${sgn(p.american)}</div><div class="muted small">${p.legs.length}-leg parlay at ${esc(book(p.book))}${p.min_american != null ? ' · take ' + sgn(p.min_american) + ' or better' : ''}</div></div>
      <div class="stats">
        <div class="stat"><span class="muted small">Hit chance</span><b>${pct(p.p_hit)}</b></div>
        <div class="stat"><span class="muted small">Expected value</span><b class="${p.ev >= 0 ? 'pos' : 'neg'}">${evTxt(p.ev)}</b></div>
        ${p.result ? `<div class="stat"><span class="muted small">Result</span><b class="${p.result === 'win' ? 'pos' : 'neg'}">${esc(p.result)}</b></div>` : ''}
      </div>
    </div>
    <ul class="legs">${p.legs.map(l => legRow(l, false)).join('')}</ul>
    ${p.book === 'any' ? `<p class="muted small" style="margin:8px 0 0">Odds shown are typical prices at the big sportsbooks. On Fliff or Rebet, find the same line and bet it if the odds are at least the "take" price, which keeps it above ${evFloor} expected value.</p>` : ''}
  </div>`;
}

function renderOthers() {
  const o = (D.today && D.today.other_parlays) || [];
  if (!o.length) return;
  document.getElementById('others').innerHTML = `<details style="margin-top:12px"><summary>Best parlay at other books (${o.length})</summary>
    ${o.map(p => `<div class="card" style="margin-top:10px"><div class="parlay-head"><div><b>${esc(book(p.book))}</b> <span class="muted">${sgn(p.american)}</span></div>
      <div class="muted small">hits ${pct(p.p_hit)} · EV <span class="${p.ev >= 0 ? 'pos' : 'neg'}">${evTxt(p.ev)}</span></div></div>
      <ul class="legs">${p.legs.map(l => legRow(l, false)).join('')}</ul></div>`).join('')}</details>`;
}

function renderGames() {
  const rows = ((D.today && D.today.game_picks) || []).filter(r => r.ev > minEv);
  const el = document.getElementById('games');
  if (!D.today) { el.innerHTML = ''; return; }
  if (!rows.length) { el.innerHTML = `<h2>Moneyline and spread picks</h2><div class="card empty muted">No moneyline or spread picks over ${evFloor} expected value today.</div>`; return; }
  const res = r => D.today.graded ? ` · <span class="${r.result === 'win' ? 'pos' : r.result === 'loss' ? 'neg' : 'muted'}">${esc(r.result)}</span>` : '';
  el.innerHTML = `<h2>Moneyline and spread picks</h2>
  <div class="card" style="padding:0 16px"><ul class="legs" style="margin:0">${rows.map((r, i) => `<li style="${i === 0 ? 'border-top:0' : ''}">
    <div><div class="pick">${esc(legName(r))}</div>
      <div class="muted small">${esc(r.team)} vs ${esc(r.opp_abbr)} · ${time(r.start)} · ${projText(r)} · ${r.kind === 'game' && r.stat === 'moneyline' ? 'wins' : 'covers'} ${pct(r.p_hit)} · take ${sgn(r.min_odds)} or better${res(r)}</div></div>
    <div style="text-align:right"><div class="pick">${sgn(r.odds)}</div>
      <div class="small"><span class="${r.ev >= 0 ? 'pos' : 'neg'}">${evTxt(r.ev)}</span></div></div>
  </li>`).join('')}</ul></div>`;
}

let marketFilter = 'All';
function renderProps() {
  const rows = ((D.today && D.today.top_props) || []).filter(r => r.ev > minEv);
  const el = document.getElementById('props');
  if (!D.today) { el.innerHTML = ''; return; }
  if (!rows.length) { el.innerHTML = `<h2>Top player props</h2><div class="card empty muted">No player props over ${evFloor} expected value today.</div>`; return; }
  const markets = ['All', ...new Set(rows.map(r => r.market))];
  const shown = rows.filter(r => marketFilter === 'All' || r.market === marketFilter);
  const res = r => D.today.graded ? ` · <span class="${r.result === 'win' ? 'pos' : r.result === 'loss' ? 'neg' : 'muted'}">${esc(r.result)}${r.actual != null ? ' (' + r.actual + ')' : ''}</span>` : '';
  el.innerHTML = `<h2>Top player props</h2>
  <div class="filters">${markets.map(m => `<button class="${m === marketFilter ? 'on' : ''}" data-m="${esc(m)}">${esc(m)}</button>`).join('')}</div>
  <div class="card" style="padding:0 16px"><ul class="legs" style="margin:0">${shown.map((r, i) => `<li style="${i === 0 ? 'border-top:0' : ''}">
    <div><div class="pick">${esc(r.player_name)} ${esc(r.side)} ${r.line} ${esc(r.market)}</div>
      <div class="muted small">${esc(r.team)} vs ${esc(r.opp_abbr)} · projects ${(+r.proj).toFixed(1)} · hits ${pct(r.p_hit)}${r.min_odds != null ? ' · take ' + sgn(r.min_odds) + ' or better' : ''}${res(r)}</div></div>
    <div style="text-align:right"><div class="pick">${r.deeplink ? `<a href="${esc(r.deeplink)}" target="_blank" rel="noopener">${sgn(r.odds)}</a>` : sgn(r.odds)}</div>
      <div class="small"><span class="muted">${esc(book(r.book))}</span> <span class="${r.ev >= 0 ? 'pos' : 'neg'}">${evTxt(r.ev)}</span></div></div>
  </li>`).join('')}</ul></div>`;
  el.querySelectorAll('button[data-m]').forEach(b => b.onclick = () => { marketFilter = b.dataset.m; renderProps(); });
}

function renderRecord() {
  const s = D.summary;
  const el = document.getElementById('record');
  const pr = s.parlays, pp = s.props;
  const recent = s.recent.filter(r => r.result !== 'none');
  el.innerHTML = `<h2>Track record</h2>
  <div class="record">
    <div class="card"><span class="muted small">Daily parlays</span><b style="display:block;font-size:20px">${pr.wins}-${pr.losses}</b>
      <span class="${pr.units >= 0 ? 'pos' : 'neg'} small">${sgn(pr.units)} units</span></div>
    <div class="card"><span class="muted small">Top player props</span><b style="display:block;font-size:20px">${pp.wins}-${pp.losses}</b>
      <span class="${pp.units >= 0 ? 'pos' : 'neg'} small">${sgn(pp.units)} units at 1 unit each</span></div>
    ${s.games ? `<div class="card"><span class="muted small">Moneyline and spread picks</span><b style="display:block;font-size:20px">${s.games.wins}-${s.games.losses}</b>
      <span class="${s.games.units >= 0 ? 'pos' : 'neg'} small">${sgn(s.games.units)} units at 1 unit each</span></div>` : ''}
  </div>
  ${recent.length ? `<div class="card" style="margin-top:10px;padding:0 16px"><ul class="legs" style="margin:0">${recent.map((r, i) => `<li style="${i === 0 ? 'border-top:0' : ''}">
    <div><div class="pick">${fmtDate(r.date)}${r.american ? ' · ' + sgn(r.american) : ''}</div>
      <div class="small">${r.legs.map(l => `<span class="${l.result === 'win' ? 'pos' : l.result === 'loss' ? 'neg' : 'muted'}">${esc(l.label || legName(l))}${l.actual != null ? ' (' + l.actual + ')' : ''}</span>`).join('<br>')}</div></div>
    <div style="text-align:right" class="pick ${r.result === 'win' ? 'pos' : r.result === 'loss' ? 'neg' : 'muted'}">${esc(r.result)}${r.profit != null ? '<div class="small">' + sgn(r.profit) + 'u</div>' : ''}</div>
  </li>`).join('')}</ul></div>` : '<p class="muted small">Results show up the morning after each game day.</p>'}`;
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

renderParlay(); renderOthers(); renderGames(); renderProps(); renderRecord(); renderModel();
</script>
</body>
</html>
"""
