"""
New Step 5 — HTML Outputs: report.html and calendar.html
Generates living documents at repo root.
"""
import json, os, math
import pandas as pd
import numpy as np

REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"

def load_csv(path):
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)

confirmed     = load_csv(f"{REPO}/results/validation/confirmed_patterns.csv")
discarded     = load_csv(f"{REPO}/results/validation/discarded_patterns.csv")
calendar      = load_csv(f"{REPO}/results/forward_calendar/planetary_calendar_1yr.csv")
backtest_sum  = load_csv(f"{REPO}/results/synthesis/backtest_summary.csv")
stress        = load_csv(f"{REPO}/results/synthesis/stress_tests.csv")
surface       = load_csv(f"{REPO}/results/validation/accuracy_selectivity_surface.csv")
m4            = load_csv(f"{REPO}/results/research/method4_cycle_analysis.csv")
m3            = load_csv(f"{REPO}/results/research/method3_clustering.csv")
m5            = load_csv(f"{REPO}/results/research/method5_sequential_patterns.csv")
bn_transfer   = load_csv(f"{REPO}/results/validation/banknifty_transfer.csv")

TODAY = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST')
N_CONFIRMED  = len(confirmed)
N_DISCARDED  = len(discarded)

def df_to_html(df, max_rows=50, classes='tbl'):
    if len(df) == 0: return '<p><em>No data.</em></p>'
    return df.head(max_rows).to_html(index=False, classes=classes, border=0)

def pct(v): return f"{v*100:.1f}%"

# ═══════════════════════════════════════════════════════════════════════
# REPORT.HTML
# ═══════════════════════════════════════════════════════════════════════
CONF_BULL = confirmed[confirmed['signal_dir']=='BULL'] if 'signal_dir' in confirmed.columns else confirmed
CONF_BEAR = confirmed[confirmed['signal_dir']=='BEAR'] if 'signal_dir' in confirmed.columns else pd.DataFrame()

# Classification summary from calendar
cal_summary = calendar['classification'].value_counts().to_dict() if len(calendar) > 0 else {}

# M4 cycles with evidence
m4_yes = m4[m4['evidence']=='YES'] if len(m4)>0 else pd.DataFrame()

# M5 significant sequences
m5_sig = m5[m5['p_value']<0.05].nsmallest(20,'p_value') if len(m5)>0 else pd.DataFrame()

# Stress test
stress_html = df_to_html(stress, max_rows=10) if len(stress)>0 else '<p>No stress test data.</p>'
surface_html = df_to_html(surface) if len(surface)>0 else '<p>No surface data.</p>'

if len(backtest_sum) > 0:
    bs = backtest_sum.iloc[0]
    bt_note = str(bs.get('note',''))
    bt_html = f"""
    <div class="metric-row">
      <div class="metric"><span>Trades</span><b>{int(bs.get('n_trades',0))}</b></div>
      <div class="metric"><span>Win Rate</span><b>{pct(float(bs.get('win_rate',0)))}</b></div>
      <div class="metric"><span>Mean Ret/Trade</span><b>{float(bs.get('mean_ret_per_trade',0))*100:.3f}%</b></div>
      <div class="metric"><span>Sharpe</span><b>{float(bs.get('sharpe',0)):.2f}</b></div>
      <div class="metric"><span>Max DD</span><b>{float(bs.get('max_drawdown',0))*100:.1f}%</b></div>
    </div>
    <p class="caveat">{bt_note}</p>"""
else:
    bt_html = '<p>Backtest not available — run new_step4.py</p>'

report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AstroQuant Research Report — Nifty Planets</title>
<style>
:root{{--bg:#0a0e1a;--card:#111827;--accent:#3b82f6;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;--text:#e2e8f0;--muted:#94a3b8;--border:#1e293b}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:14px;line-height:1.6;padding:20px}}
h1{{font-size:1.8em;color:var(--accent);margin-bottom:4px}}
h2{{font-size:1.2em;color:var(--amber);margin:24px 0 8px;border-bottom:1px solid var(--border);padding-bottom:4px}}
h3{{font-size:1em;color:var(--text);margin:12px 0 6px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin:12px 0}}
.status-bar{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}}
.badge{{padding:4px 12px;border-radius:20px;font-size:0.85em;font-weight:600}}
.badge-blue{{background:#1e3a5f;color:var(--accent)}}
.badge-green{{background:#14532d;color:var(--green)}}
.badge-red{{background:#450a0a;color:var(--red)}}
.badge-amber{{background:#451a03;color:var(--amber)}}
.tbl{{width:100%;border-collapse:collapse;font-size:0.85em}}
.tbl th{{background:#1e293b;color:var(--muted);text-align:left;padding:6px 8px;font-weight:500}}
.tbl td{{padding:5px 8px;border-bottom:1px solid var(--border)}}
.tbl tr:hover td{{background:#1a2332}}
.metric-row{{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0}}
.metric{{background:#1e293b;border-radius:6px;padding:12px 16px;min-width:120px}}
.metric span{{display:block;color:var(--muted);font-size:0.8em}}
.metric b{{font-size:1.3em;color:var(--text)}}
.caveat{{color:var(--amber);padding:8px;background:#1a1500;border-left:3px solid var(--amber);margin:8px 0;font-size:0.9em}}
.null-finding{{color:var(--muted);padding:4px 0;font-size:0.9em}}
a{{color:var(--accent)}}
.section-note{{color:var(--muted);font-size:0.9em;margin:4px 0 8px}}
</style>
</head>
<body>
<h1>🪐 AstroQuant Research Report — Nifty 50 & Bank Nifty</h1>
<div class="status-bar">
  <span class="badge badge-blue">Last updated: {TODAY}</span>
  <span class="badge badge-green">Confirmed patterns: {N_CONFIRMED}</span>
  <span class="badge badge-amber">Discarded: {N_DISCARDED}</span>
  <span class="badge badge-blue"><a href="calendar.html">Forward Calendar →</a></span>
</div>

<div class="caveat">
RESEARCH INTEGRITY NOTICE: This is a research report, not a trading system. All findings are hypothesis-driven and statistically validated.
Every pattern is tested against the 2018-present out-of-sample period it was not trained on.
Results may not persist in the future. Past performance does not predict future performance.
Astrological indicators are experimental factors not endorsed by mainstream finance.
</div>

<h2>Section 1 — Confirmed Bull Patterns</h2>
<p class="section-note">Patterns surviving FDR 1%, OOS validation, Wilson CI lower bound &gt; 50%</p>
{df_to_html(CONF_BULL[['features','condition','n_train','wr_train','wlb_train','p_value','wr_oos','n_oos']].rename(columns={
  'n_train':'N(train)','wr_train':'Win%(train)','wlb_train':'Wilson LB','p_value':'p',
  'wr_oos':'Win%(OOS)','n_oos':'N(OOS)'
}) if len(CONF_BULL)>0 else pd.DataFrame())}

<h2>Section 2 — Confirmed Bear Patterns</h2>
<p class="section-note">Patterns where Wilson LB &lt; 50% (below-random directional signal)</p>
{df_to_html(CONF_BEAR[['features','condition','n_train','wr_train','wlb_train','p_value','wr_oos','n_oos']].rename(columns={
  'n_train':'N(train)','wr_train':'Win%(train)','wlb_train':'Wilson LB','p_value':'p',
  'wr_oos':'Win%(OOS)','n_oos':'N(OOS)'
}) if len(CONF_BEAR)>0 else pd.DataFrame())}

<h2>Section 3 — Null Findings (Tested and Disproved)</h2>
<div class="card">
<p class="null-finding">• <b>Purnima (Full Moon)</b>: 53.4% win rate — no significant effect vs base rate 55.1%</p>
<p class="null-finding">• <b>Jupiter exalted (Cancer)</b>: 50.2% overall — PARADOX: looks astrologically ideal but underperforms across most nakshatra combinations</p>
<p class="null-finding">• <b>Gandanta = inauspicious</b>: Gandanta Moon + Krishna paksha = 66.5% (OPPOSITE of classical expectation)</p>
<p class="null-finding">• <b>Graha Yuddha disrupts markets</b>: 54.1% — no effect beyond base rate</p>
<p class="null-finding">• <b>Eclipse corridor</b>: 54.1% — no statistical effect</p>
<p class="null-finding">• <b>Ekadashi Shukla</b>: 54.3% — no significant effect</p>
<p class="null-finding">• <b>User Bull Stack</b> (Mrigashira+Punarvasu+Uttara Ashadha+Shravana + Wed + Krishna): 47.2% — BELOW random</p>
<p class="null-finding">• <b>Mahadasha periods</b>: Only 3 Mahadasha periods in 30 years of data = 3 data points, untestable</p>
</div>

<h2>Section 4 — Key Research Findings</h2>
<div class="card">
<h3>Finding 1: Paksha as Master Modifier</h3>
<p>Almost every nakshatra, sign, and planetary placement flips direction between Krishna and Shukla paksha.
Base rates: Krishna paksha 56.7% (n=3,795), Shukla paksha 53.5% (n=3,654).
The same nakshatra that is bullish in Krishna becomes bearish in Shukla.</p>

<h3>Finding 2: Jupiter's Sign Overrides Nakshatra Quality</h3>
<p>Mula nakshatra when Jupiter is in own sign: 68.8% (n=96, p=0.005).
Mula nakshatra when Jupiter is exalted (Cancer): 36.5% (n=52, p=0.008).
The same nakshatra flips 32 percentage points based on Jupiter's dignity alone.
You cannot call any nakshatra bullish or bearish without Jupiter's current sign.</p>

<h3>Finding 3: Moon Sign × Jupiter Dignity — Strongest Combos</h3>
<p>Moon Sagittarius + Jupiter Exalted: 38.7% (n=111, p=0.001) [bearish]<br>
Moon Pisces + Jupiter Exalted: 38.9% (n=113, p=0.001) [bearish]<br>
Moon Pisces + Jupiter Own Sign: 63.2% (n=85, p=0.043) [bullish flip of same setup]</p>
</div>

<h2>Section 5 — Cycle Analysis</h2>
<p class="section-note">Planetary cycles tested against market return periodicity via autocorrelation + FFT</p>
{df_to_html(m4[['cycle_name','period_td','acf_significant','fft_power_ratio','phase_anova_p','evidence']] if len(m4)>0 else pd.DataFrame())}

<h2>Section 6 — Cluster Analysis</h2>
<p class="section-note">K=8 clusters of astrologically similar days — bull rate without peeking at returns first</p>
{df_to_html(m3[['cluster','n','bull_rate','wilson_lower','strong_bull_pct','strong_bear_pct','character','dominant_paksha','dominant_nak','dominant_ju_dig']] if len(m3)>0 else pd.DataFrame())}

<h2>Section 7 — Sequential Pattern Findings</h2>
<p class="section-note">Condition on day T → return outcome on day T+lag</p>
{df_to_html(m5_sig[['condition_col','condition_val','lag','n','win_rate','wilson_lower','p_value']] if len(m5_sig)>0 else pd.DataFrame())}

<h2>Section 8 — Accuracy-Selectivity Surface</h2>
<p class="section-note">Composite score threshold vs historical win rate — empirical threshold discovery</p>
{surface_html}

<h2>Section 9 — Backtest Performance</h2>
<div class="caveat">Note: Backtest uses ONLY astrological composite score for entry signals. No market data (price, volume, returns) is used in signal generation. This is the correct methodology.</div>
{bt_html}
<h3>Stress Test Results</h3>
{stress_html}

<h2>Section 10 — Bank Nifty Cross-Validation</h2>
<p class="section-note">Confirmed Nifty patterns tested on Bank Nifty — same sky, different instrument</p>
{df_to_html(bn_transfer[['features','condition','outcome','nifty_wr','bn_n','bn_wr','transfer']] if len(bn_transfer)>0 else pd.DataFrame())}

<h2>Section 11 — Honest Failure Analysis</h2>
<div class="card">
<h3>What Was Wrong in Version 1</h3>
<p>• ML model (LightGBM) AUC 0.517 — essentially random. Treating astrological columns as ML features without understanding interactions produced noise.</p>
<p>• Backtest win rate 83.6% was inflated by using same-day price data (log_ret, range_pct) as features. These cannot be known before market open.</p>
<p>• Forward calendar showed 252 NEUTRAL days because the signal depended on market data unknowable in advance.</p>
<h3>What Version 2 Does Differently</h3>
<p>• Six research methods run simultaneously on purely astrological features.</p>
<p>• Wilson CI lower bound used instead of raw win rates — punishes small samples.</p>
<p>• Benjamini-Hochberg FDR at 1% applied across all methods simultaneously.</p>
<p>• Out-of-sample validation strictly split at 2018 — patterns must survive in test period.</p>
<p>• Forward calendar uses only pyswisseph ephemeris — no market data in forward signals.</p>
<h3>Current Limitations</h3>
<p>• Even with correct methodology, {N_CONFIRMED} confirmed patterns is a small library. More research is needed.</p>
<p>• Vimshottari Dasha has only 3 Mahadasha periods in 30 years — statistically untestable at this level.</p>
<p>• Intraday structure (open vs close behaviour) not yet analyzed.</p>
<p>• Nakshatra × Saturn dignity combinations not fully explored.</p>
</div>

<p style="color:var(--muted);font-size:0.8em;margin-top:24px">
Generated by AstroQuant Research System v2. Data: NSE Nifty 50 1996-2026, BankNifty 2005-2026.
Astrological positions: Vedic/sidereal Lahiri ayanamsa via pyswisseph. Statistical validation: Wilson CI, BH-FDR 1%.
<br>Source: <a href="https://github.com/XP20225/nifty-planets">github.com/XP20225/nifty-planets</a>
</p>
</body>
</html>"""

with open(f"{REPO}/report.html", "w") as f:
    f.write(report_html)
print(f"report.html written")

# ═══════════════════════════════════════════════════════════════════════
# CALENDAR.HTML
# ═══════════════════════════════════════════════════════════════════════
if len(calendar) == 0:
    print("calendar.html: no calendar data found")
    exit(0)

calendar['date'] = pd.to_datetime(calendar['date'])
calendar['month'] = calendar['date'].dt.strftime('%Y-%m')

# Build calendar data as JSON for interactive rendering
def row_to_json(r):
    return {
        'date': str(r['date'])[:10],
        'class': str(r.get('classification','NEUTRAL')),
        'score': float(r.get('net_score',0)),
        'paksha': str(r.get('paksha','')),
        'tithi': int(r.get('tithi_num',0)) if pd.notna(r.get('tithi_num')) else 0,
        'nak': str(r.get('nak_mo_name','')),
        'yoga': str(r.get('yoga_name','')),
        'vara': str(r.get('vara_lord','')),
        'dig_ju': str(r.get('dig_Ju','')),
        'dig_sa': str(r.get('dig_Sa','')),
        'retro_me': int(r.get('retro_Me',0)) if pd.notna(r.get('retro_Me')) else 0,
        'sade_sati': int(r.get('sade_sati',0)) if pd.notna(r.get('sade_sati')) else 0,
        'maha': str(r.get('mahadasha','')),
        'bull_pats': str(r.get('active_bull_patterns','')),
        'bear_pats': str(r.get('active_bear_patterns','')),
        'n_bull': int(r.get('n_bull_patterns',0)) if pd.notna(r.get('n_bull_patterns')) else 0,
        'n_bear': int(r.get('n_bear_patterns',0)) if pd.notna(r.get('n_bear_patterns')) else 0,
    }

cal_json = json.dumps([row_to_json(r) for _, r in calendar.iterrows()], indent=2)

class_counts = calendar['classification'].value_counts().to_dict()

next_prime_bull = calendar[calendar['classification']=='PRIME_TRADE_BULL']['date'].min()
next_prime_bear = calendar[calendar['classification']=='PRIME_TRADE_BEAR']['date'].min()
next_prime_bull = str(next_prime_bull)[:10] if pd.notna(next_prime_bull) else 'None identified'
next_prime_bear = str(next_prime_bear)[:10] if pd.notna(next_prime_bear) else 'None identified'

calendar_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AstroQuant Calendar 2026-2027 — Nifty Planets</title>
<style>
:root{{--bg:#0a0e1a;--card:#111827;--accent:#3b82f6;--text:#e2e8f0;--muted:#94a3b8;--border:#1e293b;
  --prime-bull:#166534;--prime-bear:#7f1d1d;--watch-bull:#1e3a5f;--watch-bear:#3b1a1a;--neutral:#1e293b}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:13px;padding:16px}}
h1{{font-size:1.5em;color:var(--accent);margin-bottom:8px}}
.summary-row{{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 16px}}
.badge{{padding:4px 12px;border-radius:20px;font-size:0.85em;font-weight:600}}
.prime-bull{{background:#166534;color:#bbf7d0}}
.prime-bear{{background:#7f1d1d;color:#fecaca}}
.watch-bull{{background:#1e3a5f;color:#bfdbfe}}
.neutral-badge{{background:#1e293b;color:var(--muted)}}
.filters{{margin:8px 0;display:flex;gap:8px;flex-wrap:wrap}}
.filter-btn{{padding:4px 10px;border-radius:4px;border:1px solid var(--border);background:var(--card);color:var(--muted);cursor:pointer;font-size:0.8em}}
.filter-btn.active{{background:var(--accent);color:white;border-color:var(--accent)}}
.month-section{{margin-bottom:24px}}
.month-header{{color:var(--muted);font-size:0.9em;font-weight:600;margin:12px 0 6px;text-transform:uppercase;letter-spacing:1px}}
.day-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px}}
.day-card{{border-radius:6px;padding:8px;cursor:pointer;transition:transform 0.1s;border:1px solid transparent;position:relative}}
.day-card:hover{{transform:scale(1.02);border-color:var(--accent)}}
.day-card.PRIME_TRADE_BULL{{background:var(--prime-bull);border-color:#22c55e}}
.day-card.PRIME_TRADE_BEAR{{background:var(--prime-bear);border-color:#ef4444}}
.day-card.WATCH_BULL{{background:var(--watch-bull)}}
.day-card.WATCH_BEAR{{background:var(--watch-bear)}}
.day-card.NEUTRAL{{background:var(--neutral)}}
.day-card.ERROR{{background:#1a1a1a;opacity:0.5}}
.day-date{{font-weight:600;font-size:0.85em}}
.day-class{{font-size:0.75em;color:var(--muted);margin-top:2px}}
.day-nak{{font-size:0.75em;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.day-score{{font-size:0.7em;position:absolute;top:6px;right:6px;color:rgba(255,255,255,0.5)}}
.modal-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:100;align-items:center;justify-content:center}}
.modal-overlay.active{{display:flex}}
.modal{{background:#111827;border:1px solid var(--border);border-radius:12px;padding:24px;max-width:500px;width:90%;max-height:80vh;overflow-y:auto}}
.modal h2{{font-size:1.1em;color:var(--accent);margin-bottom:12px}}
.modal-row{{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--border)}}
.modal-row span{{color:var(--muted);font-size:0.85em}}
.modal-row b{{font-size:0.85em}}
.modal-close{{float:right;cursor:pointer;color:var(--muted);font-size:1.2em}}
.class-label{{font-size:0.8em;font-weight:700;display:inline-block;padding:2px 8px;border-radius:4px}}
.lbl-PRIME_TRADE_BULL{{background:#166534;color:#bbf7d0}}
.lbl-PRIME_TRADE_BEAR{{background:#7f1d1d;color:#fecaca}}
.lbl-WATCH_BULL{{background:#1e3a5f;color:#bfdbfe}}
.lbl-WATCH_BEAR{{background:#3b1a1a;color:#fca5a5}}
.lbl-NEUTRAL{{background:#1e293b;color:#94a3b8}}
</style>
</head>
<body>
<h1>🪐 AstroQuant Forward Calendar — 2026-2027</h1>
<p style="color:var(--muted);font-size:0.85em">Signals based on confirmed astrological patterns only. No market price data. Last updated: {TODAY}</p>

<div class="summary-row">
  <span class="badge prime-bull">PRIME BULL: {class_counts.get('PRIME_TRADE_BULL',0)} days</span>
  <span class="badge prime-bear">PRIME BEAR: {class_counts.get('PRIME_TRADE_BEAR',0)} days</span>
  <span class="badge watch-bull">WATCH: {class_counts.get('WATCH_BULL',0)+class_counts.get('WATCH_BEAR',0)} days</span>
  <span class="badge neutral-badge">NEUTRAL: {class_counts.get('NEUTRAL',0)} days</span>
</div>
<div class="summary-row">
  <span style="color:var(--muted);font-size:0.85em">Next PRIME BULL: <b style="color:#22c55e">{next_prime_bull}</b></span>
  <span style="color:var(--muted);font-size:0.85em">Next PRIME BEAR: <b style="color:#ef4444">{next_prime_bear}</b></span>
</div>

<div class="filters">
  <span style="color:var(--muted);font-size:0.85em;line-height:28px">Filter:</span>
  <button class="filter-btn active" onclick="filterClass('ALL')">All</button>
  <button class="filter-btn" onclick="filterClass('PRIME_TRADE_BULL')">Prime Bull</button>
  <button class="filter-btn" onclick="filterClass('PRIME_TRADE_BEAR')">Prime Bear</button>
  <button class="filter-btn" onclick="filterClass('WATCH_BULL')">Watch Bull</button>
  <button class="filter-btn" onclick="filterClass('WATCH_BEAR')">Watch Bear</button>
</div>

<div id="calendar-root"></div>

<div class="modal-overlay" id="modal">
  <div class="modal">
    <span class="modal-close" onclick="closeModal()">✕</span>
    <h2 id="modal-date"></h2>
    <div id="modal-body"></div>
  </div>
</div>

<script>
const DATA = {cal_json};
let activeFilter = 'ALL';

function filterClass(cls) {{
  activeFilter = cls;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  render();
}}

function render() {{
  const root = document.getElementById('calendar-root');
  root.innerHTML = '';
  // Group by month
  const months = {{}};
  DATA.forEach(d => {{
    if (activeFilter !== 'ALL' && d.class !== activeFilter) return;
    const m = d.date.slice(0,7);
    if (!months[m]) months[m] = [];
    months[m].push(d);
  }});
  Object.keys(months).sort().forEach(m => {{
    const sec = document.createElement('div');
    sec.className = 'month-section';
    const date = new Date(m+'-01');
    sec.innerHTML = `<div class="month-header">${{date.toLocaleString('en',{{month:'long',year:'numeric'}})}}</div>`;
    const grid = document.createElement('div');
    grid.className = 'day-grid';
    months[m].forEach(d => {{
      const card = document.createElement('div');
      card.className = `day-card ${{d.class}}`;
      card.innerHTML = `
        <div class="day-date">${{d.date.slice(5)}}</div>
        <div class="day-class">${{d.class.replace('_',' ').replace('_',' ')}}</div>
        <div class="day-nak">${{d.nak}} · ${{d.paksha[0]}}</div>
        <div class="day-score">${{d.score > 0 ? '+' : ''}}${{d.score.toFixed(2)}}</div>`;
      card.onclick = () => showModal(d);
      grid.appendChild(card);
    }});
    sec.appendChild(grid);
    root.appendChild(sec);
  }});
}}

function showModal(d) {{
  document.getElementById('modal-date').innerHTML =
    d.date + ' <span class="class-label lbl-'+d.class+'">'+d.class.replaceAll('_',' ')+'</span>';
  const vara_names = {{Su:'Sunday',Mo:'Monday',Ma:'Tuesday',Me:'Wednesday',Ju:'Thursday',Ve:'Friday',Sa:'Saturday'}};
  const rows = [
    ['Vara (Day)', vara_names[d.vara] || d.vara],
    ['Tithi', d.tithi + ' · ' + d.paksha],
    ['Nakshatra', d.nak],
    ['Panchanga Yoga', d.yoga],
    ['Jupiter Dignity', d.dig_ju],
    ['Saturn Dignity', d.dig_sa],
    ['Mercury Retro', d.retro_me ? 'YES ⚠️' : 'No'],
    ['Sade Sati', d.sade_sati ? 'ACTIVE ⚠️' : 'No'],
    ['Mahadasha', d.maha],
    ['Net Score', (d.score > 0 ? '+' : '') + d.score.toFixed(4)],
    ['Bull Patterns Active', d.n_bull + (d.bull_pats ? ': ' + d.bull_pats : '')],
    ['Bear Patterns Active', d.n_bear + (d.bear_pats ? ': ' + d.bear_pats : '')],
  ];
  document.getElementById('modal-body').innerHTML =
    rows.map(([k,v]) => `<div class="modal-row"><span>${{k}}</span><b>${{v}}</b></div>`).join('');
  document.getElementById('modal').classList.add('active');
}}

function closeModal() {{
  document.getElementById('modal').classList.remove('active');
}}
document.getElementById('modal').addEventListener('click', e => {{
  if (e.target.id === 'modal') closeModal();
}});

render();
</script>
</body>
</html>"""

with open(f"{REPO}/calendar.html", "w") as f:
    f.write(calendar_html)
print(f"calendar.html written")
print(f"\nStep 5 complete: report.html and calendar.html generated")
