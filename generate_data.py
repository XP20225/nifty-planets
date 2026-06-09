"""
Nifty 50 + Planetary Positions Generator
-----------------------------------------
Fetches Nifty 50 OHLC data and calculates tropical planetary longitudes
plus ayanamsa values for five systems (Lahiri, KP, Raman, Yukteshwar,
Vakkiyam). Sign/nakshatra derivation happens client-side in index.html.

Data sources (merged):
  1. nse_data/   — local CSV files (1995–2018, Close only; Open=High=Low=Close)
  2. Yahoo Finance ^NSEI — automatic download (2007–present, full OHLC)
  Yahoo Finance takes priority for overlapping dates (has full OHLC).

Usage:
    pip install -r requirements.txt
    python generate_data.py
"""

import json, math, sys, os, glob
from datetime import datetime

try:
    import pandas as pd
    import yfinance as yf
    import swisseph as swe
except ImportError:
    sys.exit("Run: pip install -r requirements.txt")

# ── Planet keys (short to keep JSON small) ───────────────────────────────────
PLANET_IDS = [
    ('Su', swe.SUN),
    ('Mo', swe.MOON),
    ('Me', swe.MERCURY),
    ('Ve', swe.VENUS),
    ('Ma', swe.MARS),
    ('Ju', swe.JUPITER),
    ('Sa', swe.SATURN),
    ('Ra', swe.TRUE_NODE),
    # Ke = Rahu + 180, computed below
]

# Ayanamsa modes. 'va' uses Suryasiddhanta as the closest Swiss Ephem
# approximation to traditional Tamil Vakkiyam panchangam.
AYANAMSA_MODES = [
    ('la', swe.SIDM_LAHIRI),           # Lahiri / Chitrapaksha (default)
    ('kp', swe.SIDM_KRISHNAMURTI),     # KP (Krishnamurti Paddhati)
    ('ra', swe.SIDM_RAMAN),            # B.V. Raman
    ('yu', swe.SIDM_YUKTESHWAR),       # Sri Yukteshwar
    ('va', swe.SIDM_SURYASIDDHANTA),   # Vakkiyam (Suryasiddhanta approx.)
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def norm(d):
    return ((d % 360) + 360) % 360

def calc_record(jd):
    """Return (planets_dict, ayanamsas_dict) for the given Julian Day."""
    # Tropical planet longitudes (no sidereal flag)
    p = {}
    rahu_lon = None
    for key, pid in PLANET_IDS:
        xx, _ = swe.calc_ut(jd, pid, swe.FLG_SPEED)
        lon = norm(xx[0])
        if key == 'Ra':
            rahu_lon = lon
        p[key] = round(lon, 4)
    p['Ke'] = round(norm(rahu_lon + 180), 4)

    # Ayanamsa values for each system
    ayan = {}
    for key, mode in AYANAMSA_MODES:
        swe.set_sid_mode(mode)
        ayan[key] = round(swe.get_ayanamsa_ut(jd), 4)

    return p, ayan

def date_to_jd(d):
    return swe.julday(d.year, d.month, d.day, 3.75)  # 9:15 AM IST = 3.75h UTC

# ── Load NSE CSV files from nse_data/ folder ─────────────────────────────────

def load_nse_csvs(folder='nse_data'):
    """
    Loads all CSV files from nse_data/.
    Format (no header): Date, Close, Returns
    Date format: DD-Mon-YY  (e.g. 03-Apr-95)
    Only Close is available; Open/High/Low are set equal to Close.
    """
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, header=None, names=['Date','Close','chg'])
            df = df[df['chg'] != 'returns']   # drop the label row present in each file
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined['Date']  = pd.to_datetime(combined['Date'], format='%d-%b-%y', dayfirst=True)
    combined['Close'] = pd.to_numeric(combined['Close'], errors='coerce')
    combined = combined.dropna(subset=['Close'])
    combined = combined.drop_duplicates('Date').sort_values('Date').set_index('Date')

    # No OHLC in these files — set Open/High/Low = Close
    combined['Open'] = combined['Close']
    combined['High'] = combined['Close']
    combined['Low']  = combined['Close']

    result = combined[['Open','High','Low','Close']]
    print(f"  NSE CSVs: {len(result)} rows ({result.index[0].date()} → {result.index[-1].date()})")
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Yahoo Finance
    print("Fetching Nifty 50 from Yahoo Finance...")
    yf_df = yf.download('^NSEI', start='1990-01-01', auto_adjust=True, progress=True)
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = [col[0] for col in yf_df.columns]
    yf_df = yf_df[['Open','High','Low','Close']].dropna(subset=['Close'])
    yf_df.index = pd.to_datetime(yf_df.index).tz_localize(None)
    print(f"  Yahoo: {len(yf_df)} rows ({yf_df.index[0].date()} → {yf_df.index[-1].date()})")

    # 2. NSE CSV files from nse_data/
    print("Loading NSE CSV files from nse_data/...")
    nse_df = load_nse_csvs('nse_data')
    if nse_df.empty:
        print("  (No nse_data/ CSVs found — data starts from 2007)")

    # 3. Merge — Yahoo Finance takes priority for overlapping dates (full OHLC)
    if not nse_df.empty:
        nse_df.index = pd.to_datetime(nse_df.index).tz_localize(None)
        nse_only = nse_df[~nse_df.index.isin(yf_df.index)]
        combined = pd.concat([nse_only, yf_df]).sort_index()
    else:
        combined = yf_df

    # Recompute Change% across full merged series so 1995→1996 transitions are correct
    combined['Change_pct'] = combined['Close'].pct_change() * 100
    print(f"\nTotal trading days: {len(combined)}")
    print(f"Range: {combined.index[0].date()} → {combined.index[-1].date()}")

    # 4. Calculate tropical longitudes + ayanamsas
    print("\nCalculating planetary data...")
    records = []
    total = len(combined)

    for i, (idx, row) in enumerate(combined.iterrows()):
        if i % 500 == 0:
            print(f"  {i}/{total} ({i/total*100:.0f}%)...")

        d   = idx.date()
        jd  = date_to_jd(d)
        chg = float(row['Change_pct'])

        try:
            planets, ayan = calc_record(jd)
        except Exception as e:
            print(f"  Warning: planet calc failed for {d}: {e}")
            planets, ayan = {}, {}

        records.append({
            'date':  d.strftime('%Y-%m-%d'),
            'open':  round(float(row['Open']),  2),
            'high':  round(float(row['High']),  2),
            'low':   round(float(row['Low']),   2),
            'close': round(float(row['Close']), 2),
            'chg':   round(chg, 2) if not math.isnan(chg) else 0.0,
            'p':     planets,
            'ayan':  ayan,
        })

    # 5. Write output
    out = 'data.json'
    print(f"\nWriting {len(records)} records to {out}...")
    with open(out, 'w') as f:
        json.dump({'records': records, 'generated': datetime.now().isoformat()[:10]}, f, separators=(',',':'))

    size_mb = os.path.getsize(out) / 1_000_000
    print(f"Done!  {out}  ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
