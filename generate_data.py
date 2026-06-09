"""
Nifty 50 + Planetary Positions Generator
-----------------------------------------
Fetches Nifty 50 OHLC data and calculates tropical planetary longitudes
plus ayanamsa values for five systems (Lahiri, KP, Raman, Yukteshwar,
Vakkiyam). Sign/nakshatra derivation happens client-side in index.html.

Data sources (merged in order, duplicates removed):
  1. nse_historical.csv   — manually downloaded from NSE India (1996–2007)
  2. Yahoo Finance ^NSEI  — automatic download (2007–present)

How to get the NSE CSV for 1996–2007:
  1. Go to https://www.nseindia.com/reports-detail?type=equity_research
  2. Select Index → NIFTY 50, Date range: 01-01-1996 to 17-09-2007
  3. Download CSV → save as nse_historical.csv in this folder
  4. Run: python generate_data.py

Without nse_historical.csv the script still works — data starts from 2007.

Usage:
    pip install -r requirements.txt
    python generate_data.py
"""

import json, math, sys, os
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

# ── Load NSE CSV (optional, for pre-2007 data) ────────────────────────────────

def load_nse_csv(path):
    """
    NSE historical CSV columns (typical format):
      Date, Open, High, Low, Close, Shares Traded, Turnover (Rs. Cr)
    """
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if 'date' in cl:               col_map['Date']  = c
            elif cl == 'open':             col_map['Open']  = c
            elif cl == 'high':             col_map['High']  = c
            elif cl == 'low':              col_map['Low']   = c
            elif cl in ('close','closing'):col_map['Close'] = c

        for k in ('Date','Open','High','Low','Close'):
            if k not in col_map:
                raise ValueError(f"Column '{k}' not found. Columns: {list(df.columns)}")

        df = df.rename(columns={v: k for k, v in col_map.items()})
        df = df[['Date','Open','High','Low','Close']].copy()

        for fmt in ('%d-%b-%Y','%d-%m-%Y','%Y-%m-%d','%m/%d/%Y'):
            try:
                df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                break
            except Exception:
                continue
        else:
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

        for col in ('Open','High','Low','Close'):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')

        df = df.dropna(subset=['Close'])
        df = df.set_index('Date').sort_index()
        print(f"  NSE CSV: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
        return df
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return pd.DataFrame()

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

    # 2. NSE CSV (optional)
    nse_path = 'nse_historical.csv'
    nse_df = pd.DataFrame()
    if os.path.exists(nse_path):
        print(f"Loading {nse_path}...")
        nse_df = load_nse_csv(nse_path)
    else:
        print(f"  (No {nse_path} found — data starts from {yf_df.index[0].date()})")
        print(f"  To get 1996–2007 data: download from NSE India → save as {nse_path}")

    # 3. Merge — Yahoo Finance takes priority for overlapping dates
    if not nse_df.empty:
        nse_df.index = pd.to_datetime(nse_df.index).tz_localize(None)
        nse_only = nse_df[~nse_df.index.isin(yf_df.index)]
        combined = pd.concat([nse_only, yf_df]).sort_index()
    else:
        combined = yf_df

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
    if not os.path.exists(nse_path):
        print(f"\n{'─'*60}")
        print("To include 1996–2007 data:")
        print("  1. Visit https://www.nseindia.com/reports-detail?type=equity_research")
        print("  2. Select NIFTY 50, date 01-01-1996 to 17-09-2007 → Download")
        print(f"  3. Save as '{nse_path}' in this folder")
        print("  4. Re-run: python generate_data.py")

if __name__ == '__main__':
    main()
