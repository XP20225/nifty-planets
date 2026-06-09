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

import json, math, sys, os, time
from datetime import datetime, date

try:
    import pandas as pd
    import yfinance as yf
    import swisseph as swe
    from nselib import capital_market
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

# ── Fetch pre-2007 OHLC from NSE India via nselib ────────────────────────────

def fetch_nse_history(start_year=1994, end_year=2007):
    """
    Fetches full OHLC for NIFTY 50 from NSE India month by month.
    nselib caps each response at 70 rows, so yearly chunks miss data.
    """
    import calendar
    dfs = []
    total_months = (end_year - start_year + 1) * 12
    fetched = 0

    for year in range(start_year, end_year + 1):
        year_rows = 0
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            from_d = f'{1:02d}-{month:02d}-{year}'
            to_d   = f'{last_day:02d}-{month:02d}-{year}'
            try:
                df = capital_market.index_data(index='NIFTY 50', from_date=from_d, to_date=to_d)
                if df is not None and len(df):
                    dfs.append(df)
                    year_rows += len(df)
            except Exception as e:
                print(f"  {year}-{month:02d}: error — {e}")
            time.sleep(0.15)
        fetched += 1
        print(f"  {year}: {year_rows} rows  [{fetched}/{end_year - start_year + 1} years]")

    if not dfs:
        return pd.DataFrame()

    raw = pd.concat(dfs, ignore_index=True)
    raw['Date'] = pd.to_datetime(raw['TIMESTAMP'], format='%d-%b-%Y', errors='coerce')
    raw = raw.dropna(subset=['Date'])
    raw = raw.rename(columns={
        'OPEN_INDEX_VAL':  'Open',
        'HIGH_INDEX_VAL':  'High',
        'LOW_INDEX_VAL':   'Low',
        'CLOSE_INDEX_VAL': 'Close',
    })
    raw = raw[['Date','Open','High','Low','Close']].drop_duplicates('Date').sort_values('Date').set_index('Date')
    return raw

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

    # 2. NSE India — full OHLC from 1994 up to the day before Yahoo starts
    yf_start = yf_df.index[0].date()
    nse_end_year = yf_start.year  # fetch NSE up to and including that year; Yahoo takes over after
    print(f"\nFetching NSE India OHLC history (1994–{nse_end_year})...")
    nse_df = fetch_nse_history(start_year=1994, end_year=nse_end_year)

    # 3. Merge — Yahoo Finance takes priority for overlapping dates (its OHLC is more precise)
    if not nse_df.empty:
        nse_df.index = pd.to_datetime(nse_df.index).tz_localize(None)
        nse_only = nse_df[~nse_df.index.isin(yf_df.index)]
        combined = pd.concat([nse_only, yf_df]).sort_index()
        print(f"\nNSE-only rows (pre-Yahoo): {len(nse_only)}")
    else:
        print("  NSE fetch failed — using Yahoo Finance only")
        combined = yf_df

    combined['Change_pct'] = combined['Close'].pct_change() * 100
    print(f"Total trading days: {len(combined)}")
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
