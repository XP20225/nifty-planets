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

# ── Surya Siddhanta (Vakkiya) Engine ─────────────────────────────────────────
# Ported from truly-kp/src/engine/vakkiyaEngine.js
# Source: Surya Siddhanta revolution constants — Burgess translation 1859.
# Vakyakarana text: T.S. Kuppanna Sastri & K.V. Sarma 1962.
# Kali epoch: Feb 18 3102 BCE midnight (JDN 588465.5).

_CIVIL_DAYS    = 1577917828   # civil days in a Mahayuga (4,320,000 years)
_KALI_EPOCH_JD = 588465.5     # Julian Day of Kali epoch

_REV = {
    'Sun':        4320000,
    'Moon':      57753336,
    'Mars':       2296832,
    'Mercury':   17937060,
    'Jupiter':    364220,
    'Venus':      7022376,
    'Saturn':     146568,
    'MoonApogee': 488203,
    'Rahu':       232238,  # retrograde
}

_APOGEE = {
    'Sun':      77.33,
    'Mars':    130.0,
    'Mercury': 220.0,
    'Jupiter': 171.0,
    'Venus':    90.0,
    'Saturn':  236.0,
}

_MANDA_EPI = {
    'Sun':      13.67,
    'Moon':     31.83,
    'Mars':     23.33,
    'Mercury':  28.17,
    'Jupiter':  18.33,
    'Venus':    11.67,
    'Saturn':   18.33,
}

_SHEEGRA_EPI = {
    'Mars':    233.33,
    'Mercury': 133.33,
    'Jupiter':  70.0,
    'Venus':   261.67,
    'Saturn':   39.33,
}

def _norm(d):
    return ((d % 360) + 360) % 360

def _mean_lon(ahargana, rev):
    # Python integers are arbitrary precision — no BigInt needed
    int_ah  = int(ahargana)
    frac_ah = ahargana - int_ah
    int_mod = (int_ah * rev) % _CIVIL_DAYS
    return _norm(int_mod / _CIVIL_DAYS * 360 + frac_ah * rev / _CIVIL_DAYS * 360)

def _mandaphala(anomaly, epicycle):
    r = anomaly * math.pi / 180
    return math.asin(math.sin(r) * epicycle / 360) * 180 / math.pi

def _sheegraphala(anomaly, epicycle):
    r = anomaly * math.pi / 180
    return math.atan2(math.sin(r) * epicycle, 360 + math.cos(r) * epicycle) * 180 / math.pi

def _vak_planet(ah, rev, apogee, manda_epi, sheegra_epi, sun_lon):
    """4-step iterative correction for the five exterior/interior planets."""
    m  = _mean_lon(ah, rev)
    L1 = _norm(m  + _sheegraphala(_norm(sun_lon - m),  sheegra_epi) / 2)
    L2 = _norm(m  + _mandaphala(_norm(L1 - apogee),    manda_epi)   / 2)
    L3 = _norm(m  + _mandaphala(_norm(L2 - apogee),    manda_epi))
    return _norm(L3 + _sheegraphala(_norm(sun_lon - L3), sheegra_epi))

def _vak_sun_lon(jd):
    """Surya Siddhanta Sun longitude (sidereal) for a given Julian Day."""
    ah  = jd - _KALI_EPOCH_JD
    m   = _mean_lon(ah, _REV['Sun'])
    return _norm(m + _mandaphala(_norm(m - _APOGEE['Sun']), _MANDA_EPI['Sun']))

# ─────────────────────────────────────────────────────────────────────────────

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

# Swiss Ephemeris ayanamsa modes. Vakkiyam ('va') is handled separately below.
AYANAMSA_MODES = [
    ('la', swe.SIDM_LAHIRI),       # Lahiri / Chitrapaksha (default)
    ('kp', swe.SIDM_KRISHNAMURTI), # KP (Krishnamurti Paddhati)
    ('ra', swe.SIDM_RAMAN),        # B.V. Raman
    ('yu', swe.SIDM_YUKTESHWAR),   # Sri Yukteshwar
    ('pu', swe.SIDM_TRUE_PUSHYA),  # True Pushya (Pushya Paksha)
    # 'va' (Vakkiyam) is computed directly via the Surya Siddhanta engine below
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def norm(d):
    return ((d % 360) + 360) % 360

def calc_record(jd):
    """Return (planets_dict, ayanamsas_dict) for the given Julian Day."""
    # Tropical planet longitudes (no sidereal flag)
    p = {}
    rahu_lon = None
    trop_sun = None
    for key, pid in PLANET_IDS:
        xx, _ = swe.calc_ut(jd, pid, swe.FLG_SPEED)
        lon = norm(xx[0])
        if key == 'Su':
            trop_sun = lon
        if key == 'Ra':
            rahu_lon = lon
        p[key] = round(lon, 4)
    p['Ke'] = round(norm(rahu_lon + 180), 4)

    # Ayanamsa: Lahiri, KP, Raman, Yukteshwar via Swiss Ephemeris
    ayan = {}
    for key, mode in AYANAMSA_MODES:
        swe.set_sid_mode(mode)
        ayan[key] = round(swe.get_ayanamsa_ut(jd), 4)

    # Vakkiyam: authentic Surya Siddhanta engine (tropical Sun − SS Sun)
    ayan['va'] = round(norm(trop_sun - _vak_sun_lon(jd)), 4)

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

    # 2. NSE India — full OHLC from April 1996 up to the day before Yahoo starts
    yf_start = yf_df.index[0].date()
    nse_end_year = yf_start.year
    print(f"\nFetching NSE India OHLC history (1996–{nse_end_year})...")
    nse_df = fetch_nse_history(start_year=1996, end_year=nse_end_year)

    # 3. Merge — Yahoo Finance takes priority for overlapping dates (its OHLC is more precise)
    if not nse_df.empty:
        nse_df.index = pd.to_datetime(nse_df.index).tz_localize(None)
        nse_only = nse_df[~nse_df.index.isin(yf_df.index)]
        combined = pd.concat([nse_only, yf_df]).sort_index()
        print(f"\nNSE-only rows (pre-Yahoo): {len(nse_only)}")
    else:
        print("  NSE fetch failed — using Yahoo Finance only")
        combined = yf_df

    combined = combined[combined.index >= pd.Timestamp('1996-04-22')]
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
