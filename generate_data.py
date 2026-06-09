"""
Nifty 50 + Planetary Positions Generator
-----------------------------------------
Fetches Nifty 50 OHLC data from Yahoo Finance and calculates
sidereal (KP/Krishnamurti) planetary positions for each trading day.

Output: data.json  (loaded by index.html)

Usage:
    pip install -r requirements.txt
    python generate_data.py
"""

import json
import math
import sys
from datetime import datetime, timezone

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing yfinance. Run: pip install -r requirements.txt")

try:
    import swisseph as swe
except ImportError:
    sys.exit("Missing pyswisseph. Run: pip install -r requirements.txt")

# ── Constants ─────────────────────────────────────────────────────────────────

SIGNS = [
    'Aries','Taurus','Gemini','Cancer','Leo','Virgo',
    'Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces',
]
SIGN_ABBR = [
    'Ari','Tau','Gem','Can','Leo','Vir',
    'Lib','Sco','Sag','Cap','Aqu','Pis',
]
NAKSHATRAS = [
    'Ashwini','Bharani','Krittika','Rohini','Mrigashira','Ardra',
    'Punarvasu','Pushya','Ashlesha','Magha','Purva Phalguni','Uttara Phalguni',
    'Hasta','Chitra','Swati','Vishakha','Anuradha','Jyeshtha',
    'Mula','Purva Ashadha','Uttara Ashadha','Shravana','Dhanishtha','Shatabhisha',
    'Purva Bhadrapada','Uttara Bhadrapada','Revati',
]
NAK_ABBR = [
    'Ashw','Bhar','Krit','Rohi','Mrig','Ardr',
    'Puna','Push','Ashl','Magh','PPha','UPha',
    'Hast','Chit','Swat','Vish','Anur','Jyes',
    'Mula','PAsh','UAsh','Shra','Dhan','Shat',
    'PBha','UBha','Reva',
]

PLANETS = [
    ('Sun',     swe.SUN),
    ('Moon',    swe.MOON),
    ('Mercury', swe.MERCURY),
    ('Venus',   swe.VENUS),
    ('Mars',    swe.MARS),
    ('Jupiter', swe.JUPITER),
    ('Saturn',  swe.SATURN),
    ('Rahu',    swe.TRUE_NODE),
    ('Ketu',    None),           # Ketu = Rahu + 180°
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def norm(d):
    return ((d % 360) + 360) % 360

def lon_info(lon):
    lon = norm(lon)
    sign_idx = int(lon / 30)
    within   = lon - sign_idx * 30
    deg      = int(within)
    minutes  = int((within - deg) * 60)
    nak_idx  = int(lon / (360 / 27))
    pada     = int((lon % (360 / 27)) / (360 / 108)) + 1
    return {
        'lon':  round(lon, 2),
        'deg':  f"{deg}°{minutes:02d}'",
        'sign': SIGN_ABBR[sign_idx],
        'nak':  NAK_ABBR[min(nak_idx, 26)],
        'pada': min(pada, 4),
    }

def calc_planets(jd):
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)  # KP ayanamsa
    result = {}
    rahu_lon = None
    for name, pid in PLANETS:
        if name == 'Ketu':
            lon = norm(rahu_lon + 180)
            result[name] = lon_info(lon)
        else:
            xx, _ = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL | swe.FLG_SPEED)
            lon = norm(xx[0])
            if name == 'Rahu':
                rahu_lon = lon
            result[name] = lon_info(lon)
    return result

def date_to_jd(d):
    # NSE opens at 9:15 AM IST = 3:45 AM UTC
    return swe.julday(d.year, d.month, d.day, 3.75)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching Nifty 50 data from Yahoo Finance...")
    df = yf.download('^NSEI', start='1990-01-01', auto_adjust=True, progress=True)
    df = df.dropna(subset=['Close'])
    df['Change_pct'] = df['Close'].pct_change() * 100
    print(f"  Got {len(df)} trading days")

    records = []
    total = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 250 == 0:
            pct = i / total * 100
            print(f"  Calculating planets: {i}/{total} ({pct:.0f}%)...")

        d   = idx.date() if hasattr(idx, 'date') else idx
        jd  = date_to_jd(d)
        chg = float(row['Change_pct'])

        try:
            planets = calc_planets(jd)
        except Exception as e:
            print(f"  Warning: planet calc failed for {d}: {e}")
            planets = {}

        records.append({
            'date':   d.strftime('%Y-%m-%d'),
            'open':   round(float(row['Open']),  2),
            'high':   round(float(row['High']),  2),
            'low':    round(float(row['Low']),   2),
            'close':  round(float(row['Close']), 2),
            'chg':    round(chg, 2) if not math.isnan(chg) else 0.0,
            'p':      planets,
        })

    out_path = 'data.json'
    print(f"Writing {len(records)} records to {out_path}...")
    with open(out_path, 'w') as f:
        json.dump({'records': records}, f, separators=(',', ':'))

    size_mb = len(open(out_path).read()) / 1_000_000
    print(f"Done! {out_path} is {size_mb:.1f} MB")
    print(f"\nNext step: open index.html in a browser (or deploy with data.json alongside it)")

if __name__ == '__main__':
    main()
