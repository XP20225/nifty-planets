"""
Fetch OHLC data for any NSE index and write a lightweight JSON file.
Only stores date, open, high, low, close, chg — no planetary data.

For each product, tries NSE India (via nselib) for pre-Yahoo data,
then Yahoo Finance for the rest. Yahoo takes priority on overlap.

Usage:
    python fetch_ohlc.py BANKNIFTY   ->  ohlc_banknifty.json
    python fetch_ohlc.py NIFTYIT     ->  ohlc_niftyit.json
"""

import json, math, sys, os, time, calendar
from datetime import datetime, date

try:
    import pandas as pd
    import yfinance as yf
    from nselib import capital_market
except ImportError:
    sys.exit("Run: pip install yfinance pandas nselib")

# Yahoo Finance ticker, NSE index name, inception date
PRODUCTS = {
    'BANKNIFTY':   ('^NSEBANK',  'NIFTY BANK',       date(2003, 9, 15)),
    'NIFTYIT':     ('^CNXIT',    'NIFTY IT',          date(1996, 1,  1)),
    'NIFTYMIDCAP': ('^NSMIDCP',  'NIFTY MIDCAP 100',  date(2001, 1,  1)),
}

def fetch_nse(index_name, start_year, end_year):
    """Fetch OHLC from NSE India month by month via nselib."""
    dfs = []
    for year in range(start_year, end_year + 1):
        year_rows = 0
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            from_d = f'{1:02d}-{month:02d}-{year}'
            to_d   = f'{last_day:02d}-{month:02d}-{year}'
            try:
                df = capital_market.index_data(index=index_name, from_date=from_d, to_date=to_d)
                if df is not None and len(df):
                    dfs.append(df)
                    year_rows += len(df)
            except Exception as e:
                print(f"  {year}-{month:02d}: {e}")
            time.sleep(0.15)
        print(f"  NSE {year}: {year_rows} rows")

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

def fetch(product):
    product = product.upper()
    if product not in PRODUCTS:
        sys.exit(f"Unknown product '{product}'. Known: {', '.join(PRODUCTS)}")

    ticker, nse_name, inception = PRODUCTS[product]

    # 1. Yahoo Finance
    print(f"\nFetching {product} from Yahoo Finance ({ticker})...")
    yf_df = yf.download(ticker, start='1994-01-01', auto_adjust=True, progress=True)
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = [col[0] for col in yf_df.columns]
    yf_df = yf_df[['Open','High','Low','Close']].dropna(subset=['Close'])
    yf_df.index = pd.to_datetime(yf_df.index).tz_localize(None)
    print(f"  Yahoo: {len(yf_df)} rows  ({yf_df.index[0].date()} → {yf_df.index[-1].date()})")

    # 2. NSE India for pre-Yahoo period (from inception up to Yahoo's start year)
    yf_start = yf_df.index[0].date()
    if inception < yf_start:
        print(f"\nFetching pre-Yahoo history from NSE India ({inception.year}–{yf_start.year})...")
        nse_df = fetch_nse(nse_name, inception.year, yf_start.year)
        if not nse_df.empty:
            nse_df.index = pd.to_datetime(nse_df.index).tz_localize(None)
            nse_only = nse_df[~nse_df.index.isin(yf_df.index)]
            combined = pd.concat([nse_only, yf_df]).sort_index()
            print(f"  NSE-only rows added: {len(nse_only)}")
        else:
            print("  NSE fetch returned nothing — using Yahoo only")
            combined = yf_df
    else:
        combined = yf_df

    # Trim to inception date, drop rows with any NaN OHLC, then compute chg
    combined = combined[combined.index >= pd.Timestamp(inception)]
    combined = combined.dropna(subset=['Open','High','Low','Close'])
    combined['Change_pct'] = combined['Close'].pct_change() * 100
    print(f"\nTotal: {len(combined)} rows  ({combined.index[0].date()} → {combined.index[-1].date()})")

    # Build output dict keyed by date string
    records = {}
    for idx, row in combined.iterrows():
        chg = float(row['Change_pct'])
        if math.isnan(chg) or math.isinf(chg):
            chg = 0.0
        records[idx.strftime('%Y-%m-%d')] = {
            'o': round(float(row['Open']),  2),
            'h': round(float(row['High']),  2),
            'l': round(float(row['Low']),   2),
            'c': round(float(row['Close']), 2),
            'g': round(chg, 2),
        }

    out = f"ohlc_{product.lower()}.json"
    with open(out, 'w') as f:
        json.dump({'product': product, 'generated': datetime.now().isoformat()[:10],
                   'records': records}, f, separators=(',',':'))
    size_kb = os.path.getsize(out) / 1000
    print(f"Written: {out}  ({size_kb:.0f} KB)")

if __name__ == '__main__':
    product = sys.argv[1] if len(sys.argv) > 1 else 'BANKNIFTY'
    fetch(product)
