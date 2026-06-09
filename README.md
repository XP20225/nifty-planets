# Nifty 50 · Planetary Positions

A standalone web page showing every Nifty 50 trading day since **April 22, 1996** alongside the sidereal positions of all 9 planets (KP / Lahiri / Raman / Yukteshwar / Vakkiyam ayanamsas). No framework, no build step — a single `index.html` file you can open in any browser.

---

## Live Access

Once deployed (see [Deploy](#deploy) below), the page is publicly shareable via a URL — no login, no app install required.

To run it **locally right now:**

```bash
cd nifty-planets
python -m http.server 8765
# open http://localhost:8765
```

> `index.html` fetches `data.json` via a relative path, so it must be served over HTTP (not opened as a `file://` URL).

---

## Features

| Feature | Details |
|---|---|
| **Date range** | Apr 22, 1996 → present (~7,450 trading days) |
| **OHLC data** | 1996–2007 from NSE India (via `nselib`); 2007–present from Yahoo Finance |
| **Planets** | Sun · Moon · Mercury · Venus · Mars · Jupiter · Saturn · Rahu · Ketu |
| **Ayanamsa selector** | Lahiri (default) · KP · Raman · Yukteshwar · Vakkiyam — switch instantly, no reload |
| **Tamil / English** | Toggle all labels, planet names, sign and nakshatra names to Tamil script |
| **Sortable columns** | Click any column header to sort ascending / descending |
| **Date range filter** | Filter rows to any date window |
| **Sign / Nak search** | Free-text search across all planets' sign and nakshatra |
| **CSV export** | Downloads filtered data with the active ayanamsa in the filename |
| **Pagination** | 100 rows per page with page range navigation |

---

## Deploy

### GitHub Pages (free, recommended)

1. Create a new repo on GitHub (e.g. `nifty-planets`)
2. Push this repo:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/nifty-planets.git
   git push -u origin master
   ```
3. Go to **Settings → Pages → Deploy from branch → `master` / `root`**
4. GitHub will publish the page at:
   ```
   https://YOUR_USERNAME.github.io/nifty-planets/
   ```

> `data.json` is in `.gitignore` — you need to either commit it once (remove it from `.gitignore`) or serve it separately. The simplest option is to remove the line `data.json` from `.gitignore`, then `git add data.json && git push`.

### Netlify Drop (instant, no account needed)

1. Generate `data.json` (see below)
2. Drag the entire `nifty-planets/` folder onto **drop.netlify.com**
3. Get a shareable URL immediately

---

## Regenerate Data

Data is pre-generated into `data.json`. To refresh it (e.g. to add recent trading days):

```bash
# Install dependencies once
pip install -r requirements.txt

# Regenerate
python generate_data.py
```

This fetches:
- **1996–2007** — OHLC from NSE India via `nselib` (fetched month by month; takes ~3 min)
- **2007–present** — OHLC from Yahoo Finance (`^NSEI`)

Planets are calculated using Swiss Ephemeris at 9:15 AM IST (market open). Five ayanamsa values are stored per day so the browser can switch between them instantly without a reload.

### Requirements

```
yfinance>=0.2.40
pyswisseph>=2.10.3.2
nselib>=2.5.1
pandas
```

---

## Data Notes

| Period | Source | OHLC |
|---|---|---|
| Apr 1996 – Sep 2007 | NSE India (`nselib`) | Full O/H/L/C |
| Sep 2007 – present | Yahoo Finance (`^NSEI`) | Full O/H/L/C |

- **Ayanamsa systems**: Lahiri, KP (Krishnamurti), B.V. Raman, Sri Yukteshwar, Vakkiyam (Swiss Ephemeris `SIDM_SURYASIDDHANTA` — closest standard approximation to traditional Tamil Vakkiyam panchangam)
- **Planet calculation time**: 9:15 AM IST = 3:45 AM UTC = Julian Day fraction 3.75h
- **Rahu**: True Node; **Ketu**: Rahu + 180°
