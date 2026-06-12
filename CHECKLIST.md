# Astro Quant Research System — Live Checklist
Last updated: 2026-06-13 IST
Overall status: 9 of 54 items complete

---

## STEP 0: Repo Understanding
- [x] DONE — 2026-06-13 — All repo files listed recursively (15 files + nse_data/ CSVs)
- [x] DONE — 2026-06-13 — HTML generation code: index.html (single-file app, client-side rendering)
- [x] DONE — 2026-06-13 — Market data source: data.json (Nifty50 OHLC + planets), ohlc_banknifty.json (BankNifty OHLC)
- [x] DONE — 2026-06-13 — Astrological data: computed by generate_data.py via swisseph, tropical longitudes stored; sidereal = tropical - ayanamsa
- [x] DONE — 2026-06-13 — Full schema printed: date/open/high/low/close/chg/p{Su,Mo,Me,Ve,Ma,Ju,Sa,Ra,Ke}/ayan{la,kp,ra,yu,pu,va}
- [x] DONE — 2026-06-13 — Date ranges confirmed: Nifty50 1996-04-22→2026-06-11 (7,452 days), BankNifty 2005-06-09→2026-06-11 (5,177 days)
- [x] DONE — 2026-06-13 — All 9 planet columns confirmed present: Su Mo Me Ve Ma Ju Sa Ra Ke, zero missing
- [x] DONE — 2026-06-13 — Nakshatra/Karana/Yogi NOT stored in JSON — must be derived in Step 1 from Su/Mo + ayanamsa (formulas confirmed from index.html)
- [x] DONE — 2026-06-13 — Data quality report saved to reports/00_data_quality.txt — PASS: 0 missing, 0 out-of-range, 0 duplicates

## STEP 1: Feature Engineering
- [x] DONE — 2026-06-13 — Planetary tropical longitudes parsed to float64, sidereal = (tropical - ayan_la + 360) % 360
- [x] DONE — 2026-06-13 — Forward returns verified zero lookahead: fwd_ret/dir 1/2/3/5/10d all NaN in final N rows — PASS
- [x] DONE — 2026-06-13 — All Vedic features computed: Tithi, Paksha, Karana, Yogi/Avayogi, Nakshatra+lord+quality, Dasha, Ashtakavarga, Jupiter/Saturn dignity, Moon strength, Gandanta, Eclipse corridor, Graha Yuddha, 15 interaction features
- [x] DONE — 2026-06-13 — data/nifty_enriched.csv (7452 × 187) and data/banknifty_enriched.csv (5177 × 187) saved
- [x] DONE — 2026-06-13 — 187 columns confirmed, nulls only in warm-up rows (SMA200 needs 200 rows etc.) — all expected

## STEP 2: Discovery Loop
- [ ] PENDING — Unconditional signal scan complete
- [ ] PENDING — Conditional scan (Bull / Bear / Transitional) complete
- [ ] PENDING — Extreme day pattern analysis complete
- [ ] PENDING — Cosmic coherence scores computed
- [ ] PENDING — Zero lookahead audit passed

## STEP 3: Validation Loop
- [ ] PENDING — Monte Carlo 10,000 shuffles complete for all candidates
- [ ] PENDING — Benjamini-Hochberg FDR correction applied
- [ ] PENDING — Temporal stability across rolling 5-year windows tested
- [ ] PENDING — Regime robustness classification complete
- [ ] PENDING — Bootstrap 99% confidence intervals computed
- [ ] PENDING — Accuracy-selectivity surface table saved
- [ ] PENDING — Empirical optimal threshold identified

## STEP 4: Synthesis Loop
- [ ] PENDING — ML Model A (Volatility) trained and saved
- [ ] PENDING — ML Model B (Directional) trained and saved
- [ ] PENDING — SHAP values computed for both models
- [ ] PENDING — Out-of-sample test on final 3 years complete
- [ ] PENDING — Composite score computed for full history
- [ ] PENDING — Backtest complete with full trade log
- [ ] PENDING — All benchmark comparisons complete
- [ ] PENDING — Stress tests across all crash periods complete
- [ ] PENDING — Cross-market Bank Nifty transfer test complete
- [ ] PENDING — Forward 1-year planetary calendar complete
- [ ] PENDING — Day classifications assigned to all 365 future days
- [ ] PENDING — Campaign windows and dead zones identified

## STEP 5: Daily Signal Generator
- [ ] PENDING — generate_signal.py built
- [ ] PENDING — Tested on 5 historical dates, output matches stored scores
- [ ] PENDING — Runs in under 10 seconds confirmed

## STEP 6: HTML Outputs and GitHub Pages
- [ ] PENDING — calendar.html built and pushed
- [ ] PENDING — report.html built and pushed
- [ ] PENDING — RESEARCH_REPORT.md built and pushed
- [ ] PENDING — TRADING_MANUAL.md built and pushed
- [ ] PENDING — All files accessible on GitHub Pages

## FINAL CHECKS
- [ ] PENDING — No step used future data (date-shift audit)
- [ ] PENDING — Underperformance documented honestly
- [ ] PENDING — All PNGs saved
- [ ] PENDING — All CSVs saved with consistent date indexing
- [ ] PENDING — Models loadable from models/ directory
- [ ] PENDING — CHECKLIST.md itself up to date and pushed
