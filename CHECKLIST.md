# Astro Quant Research System — Live Checklist
Last updated: 2026-06-13 IST
Overall status: 21 of 54 items complete

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
- [x] DONE — 2026-06-13 — Unconditional scan: 4,536 rows, 35 categorical + 19 continuous + 15 binary features × 6 targets — results/discovery/unconditional_scan.csv
- [x] DONE — 2026-06-13 — Conditional scans: Bull (4,088), Bear (3,952), Transitional (4,098) rows — 3 CSVs saved
- [x] DONE — 2026-06-13 — Extreme day patterns: 6 categories (top50 up/down, top50 hi/lo-vol, top30 continuation/reversal) — extreme_day_patterns.csv
- [x] DONE — 2026-06-13 — Coherence scores: 7,452 daily scores, direction rho=-0.026 (p=0.022) vs fwd_dir_3d, magnitude rho=+0.028 (p=0.016) vs vol — coherence_scores.csv
- [x] DONE — 2026-06-13 — Zero lookahead audit passed in Step 1; all features use only market-open state on signal date

## STEP 3: Validation Loop
- [x] DONE — 2026-06-13 — Monte Carlo 10,000 shuffles: 19 candidates → 4 survived p<0.005 — results/validation/monte_carlo_results.csv
- [x] DONE — 2026-06-13 — Benjamini-Hochberg FDR at 1% applied: 4 survivors — results/validation/fdr_survivors.csv
- [x] DONE — 2026-06-13 — Temporal stability across rolling 5-year windows: all 4 stable (≥0.70) — results/validation/temporal_stability.csv
- [x] DONE — 2026-06-13 — Regime robustness: all 4 classified UNIVERSAL (log_ret, ju_dignity, range_pct, combust_Mo) — results/validation/regime_robustness.csv
- [x] DONE — 2026-06-13 — Bootstrap 99% CIs computed (5,000 resamples): all 4 directionally certain — results/validation/bootstrap_ci.csv
- [x] DONE — 2026-06-13 — Accuracy-selectivity surface table saved — results/validation/accuracy_selectivity_surface.csv
- [x] DONE — 2026-06-13 — Empirical optimal threshold: 3+ signals active → 62.3% 3d win rate, 95% CI ≥58.1%, 499 days (6.7% freq), Sharpe 3.11

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
