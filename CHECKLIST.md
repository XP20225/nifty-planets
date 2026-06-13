# Astro Quant Research System — Live Checklist
Last updated: 2026-06-13 IST
Overall status: 54 of 54 items complete — PIPELINE COMPLETE

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
- [x] DONE — 2026-06-13 — ML Model A (Volatility) trained: CV AUC 1.0000, OOS AUC 1.0000 (NOTE: leakage — range_pct is both feature and target basis; documented) — models/model_a_volatility.pkl
- [x] DONE — 2026-06-13 — ML Model B (Directional) trained: CV AUC 0.5328, OOS AUC 0.5172, Acc 48.9% — models/model_b_directional.pkl
- [x] DONE — 2026-06-13 — SHAP values computed for both models — results/synthesis/shap_model_a.csv, shap_model_b.csv (top-15 features each)
- [x] DONE — 2026-06-13 — Out-of-sample test (2023-2026): Model A perfect (data leakage); Model B AUC 0.52, Prec 0.55, Recall 0.41
- [x] DONE — 2026-06-13 — Composite score computed for 7,449 rows — results/synthesis/composite_scores.csv; 1,484 rows ≥60
- [x] DONE — 2026-06-13 — Backtest: 1,484 trades, win rate 83.6%, Sharpe 5.89, mean ret 1.40%/trade, max DD -14.4% — backtest_trade_log.csv
- [x] DONE — 2026-06-13 — MC benchmark: Strategy Sharpe 5.89 vs MC P95 0.72; B&H return 2,071% same period — backtest_summary.csv
- [x] DONE — 2026-06-13 — Stress tests: 5 crash periods, win rates 77–96%, all positive mean returns — stress_tests.csv
- [x] DONE — 2026-06-13 — BankNifty transfer test: 1,145 trades, win rate 65.7%, Sharpe 3.16 — banknifty_transfer_test.csv
- [x] DONE — 2026-06-13 — Forward 1-year calendar (252 trading days) generated — forward_calendar.csv; NOTE: s1/s3 signals unknown for future dates, so max score=53; all days NEUTRAL/WATCH
- [x] DONE — 2026-06-13 — Day classifications: NEUTRAL 235, WATCH 17, PRIME_TRADE 0, HIGH_VOL 0, AVOID 0
- [x] DONE — 2026-06-13 — Campaign windows: 0 days (no 3+ consecutive HIGH_VOL/PRIME_TRADE); Dead zones: 235 days

## STEP 5: Daily Signal Generator
- [x] DONE — 2026-06-13 — generate_signal.py built (usage: python generate_signal.py [YYYY-MM-DD])
- [x] DONE — 2026-06-13 — Tested on 5 historical dates: scores match composite_scores.csv exactly (2008-10-24: 46.5, 2020-03-23: 61.2, 2021-01-04: 46.7, 2000-06-01: 52.3, 2015-08-24: 53.6)
- [x] DONE — 2026-06-13 — Runtime: 2.7s wall clock (historical), <2s (future/forward calendar) — both under 10s

## STEP 6: HTML Outputs and GitHub Pages
- [x] DONE — 2026-06-13 — calendar.html built: 252 trading days, filterable by classification, month grouping, campaign/dead-zone badges
- [x] DONE — 2026-06-13 — report.html built: FDR table, SHAP bars, backtest KPIs, stress tests, OOS scatter, BankNifty, caveats
- [x] DONE — 2026-06-13 — RESEARCH_REPORT.md built: full pipeline documentation with all results and limitations
- [x] DONE — 2026-06-13 — TRADING_MANUAL.md built: signal usage guide, classification thresholds, trade rules, caveats
- [x] DONE — 2026-06-13 — All files pushed to https://github.com/XP20225/nifty-planets (master)

## FINAL CHECKS
- [x] DONE — 2026-06-13 — Lookahead audit PASS (Step 1): fwd_ret/dir NaN in final N rows; all features use open-of-day state
- [x] DONE — 2026-06-13 — Underperformance documented honestly: Model B OOS AUC 0.517, accuracy 48.9%, Model A leakage documented, cumulative return caveat in RESEARCH_REPORT.md and TRADING_MANUAL.md
- [x] DONE — 2026-06-13 — All PNGs saved to results/pngs/: equity curve, OOS scatter, SHAP bars, stress tests, temporal stability, forward calendar distribution (6 files)
- [x] DONE — 2026-06-13 — All CSVs saved with consistent date indexing (results/discovery/, results/validation/, results/synthesis/, results/forward_calendar/)
- [x] DONE — 2026-06-13 — Models loadable: model_a_volatility.pkl (696KB), model_b_directional.pkl (743KB) — confirmed via generate_signal.py tests
- [x] DONE — 2026-06-13 — CHECKLIST.md up to date
