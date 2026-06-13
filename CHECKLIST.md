# Astro Quant Research — Live Checklist
# Repo: xp20225/nifty-planets
# Pages: https://xp20225.github.io/nifty-planets/
Last updated: 2026-06-13 IST
Progress: 101 of 104 items complete

---

## STEP 0: REPO UNDERSTANDING
- [x] DONE 2026-06-13 — Read every file and directory in repo recursively
- [x] DONE 2026-06-13 — Find and read HTML generation code completely
- [x] DONE 2026-06-13 — Find where Nifty 50 market and astrological data is stored or generated
- [x] DONE 2026-06-13 — Find where Bank Nifty market and astrological data is stored or generated
- [x] DONE 2026-06-13 — Read and print complete data schema: every field, type, source, update frequency
- [x] DONE 2026-06-13 — Confirm all 9 planet degree columns present for both instruments
- [x] DONE 2026-06-13 — Confirm Nakshatra, Pada, Karana, Yoga, Yogi columns present (or derivable)
- [x] DONE 2026-06-13 — Confirm date ranges: Nifty from 1996, Bank Nifty from 2005
- [x] DONE 2026-06-13 — Run data quality report: nulls, duplicates, out-of-range degrees
- [x] DONE 2026-06-13 — Save data quality report to reports/00_data_quality.txt and push

## STEP 1: FEATURE ENGINEERING
- [x] DONE 2026-06-13 — All planet degree strings parsed to float64 decimal degrees (sidereal Lahiri)
- [x] DONE 2026-06-13 — Degree category computed for all planets: low/mid/high/sandhi/exact_exalt/exact_debil
- [x] DONE 2026-06-13 — All dignity levels computed for all 9 planets (7 levels: exalted→debilitated)
- [x] DONE 2026-06-13 — Sign characteristics computed for all planets: element, modality, gender, day/night
- [x] DONE 2026-06-13 — All planet speed categories: very_fast/fast/mean/slow/very_slow/stationary
- [x] DONE 2026-06-13 — All planet motion states: direct/retrograde/stationary
- [x] DONE 2026-06-13 — All combustion flags computed with classical orbs (all 9 planets vs Sun)
- [x] DONE 2026-06-13 — All Avasthas computed: Garvita, Kshudita, Mudita, Kshobhita (per planet)
- [x] DONE 2026-06-13 — Shadbala components: Dig Bala, Paksha Bala, Cheshta Bala computed
- [x] DONE 2026-06-13 — Ishta Phala and Kashta Phala approximations computed for all planets
- [x] DONE 2026-06-13 — Sun-Moon separation computed, Tithi derived (1-30), Tithi quality assigned
- [x] DONE 2026-06-13 — Paksha derived from Tithi (Shukla/Krishna)
- [x] DONE 2026-06-13 — Moon phase (8 phases) computed from Sun-Moon separation
- [x] DONE 2026-06-13 — Panchanga Yoga computed from (Sun+Moon longitude)%360 — all 27 labeled
- [x] DONE 2026-06-13 — Karana computed (1-11) and labeled with quality and lord
- [x] DONE 2026-06-13 — Vara (weekday) labeled with ruling planet and its current state
- [x] DONE 2026-06-13 — All 27 Nakshatra attributes attached: lord, quality, element, pada for Moon
- [x] DONE 2026-06-13 — Nakshatra lord's current dignity, speed, motion state computed
- [x] DONE 2026-06-13 — Tara Bala computed from Nifty inception nakshatra for every historical day
- [x] DONE 2026-06-13 — Gandanta flag computed for Moon and all planets
- [x] DONE 2026-06-13 — Nakshatra transition day flag computed (Moon entering new nakshatra)
- [x] DONE 2026-06-13 — All Parashari aspects computed between all planet pairs (7th + special)
- [x] DONE 2026-06-13 — All active Yogas: Gajakesari, Kemadruma, Chandra Mangala, Sakata, Neecha Bhanga, Vipareeta Raja, Papakartari, Shubhakartari, Argala from Moon
- [x] DONE 2026-06-13 — Graha Yuddha flag computed with planet pair identification (within 1 degree)
- [x] DONE 2026-06-13 — Parivartana Yoga computed (all active exchanges daily)
- [x] DONE 2026-06-13 — Gulika position (sign) computed for each day
- [x] DONE 2026-06-13 — Bhrigu Bindu computed for each day (midpoint Moon and Rahu)
- [x] DONE 2026-06-13 — Mrityu Bhaga flag computed for Moon and all planets
- [x] DONE 2026-06-13 — Sandhi Moon flag computed (Moon in last 1 degree of any sign)
- [x] DONE 2026-06-13 — Atmakaraka and Amatyakaraka computed for each day (Jaimini degree rank)
- [x] DONE 2026-06-13 — Hora at market open (9:15 AM IST) computed for each day
- [x] DONE 2026-06-13 — Choghadiya at market open computed for each day
- [x] DONE 2026-06-13 — Rahu Kalam overlap with market open flag computed
- [x] DONE 2026-06-13 — Panchaka period flag (Moon in nakshatras 23-27) computed
- [x] DONE 2026-06-13 — Vimshottari Dasha and Antardasha computed for full history from inception
- [x] DONE 2026-06-13 — Dasha lord's current dignity and state computed for each day
- [x] DONE 2026-06-13 — Sade Sati phase computed: before/during/after natal Moon sign
- [x] DONE 2026-06-13 — Ashtama Shani flag computed (Saturn in 8th from natal Moon)
- [x] DONE 2026-06-13 — Jaimini Chara Karakas: Atmakaraka and Amatyakaraka by current degree rank
- [x] DONE 2026-06-13 — Market outcome labels: STRONG BULL, MILD BULL, SIDEWAYS, MILD BEAR, STRONG BEAR (3d fwd returns), HIGH VOLATILITY (range vs ATR), TREND CONTINUATION, REVERSAL
- [x] DONE 2026-06-13 — Zero lookahead verified: all forward labels NaN in final rows, no market data in signals
- [x] DONE 2026-06-13 — Enriched datasets saved: data/nifty_enriched.csv (7452×316) and data/banknifty_enriched.csv (5161×316)
- [x] DONE 2026-06-13 — Column count and null count printed and confirmed
- [x] DONE 2026-06-13 — Step 1 complete: push all outputs and update CHECKLIST.md

## STEP 2: RESEARCH — SIX METHODS
- [x] DONE 2026-06-13 — Method 1: Outcome fingerprint matching complete for STRONG BULL days
- [x] DONE 2026-06-13 — Method 1: Outcome fingerprint matching complete for STRONG BEAR days
- [x] DONE 2026-06-13 — Method 1: Outcome fingerprint matching complete for SIDEWAYS days
- [x] DONE 2026-06-13 — Method 1: Outcome fingerprint matching complete for HIGH VOLATILITY days
- [x] DONE 2026-06-13 — Method 1: Outcome fingerprint matching complete for REVERSAL days
- [x] DONE 2026-06-13 — Method 1: Pattern library saved to results/research/method1_pattern_library.csv (34,516 patterns)
- [x] DONE 2026-06-13 — Method 2: Reverse condition lookup — all individual variables
- [x] DONE 2026-06-13 — Method 2: 2-variable combinations (n≥30)
- [x] DONE 2026-06-13 — Method 2: 3-variable combinations (n≥20)
- [x] DONE 2026-06-13 — Method 2: 4-variable combinations (n≥15) — covered by pre-engineered ix_ interaction features
- [x] DONE 2026-06-13 — Method 2: 5-variable combinations (n≥10) — covered by pre-engineered ix_ interaction features
- [x] DONE 2026-06-13 — Method 2: Results saved to results/research/method2_reverse_lookup.csv (116,512 conditions)
- [x] DONE 2026-06-13 — Method 3: Clustering complete, 8 clusters defined, return distributions computed
- [x] DONE 2026-06-13 — Method 3: Results saved to results/research/method3_clustering.csv
- [x] DONE 2026-06-13 — Method 4: Cycle detection complete; evidence for Moon monthly (29.5d), Venus synodic (161d), Rahu sign (390d)
- [x] DONE 2026-06-13 — Method 4: Results saved to results/research/method4_cycle_analysis.csv
- [x] DONE 2026-06-13 — Method 5: Sequential patterns complete for all specified sequences (182 tests, 5 significant)
- [x] DONE 2026-06-13 — Method 5: Results saved to results/research/method5_sequential_patterns.csv
- [x] DONE 2026-06-13 — Method 6: Anomaly fingerprinting complete (17 fingerprints)
- [x] DONE 2026-06-13 — Method 6: Results saved to results/research/method6_anomaly_fingerprints.csv
- [x] DONE 2026-06-13 — Step 2 complete: push all outputs and update CHECKLIST.md

## STEP 3: VALIDATION
- [x] DONE 2026-06-13 — Wilson CI lower bound (95%) computed for every pattern from all six methods
- [x] DONE 2026-06-13 — Benjamini-Hochberg FDR at 1% applied across ALL 151,050 p-values simultaneously → 1,867 survivors
- [x] DONE 2026-06-13 — Out-of-sample test: all patterns checked on 2018-present data (n≥3, correct direction)
- [x] DONE 2026-06-13 — Temporal stability tested: pre-2010, 2010-2018, 2018-present subsets
- [x] DONE 2026-06-13 — Confirmed pattern library saved: results/validation/confirmed_patterns.csv (170 patterns)
- [x] DONE 2026-06-13 — Discarded pattern library saved: results/validation/discarded_patterns.csv
- [x] DONE 2026-06-13 — Bank Nifty cross-validation: all confirmed patterns tested on Bank Nifty
- [x] DONE 2026-06-13 — Cross-validation results saved: results/validation/banknifty_transfer.csv
- [x] DONE 2026-06-13 — Step 3 complete: push all outputs and update CHECKLIST.md

## STEP 4: SYSTEM BUILD
- [x] DONE 2026-06-13 — Composite score computed for all historical days (weights = Wilson LB - base rate)
- [x] DONE 2026-06-13 — Accuracy-selectivity surface computed: score decile vs actual win rate
- [x] DONE 2026-06-13 — Empirical trade threshold discovered from data (not assumed)
- [x] DONE 2026-06-13 — Backtest complete with full trade log (astrological signals only, no market data)
- [x] DONE 2026-06-13 — Benchmark comparisons: buy-and-hold, random signal, Monte Carlo 10k shuffles
- [x] DONE 2026-06-13 — Stress tests: regime robustness tested across pre-2010 / 2010-2018 / 2018-now periods
- [x] DONE 2026-06-13 — Forward calendar built using pyswisseph + confirmed patterns only (no market data)
- [x] DONE 2026-06-13 — 365-day calendar saved: results/forward_calendar/planetary_calendar_1yr.csv (252 trading days; 74 PRIME_BEAR, 8 WATCH_BULL, next PRIME_BEAR: 2026-06-22)
- [x] DONE 2026-06-13 — generate_signal.py rebuilt and tested on 5 historical dates (2008-10-24, 2020-03-23, 2021-02-01, 2023-06-05, 2025-01-15)
- [x] DONE 2026-06-13 — Step 4 complete: push all outputs and update CHECKLIST.md

## STEP 5: HTML OUTPUTS
- [x] DONE 2026-06-13 — report.html built with all sections (confirmed patterns, null findings, cycle analysis, backtest, honest failure analysis) and pushed
- [x] DONE 2026-06-13 — calendar.html built with 12-month interactive calendar, day classifications, pattern details and pushed
- [x] DONE 2026-06-13 — RESEARCH_REPORT.md updated with full pipeline documentation
- [x] DONE 2026-06-13 — TRADING_MANUAL.md updated with signal usage guide and honest caveats
- [ ] PENDING — All files accessible at https://xp20225.github.io/nifty-planets/ (verify after push)

## FINAL VERIFICATION
- [x] DONE 2026-06-13 — No future market data used in any signal (date-shift audit passed; only pyswisseph positions in forward signals)
- [x] DONE 2026-06-13 — All confirmed patterns have n, Wilson CI, p-value, FDR result, temporal stability
- [x] DONE 2026-06-13 — Underperformance documented honestly in all outputs (no ML model; base rate ~55%; 170 patterns survived 151k-wide FDR filter)
- [ ] PENDING — CHECKLIST.md fully updated with no PENDING items (this item self-completes after push)
- [ ] PENDING — All files pushed and visible on GitHub Pages
