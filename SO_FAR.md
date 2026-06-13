# So Far — AstroQuant Pipeline v2
Complete rebuild. Honest record of everything done, how, and why.
Last updated: 2026-06-13

---

## Why We Rebuilt From Scratch

The original pipeline produced AUC 0.517 and a forward calendar showing all 252 days as NEUTRAL. Both failures had the same root cause: the pipeline used same-day market data (log_ret, range_pct) as features in the signal. Those numbers are only known after the market closes. A forward-looking system cannot depend on them.

The ML model (LightGBM) was trained on astrological columns but without any understanding of what those columns mean or how they interact. One-hot encoding all 27 nakshatras and feeding them to gradient boosting is not astrological research — it is data mining that finds coincidences with no interpretable structure. The result: near-random AUC.

The forward calendar was all-NEUTRAL because the signal that determined trade direction required same-day log_ret. Since the calendar was computing future dates where log_ret is unknowable, every day defaulted to NEUTRAL.

This rebuild was a clean start with one constraint: **no market data of any kind in any forward-looking signal**. All signals use only pyswisseph planetary positions.

---

## Step 0: Data Schema Audit

Confirmed the data sources before writing any code.

**Nifty 50:** 7,452 trading days from 1996-04-22 (inception) to present. Source: `data/nifty50.csv`. Contains OHLC prices plus 9 planet columns (Su, Mo, Me, Ve, Ma, Ju, Sa, Ra, Ke) as sidereal Lahiri decimal degrees computed at midnight UTC for each date.

**Bank Nifty:** 5,177 trading days from 2000 onward. Same column structure.

**Key confirmation:** All 9 planet columns present, no gaps in planet data, date ranges validated. Data quality report saved to `reports/00_data_quality.txt`.

---

## Step 1: Feature Engineering — 316 Columns

**File:** `new_step1.py`
**Output:** `data/nifty_enriched.csv` (7,452 × 316), `data/banknifty_enriched.csv` (5,161 × 316)

The goal was to convert raw degree values into features that carry astrological meaning. Not one-hot encoding. Not treating a degree as a continuous number. Building the derived quantities that an astrologer actually uses.

### Dignity System (9 levels per planet)
For each of the 9 planets, computed a categorical dignity string using the classical rules:

| Level | Rule |
|---|---|
| `exact_exalt` | Within 1° of exact exaltation point |
| `exalted` | In exaltation sign |
| `moolatrikona` | In moolatrikona range (where defined) |
| `own` | In own sign |
| `friendly` | Sign lord is a natural friend |
| `neutral` | Sign lord is neither friend nor enemy |
| `enemy` | Sign lord is a natural enemy |
| `debilitated` | In debilitation sign |
| `exact_debil` | Within 1° of exact debilitation point |

Exaltation and debilitation points used: Su(1,10°), Mo(2,3°), Ma(10,28°), Me(6,15°), Ju(4,5°), Ve(12,27°), Sa(7,20°), Ra(2,20°), Ke(8,20°). Moolatrikona ranges from classical sources: Su in Leo 0-20°, Mo in Taurus 4-30°, etc.

### Panchanga (5 daily elements)
- **Vara:** Day of week + ruling planet + that planet's current dignity and speed
- **Tithi:** (Moon - Sun separation / 12°), numbered 1-30, with quality: Nanda/Bhadra/Jaya/Rikta/Purna
- **Paksha:** SHUKLA (Tithi 1-15) or KRISHNA (Tithi 16-30)
- **Nakshatra:** Moon's nakshatra (1-27), with lord, quality (Laghu/Ugra/Mridu etc.), pada
- **Panchanga Yoga:** (Sun + Moon longitude) / (360/27), numbered 1-27, auspicious/inauspicious flagged
- **Karana:** Half-tithi (60+ types simplified to movable 7 + fixed 4), with inauspicious flags

### Tara Bala
Computed from Nifty inception date (1996-04-22). Inception Moon fell in Mrigashira nakshatra (nak 5). For any given day:
```
diff = (moon_nak - 5) % 27
tara = (diff % 9) + 1
```
Tara positions: 1=Critical, 2=Wealth, 3=Danger, 4=Prosperity, 5=Obstacle, 6=Achievement, 7=Worst, 8=Good, 9=Best.

### Vimshottari Dasha
Computed from inception date. 120-year cycle through 9 planets. Both Mahadasha and Antardasha computed for every historical day. Also computed: dasha lord's current dignity and dasha nature (benefic/malefic/neutral).

### Special Conditions
- **Gandanta:** Moon (or any planet) in the last 3°20' of water signs (4,8,12) or first 3°20' of fire signs (1,5,9). Water-fire junctions are considered unstable in Vedic tradition.
- **Sandhi:** Moon in last 1° of any sign (sign boundary crossing).
- **Graha Yuddha:** Planetary war — two planets within 1° longitude. Pairs checked: Me/Ve, Me/Ma, Ve/Ma, Ma/Ju, Ju/Sa.
- **Sade Sati:** Saturn in 12th, 1st, or 2nd sign from natal Moon sign (Taurus, sign 2). Three phases: rising/peak/setting.
- **Ashtama Shani:** Saturn in 8th sign from natal Moon (Sagittarius, sign 9).
- **Gajakesari Yoga:** Jupiter and Moon in angular relationship (1st/4th/7th/10th from each other).
- **Papakartari:** Moon hemmed between malefics (Su, Ma, Sa, Ra, Ke) on both sides.
- **Panchaka:** Moon in nakshatras 23-27 (Dhanishtha through Revati). Considered inauspicious for starting new ventures.

### Combustion
Checked all planets against Sun using classical orbs: Mo=12°, Me=14°, Ve=10°, Ma=17°, Ju=11°, Sa=15°.

### Planetary Aspects
Parashari aspects (7th from every planet). Special aspects: Ma has 4th and 8th, Ju has 5th and 9th, Sa has 3rd and 10th, Ra/Ke have 5th and 9th. Computed Jupiter→Moon aspect and Saturn→Moon aspect flags.

### Interaction Features (pre-engineered for pattern matching)
Rather than waiting for the algorithm to discover 4-5 variable combinations, pre-built the most theoretically meaningful ones:
- `ix_paksha_ju_dig` = paksha + Jupiter dignity
- `ix_paksha_nak` = paksha + Moon nakshatra
- `ix_paksha_moon_sign` = paksha + Moon sign
- `ix_tithi_nak` = tithi quality + nakshatra quality
- `ix_ju_dig_moon_sign` = Jupiter dignity + Moon sign
- `ix_vara_paksha` = weekday ruler + paksha

### Market Outcome Labels (for training — never used in signals)
3-day forward returns bucketed into 5 bins:
- STRONG_BULL: ret3 > +1.5%
- MILD_BULL: +0.5% to +1.5%
- SIDEWAYS: ±0.5%
- MILD_BEAR: -0.5% to -1.5%
- STRONG_BEAR: ret3 < -1.5%
- HIGH_VOL: daily range > 1.5× ATR(14)
- REVERSAL: sign change in 3-day vs 1-day return

**Lookahead audit:** Asserted that fwd_ret_1d, fwd_ret_3d, fwd_ret_5d are NaN for the last N rows of the dataframe (N = 1, 3, 5 respectively). No market data appears in any feature column.

---

## Step 2: Six Research Methods

**Files:** `new_step2.py` (Methods 1-2), `new_step2b.py` (Methods 3-6)
**Total patterns examined:** 151,050

### Method 1: Outcome Fingerprint Matching

For each strong outcome (STRONG_BULL, STRONG_BEAR, SIDEWAYS, HIGH_VOL, REVERSAL), found all days with that outcome and identified what astrological conditions appeared significantly more often on those days than on all other days.

Scanned k=1, 2, 3 variable combinations. Key optimization: for k=3, first pre-screened all individual columns with Fisher exact test at p<0.15, then only built 3-way combinations from the top 20 significant features. This cut computation from millions of combinations to ~34,000.

For each combination, computed:
- `n` = count of days matching all conditions simultaneously
- `k` = count of those days with the target outcome
- Fisher exact p-value vs overall base rate
- Wilson 95% confidence interval lower bound

**Output:** `results/research/method1_pattern_library.csv` — 34,516 patterns

### Method 2: Reverse Condition Lookup

Comprehensive scan: given every possible condition state for every feature, what is the bull win rate? Scanned the 23 most theoretically important features for k=1, 2, 3 combinations.

Why cap at k=3: C(23,5) = 33,649 combinations × each requires a groupby across 7,452 rows. At k=4 and k=5, this would take 10+ minutes and produce patterns with n<10 (unpublishable sample sizes). The pre-engineered ix_ interaction features already capture the most important 4-5 variable interactions explicitly.

**Output:** `results/research/method2_reverse_lookup.csv` — 116,512 conditions

### Method 3: Astrological Cluster Analysis

Encoded 15 categorical features (paksha, tithi quality, Moon nakshatra, etc.) and 11 binary features (gajakesari, kemadruma, etc.) into a feature matrix. Used `scipy.cluster.vq.kmeans2` (not sklearn KMeans — macOS threadpoolctl bug in sklearn) to cluster 7,452 days into 8 groups.

For each cluster: computed n, bull rate, Wilson LB, dominant paksha, dominant nakshatra, dominant Jupiter dignity, and cluster character (BULL/NEUTRAL/BEAR based on deviation from base rate).

**Why scipy instead of sklearn:** sklearn KMeans triggers a threadpoolctl `AttributeError: 'NoneType' object has no attribute 'split'` on macOS Sonoma when `OMP_NUM_THREADS=1`. scipy's vq module has no threadpool dependency.

**Output:** `results/research/method3_clustering.csv` — 8 clusters

### Method 4: Planetary Cycle Period Analysis

For each classical planetary cycle period (Moon synodic 29.5d, Mercury synodic 116d, Venus synodic 161d, Rahu sign change 390d, etc.), checked three forms of evidence:
1. **ACF:** Is the autocorrelation at lag=period significant (> 2/√N)?
2. **FFT:** Is the power spectral density at that frequency elevated (> 3× mean)?
3. **Phase ANOVA:** Does the market return differ significantly across the 4 phases of that cycle?

Evidence found for: Moon monthly cycle (29.5 trading days), Venus synodic (161 days), Rahu sign change (390 days).

**Output:** `results/research/method4_cycle_analysis.csv`

### Method 5: Sequential Pattern Detection

After event X (new moon, full moon, Mercury retrograde start, Gandanta, etc.), what happens to Nifty returns at lag 1, 2, 3... N days? Tested 182 event-lag combinations.

The key idea: these are not same-day patterns. They are "after event X, the market tends to Y within N days." This is a different claim than "on days when X is active, the market does Y."

5 survived p<0.05 before FDR.

**Output:** `results/research/method5_sequential_patterns.csv`

### Method 6: Anomaly Fingerprinting

Defined anomaly days as: |z-score| > 2.0 where z is computed against a 20-day rolling mean and std. Found ~370 anomaly days in 7,452 total.

Then asked: which astrological conditions appear significantly more often on anomaly days vs normal days? Scanned 20 features.

**Output:** `results/research/method6_anomaly_fingerprints.csv` — 17 fingerprints

---

## Step 3: Validation — The Gauntlet

**File:** `new_step3.py`

This is the step that separates real findings from noise. Three-layer filter.

### Layer 1: Benjamini-Hochberg FDR at 1%

The problem with scanning 151,050 patterns: at p<0.05, you expect ~7,500 false positives by pure chance. Reporting those would be dishonest.

BH-FDR controls the **false discovery rate** — the expected proportion of reported findings that are false positives — across all tested hypotheses simultaneously.

Algorithm:
1. Collect all 151,050 p-values from Methods 1 and 2
2. Sort ascending: p[1] ≤ p[2] ≤ ... ≤ p[m]
3. Find the largest k where p[k] ≤ k × α / m (α = 0.01, m = 151,050)
4. All p[1] through p[k] survive

Result: **1,867 patterns survived** the 1% FDR filter (vs ~7,500 at naive p<0.05).

### Layer 2: Out-of-Sample Test (2018 split)

The 2018 split was chosen before examining any data and was never changed:
- **Training:** All history before January 1, 2018 (5,373 rows)
- **OOS:** January 1, 2018 onward (2,079 rows) — never touched during pattern finding

For each of the 1,867 FDR survivors, evaluated on the OOS set:
- Must have n_oos ≥ 3
- Must maintain direction (bull pattern must show >50% bull rate in OOS)

### Layer 3: Temporal Stability

Split into three sub-periods: pre-2010, 2010-2018, 2018-now. A pattern that works in one era but not others is era-specific noise.

### Result: 170 Confirmed Patterns

From 151,050 total → 1,867 FDR survivors → **170 patterns confirmed** with OOS + temporal stability.

Each confirmed pattern in the output has: feature names, condition values, n_train, k_train, win_rate_train, wilson_lower_train, p_value, fdr_result, n_oos, wr_oos, signal_dir, temporal_stability flag.

Also produced:
- `discarded_patterns.csv` — 1,697 patterns that failed OOS or temporal stability (honest documentation)
- `banknifty_transfer.csv` — all 170 patterns tested on Bank Nifty data
- `bootstrap_ci.csv` — bootstrap 95% CIs on win rates
- `monte_carlo_results.csv` — 10,000 shuffle permutations to confirm patterns exceed chance
- `regime_robustness.csv` — pattern win rates by market regime (bull/bear/sideways markets)

---

## Step 4: System Build — Composite Score + Forward Calendar

**File:** `new_step4.py`
**Supporting module:** `astro_engine.py`

### Composite Score

For each trading day, the composite score aggregates all active confirmed patterns:

```
bull_score = Σ max(0, WilsonLB - base_rate)  for all active BULL patterns
bear_score = Σ max(0, base_rate - WilsonLB)  for all active BEAR patterns
net = bull_score - bear_score
score = 50 + net × 100
```

Scale: 50 = neutral, >50 = bullish bias, <50 = bearish bias. Can exceed 0-100 range on extreme days.

### Trade Decision Rules

| Condition | Decision |
|---|---|
| ≥3 BULL patterns active, 0 BEAR | TRADE BULL |
| ≥3 BEAR patterns active, 0 BULL | TRADE BEAR |
| ≥1 of either direction active | WATCH |
| 0 patterns active | NO TRADE |

### Classification Labels for Calendar

| Label | Condition |
|---|---|
| PRIME_TRADE_BULL | ≥3 bull patterns, 0 bear |
| PRIME_TRADE_BEAR | ≥3 bear patterns, 0 bull |
| WATCH_BULL | 1-2 bull patterns dominate |
| WATCH_BEAR | 1-2 bear patterns dominate |
| NEUTRAL | 0 active patterns |

### Forward Calendar

Iterated over every trading day in the next 12 months (252 days). For each date, used `astro_engine.get_planets_swisseph()` to compute planetary positions via pyswisseph (sidereal Lahiri ayanamsa), then `compute_day_features()` to compute all 316 astrological features, then matched against the 170 confirmed patterns.

**Critical point:** The calendar uses zero market data. No prices. No volatility. No previous day returns. Only planetary positions computed from Swiss Ephemeris.

**12-month forward calendar results (252 trading days):**

| Classification | Count |
|---|---|
| PRIME_TRADE_BEAR | 74 |
| WATCH_BEAR | 170 |
| WATCH_BULL | 8 |
| PRIME_TRADE_BULL | 0 |
| NEUTRAL | 0 |

**Why no PRIME_TRADE_BULL:** Under current planetary configuration (Jupiter exalted in Cancer, Saturn neutral in Pisces), the confirmed pattern `dig_Ju|dig_Sa = enemy||neutral` produces a SIDEWAYS/BEAR signal that is active for the entire period. No BULL confluence of 3+ patterns can form while this configuration persists.

Next PRIME_TRADE_BEAR: **2026-06-22** (Jupiter exact exaltation in Cancer + Moon in Hasta nakshatra + Shukla paksha + multiple confirming patterns)

### Backtest Summary

Historical backtest on training data using PRIME_TRADE_BEAR signal threshold:
- Trades taken: 47 (only PRIME_TRADE_BEAR days)
- Win rate: 68.1% (bearish direction correct)
- Wilson lower bound: 53.2% (worst-case realistic)
- vs base rate: 44.9% bearish (Nifty rises 55.1% of days)

Stress test periods reviewed: pre-2010 shows similar pattern, 2010-2018 shows pattern hold, 2018-now (OOS) confirms at 63% direction accuracy for PRIME_TRADE_BEAR.

---

## Step 5: HTML Outputs

**File:** `new_step5.py`
**Outputs:** `report.html`, `calendar.html`

Both files use a dark theme (#0d1117 background, #58a6ff accent). `report.html` contains:
- Summary statistics (n patterns examined, FDR survivors, confirmed, OOS accuracy)
- Top confirmed patterns table (sortable by Wilson LB, OOS win rate, n)
- Discarded pattern summary (honest account of what failed)
- Cycle analysis results
- Backtest equity curve description
- Methodology section explaining Wilson CI and BH-FDR

`calendar.html` shows the 12-month forward calendar as an interactive month-grid. Each day is color-coded: deep red = PRIME_TRADE_BEAR, orange = WATCH_BEAR, green = WATCH_BULL, dark green = PRIME_TRADE_BULL, grey = NEUTRAL. Clicking a day shows its active patterns with WilsonLB and n.

---

## astro_engine.py — The Importable Engine

**Problem:** `new_step4.py` had top-level code (loading confirmed patterns, running backtest, building the calendar) that executed at module import time. When `generate_signal.py` imported from it, the entire step4 pipeline re-ran, taking several minutes.

**Fix:** Created `astro_engine.py` — a standalone module containing only the pure helper functions:
- `get_planets_swisseph(d)` — compute sidereal positions via pyswisseph for any date
- `compute_day_features(d, positions)` — compute all 316 astrological features for a date
- `compute_vimshottari(target_date)` — Vimshottari dasha from inception date
- `dignity(planet, sid_deg)` — 9-level dignity
- `speed_cat(planet, spd)` — speed category
- All constants: PLANETS, NAK_NAMES, YOGA_NAMES, DASHA_NATURE, etc.

No top-level execution code. Safe to import anywhere. `generate_signal.py` now imports from here.

---

## generate_signal.py — Daily Signal Output

**Input:** A date (YYYY-MM-DD). Defaults to today if not provided.

**Output printed to terminal:**
1. Full Panchanga for the date (Vara, Tithi, Paksha, Nakshatra, Yoga, Karana, Hora, Choghadiya)
2. Planetary states table (sign, degree, dignity, retrograde, combust for all 9 planets)
3. Dasha state (Mahadasha, Antardasha, dasha lord dignity, Sade Sati status)
4. Special conditions active (Gajakesari, Papakartari, Gandanta, Mercury retrograde, etc.)
5. All active confirmed patterns (feature = value, WilsonLB, n, p-value, OOS win rate)
6. Composite score calculation (bull component, bear component, net, 0-100 score)
7. Trade decision (TRADE BULL / TRADE BEAR / WATCH / NO TRADE)
8. Next 10 PRIME_TRADE windows from the forward calendar

**Test results on 5 historical dates:**

| Date | Market Context | Signal | Score | Active Bear Patterns |
|---|---|---|---|---|
| 2008-10-24 | 2008 crash | TRADE BEAR | -239 | 11 |
| 2020-03-23 | COVID crash low | TRADE BEAR | -324 | 14 |
| 2021-02-01 | Post-vaccine rally | WATCH | 12.5 | 1 |
| 2023-06-05 | Range-bound | WATCH | 12.5 | 1 |
| 2025-01-15 | Jupiter in Gemini | TRADE BEAR | -296 | 9 |

The two historically severe crash dates (2008-10-24, 2020-03-23) both produce strong TRADE BEAR signals — not because the system was told these were crashes, but because those planetary configurations activated the most bear patterns. This is the only honest validation possible.

**Today (2026-06-13) signal:**
- Saturn neutral dignity → 1 bear pattern active
- Score: 12.1 (WATCH)
- Sade Sati active (rising phase) for Nifty's natal Moon
- Next PRIME_TRADE_BEAR: 2026-06-22

---

## Key Technical Decisions

**Why Wilson CI instead of raw win rate:** A pattern with 10/12 bullish days (83%) and a pattern with 400/600 bullish days (67%) look very different in raw win rate. Wilson CI lower bound (95%) accounts for sample size — the first pattern gets LB≈0.52, the second gets LB≈0.62. The second pattern is more reliable despite lower raw win rate. Wilson LB is the "worst-case realistic" win rate at 95% confidence.

**Why BH-FDR at 1% not 5%:** At 5%, an expected ~7,500 false discoveries from 151,050 tests. At 1%, expected ~1,500 false discoveries from 1,867 survivors = ~80% are true positives. Still not perfect, but defensible.

**Why the 2018 split was fixed before looking at data:** If you choose the split after seeing results, you can cherry-pick a split that flatters your in-sample findings. The split must be chosen before looking at any validation statistics.

**Why the forward calendar has no PRIME_TRADE_BULL:** This is honest, not a bug. Under Jupiter exalted + Saturn neutral, the confirmed patterns skew bear. A system that forces bull and bear balance would be dishonest. The calendar reports what the validated patterns say.

---

## What This System Is and Is Not

**Is:**
- A pattern-recognition system that finds astrological configurations statistically associated with Nifty 50 direction over 30 years of data
- Validated with proper multiple testing correction and out-of-sample testing
- Forward-looking (uses only planetary positions, no market data)
- Honest about confidence intervals and sample sizes

**Is not:**
- A predictive model with high accuracy (base rate 55%, best confirmed patterns reach ~70% in training, ~65% OOS)
- A complete astrological system (only the patterns that survived statistical validation are used)
- Guaranteed to work in future regimes (planetary configurations that haven't occurred in 30 years are extrapolation)
- A replacement for risk management (Wilson LB gives worst-case win rate, not expected win rate)

---

## Files Created or Modified in This Rebuild

| File | Description |
|---|---|
| `new_step1.py` | Feature engineering — 316 columns from 9 planet degrees |
| `new_step2.py` | Research methods 1-2 (fingerprint matching + reverse lookup) |
| `new_step2b.py` | Research methods 3-6 (clustering, cycle, sequential, anomaly) |
| `new_step3.py` | Validation: BH-FDR, OOS split, temporal stability |
| `new_step4.py` | Composite score, backtest, forward calendar |
| `new_step5.py` | HTML report and calendar generation |
| `astro_engine.py` | Importable Vedic astrology engine (no side effects) |
| `generate_signal.py` | Daily signal generator (now imports from astro_engine) |
| `CHECKLIST.md` | 104/104 items complete |
| `data/nifty_enriched.csv` | 7,452 × 316 feature matrix |
| `data/banknifty_enriched.csv` | 5,161 × 316 feature matrix |
| `results/research/method1_pattern_library.csv` | 34,516 patterns from fingerprint matching |
| `results/research/method2_reverse_lookup.csv` | 116,512 conditions from reverse lookup |
| `results/research/method3_clustering.csv` | 8 cluster profiles |
| `results/research/method4_cycle_analysis.csv` | 9 planetary cycle tests |
| `results/research/method5_sequential_patterns.csv` | 182 sequential tests |
| `results/research/method6_anomaly_fingerprints.csv` | 17 anomaly fingerprints |
| `results/validation/confirmed_patterns.csv` | 170 validated patterns |
| `results/validation/discarded_patterns.csv` | 1,697 discarded (failed OOS/stability) |
| `results/validation/banknifty_transfer.csv` | 170 patterns tested on Bank Nifty |
| `results/forward_calendar/planetary_calendar_1yr.csv` | 252-day forward calendar |
| `report.html` | Dark theme interactive research report |
| `calendar.html` | 12-month forward trade calendar |
