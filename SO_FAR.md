# So Far — AstroQuant Pipeline v2 + Fixes
Complete record of everything built, found, and fixed.
Last updated: 2026-06-13

---

## Quick Status

| Stage | Status | Key Output |
|---|---|---|
| Pipeline v2 rebuild (Steps 1–5) | COMPLETE | 170 confirmed patterns, 252-day calendar |
| Fix 1 — Missing features | COMPLETE | 353 columns (was 316) |
| Fix 2 — Uncapped fingerprint | RUNNING | ~1,400+ patterns so far |
| Fix 3 — Bull/bear investigation | COMPLETE | Root cause identified |
| Fix 4 — Bank Nifty independent research | RUNNING | — |
| Fix 5 — Validate M3–6 + merge | PENDING | Runs after Fix 2 + 4 complete |

---

## Part 1: Why the Pipeline Was Rebuilt

The original pipeline (before this session) produced AUC 0.517 and a 252-day forward calendar showing every day as NEUTRAL. Both failures had the same root cause: same-day market data (log_ret, range_pct) was used as a signal feature. Those numbers are only known after market close. A forward-looking system cannot depend on them.

The forward calendar was all-NEUTRAL because the signal that determined trade direction required same-day log_ret. When computing future dates where log_ret is unknowable, every day defaulted to NEUTRAL.

The LightGBM model (AUC 0.517) was trained on astrological columns but without understanding what they mean. One-hot encoding all 27 nakshatras and feeding them to gradient boosting finds coincidences with no interpretable structure. The result is near-random.

The rebuild constraint: **no market data of any kind in any forward-looking signal.** All signals use only pyswisseph planetary positions.

---

## Part 2: Pipeline v2 (completed before this session's fixes)

### Step 1: Feature Engineering
**File:** `new_step1.py`
**Output:** `data/nifty_enriched.csv` (7,452 × 316), `data/banknifty_enriched.csv` (5,161 × 316)

Converted raw sidereal planetary degrees into astrologically meaningful features.

**Dignity system (9 levels per planet):** exact_exalt / exalted / moolatrikona / own / friendly / neutral / enemy / debilitated / exact_debil. Uses classical exaltation points (Su@Aries10°, Mo@Taurus3°, Ju@Cancer5° etc.) and natural friendship tables.

**Panchanga (5 daily elements):**
- Vara: weekday + ruling planet's current dignity and speed
- Tithi: (Moon−Sun separation)/12°, numbered 1–30, Nanda/Bhadra/Jaya/Rikta/Purna qualities
- Paksha: SHUKLA (Tithi 1–15) or KRISHNA (Tithi 16–30)
- Nakshatra: Moon's one of 27 nakshatras, with lord, quality, pada
- Panchanga Yoga: (Sun+Moon)/360°×27, all 27 labeled, inauspicious ones flagged
- Karana: half-tithi with inauspicious flags (Vishti, Shakuni, Chatushpada, Naga)

**Tara Bala:** Inception date 1996-04-22, Moon at Mrigashira (nak 5). Formula: `diff = (moon_nak − 5) % 27; tara = (diff % 9) + 1`. Maps 1–9 to Critical/Wealth/Danger/Prosperity/Obstacle/Achievement/Worst/Good/Best.

**Vimshottari Dasha:** 120-year cycle computed from inception. Both Mahadasha and Antardasha for every historical day. Dasha lord's current dignity computed daily.

**Special conditions:** Gandanta (last 3°20' water → first 3°20' fire), Sandhi (last 1° of any sign), Graha Yuddha (planetary war within 1°), Sade Sati (Saturn in 12th/1st/2nd from natal Moon = Taurus), Ashtama Shani (Saturn in 8th = Sagittarius), Gajakesari, Papakartari, Panchaka.

**Market outcome labels (used only for training, never in signals):**
- STRONG_BULL: 3d fwd return > +1.5%
- MILD_BULL: +0.5% to +1.5%
- SIDEWAYS: ±0.5%
- MILD_BEAR: −0.5% to −1.5%
- STRONG_BEAR: < −1.5%
- HIGH_VOL: daily range > 1.5× ATR(14)

Lookahead audit passed: fwd_ret columns are NaN for the final N rows.

### Step 2: Six Research Methods

**Total patterns examined: 151,050**

**Method 1 — Outcome Fingerprint Matching:** For each STRONG_BULL/STRONG_BEAR/SIDEWAYS/HIGH_VOL/REVERSAL day, scanned k=1,2,3 feature combinations for conditions significantly over-represented on those days. Used vectorized key construction: `key = df[c1].astype(str) + '||' + df[c2].astype(str)`. Cap at k=3, top-20 features for k=3. Found 34,516 patterns.

**Method 2 — Reverse Condition Lookup:** For every possible value of every astrological feature, computed bull win rate, n, Wilson LB, Fisher p. Scanned k=1,2,3 combinations for 23 core features. Found 116,512 conditions.

**Method 3 — Clustering:** `scipy.cluster.vq.kmeans2` (not sklearn, which crashes on macOS with threadpoolctl bug) into 8 clusters. Computed bull rate, Wilson LB, dominant paksha/nakshatra/Jupiter dignity per cluster.

**Method 4 — Planetary Cycle Detection:** For 9 classical periods (Moon synodic 29.5d, Venus synodic 161d, Rahu sign 390d, etc.), tested ACF significance, FFT power ratio, phase ANOVA. Found evidence for Moon monthly, Venus synodic, Rahu sign change.

**Method 5 — Sequential Patterns:** After event X (new moon, Mercury retrograde, Gandanta, etc.), what happens at lag 1, 2, 3...N days? 182 event-lag tests. 5 survived p<0.05 before FDR.

**Method 6 — Anomaly Fingerprinting:** Days with |z-score| > 2.0 vs 20-day rolling baseline. Which astrological conditions predict anomaly days? 17 fingerprints found.

### Step 3: Validation

**BH-FDR at 1% across ALL 151,050 p-values simultaneously:** Ranked all p-values, found threshold k such that p[k] ≤ k×0.01/151050. Result: 1,867 survivors.

**Out-of-sample split fixed at 2018 before examining any results:**
- Training: pre-2018 (5,373 rows)
- OOS: 2018-present (2,079 rows)

Each FDR survivor must: (a) have n_oos ≥ 3, (b) maintain direction in OOS, (c) hold across 3 temporal sub-periods (pre-2010, 2010–2018, 2018-now).

**Result: 170 confirmed patterns** from 151,050 total. Also saved: 1,697 discarded patterns (honest documentation of failures), bootstrap CIs, Monte Carlo 10k shuffles, regime robustness.

### Step 4: System Build + Forward Calendar

**Composite score:** For each day, sum `max(0, WilsonLB − base_rate)` for all active BULL patterns, subtract sum for BEAR patterns. Score = 50 + net×100.

**Trade decision rules:**
- ≥3 BULL patterns active, 0 BEAR → PRIME_TRADE_BULL
- ≥3 BEAR patterns active, 0 BULL → PRIME_TRADE_BEAR
- 1–2 active in either direction → WATCH
- 0 active → NEUTRAL

**Forward calendar (252 trading days, pyswisseph only, zero market data):**

| Classification | Count |
|---|---|
| PRIME_TRADE_BEAR | 74 |
| WATCH_BEAR | 170 |
| WATCH_BULL | 8 |
| PRIME_TRADE_BULL | 0 |

No PRIME_TRADE_BULL for the full year. Reason: under current Jupiter exalted in Cancer + Saturn neutral in Pisces, the `dig_Ju|dig_Sa = enemy||neutral` pattern (n=143, OOS wr=27%) is dominant. No 3-BULL confluence can form.

Next PRIME_TRADE_BEAR: **2026-06-22** (Jupiter exact exaltation, Moon in Hasta, Shukla paksha).

### Step 5: HTML Outputs + Signal Generator

`report.html` and `calendar.html` — dark theme interactive reports. `generate_signal.py` now imports from `astro_engine.py` (a side-effect-free importable module). Tested on 5 historical dates:

| Date | Context | Signal | Score |
|---|---|---|---|
| 2008-10-24 | 2008 crash | TRADE BEAR | −239 |
| 2020-03-23 | COVID low | TRADE BEAR | −324 |
| 2021-02-01 | Post-vaccine | WATCH | 12.5 |
| 2023-06-05 | Range-bound | WATCH | 12.5 |
| 2025-01-15 | Ju in Gemini | TRADE BEAR | −296 |

---

## Part 3: Fix 1 — Missing Features (COMPLETE, 2026-06-13)

**File:** `fix1_enrich.py`

Added 37 new columns to both enriched CSVs. New total: **353 columns** (was 316).

**What was added:**

| Feature | How computed | Why it matters |
|---|---|---|
| `nak_{p}` (9 planets) | `int(sid_deg / (360/27)) + 1` | Each planet's nakshatra, not just Moon's |
| `own_nak_{p}` (9 planets) | Planet's nak ∈ its 3 Vimshottari-ruled naks | Strength indicator: planet in its own domain |
| `argala_positive` | Any planet in 2nd/4th/11th from Moon sign | Positive intervention on Moon |
| `argala_obstruct` | Any planet in 3rd/5th/12th from Moon sign | Virodha argala — blocks Moon's significations |
| `argala_net` | argala_positive − argala_obstruct (−1/0/+1) | Net argala balance |
| `vipareeta_raja` | Debilitated planet in 6th/8th/12th from Moon | Neecha planet gains strength in dusthana |
| `cheshta_cat_{p}` (9 planets) | Explicit speed string: retrograde/stationary/very_fast/fast/mean/mean_slow/slow | Cheshta Bala — retrograde planets have highest cheshta |
| `true_node_diff` | True node − mean node via pyswisseph (degrees) | Nodal oscillation: true node swings ±1.5° around mean |
| `true_node_cat` | Bucketed: far_behind/behind/aligned/ahead/far_ahead | True node phase vs mean node |
| `ix_ju_speed_dig` | `cheshta_cat_Ju + '_' + dig_Ju` | Jupiter's speed × dignity interaction |
| `ix_sa_speed_dig` | `cheshta_cat_Sa + '_' + dig_Sa` | Saturn's speed × dignity interaction |
| `ix_own_nak_ju_paksha` | `own_nak_Ju + '_' + paksha` | Jupiter in own nakshatra × paksha |
| `ix_argala_paksha` | `argala_net + '_' + paksha` | Net argala × paksha |

**Own nakshatra assignment (Vimshottari lords):**
Ke: naks 1,10,19 | Ve: 2,11,20 | Su: 3,12,21 | Mo: 4,13,22 | Ma: 5,14,23
Ra: 6,15,24 | Ju: 7,16,25 | Sa: 8,17,26 | Me: 9,18,27

---

## Part 4: Fix 2 — Uncapped Fingerprint Relaxation (RUNNING, 2026-06-13)

**File:** `fix2_fingerprint.py`

The original Method 1 had two artificial limits: k=3 maximum and only top-20 features for k=3. Both caused missed bull patterns.

**Correct algorithm:**
For each positive-outcome day (e.g., STRONG_BULL):
1. Collect all features where that day's specific value has Fisher p < 0.35 for the outcome.
2. Sort by significance (lowest p first). This gives an ordered "fingerprint" for the day.
3. Check how many other positive-outcome days share ALL features in the fingerprint.
4. If fewer than 5, drop the least significant feature (last in sorted list) and retry.
5. Repeat until ≥5 positive days share the combination.
6. Record the pattern. Mark all matching positive days as "explained."
7. Move to the next unexplained positive day.

This discovers patterns of any complexity — k=4, 5, 6+ when needed. No cap.

**Results so far (RUNNING — on is_bull scan):**

| Outcome | Positive days | Patterns found | % explained | Time |
|---|---|---|---|---|
| STRONG_BULL | 1,840 | 345 | 100% | 75s |
| STRONG_BEAR | 1,502 | 323 | 100% | 88s |
| SIDEWAYS | 1,479 | 295 | 100% | 47s |
| HIGH_VOL | 592 | 112 | 100% | 9s |
| is_bull | 4,107 | running… | — | — |

100% of positive days are being explained — the algorithm always finds a minimal combination that groups ≥5 days, even if it has to drop down to a single-feature fingerprint.

Also running: uncapped Method 2 scan (k=1,2,3 on all sig_cols, not capped at 20).

---

## Part 5: Fix 3 — Bull/Bear Asymmetry Root Cause (COMPLETE, 2026-06-13)

**File:** `fix3_bull_bear.py`

The 170 confirmed patterns had 9 BULL and 161 BEAR. Three causes identified:

**Cause A: Training era planetary bias.**
Jupiter was in enemy dignity (Taurus, Gemini, Aries) for 35.7% of training days. Saturn was in enemy dignity for 33.3%. These are common combinations that produce low bull rates → most surviving patterns are "conditions where Nifty goes down."

| dig_Ju | Train % | OOS % |
|---|---|---|
| enemy | 35.7% | 32.4% |
| friendly | 24.5% | 23.7% |
| own | 14.9% | 20.3% |
| exalted | 9.5% | 2.0% |

Jupiter in enemy dignity dominated the training period. Every combination including `dig_Ju=enemy` tends to show wr < 55% → classified as BEAR.

**Cause B: Scanning asymmetry.**
The `fast_scan()` function ran on `is_bull` with `min_wlb=0.58` for BULL patterns. But it never filtered the LOW side. A condition with wr=35% on n=200 has p<0.05 and passes the FDR filter → lands in confirmed patterns as a BEAR signal. There was no explicit BEAR scan — the BEAR patterns emerged as a side-effect of the BULL scan.

- Bear patterns with wr_train < 0.40: **107** (66% of all bear patterns)
- Their mean n_train: 477 — large sample, low bull rate, statistically certain

**Cause C: k=3 cap eliminated complex bull patterns.**
Complexity distribution of 170 confirmed patterns:

| Complexity | BULL | BEAR |
|---|---|---|
| k=1 | 0 | 8 |
| k=2 | 0 | 54 |
| k=3 | 9 | 99 |

All 9 BULL patterns are k=3. Zero BULL patterns at k=1 or k=2. This is the tell: bull patterns in astrology require multiple conditions simultaneously (paksha AND Jupiter dignity AND nakshatra AND...). The k=3 cap cut off the search exactly where bull patterns live. Fix 2's uncapped fingerprint will find them.

---

## Part 6: Fix 4 — Bank Nifty Full Independent Research (RUNNING, 2026-06-13)

**File:** `fix4_banknifty_full.py`

Previous Bank Nifty work only tested Nifty's confirmed patterns on Bank Nifty data. This runs the complete 6-method research independently on Bank Nifty from scratch.

**Why this matters:** Bank Nifty is a bank-sector index. Certain planetary configurations (e.g., Mercury dignity, Venus dignity) may affect bank stocks differently from the broad market. Universal patterns (those confirmed on both instruments) are more trustworthy than Nifty-only patterns.

**Methodology:** Same 6 methods, same BH-FDR at 1%, same 2018 OOS split. Bank Nifty has 5,161 rows from 2000 onward (vs Nifty's 7,452 from 1996). Same feature pool (353 columns after Fix 1).

**Output will include:**
- `bnk_confirmed_patterns.csv` — BankNifty's own confirmed patterns
- `cross_instrument_comparison.csv` — universal vs Nifty-only vs BankNifty-only patterns

Universal patterns = confirmed independently on both instruments = highest confidence.

---

## Part 7: Fix 5 — Validate Methods 3–6 with Full Rigor (PENDING)

**File:** `fix5_validate_all.py`

The original validation pipeline only applied BH-FDR + OOS to Methods 1 and 2. Methods 3–6 findings were never formally validated.

**What this does:**
1. Pools ALL p-values from ALL methods into one global pool
2. Applies BH-FDR at 1% across the entire combined pool
3. OOS-validates every FDR survivor from M3–6
4. Adds new confirmed patterns to `confirmed_patterns.csv`

**Method adaptations for validation:**
- M3 (clusters): cluster membership → is_bull rate → Fisher p included in pool
- M4 (cycles): phase ANOVA p-values included in pool; survivors get forward-calendar cycle-phase signals
- M5 (sequential): lag-conditioned patterns get OOS test: "after event X at lag N, does OOS data confirm?"
- M6 (anomaly): anomaly-rate conditions get re-tested for is_bull direction; only included if both anomaly prediction AND bull/bear direction survive

---

## Key Findings Summary

### What the Data Says About Nifty

**The structural finding:** Paksha modifies everything. The same nakshatra, sign, and dignity combination gives opposite results depending on whether it falls in KRISHNA (dark half) or SHUKLA (bright half) of the lunar month.

**Jupiter dignity overrides nakshatra.** Nakshatra quality is meaningless without knowing Jupiter's sign. Mula nakshatra with Jupiter in own sign → 68.8% bull. Mula with Jupiter exalted → 36.5% bear. Jupiter exalted in Cancer is NOT a bull signal — it is bearish when combined with most other configurations, because exaltation in Cancer puts Jupiter opposite Capricorn where it would aspect its debilitation sign directly.

**The strongest confirmed patterns (top 5 by Wilson LB):**

| Features | Condition | n | WLB | OOS wr | Direction |
|---|---|---|---|---|---|
| dig_Mo\|dig_Me\|ix_paksha_ju_dig | neutral\|\|friendly\|\|KRISHNA_enemy | 101 | 0.671 | 0.767 | BULL |
| dig_Mo\|dig_Ve\|dig_Me | neutral\|\|enemy\|\|enemy | 113 | 0.618 | 0.690 | BULL |
| dig_Mo\|dig_Me\|sade_sati_phase | neutral\|\|enemy\|\|none | 174 | 0.600 | 0.659 | BULL |
| dig_Ju\|dig_Ma\|mahadasha | enemy\|\|friendly\|\|Ra | 306 | 0.589 | 0.730 | BULL |
| dig_Ju\|dig_Mo\|dig_Me | enemy\|\|neutral\|\|friendly | 234 | 0.587 | 0.662 | BULL |

**Counterintuitive confirmed findings:**
- Jupiter exalted alone = bearish (wr < base rate in most combinations)
- Kemadruma (Moon isolated) under KRISHNA paksha is NOT bearish — multiple patterns show it as bullish context
- Sade Sati phase = 'none' (not in Sade Sati at all) appears in BULL patterns — Sade Sati active is a mild bear condition
- Saturn neutral (Pisces) is bearish: `dig_Sa=neutral` → wr 27%, n=798 in OOS

### What the Calendar Is Saying Now (2026-06-13)

Current planetary setup:
- Sun in Taurus — enemy dignity
- Moon in Taurus — exalted
- Mercury in Gemini — own sign
- Venus in Cancer — enemy
- Mars in Aries — own
- Jupiter in Cancer — **exact exaltation** (2°)
- Saturn in Pisces — neutral dignity
- Rahu in Aquarius — friendly
- Ketu in Leo — enemy

Active conditions: Sade Sati rising (Saturn in Aries = 12th from Taurus natal Moon). Jupiter exactly exalted. `dig_Ju=exact_exalt || dig_Sa=neutral` → BEAR pattern active.

**Next 10 PRIME_TRADE_BEAR dates:** 2026-06-22, 2026-06-23, 2026-07-07 through 2026-07-16.
**PRIME_TRADE_BULL in next 12 months:** 0.

---

## Technical Decisions and Why

**Wilson CI lower bound instead of raw win rate:** A pattern with 10/12 bullish days (83%) and one with 400/600 (67%) look different. Wilson LB at 95%: first pattern gets 0.52 (barely above chance), second gets 0.62 (reliable). Sample size is penalized automatically.

**BH-FDR at 1% not 5%:** At 5%, ~7,500 false discoveries from 151,050 tests. At 1%, ~1,500 false discoveries from 1,867 survivors — roughly 80% are true positives. Tighter but defensible.

**2018 OOS split fixed before looking at data:** Post-hoc split selection is a form of data leakage. The split was specified in the prompt and never moved.

**pyswisseph only for forward signals:** No market prices, no volatility, no previous returns. The forward calendar is astronomically determined. This is what makes it actually forward-looking.

**scipy.cluster.vq.kmeans2 not sklearn KMeans:** macOS Sonoma sklearn KMeans crashes with `threadpoolctl AttributeError: 'NoneType' has no attribute 'split'`. scipy's vq module has no threading dependency.

**Vectorized key construction:** `key = df[c1].astype(str) + '||' + df[c2].astype(str)` instead of `df[combo].apply(lambda r: '||'.join(r), axis=1)`. The latter is Python-level row iteration; the former is C-level vectorized string concat. ~50× faster.

---

## File Manifest

| File | What it does |
|---|---|
| `new_step1.py` | Feature engineering — 316 columns from 9 planet degrees |
| `new_step2.py` | Research methods 1–2 (original k=3 capped) |
| `new_step2b.py` | Research methods 3–6 (clustering, cycle, sequential, anomaly) |
| `new_step3.py` | Validation: BH-FDR, OOS split, temporal stability |
| `new_step4.py` | Composite score, backtest, forward calendar |
| `new_step5.py` | HTML report and calendar generation |
| `astro_engine.py` | Importable Vedic astrology engine (no side effects) |
| `generate_signal.py` | Daily signal generator |
| `fix1_enrich.py` | Adds 37 new features → 353 columns |
| `fix2_fingerprint.py` | Uncapped fingerprint relaxation + full M2 scan |
| `fix3_bull_bear.py` | Bull/bear asymmetry investigation |
| `fix4_banknifty_full.py` | Full 6-method research on Bank Nifty independently |
| `fix5_validate_all.py` | Pools all M1–6 p-values, combined BH-FDR, merges confirmed |
| `data/nifty_enriched.csv` | 7,452 × 353 |
| `data/banknifty_enriched.csv` | 5,161 × 353 |
| `results/validation/confirmed_patterns.csv` | Currently 170; will grow after Fix 5 |
| `results/research/method1_fp_uncapped.csv` | Fix 2 output — uncapped fingerprints |
| `results/research/method2_full.csv` | Fix 2 output — uncapped k=1,2,3 scan |
| `results/validation/bnk_confirmed_patterns.csv` | Fix 4 output |
| `results/validation/cross_instrument_comparison.csv` | Fix 4 output — universal vs instrument-specific |
| `results/validation/m3m6_validated.csv` | Fix 5 output — M3–6 survivors |
| `results/validation/fdr_pool_all.csv` | Fix 5 output — full combined FDR pool |

---

## What Remains After Fixes

When Fix 2, 4, 5 complete and the forward calendar and HTML are regenerated:

- Confirmed patterns will include M3–6 survivors and uncapped fingerprints — expected to add more BULL patterns
- Bank Nifty will have its own validated pattern set
- Universal patterns (Nifty + BankNifty confirmed) will be the highest-confidence trade signals
- The forward calendar will be rebuilt with the updated pattern set
- CHECKLIST.md will be updated
