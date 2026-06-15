# So Far — AstroQuant Pipeline v2 + Fixes
Complete record of everything built, found, and fixed.
Last updated: 2026-06-14

---

## Quick Status

| Stage | Status | Key Output |
|---|---|---|
| Pipeline v2 rebuild (Steps 1–5) | COMPLETE | 170 confirmed patterns, 252-day calendar |
| Fix 1 — Missing Vedic features | COMPLETE | 353 columns (was 316), 37 new features |
| Fix 2 — Uncapped fingerprint | COMPLETE | 1,797 M1 patterns in 93s (851× speedup) |
| Fix 3 — Bull/bear investigation | COMPLETE | 3 root causes identified and documented |
| Fix 4 — Bank Nifty independent research | COMPLETE | 642 bnk patterns, 1 universal |
| Fix 5 — Validate M3–6 + full FDR merge | COMPLETE | **335 confirmed patterns** (128 BULL, 207 BEAR) |
| Fix 6 — Comprehensive inter-planetary aspects | COMPLETE | 398 new features → **751 columns** |
| Aspect scan (k=1,2 on aspect features) | IN PROGRESS | Finding simple aspect-only patterns |

---

## Part 1: Why the Pipeline Was Rebuilt

The original pipeline (before this session) produced AUC 0.517 and a 252-day forward calendar showing every day as NEUTRAL. Both failures had the same root cause: same-day market data (log_ret, range_pct) was used as a signal feature. Those numbers are only known after market close. A forward-looking system cannot depend on them.

The forward calendar was all-NEUTRAL because the signal that determined trade direction required same-day log_ret. When computing future dates where log_ret is unknowable, every day defaulted to NEUTRAL.

The LightGBM model (AUC 0.517) was trained on astrological columns but without understanding what they mean. One-hot encoding all 27 nakshatras and feeding them to gradient boosting finds coincidences with no interpretable structure. The result is near-random.

The rebuild constraint: **no market data of any kind in any forward-looking signal.** All signals use only pyswisseph planetary positions.

---

## Part 2: Pipeline v2

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

**Result: 170 confirmed patterns** from 151,050 total.

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

No PRIME_TRADE_BULL for the full year. Reason: under current Jupiter exalted in Cancer + Saturn neutral in Pisces, the `dig_Ju|dig_Sa = enemy||neutral` pattern (n=143, OOS wr=27%) is dominant.

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

## Part 3: Fix 1 — Missing Vedic Features (COMPLETE, 2026-06-13)

**File:** `fix1_enrich.py`
**Runtime:** 4.4 seconds

Added 37 new columns to both enriched CSVs. New total: **353 columns** (was 316).

| Feature | How computed | Why it matters |
|---|---|---|
| `nak_{p}` (9 planets) | `int(sid_deg / (360/27)) + 1` | Each planet's nakshatra, not just Moon's |
| `own_nak_{p}` (9 planets) | Planet's nak ∈ its 3 Vimshottari-ruled naks | Strength indicator: planet in its own domain |
| `argala_positive` | Any planet in 2nd/4th/11th from Moon sign | Positive intervention on Moon's significations |
| `argala_obstruct` | Any planet in 3rd/5th/12th from Moon sign | Virodha argala — blocks Moon's significations |
| `argala_net` | argala_positive − argala_obstruct (−1/0/+1) | Net argala balance |
| `vipareeta_raja` | Debilitated planet in 6th/8th/12th from Moon | Neecha planet gains strength in dusthana |
| `cheshta_cat_{p}` (9 planets) | retrograde/stationary/very_fast/fast/mean/slow | Cheshta Bala — retrograde planets highest |
| `true_node_diff` | True node − mean node via pyswisseph (degrees) | Nodal oscillation ±1.5° around mean |
| `true_node_cat` | far_behind/behind/aligned/ahead/far_ahead | True node phase vs mean node |
| `ix_ju_speed_dig` | `cheshta_cat_Ju + '_' + dig_Ju` | Jupiter speed × dignity |
| `ix_sa_speed_dig` | `cheshta_cat_Sa + '_' + dig_Sa` | Saturn speed × dignity |

---

## Part 4: Fix 2 — Uncapped Fingerprint Relaxation (COMPLETE, 2026-06-13)

**File:** `fix2_fingerprint.py`
**Feature pool:** 668 columns (auto-discovered after Fix 6 added aspect features)
**M1 result:** 1,797 patterns in 93s | **M2:** killed (330 sig_cols × k=3 = 5.9M combos, infeasible)

The original Method 1 had two artificial limits: k=3 maximum and only top-20 features for k=3. Both caused missed bull patterns (all 9 confirmed BULL patterns were exactly k=3 — the cap hit exactly where complex bull patterns live).

### Algorithm

For each positive-outcome day:
1. Collect all features where that day's specific value has Fisher p < 0.35.
2. Sort by significance. This is the day's ordered "fingerprint."
3. Check how many other positive-outcome days share ALL features.
4. If fewer than 5, drop the least significant feature and retry.
5. Repeat until ≥5 days share the combination. No k cap.
6. Record the pattern. Mark matching days "explained." Move to next unexplained day.

### Performance fix (851× speedup)

Original: `df_pos[col].astype(str) == val` rebuilt inside every while-loop iteration.
For `is_bull` with 4,107 positive days × 41 features × 28 while-loop iterations = 4.7M pandas calls → 38+ minutes.

Fix: precompute numpy boolean arrays ONCE:
```python
pos_str   = {c: df_pos[c].astype(str).values   for c in present_cols}  # once
pos_bool  = {(col,val): (pos_str[col] == val) for (col,val) in pval_cache}  # once

# Inner loop: numpy bitwise AND
mask = np.ones(n_pos, dtype=bool)
for col, val, _ in active:
    mask &= pos_bool[(col, val)]
```
Also replaced `df_pos.iloc[i][c]` (builds full pandas Series) with `pos_str[c][i]` (direct array index). Combined: 851× speedup. is_bull: 2.0s (was 38+ minutes).

### Method 1 results (after Fix 6, 668-col pool)

| Outcome | Patterns | Time |
|---|---|---|
| STRONG_BULL | 345 | 1.3s |
| STRONG_BEAR | 323 | 1.3s |
| SIDEWAYS | 295 | 1.2s |
| HIGH_VOL | 112 | 0.8s |
| BULL_DIR | 722 | 2.0s |
| **Total** | **1,797** | **93s total** |

Note: fingerprint patterns use 27-40 features each (complex, sparse intersections). They explain 100% of positive days but are too overfit (n=5-10 in training → n=0-1 in OOS) to survive Fix 5 validation directly. Only patterns with p<0.05 AND unique vs original M1 are added to the Fix 5 pool.

---

## Part 5: Fix 3 — Bull/Bear Asymmetry Root Cause (COMPLETE, 2026-06-13)

**File:** `fix3_bull_bear.py`

The 170 confirmed patterns had **9 BULL and 161 BEAR**. Three causes identified:

**Cause A: Training era planetary bias.**
Jupiter was in enemy dignity for 35.7% of training days. Saturn in enemy for 33.3%. These produce low bull rates → most surviving patterns are bearish conditions.

**Cause B: Scanning asymmetry.**
`fast_scan()` ran on `is_bull` with `min_wlb=0.58` for BULL. No explicit BEAR scan — BEAR patterns emerged as side-effect. Bear patterns (wr < 40%) automatically pass significance tests when n > 200.

**Cause C: k=3 cap eliminated complex bull patterns.**
All 9 confirmed BULL patterns were exactly k=3. Fix 2's uncapped M1 found 845 BULL_DIR patterns — they exist, they needed k > 3.

---

## Part 6: Fix 4 — Bank Nifty Full Independent Research (COMPLETE, 2026-06-13)

**File:** `fix4_banknifty_full.py`

Runs complete 6-method research on Bank Nifty from scratch (5,161 rows, 2000–present). Bank Nifty: 93,952 raw patterns → 1,234 FDR survivors → **642 confirmed patterns** (141 BULL, 501 BEAR).

**Best Bank Nifty BULL pattern:** `ix_paksha_ju_dig = KRISHNA_moolatrikona` (Jupiter in own sign during dark fortnight) — OOS win rate 68.2%. With fast Jupiter speed: 82.4% OOS.

**The single universal pattern** (confirmed on BOTH instruments):
`dig_Ju|dig_Me = own||neutral` → BEAR. Nifty WLB=0.286, Bank Nifty WLB=0.341. When Jupiter is in its own sign AND Mercury is neutral simultaneously, both indices are bearish.

Cross-instrument: 1 universal, 163 Nifty-only, 641 Bank Nifty-only. Near-zero overlap means patterns are largely instrument-specific.

---

## Part 7: Fix 5 — Validate All Methods with Full BH-FDR (COMPLETE, 2026-06-13)

**File:** `fix5_validate_all.py`

### Pool construction
| Source | Rows | Notes |
|---|---|---|
| Original M1 (`method1_pattern_library.csv`) | 34,516 | All p<0.05, always included |
| Fingerprint M1 (`method1_fp_uncapped.csv`) | +531 | New patterns not in M1, p<0.05 only |
| Original M2 (`method2_reverse_lookup.csv`) | varies | Fallback (M2 full was killed) |
| M3 clusters | 8 | Fisher p for each cluster |
| M4 cycles | varies | ANOVA p-values for phase effects |
| M5 sequential | 5 | Event-lag patterns |
| M6 anomaly | 17 | Anomaly fingerprints |
| **Total pool** | **~151,728** | |

### BH-FDR fix
Original bug: fix5 used fingerprint (1,921 patterns, most p>0.35) INSTEAD of original M1 (34,516 patterns, all p<0.05). This collapsed FDR survivors from 1,867 to 22.

Fix: always include original M1, then ADD fingerprint patterns that are (a) new vs M1 AND (b) p<0.05.

### Results: 335 confirmed patterns (128 BULL, 207 BEAR)

**Source breakdown:**
- 170 from original pipeline v2
- 165 new from Fix 5 (M1 fingerprint + M3-6 survivors)

**Top 5 BULL patterns by Wilson LB:**

| Features | Condition | n | WLB | OOS wr |
|---|---|---|---|---|
| dig_Mo\|dig_Me\|ix_paksha_ju_dig | neutral\|\|friendly\|\|KRISHNA_enemy | 101 | 0.671 | 0.767 |
| dig_Mo\|dig_Ve\|dig_Me | neutral\|\|enemy\|\|enemy | 113 | 0.618 | 0.690 |
| dig_Mo\|dig_Me\|sade_sati_phase | neutral\|\|enemy\|\|none | 174 | 0.600 | 0.659 |
| dig_Ju\|dig_Ma\|mahadasha | enemy\|\|friendly\|\|Ra | 306 | 0.589 | 0.730 |
| dig_Ju\|dig_Mo\|dig_Me | enemy\|\|neutral\|\|friendly | 234 | 0.587 | 0.662 |

---

## Part 8: Fix 6 — Comprehensive Inter-Planetary Aspects (COMPLETE, 2026-06-14)

**File:** `fix6_aspects.py`
**Output:** Both enriched CSVs expanded from 353 → **751 columns** (+398 new features)

### Why this was missing

The original pipeline had only 4 aspect features: `ju_asp_mo`, `sa_asp_mo`, `ma_asp_mo`, `ju_sa_aspect` (Jupiter and Saturn aspecting Moon). Missing:
- Aspects between non-Moon planets (Jupiter→Saturn, Saturn→Mars, etc.)
- Full 7th-house (opposition) aspect from ALL planets
- Special house aspects: Saturn's 3rd/10th, Mars's 4th/8th, Jupiter's 5th/9th (Vedic rules)
- Lord-chain: Saturn aspecting Sagittarius influences Jupiter (lord of Sag) — `asp_Sa_dom_Ju`
- Degree-based aspects (Western-style trines/squares/sextiles with orbital tolerances)
- Natal chart aspects (transiting planets aspecting Nifty inception Moon sign = Taurus)

### Vedic aspect rules (planetary drishtis)

| Planet | Houses it aspects (from its own sign) |
|---|---|
| Sun, Moon, Mercury, Venus | 7th only |
| Mars | 4th, 7th, 8th |
| Jupiter | 5th, 7th, 9th |
| Saturn | 3rd, 7th, 10th |
| Rahu, Ketu | 5th, 7th, 9th (Jupiter-like) |

Formula: `aspected_sign = ((sign_P1 - 1 + H - 1) % 12) + 1`
Planet's own sign = H=1, so 7th house = H=7 gives 6 signs forward (inclusive counting).

### Lord-domain chain

When Saturn (sign 12 = Pisces) aspects house 3 → Taurus (sign 2) → lord of Taurus is Venus. So Saturn's 3rd-house drishti influences Venus's domain. Feature: `asp3_Sa_dom_Ve=1`.

Formula: for each aspected sign, look up its lord. Check if that lord rules any of the aspected signs.

**Sign lord table:** {1:Ma, 2:Ve, 3:Me, 4:Mo, 5:Su, 6:Me, 7:Ve, 8:Ma, 9:Ju, 10:Sa, 11:Sa, 12:Ju}

### Feature categories added (+398 total)

| Category | Count | Example |
|---|---|---|
| Vedic sign-based aspects (9 planets × their houses × 8 targets) | ~255 | `asp7_Ju_Sa=1` (Jupiter's 7th asp hits Saturn) |
| Lord-domain chain features | 56 | `asp_Sa_dom_Ju=1` (Saturn aspects Ju's sign) |
| Aspected sign & lord labels | 40 | `aspected_sign_Sa_3=2` (Sa's 3rd = Taurus) |
| Natal Moon (Taurus) aspect tracking | 20 | `asp7_Ju_natal_mo=1` (Jupiter opposes natal Moon) |
| Degree-based aspects (20 pairs × 5 types) | 100 | `deg_trine_Ju_Sa=1` (120° ± 7°) |
| Exact aspects ≤3° (20 pairs × 5 types) | 100 | `ex_conj_Ju_Sa=1` (conjunction within 3°) |
| Aggregate planet counts under slow planet's field | 3 | `n_asp_under_Sa=3` (3 planets in Sa's aspect field) |

**Degree-based aspect tolerances:**
- Conjunction (0°): ±8° orb
- Opposition (180°): ±8° orb
- Trine (120°): ±7° orb
- Square (90°): ±7° orb
- Sextile (60°): ±6° orb

**20 key planet pairs for degree-based aspects:**
Ju-Sa, Ju-Ma, Sa-Ma, Ju-Su, Sa-Su, Ma-Su, Ju-Mo, Sa-Mo, Ma-Mo, Ve-Ju, Ve-Sa, Ve-Ma, Me-Ju, Me-Sa, Me-Ma, Ra-Ju, Ra-Sa, Ra-Mo, Su-Mo, Me-Ve

### Forward calendar integration

`compute_day_features` in `new_step4.py` now computes all 398 aspect features for each future date using `sid[p]` (sidereal degrees already available). This ensures confirmed patterns that reference aspect features can fire in the forward calendar.

### Why fingerprint patterns don't use aspects (yet)

The fingerprint relaxation with 668 columns creates 27-40 feature patterns (because with more columns, more features have p<0.35 for a given day, and the intersection n_match ≥ 5 takes many features). These over-fit patterns (n=5-10 in training) don't survive OOS validation.

Simple k=1,2 patterns using aspect features require a targeted scan. The `aspect_scan.py` script runs k=1,2 scan on the ~350 binary/categorical aspect columns only. Results feed into Fix 5's pool for FDR + OOS validation.

---

## Key Findings Summary

### What the Data Says About Nifty

**The structural finding:** Paksha modifies everything. The same nakshatra, sign, and dignity combination gives opposite results depending on KRISHNA (dark half) vs SHUKLA (bright half).

**Jupiter dignity overrides nakshatra.** Nakshatra quality is meaningless without knowing Jupiter's sign. Mula nakshatra with Jupiter in own sign → 68.8% bull. Mula with Jupiter exalted → 36.5% bear. Jupiter exalted in Cancer is NOT a bull signal.

**Counterintuitive confirmed findings:**
- Jupiter exalted alone = bearish in most combinations
- Kemadruma (Moon isolated) under KRISHNA paksha is bullish context in multiple patterns
- Sade Sati phase = 'none' (not in Sade Sati) appears in BULL patterns
- Saturn neutral (Pisces) is bearish: wr 27%, n=798 in OOS

### What the Calendar Is Saying Now (2026-06-14)

Current planetary setup:
- Jupiter in Cancer — **exact exaltation** (2°)
- Saturn in Pisces — neutral dignity
- `dig_Ju=exact_exalt + dig_Sa=neutral` → dominant BEAR pattern active

**Next PRIME_TRADE_BEAR dates:** 2026-06-22, 2026-06-23, 2026-07-07 through 2026-07-16.
**PRIME_TRADE_BULL in next 12 months:** 0.

---

## Technical Decisions and Why

**Wilson CI lower bound instead of raw win rate:** Sample size is penalized automatically. A 10/12 result gets WLB=0.52; a 400/600 result gets WLB=0.62.

**BH-FDR at 1% not 5%:** At 5%, ~7,500 false discoveries from 151,050 tests. At 1%, ~1,500 false from 1,867 survivors — ~80% true positives.

**2018 OOS split fixed before looking at data:** Post-hoc split selection is data leakage. The split was specified in the prompt and never moved.

**pyswisseph only for forward signals:** No market prices, no volatility, no previous returns. Astronomically determined.

**scipy.cluster.vq.kmeans2 not sklearn KMeans:** macOS Sonoma sklearn KMeans crashes with `threadpoolctl AttributeError`. scipy's vq module has no threading dependency.

**Numpy bitmask precomputation:** Any algorithm checking feature-value combinations in a while loop must precompute boolean arrays once. 851× speedup documented above.

**Vedic aspect formula (inclusive counting):** `aspected_sign = ((sign_P1 - 1 + H - 1) % 12) + 1`. Planet's own sign = H=1 (identity). The "-1" before modulo and "+1" after implements 1-indexed inclusive house counting.

---

## File Manifest

| File | What it does |
|---|---|
| `new_step1.py` | Feature engineering — 316 columns from 9 planet degrees |
| `new_step2.py` | Research methods 1–2 (original k=3 capped) |
| `new_step2b.py` | Research methods 3–6 (clustering, cycle, sequential, anomaly) |
| `new_step3.py` | Validation: BH-FDR, OOS split, temporal stability |
| `new_step4.py` | Composite score, backtest, forward calendar (includes aspect computation) |
| `new_step5.py` | HTML report and calendar generation |
| `astro_engine.py` | Importable Vedic astrology engine (no side effects on import) |
| `generate_signal.py` | Daily signal generator |
| `fix1_enrich.py` | Adds 37 new Vedic features → 353 columns |
| `fix2_fingerprint.py` | Uncapped fingerprint relaxation (851× speedup, 668-col pool) |
| `fix3_bull_bear.py` | Bull/bear asymmetry investigation |
| `fix4_banknifty_full.py` | Full 6-method research on Bank Nifty independently |
| `fix5_validate_all.py` | Global BH-FDR + OOS, merges all methods, outputs confirmed_patterns.csv |
| `fix6_aspects.py` | Adds 398 inter-planetary aspect features to both enriched CSVs |
| `aspect_scan.py` | Targeted k=1,2 scan on aspect features only |
| `data/nifty_enriched.csv` | 7,452 × **751** (after Fix 6) |
| `data/banknifty_enriched.csv` | 5,161 × **751** (after Fix 6) |
| `results/validation/confirmed_patterns.csv` | **335 patterns** (128 BULL, 207 BEAR) |
| `results/research/method1_fp_uncapped.csv` | Fix 2 M1: 1,797 fingerprint patterns |
| `results/research/method1_asp_scan.csv` | Aspect scan: k=1,2 aspect-only patterns (in progress) |
| `results/validation/bnk_confirmed_patterns.csv` | Fix 4: 642 Bank Nifty confirmed patterns |
| `results/validation/cross_instrument_comparison.csv` | Fix 4: 805-row universal/nifty/bnk comparison |
| `results/validation/m3m6_validated.csv` | Fix 5: M3-6 survivors |
| `results/forward_calendar/planetary_calendar_1yr.csv` | 252-day forward calendar |

---

## What Remains

1. **Aspect scan** — k=1,2 scan on ~350 aspect columns (running). Finds simple patterns like "asp7_Ju_Sa=1 → BEAR".
2. **Fix 5 re-run** — After aspect scan: re-pool with aspect patterns, re-run BH-FDR + OOS. New count of confirmed patterns expected.
3. **Rebuild forward calendar** — `new_step4.py` with updated confirmed_patterns.csv.
4. **Regenerate HTML reports** — `new_step5.py`.
5. **Bank Nifty re-run** — `fix4_banknifty_full.py` with 751-column feature set to find aspect-based Bank Nifty patterns.
6. **Push to GitHub** — all Fix 6 + aspect scan results.
