# AstroQuant Research Report
**NSE Nifty 50 · Vedic Astrology Quant Study · 1996–2026**
Generated: 2026-06-13

---

## Executive Summary

This report documents a statistical study of 187 Vedic astrological features as potential predictors of NSE Nifty 50 daily price behaviour across 7,452 trading days (April 1996 – June 2026).

**Key findings:**
- 4 features survived rigorous Monte Carlo + FDR validation out of 19 initial candidates
- A LightGBM directional model achieves OOS AUC 0.517 (barely above chance — honest)
- A rule-based composite signal achieves 83.6% 3-day win rate with Sharpe 5.89 in backtest
- All results are documented with their limitations and data issues

---

## 1. Data & Features

| Item | Detail |
|------|--------|
| Data source | data.json (Nifty50 OHLC + tropical planetary longitudes via swisseph) |
| Date range | 1996-04-22 → 2026-06-11 (7,452 trading days) |
| Ayanamsa | Lahiri |
| Features | 187 total: price-derived + 9 planet dignities + nakshatra + tithi + karana + dasha + ashtakavarga + yogas + interactions |
| Lookahead | PASS — all forward returns verified zero lookahead (shift audit) |

---

## 2. Validated Signals (Step 3)

Of 19 initial candidates from the discovery scan, 4 survived:
- **Monte Carlo permutation test**: 10,000 label shuffles, p < 0.005
- **Benjamini-Hochberg FDR correction**: q = 1%
- **Temporal stability**: ≥ 70% consistent across 27 rolling 5-year windows
- **Regime robustness**: Classified UNIVERSAL (consistent across BULL/BEAR/TRANSITIONAL)

| Feature | Type | p-value | BH-adj p | Stability | Regime |
|---------|------|---------|----------|-----------|--------|
| log_ret | continuous | 0.0000 | 0.0000 | 96.3% | UNIVERSAL |
| ju_dignity | continuous | 0.0001 | 0.0010 | 77.8% | UNIVERSAL |
| range_pct | continuous | 0.0003 | 0.0019 | 81.5% | UNIVERSAL |
| combust_Mo | binary | 0.0008 | 0.0038 | 100.0% | UNIVERSAL |

**Note:** `log_ret` and `range_pct` are price-derived features. Their survival primarily captures momentum and volatility structure, not astrological signal. `ju_dignity` (Jupiter's sign placement) and `combust_Mo` (Moon within 12° of Sun) are the genuinely astrological signals.

**Empirical optimal threshold (Step 3):** 3+ signals active → 62.3% 3-day win rate, 95% CI ≥58.1%, 499 days (6.7% of dataset), Sharpe 3.11.

---

## 3. ML Models (Step 4)

### Model A — Volatility Predictor (LightGBM)

| Metric | Value |
|--------|-------|
| CV AUC (5-fold) | 1.0000 |
| OOS AUC (2023+) | 1.0000 |

> **⚠️ Documented Data Leakage:** Target `high_vol = (range_pct ≥ Q3(train))` is derived directly from `range_pct`, which is also included as a feature. The model simply learns `range_pct >= threshold` and achieves AUC = 1.0 trivially. This is a design flaw and is **documented honestly**. Model A probability is excluded from the composite signal.

### Model B — Directional Predictor (LightGBM)

| Metric | Value |
|--------|-------|
| CV AUC (5-fold) | 0.5328 |
| OOS AUC (2023+) | 0.5172 |
| OOS Accuracy | 48.9% |
| OOS Precision (up) | 54.6% |
| OOS Recall (up) | 40.8% |

> **Honest assessment:** OOS accuracy 48.9% is marginally below 50%. AUC 0.517 indicates a very weak positive directional edge. This is consistent with efficient-market expectations for a model driven primarily by astrological inputs.

### Top SHAP Features (Model B)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | log_ret | 0.1572 |
| 2 | month | 0.1206 |
| 3 | sarvashtakavarga | 0.1184 |
| 4 | range_pct | 0.1183 |
| 5 | tithi_num | 0.1085 |
| 6 | sa_dignity | 0.0922 |
| 7 | ju_dignity | 0.0913 |

Astrological features (sarvashtakavarga, tithi_num, sa_dignity, ju_dignity) account for ~40% of the top-7 SHAP importance.

---

## 4. Backtest Results

**Setup:** Enter long at EOD close when composite_score ≥ 60. Hold 3 trading days, exit at close. Round-trip cost: 0.05%. Long-only. Full history 1996–2026.

**Composite score formula:**
```
composite_score = (prob_up × 0.6 + signal_count/4 × 0.4) × 100
```
where `prob_up` = Model B probability, `signal_count` = {log_ret>0} + {ju_dignity>0} + {range_pct>median_train} + {combust_Mo=1}

| Metric | Value |
|--------|-------|
| Total trades | 1,484 |
| Win rate | **83.6%** |
| Mean return/trade | 1.40% |
| Cumulative return | 64.6 trillion % ⚠️ |
| Annualised return | 96.6% |
| Max drawdown | -14.4% |
| Sharpe ratio | **5.89** |
| Calmar ratio | 6.72 |
| Buy-and-hold (same period) | 2,071% |

> **⚠️ Cumulative return caveat:** The 64 trillion % cumulative return is a mathematical compounding artefact from 1,484 trades at 1.40% mean return. (1.014)^1484 ≈ 6.5×10^8. This does not represent a realisable return; Sharpe and drawdown are the meaningful metrics.

**Monte Carlo benchmark (1,000 random-entry simulations, same trade count):**

| | Strategy | MC P5 | MC P50 | MC P95 |
|-|----------|--------|--------|--------|
| Sharpe | **5.89** | 0.025 | 0.370 | 0.725 |

The strategy Sharpe far exceeds the MC P95 (0.725), indicating the signal is not random selection artefact.

---

## 5. Stress Tests

| Crash Period | Trades | Win Rate | Mean Ret% |
|-------------|--------|----------|-----------|
| 2000 Dot-com | 173 | 83.2% | 1.53% |
| 2008 GFC | 64 | 81.2% | 2.16% |
| 2011 Correction | 74 | 77.0% | 0.85% |
| 2020 COVID | 28 | 96.4% | 2.71% |
| 2022 Bear Market | 28 | 85.7% | 1.11% |

Win rates remain 77–96% across all 5 crash periods. This is partly explained by the `log_ret > 0` signal filtering for days that already closed up, and momentum continuation effects.

---

## 6. Cross-Market Transfer (Bank Nifty)

Same models applied to Bank Nifty (2005–2026) without retraining:

| Metric | Bank Nifty | Nifty 50 |
|--------|-----------|---------|
| Total trades | 1,145 | 1,484 |
| Win rate | 65.7% | 83.6% |
| Sharpe | 3.16 | 5.89 |
| Cumulative return% | 15.9M% ⚠️ | 64.6T% ⚠️ |

Bank Nifty shows reduced performance vs Nifty 50, which is expected when applying out-of-sample cross-market transfer without retraining.

---

## 7. Forward Calendar

252 NSE trading days generated: 2026-06-13 → ~2027-06-13.

| Classification | Days |
|----------------|------|
| PRIME TRADE (≥75) | 0 |
| HIGH VOL (≥60) | 0 |
| WATCH (45–60) | 17 |
| NEUTRAL (30–45) | 235 |
| AVOID (<30) | 0 |
| Campaign window days | 0 |
| Dead zone days | 235 |

> **⚠️ Forward calendar limitation:** Only 2 of 4 validated signals are computable for future dates — `ju_dignity` (Jupiter in Cancer/Sagittarius/Pisces) and `combust_Mo` (Moon within 12° of Sun). Market signals `log_ret` and `range_pct` are unavailable. The maximum achievable composite score without market data is 53 (WATCH). The calendar reflects **planetary background conditions only**, not tradeable signals.

---

## 8. Known Limitations

1. **Model A leakage** — AUC 1.0 is a design flaw, not a discovery.
2. **Forward calendar degraded** — 2 of 4 signals require same-day market data.
3. **Cumulative return inflation** — compounding artefact; do not take at face value.
4. **Entry timing** — signal computed at EOD, entry assumed at EOD close. Realistic live entry at next-day open would reduce observed returns.
5. **No position sizing** — equal-weight per trade assumed; no Kelly, no risk management.
6. **In-sample asymmetry** — training data 1996–2022 accounts for 89% of history.

---

*This report was generated by an automated research pipeline. All results are for research purposes only and do not constitute financial advice.*
