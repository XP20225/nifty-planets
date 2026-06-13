# AstroQuant Trading Manual
**NSE Nifty 50 · Composite Signal System**
Generated: 2026-06-13

> **Disclaimer:** This is a research system. All strategies carry risk of capital loss. Past backtest results do not guarantee future performance. This document is for educational and research purposes only, not financial advice.

---

## Quick Reference

```
python generate_signal.py YYYY-MM-DD
```

Prints the composite score (0–100) and classification for any NSE trading date.

---

## Signal Classification

| Score | Class | Interpretation | Action |
|-------|-------|----------------|--------|
| ≥ 75 | **PRIME TRADE** | Strong multi-signal confluence | Consider long entry (3-day hold) |
| 60–75 | **HIGH VOL** | Elevated opportunity | Monitor; enter on confirmation |
| 45–60 | **WATCH** | Marginal setup | Watch, no position |
| 30–45 | **NEUTRAL** | No signal | Stay flat or hold existing |
| < 30 | **AVOID** | Unfavourable | Avoid new longs |

---

## Composite Score Formula

```
score = (prob_up × 0.6 + signal_count/4 × 0.4) × 100
```

**Signal components (each 0 or 1):**

| Signal | Condition | Type |
|--------|-----------|------|
| S1 | Today's log return > 0 | Price momentum |
| S2 | Jupiter in Cancer / Sagittarius / Pisces | Astrological |
| S3 | Today's intraday range > training median (1.42%) | Volatility |
| S4 | Moon within 12° of Sun (combust) | Astrological |

`prob_up` = LightGBM Model B directional probability (OOS AUC 0.517).

**Important:** S1 and S3 require same-day EOD data. The signal is computed after market close and applied the next trading day (or at EOD close if immediate execution is possible).

---

## How to Use — Historical Dates

For any date already in the dataset (up to 2026-06-11):

```bash
python generate_signal.py 2024-03-15
```

The output includes:
- Composite score (0–100)
- Classification label
- All 4 signal component values
- prob_up from Model B
- Tithi, Moon nakshatra, Jupiter dignity, combust status

---

## How to Use — Future Dates

For dates after the last data point:

```bash
python generate_signal.py 2026-09-01
```

The output comes from the pre-computed forward_calendar.csv. Only astrological signals (S2, S3) are available; market signals (S1, S4) are set to 0.

**Maximum future score = 53 (WATCH)** — no future date will reach HIGH VOL or PRIME TRADE until market data is available.

---

## Trade Rules (Research System)

### Entry
- Signal computed at market close
- Enter long at **next open** (conservative) or **same-day close** (aggressive)
- Enter only when score ≥ 60 (HIGH VOL or PRIME TRADE)
- Prioritise days with score ≥ 75 and signal_count = 3 or 4

### Exit
- **Default:** Exit at close 3 trading days after entry
- **Stop-loss:** Consider -2% intraday stop (not modelled in backtest)

### Position Sizing
- The backtest assumes equal-weight per trade with no position sizing
- For live use, consider risking 1–2% of portfolio per trade maximum
- Avoid stacking multiple open positions from consecutive signals

### What to Avoid
- Avoid entering on days flagged as **dead zones** (5+ consecutive low-signal days)
- Avoid low-liquidity market hours
- Never chase entry if the open gaps significantly above entry price

---

## Campaign Windows

A **campaign window** is 3 or more consecutive HIGH VOL / PRIME TRADE days. Historically, these represent planetary clusters where momentum trades have higher follow-through probability.

The forward calendar (2026–2027) shows **0 campaign windows** because market signals are unavailable for future dates. Campaign windows will only be identifiable in real-time as market data accumulates.

---

## Dead Zones

A **dead zone** is 5 or more consecutive NEUTRAL / AVOID days. These indicate prolonged low-signal periods where the astrological environment is unfavourable.

The forward calendar shows **235 dead zone days** (93% of the forecast period). This is primarily because Jupiter enters Gemini in mid-2026, leaving its own/exaltation signs, and both key market signals (log_ret and range_pct) are unknown.

---

## Model Files

| File | Description |
|------|-------------|
| `models/model_a_volatility.pkl` | LightGBM volatility classifier (note: leakage documented) |
| `models/model_b_directional.pkl` | LightGBM directional classifier (OOS AUC 0.517) |
| `results/synthesis/composite_scores.csv` | Historical scores 1996–2026 |
| `results/synthesis/forward_calendar.csv` | 252 future trading days |

Load models:
```python
import joblib
model_b = joblib.load("models/model_b_directional.pkl")
```

---

## Performance Summary (Backtest Reference)

| Metric | Nifty 50 | Bank Nifty |
|--------|---------|---------|
| Total trades | 1,484 | 1,145 |
| Win rate | 83.6% | 65.7% |
| Mean return/trade | 1.40% | varies |
| Sharpe ratio | 5.89 | 3.16 |
| Max drawdown | -14.4% | — |

**The above reflects in-sample and out-of-sample backtest. It should not be used as an expectation for live trading.**

---

## Known Issues and Caveats

1. **Model A AUC = 1.0** — this is a data leakage artefact, not a genuine edge. It does not affect the composite score.
2. **83.6% win rate** — high because S1 (log_ret > 0) filters for days that already moved up; the signal captures EOD momentum continuation.
3. **Future calendar is degraded** — treat as a planetary background indicator only.
4. **Cumulative return (64 trillion %)** — do not interpret literally; it is a compounding artefact.

---

*For research questions and improvements, see RESEARCH_REPORT.md and the pipeline scripts (step1_features.py through step4_synthesis.py).*
