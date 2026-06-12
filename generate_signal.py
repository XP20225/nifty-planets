"""
Daily Signal Generator — Astro Quant Research System
Usage:  python generate_signal.py [YYYY-MM-DD]
        (defaults to today if no date given)

For historical dates (in nifty_enriched.csv): reads stored features + models.
For future dates (in forward_calendar.csv): reads pre-computed planetary estimates.
Prints a one-line classification and full signal breakdown.
Runs in under 10 seconds.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE          = Path("/Users/vasanthakumaranpalanisamy/Nifty Planets")
DATA_DIR      = BASE / "data"
RESULTS_DIR   = BASE / "results" / "synthesis"
MODELS_DIR    = BASE / "models"

FEATURES = [
    "log_ret", "ju_dignity", "range_pct", "combust_Mo",
    "tithi_num", "paksha", "karana", "moon_nak_num", "moon_nak_quality",
    "sa_dignity", "moon_strength", "mahadasha", "antardasha",
    "sarvashtakavarga", "combust_Me", "combust_Ve", "combust_Ma",
    "combust_Ju", "combust_Sa", "gandanta_Su", "gandanta_Mo",
    "eclipse_corridor", "graha_yuddha", "dow", "month",
]

RANGE_PCT_MEDIAN_TRAIN = 0.0142  # from step 4 training set


def classification_label(score: float) -> str:
    if score >= 75:
        return "PRIME_TRADE"
    elif score >= 60:
        return "HIGH_VOL"
    elif score >= 45:
        return "WATCH"
    elif score >= 30:
        return "NEUTRAL"
    return "AVOID"


def load_models():
    model_a = joblib.load(MODELS_DIR / "model_a_volatility.pkl")
    model_b = joblib.load(MODELS_DIR / "model_b_directional.pkl")
    return model_a, model_b


def signal_for_historical(target_date: pd.Timestamp):
    """Read row from enriched CSV, run models, print signal."""
    df = pd.read_csv(DATA_DIR / "nifty_enriched.csv", parse_dates=["date"])
    row_mask = df["date"] == target_date
    if not row_mask.any():
        return None, "Date not found in historical data (nifty_enriched.csv)."

    # Encode categoricals on full column (must match step4 approach)
    cat_cols = ["paksha", "moon_nak_quality", "karana", "mahadasha", "antardasha"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    row = df[row_mask].iloc[0]
    X = pd.DataFrame([row])[FEATURES].fillna(0)

    model_a, model_b = load_models()
    prob_up  = model_b.predict_proba(X)[0, 1]
    prob_vol = model_a.predict_proba(X)[0, 1]

    s1 = int(row["log_ret"] > 0)
    s2 = int(row["ju_dignity"] > 0)
    s3 = int(row["range_pct"] > RANGE_PCT_MEDIAN_TRAIN)
    s4 = int(row["combust_Mo"] == 1)
    signal_count = s1 + s2 + s3 + s4

    composite = round((prob_up * 0.6 + signal_count / 4 * 0.4) * 100, 1)
    composite = float(np.clip(composite, 0, 100))
    label = classification_label(composite)

    return composite, label, {
        "source":        "historical",
        "prob_up":       round(prob_up, 4),
        "prob_vol":      round(prob_vol, 4),
        "signal_count":  signal_count,
        "s1_log_ret_pos":   s1,
        "s2_ju_dignity_pos":s2,
        "s3_range_above_med":s3,
        "s4_moon_combust":  s4,
        "log_ret":       round(float(row["log_ret"]), 6) if not pd.isna(row["log_ret"]) else None,
        "range_pct":     round(float(row["range_pct"]), 6) if not pd.isna(row["range_pct"]) else None,
        "ju_dignity":    int(row["ju_dignity"]) if not pd.isna(row["ju_dignity"]) else None,
        "combust_Mo":    int(row["combust_Mo"]) if not pd.isna(row["combust_Mo"]) else None,
        "tithi":         int(row["tithi_num"]) if not pd.isna(row["tithi_num"]) else None,
        "moon_nakshatra":int(row["moon_nak_num"]) if not pd.isna(row["moon_nak_num"]) else None,
    }


def signal_for_future(target_date: pd.Timestamp):
    """Look up pre-computed forward calendar row."""
    cal = pd.read_csv(RESULTS_DIR / "forward_calendar.csv", parse_dates=["date"])
    row = cal[cal["date"] == target_date]
    if row.empty:
        return None, "Date not found in forward_calendar.csv (may be a holiday or weekend)."

    row = row.iloc[0]
    composite = float(row["composite_score"])
    label = classification_label(composite)

    return composite, label, {
        "source":               "forward_calendar (planetary estimates only)",
        "signal_count":         int(row["signal_count"]),
        "s1_log_ret_pos":       "N/A (future)",
        "s2_ju_dignity_pos":    int(row["ju_dignity_est"] > 0),
        "s3_range_above_med":   "N/A (future)",
        "s4_moon_combust":      int(row["moon_combust_est"]),
        "ju_dignity_est":       int(row["ju_dignity_est"]),
        "moon_combust_est":     int(row["moon_combust_est"]),
        "campaign_window":      bool(row["campaign_window"]),
        "dead_zone":            bool(row["dead_zone"]),
        "note":                 "prob_up fixed at 0.55; market signals unavailable for future dates",
    }


def print_signal(target_date: pd.Timestamp, composite, label, details: dict):
    divider = "─" * 60
    print(divider)
    print(f"  ASTRO QUANT SIGNAL — {target_date.date()}")
    print(divider)
    print(f"  Composite Score : {composite:.1f} / 100")
    print(f"  Classification  : {label}")
    print(divider)
    print("  Signal breakdown:")
    for k, v in details.items():
        if k == "source":
            continue
        if isinstance(v, float):
            print(f"    {k:<28}: {v:.4f}")
        else:
            print(f"    {k:<28}: {v}")
    print(f"  Source: {details.get('source', '')}")
    print(divider)

    # Trading guidance
    if label == "PRIME_TRADE":
        print("  ACTION: Strong bullish setup — consider long entry (3-day hold)")
    elif label == "HIGH_VOL":
        print("  ACTION: Elevated opportunity — monitor for long entry")
    elif label == "WATCH":
        print("  ACTION: Marginal setup — watch but stay cautious")
    elif label == "NEUTRAL":
        print("  ACTION: No signal — stay flat or hold existing positions")
    else:
        print("  ACTION: AVOID — unfavorable astrological conditions")

    if details.get("dead_zone"):
        print("  WARNING: Dead zone — extended low-signal period")
    if details.get("campaign_window"):
        print("  NOTE: Campaign window — part of sustained high-signal cluster")
    print(divider)


def main():
    if len(sys.argv) > 1:
        try:
            target_date = pd.Timestamp(sys.argv[1])
        except Exception:
            print(f"Error: invalid date '{sys.argv[1]}'. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        target_date = pd.Timestamp.now().normalize()

    today = pd.Timestamp.now().normalize()

    print(f"\nGenerating signal for {target_date.date()} ...")

    if target_date <= today:
        result = signal_for_historical(target_date)
    else:
        result = signal_for_future(target_date)

    if result[0] is None:
        print(f"Error: {result[1]}")
        sys.exit(1)

    composite, label, details = result
    print_signal(target_date, composite, label, details)


if __name__ == "__main__":
    main()
