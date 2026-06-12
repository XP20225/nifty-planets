"""
Step 4: Synthesis — ML Models, Backtesting, Forward Calendar
Astro Quant Research System for NSE Nifty 50

NOTE: Composite score for backtesting is computed post-hoc using models trained
on the training set only (pre-2023). Scores for training period rows are
in-sample, while scores for 2023+ rows are out-of-sample. This is a research
system and is documented accordingly.
"""

import os
import sys
import json
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_recall_fscore_support, classification_report)
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/vasanthakumaranpalanisamy/Nifty Planets")
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results" / "synthesis"
MODELS_DIR = BASE / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STEP 4: Synthesis — ML, Backtesting, Forward Calendar")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 4.1 — Feature Selection for ML
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.1] Loading and preparing features...")

try:
    df_raw = pd.read_csv(DATA_DIR / "nifty_enriched.csv", parse_dates=["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    print(f"  Loaded nifty_enriched.csv: {df_raw.shape}")

    # Map spec column names → actual column names
    COLUMN_MAP = {
        "tithi":               "tithi_num",
        "paksha":              "paksha",
        "karana_id":           "karana",
        "moon_nak_id":         "moon_nak_num",
        "moon_nak_quality":    "moon_nak_quality",
        "sun_nak_id":          None,          # not present
        "su_dignity":          None,          # not present
        "mo_dignity":          None,          # not present
        "me_dignity":          None,          # not present
        "ve_dignity":          None,          # not present
        "ma_dignity":          None,          # not present
        "sa_dignity":          "sa_dignity",
        "ra_dignity":          None,          # not present
        "ke_dignity":          None,          # not present
        "moon_strength":       "moon_strength",
        "dasha_lord_id":       "mahadasha",
        "antardasha_lord_id":  "antardasha",
        "avg_ashtak":          "sarvashtakavarga",
        "su_ashtak":           None,
        "mo_ashtak":           None,
        "ju_ashtak":           None,
        "sa_ashtak":           None,
        "combust_Me":          "combust_Me",
        "combust_Ve":          "combust_Ve",
        "combust_Ma":          "combust_Ma",
        "combust_Ju":          "combust_Ju",
        "combust_Sa":          "combust_Sa",
        "gandanta_Su":         "gandanta_Su",
        "gandanta_Mo":         "gandanta_Mo",
        "eclipse_zone":        "eclipse_corridor",   # closest match
        "graha_yuddha":        "graha_yuddha",
        "day_of_week":         "dow",
        "month":               "month",
    }

    VALIDATED = ["log_ret", "ju_dignity", "range_pct", "combust_Mo"]

    # Build actual feature list
    FEATURES = list(VALIDATED)
    for spec_name, actual_name in COLUMN_MAP.items():
        if actual_name and actual_name in df_raw.columns:
            if actual_name not in FEATURES:
                FEATURES.append(actual_name)

    print(f"  Feature candidates: {len(FEATURES)}")

    # Encode categoricals before dropping
    cat_cols = ["paksha", "moon_nak_quality", "karana", "mahadasha", "antardasha"]
    for col in cat_cols:
        if col in df_raw.columns and col in FEATURES:
            df_raw[col] = pd.Categorical(df_raw[col]).codes

    # Drop rows where targets are NaN
    df = df_raw.dropna(subset=["fwd_ret_3d", "fwd_dir_3d"]).copy()
    print(f"  After dropping NaN targets: {df.shape}")

    # Keep only features that exist
    FEATURES = [f for f in FEATURES if f in df.columns]
    print(f"  Final features used: {len(FEATURES)}: {FEATURES}")

    # Train/test split
    TRAIN_END = "2022-12-31"
    TEST_START = "2023-01-01"
    train_mask = df["date"] <= TRAIN_END
    test_mask  = df["date"] >= TEST_START

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()
    print(f"  Train: {df_train.shape} ({df_train['date'].min()} to {df_train['date'].max()})")
    print(f"  Test:  {df_test.shape}  ({df_test['date'].min()} to {df_test['date'].max()})")

    STEP41_OK = True

except Exception as e:
    print(f"  ERROR in 4.1: {e}")
    traceback.print_exc()
    STEP41_OK = False
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 4.2 — Model A: Volatility Predictor
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.2] Training Model A: Volatility Predictor...")

MODEL_A = None
MODEL_A_CV_AUC = None
MODEL_A_OOS_AUC = None

try:
    import lightgbm as lgb

    # Compute high_vol threshold from training data only
    vol_threshold = df_train["range_pct"].quantile(0.75)
    print(f"  Volatility threshold (train Q3): {vol_threshold:.4f}")

    df_train = df_train.copy()
    df_test  = df_test.copy()
    df       = df.copy()

    df_train["high_vol"] = (df_train["range_pct"] >= vol_threshold).astype(int)
    df_test["high_vol"]  = (df_test["range_pct"]  >= vol_threshold).astype(int)
    df["high_vol"]       = (df["range_pct"]        >= vol_threshold).astype(int)

    X_train_a = df_train[FEATURES].fillna(0)
    y_train_a = df_train["high_vol"]
    X_test_a  = df_test[FEATURES].fillna(0)
    y_test_a  = df_test["high_vol"]

    lgb_params = dict(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=42, class_weight="balanced",
        verbose=-1, n_jobs=-1
    )

    # 5-fold time-series CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_a)):
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_train_a.iloc[tr_idx], y_train_a.iloc[tr_idx],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
                eval_set=[(X_train_a.iloc[val_idx], y_train_a.iloc[val_idx])])
        prob = clf.predict_proba(X_train_a.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y_train_a.iloc[val_idx], prob)
        cv_aucs.append(auc)
        print(f"    Fold {fold+1} AUC: {auc:.4f}")

    MODEL_A_CV_AUC = np.mean(cv_aucs)
    print(f"  Model A CV AUC (mean): {MODEL_A_CV_AUC:.4f}")

    # Train final model on full training set
    MODEL_A = lgb.LGBMClassifier(**lgb_params)
    MODEL_A.fit(X_train_a, y_train_a,
                callbacks=[lgb.log_evaluation(-1)])

    # Feature importance
    fi_a = pd.DataFrame({
        "feature": FEATURES,
        "importance": MODEL_A.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_a.to_csv(RESULTS_DIR / "model_a_feature_importance.csv", index=False)
    print(f"  Saved model_a_feature_importance.csv ({len(fi_a)} rows)")

    # Save model
    joblib.dump(MODEL_A, MODELS_DIR / "model_a_volatility.pkl")
    print(f"  Saved model_a_volatility.pkl")

    # OOS evaluation
    prob_test_a = MODEL_A.predict_proba(X_test_a.fillna(0))[:, 1]
    MODEL_A_OOS_AUC = roc_auc_score(y_test_a, prob_test_a)
    print(f"  Model A OOS AUC: {MODEL_A_OOS_AUC:.4f}")

    STEP42_OK = True

except Exception as e:
    print(f"  ERROR in 4.2: {e}")
    traceback.print_exc()
    STEP42_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# 4.3 — Model B: Directional Predictor
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.3] Training Model B: Directional Predictor...")

MODEL_B = None
MODEL_B_CV_AUC = None
MODEL_B_OOS_AUC = None
MODEL_B_OOS_ACC = None

try:
    X_train_b = df_train[FEATURES].fillna(0)
    y_train_b = df_train["fwd_dir_3d"].astype(int)
    X_test_b  = df_test[FEATURES].fillna(0)
    y_test_b  = df_test["fwd_dir_3d"].astype(int)

    # 5-fold time-series CV
    cv_aucs_b = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_b)):
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_train_b.iloc[tr_idx], y_train_b.iloc[tr_idx],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
                eval_set=[(X_train_b.iloc[val_idx], y_train_b.iloc[val_idx])])
        prob = clf.predict_proba(X_train_b.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y_train_b.iloc[val_idx], prob)
        cv_aucs_b.append(auc)
        print(f"    Fold {fold+1} AUC: {auc:.4f}")

    MODEL_B_CV_AUC = np.mean(cv_aucs_b)
    print(f"  Model B CV AUC (mean): {MODEL_B_CV_AUC:.4f}")

    # Train final model
    MODEL_B = lgb.LGBMClassifier(**lgb_params)
    MODEL_B.fit(X_train_b, y_train_b,
                callbacks=[lgb.log_evaluation(-1)])

    # Feature importance
    fi_b = pd.DataFrame({
        "feature": FEATURES,
        "importance": MODEL_B.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_b.to_csv(RESULTS_DIR / "model_b_feature_importance.csv", index=False)
    print(f"  Saved model_b_feature_importance.csv ({len(fi_b)} rows)")

    # Save model
    joblib.dump(MODEL_B, MODELS_DIR / "model_b_directional.pkl")
    print(f"  Saved model_b_directional.pkl")

    # OOS evaluation
    prob_test_b = MODEL_B.predict_proba(X_test_b.fillna(0))[:, 1]
    pred_test_b = MODEL_B.predict(X_test_b.fillna(0))
    MODEL_B_OOS_AUC = roc_auc_score(y_test_b, prob_test_b)
    MODEL_B_OOS_ACC = accuracy_score(y_test_b, pred_test_b)
    print(f"  Model B OOS AUC: {MODEL_B_OOS_AUC:.4f}, Accuracy: {MODEL_B_OOS_ACC:.4f}")

    STEP43_OK = True

except Exception as e:
    print(f"  ERROR in 4.3: {e}")
    traceback.print_exc()
    STEP43_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# 4.4 — SHAP Values
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.4] Computing SHAP values...")

try:
    import shap
    print(f"  shap version: {shap.__version__}")

    if MODEL_A is not None:
        explainer_a = shap.TreeExplainer(MODEL_A)
        shap_vals_a = explainer_a.shap_values(X_train_a.fillna(0))
        # For binary classification, shap_values returns list; take class-1 values
        if isinstance(shap_vals_a, list):
            sv_a = shap_vals_a[1]
        else:
            sv_a = shap_vals_a
        mean_shap_a = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": np.abs(sv_a).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False).head(15)
        mean_shap_a.to_csv(RESULTS_DIR / "shap_model_a.csv", index=False)
        print(f"  Saved shap_model_a.csv (top 15)")
        print("  Top-5 SHAP features (Model A):")
        print(mean_shap_a.head(5).to_string(index=False))

    if MODEL_B is not None:
        explainer_b = shap.TreeExplainer(MODEL_B)
        shap_vals_b = explainer_b.shap_values(X_train_b.fillna(0))
        if isinstance(shap_vals_b, list):
            sv_b = shap_vals_b[1]
        else:
            sv_b = shap_vals_b
        mean_shap_b = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": np.abs(sv_b).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False).head(15)
        mean_shap_b.to_csv(RESULTS_DIR / "shap_model_b.csv", index=False)
        print(f"  Saved shap_model_b.csv (top 15)")
        print("  Top-5 SHAP features (Model B):")
        print(mean_shap_b.head(5).to_string(index=False))

    STEP44_OK = True

except Exception as e:
    print(f"  ERROR in 4.4: {e}")
    traceback.print_exc()
    STEP44_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# 4.5 — Out-of-Sample Test
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.5] Out-of-Sample Evaluation...")

try:
    if MODEL_A is not None and MODEL_B is not None:
        # Model A OOS detailed
        prob_a = MODEL_A.predict_proba(X_test_a.fillna(0))[:, 1]
        pred_a = (prob_a >= 0.5).astype(int)
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
            y_test_a, pred_a, labels=[1], zero_division=0)
        print(f"  Model A (high_vol=1):")
        print(f"    AUC={MODEL_A_OOS_AUC:.4f}  Precision={prec_a[0]:.4f}  "
              f"Recall={rec_a[0]:.4f}  F1={f1_a[0]:.4f}")

        # Model B OOS detailed
        prob_b = MODEL_B.predict_proba(X_test_b.fillna(0))[:, 1]
        pred_b = (prob_b >= 0.5).astype(int)
        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
            y_test_b, pred_b, labels=[1], zero_division=0)
        print(f"  Model B (fwd_dir_3d=1):")
        print(f"    AUC={MODEL_B_OOS_AUC:.4f}  Acc={MODEL_B_OOS_ACC:.4f}  "
              f"Precision={prec_b[0]:.4f}  Recall={rec_b[0]:.4f}  F1={f1_b[0]:.4f}")

    STEP45_OK = True
except Exception as e:
    print(f"  ERROR in 4.5: {e}")
    traceback.print_exc()
    STEP45_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# 4.6 — Composite Score
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.6] Computing Composite Scores...")

COMPOSITE_DF = None
RANGE_PCT_MEDIAN_TRAIN = None

try:
    if MODEL_A is not None and MODEL_B is not None:
        X_all = df[FEATURES].fillna(0)

        prob_up_all  = MODEL_B.predict_proba(X_all)[:, 1]
        prob_vol_all = MODEL_A.predict_proba(X_all)[:, 1]

        # Signal count — compute from original (pre-encoded) df_raw aligned to df
        # We need to work with df (the filtered version)
        RANGE_PCT_MEDIAN_TRAIN = df_train["range_pct"].median()
        print(f"  range_pct training median: {RANGE_PCT_MEDIAN_TRAIN:.4f}")

        # Signals:
        # 1. log_ret > 0
        # 2. ju_dignity > 0
        # 3. range_pct > median training range_pct
        # 4. combust_Mo == 1
        s1 = (df["log_ret"] > 0).astype(int)
        s2 = (df["ju_dignity"] > 0).astype(int)
        s3 = (df["range_pct"] > RANGE_PCT_MEDIAN_TRAIN).astype(int)
        s4 = (df["combust_Mo"] == 1).astype(int)
        signal_count = s1 + s2 + s3 + s4

        # Composite score (simplified version from spec):
        # composite_score = round((prob_up * 0.6 + signal_count/4 * 0.4) * 100, 1)
        composite_score = np.round((prob_up_all * 0.6 + signal_count.values / 4 * 0.4) * 100, 1)
        composite_score = np.clip(composite_score, 0, 100)

        COMPOSITE_DF = pd.DataFrame({
            "date":           df["date"].values,
            "composite_score": composite_score,
            "prob_up":        np.round(prob_up_all, 4),
            "prob_vol":       np.round(prob_vol_all, 4),
            "signal_count":   signal_count.values,
            "fwd_ret_3d":     df["fwd_ret_3d"].values,
            "fwd_dir_3d":     df["fwd_dir_3d"].values,
        })

        COMPOSITE_DF.to_csv(RESULTS_DIR / "composite_scores.csv", index=False)
        print(f"  Saved composite_scores.csv ({len(COMPOSITE_DF)} rows)")
        print(f"  Score distribution: min={composite_score.min():.1f} "
              f"mean={composite_score.mean():.1f} max={composite_score.max():.1f}")
        print(f"  Rows with score >= 60: {(composite_score >= 60).sum()}")

    STEP46_OK = True
except Exception as e:
    print(f"  ERROR in 4.6: {e}")
    traceback.print_exc()
    STEP46_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Helper: compute backtest metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_backtest_metrics(trades_df, price_df, label="Strategy"):
    """Given a trades log DataFrame, compute aggregate metrics."""
    if len(trades_df) == 0:
        return {}, 0

    total_trades = len(trades_df)
    win_rate     = trades_df["hit"].mean()
    mean_ret     = trades_df["return_pct"].mean()
    cum_ret      = (1 + trades_df["return_pct"] / 100).prod() - 1

    # Annualised return (assume ~252 trading days/year)
    # Count calendar days
    if len(trades_df) > 1:
        total_calendar_days = (
            pd.to_datetime(trades_df["exit_date"].max()) -
            pd.to_datetime(trades_df["entry_date"].min())
        ).days
        years = max(total_calendar_days / 365.25, 0.1)
    else:
        years = 1.0
    ann_ret = (1 + cum_ret) ** (1 / years) - 1

    # Equity curve (starting capital = 100)
    equity = [100.0]
    for r in trades_df["return_pct"]:
        equity.append(equity[-1] * (1 + r / 100))
    equity = np.array(equity)

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown    = (equity - running_max) / running_max
    max_dd      = drawdown.min()

    # Sharpe (annualised) — using per-trade returns
    trade_rets = trades_df["return_pct"].values / 100
    if trade_rets.std() > 0:
        # Annualise: assume each trade held 3 days, so ~84 trades/year
        ann_factor = np.sqrt(252 / 3)
        sharpe = (trade_rets.mean() / trade_rets.std()) * ann_factor
    else:
        sharpe = 0.0

    # Calmar
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else np.nan

    return {
        "label":         label,
        "total_trades":  total_trades,
        "win_rate":      round(win_rate, 4),
        "mean_ret_pct":  round(mean_ret, 4),
        "cum_ret_pct":   round(cum_ret * 100, 2),
        "ann_ret_pct":   round(ann_ret * 100, 2),
        "max_drawdown":  round(max_dd, 4),
        "sharpe":        round(sharpe, 4),
        "calmar":        round(calmar, 4) if not np.isnan(calmar) else None,
    }, equity


def run_backtest(composite_df, price_df, threshold=60, cost_pct=0.05,
                 label="Strategy", close_col="close"):
    """
    Run a backtest: enter long when composite_score >= threshold.
    Hold 3 trading days. Exit at close on day 3.
    """
    # Merge composite scores with price data
    merged = composite_df[["date", "composite_score"]].merge(
        price_df[["date", close_col]].rename(columns={close_col: "price"}),
        on="date", how="inner"
    ).sort_values("date").reset_index(drop=True)

    # Filter by score >= threshold
    signal_idx = merged[merged["composite_score"] >= threshold].index.tolist()

    trades = []
    i = 0
    used_exit_days = set()  # prevent overlapping trades (optional — keep simple)

    all_dates = merged["date"].tolist()
    all_prices = dict(zip(merged["date"], merged["price"]))

    for idx in signal_idx:
        entry_date  = merged.loc[idx, "date"]
        entry_price = merged.loc[idx, "price"]

        # Exit 3 trading days later
        exit_idx = idx + 3
        if exit_idx >= len(merged):
            continue

        exit_date  = merged.loc[exit_idx, "date"]
        exit_price = merged.loc[exit_idx, "price"]

        ret_gross = (exit_price - entry_price) / entry_price * 100
        ret_net   = ret_gross - cost_pct  # entry + exit combined cost
        hit       = 1 if ret_net > 0 else 0

        trades.append({
            "entry_date":   entry_date,
            "entry_price":  round(entry_price, 2),
            "exit_date":    exit_date,
            "exit_price":   round(exit_price, 2),
            "return_pct":   round(ret_net, 4),
            "hit":          hit,
        })

    trades_df = pd.DataFrame(trades)
    return trades_df


# ══════════════════════════════════════════════════════════════════════════════
# 4.7 — Backtesting Engine
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.7] Backtesting Engine...")

BT_SUMMARY = None
TRADE_LOG = None
BUY_HOLD_RET = None
SHARPE_STRATEGY = None

try:
    if COMPOSITE_DF is not None:
        price_df_nifty = df_raw[["date", "close"]].copy()

        # Run strategy backtest
        TRADE_LOG = run_backtest(
            COMPOSITE_DF, price_df_nifty, threshold=60,
            cost_pct=0.05, label="AstroQuant-Nifty"
        )
        print(f"  Total trades: {len(TRADE_LOG)}")

        if len(TRADE_LOG) > 0:
            metrics, equity_curve = compute_backtest_metrics(
                TRADE_LOG, price_df_nifty, label="AstroQuant-Nifty"
            )
            print(f"  Win rate: {metrics['win_rate']:.1%}")
            print(f"  Mean return/trade: {metrics['mean_ret_pct']:.4f}%")
            print(f"  Cumulative return: {metrics['cum_ret_pct']:.2f}%")
            print(f"  Annualised return: {metrics['ann_ret_pct']:.2f}%")
            print(f"  Max drawdown: {metrics['max_drawdown']:.4f}")
            print(f"  Sharpe: {metrics['sharpe']:.4f}")
            print(f"  Calmar: {metrics['calmar']}")

            SHARPE_STRATEGY = metrics["sharpe"]

            # ── Buy and hold benchmark ─────────────────────────────────────
            first_date = TRADE_LOG["entry_date"].min()
            last_date  = TRADE_LOG["exit_date"].max()
            bh_prices  = price_df_nifty[
                (price_df_nifty["date"] >= first_date) &
                (price_df_nifty["date"] <= last_date)
            ]["close"].values
            if len(bh_prices) >= 2:
                BUY_HOLD_RET = (bh_prices[-1] - bh_prices[0]) / bh_prices[0] * 100
            else:
                BUY_HOLD_RET = 0.0
            print(f"  Buy-and-hold return (same period): {BUY_HOLD_RET:.2f}%")

            # ── Monte Carlo random benchmark ───────────────────────────────
            print("  Running 1000 Monte Carlo random-entry benchmarks...")
            n_trades    = len(TRADE_LOG)
            mc_cum_rets = []
            mc_sharpes  = []
            rng         = np.random.default_rng(42)
            all_idx     = list(range(len(COMPOSITE_DF) - 3))  # need 3 days ahead

            # Reindex composite_df for fast access
            cs_arr = COMPOSITE_DF.reset_index(drop=True)

            for _ in range(1000):
                rand_entries = rng.choice(all_idx, size=n_trades, replace=False)
                rand_entries = np.sort(rand_entries)
                rets = []
                for eidx in rand_entries:
                    entry_date  = cs_arr.loc[eidx, "date"]
                    exit_idx    = eidx + 3
                    if exit_idx >= len(cs_arr):
                        continue
                    exit_date   = cs_arr.loc[exit_idx, "date"]
                    ep = price_df_nifty.loc[price_df_nifty["date"] == entry_date, "close"]
                    xp = price_df_nifty.loc[price_df_nifty["date"] == exit_date,  "close"]
                    if ep.empty or xp.empty:
                        continue
                    r = (xp.values[0] - ep.values[0]) / ep.values[0] * 100 - 0.05
                    rets.append(r)
                if not rets:
                    continue
                rets = np.array(rets)
                cum = (1 + rets / 100).prod() - 1
                mc_cum_rets.append(cum * 100)
                if rets.std() > 0:
                    mc_sharpes.append((rets.mean() / rets.std()) * np.sqrt(252 / 3))

            mc_cum_rets = np.array(mc_cum_rets)
            mc_sharpes  = np.array(mc_sharpes)
            print(f"  MC Cum Return  — P5:{np.percentile(mc_cum_rets,5):.2f}%  "
                  f"P50:{np.percentile(mc_cum_rets,50):.2f}%  "
                  f"P95:{np.percentile(mc_cum_rets,95):.2f}%")
            print(f"  MC Sharpe      — P5:{np.percentile(mc_sharpes,5):.4f}  "
                  f"P50:{np.percentile(mc_sharpes,50):.4f}  "
                  f"P95:{np.percentile(mc_sharpes,95):.4f}")

            metrics["bh_return_pct"]      = round(BUY_HOLD_RET, 2)
            metrics["mc_cum_ret_p5"]      = round(float(np.percentile(mc_cum_rets, 5)), 2)
            metrics["mc_cum_ret_p50"]     = round(float(np.percentile(mc_cum_rets, 50)), 2)
            metrics["mc_cum_ret_p95"]     = round(float(np.percentile(mc_cum_rets, 95)), 2)
            metrics["mc_sharpe_p5"]       = round(float(np.percentile(mc_sharpes, 5)), 4)
            metrics["mc_sharpe_p50"]      = round(float(np.percentile(mc_sharpes, 50)), 4)
            metrics["mc_sharpe_p95"]      = round(float(np.percentile(mc_sharpes, 95)), 4)

            BT_SUMMARY = pd.DataFrame([metrics])

            # ── Stress Tests ───────────────────────────────────────────────
            stress_periods = {
                "2000_dot_com_crash":  ("2000-01-01", "2002-12-31"),
                "2008_GFC":            ("2008-01-01", "2009-06-30"),
                "2011_correction":     ("2011-01-01", "2012-06-30"),
                "2020_COVID":          ("2020-01-01", "2020-09-30"),
                "2022_bear_market":    ("2022-01-01", "2022-12-31"),
            }

            stress_rows = []
            for period_name, (start, end) in stress_periods.items():
                period_trades = TRADE_LOG[
                    (TRADE_LOG["entry_date"] >= start) &
                    (TRADE_LOG["entry_date"] <= end)
                ]
                if len(period_trades) == 0:
                    stress_rows.append({
                        "period": period_name,
                        "start": start, "end": end,
                        "n_trades": 0, "win_rate": None, "mean_ret_pct": None
                    })
                    print(f"  Stress [{period_name}]: 0 trades")
                else:
                    wr  = period_trades["hit"].mean()
                    mr  = period_trades["return_pct"].mean()
                    stress_rows.append({
                        "period": period_name,
                        "start": start, "end": end,
                        "n_trades": len(period_trades),
                        "win_rate": round(wr, 4),
                        "mean_ret_pct": round(mr, 4),
                    })
                    print(f"  Stress [{period_name}]: n={len(period_trades)}, "
                          f"win_rate={wr:.1%}, mean_ret={mr:.4f}%")

            stress_df = pd.DataFrame(stress_rows)
            stress_df.to_csv(RESULTS_DIR / "stress_tests.csv", index=False)
            print(f"  Saved stress_tests.csv ({len(stress_df)} rows)")

            # Save trade log and summary
            TRADE_LOG.to_csv(RESULTS_DIR / "backtest_trade_log.csv", index=False)
            BT_SUMMARY.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)
            print(f"  Saved backtest_trade_log.csv ({len(TRADE_LOG)} rows)")
            print(f"  Saved backtest_summary.csv")

    STEP47_OK = True
except Exception as e:
    print(f"  ERROR in 4.7: {e}")
    traceback.print_exc()
    STEP47_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# 4.8 — Bank Nifty Transfer Test
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.8] Bank Nifty Transfer Test...")

try:
    if MODEL_A is not None and MODEL_B is not None:
        df_bn_raw = pd.read_csv(DATA_DIR / "banknifty_enriched.csv", parse_dates=["date"])
        df_bn_raw = df_bn_raw.sort_values("date").reset_index(drop=True)
        print(f"  Loaded banknifty_enriched.csv: {df_bn_raw.shape}")

        # Encode categoricals
        for col in cat_cols:
            if col in df_bn_raw.columns:
                df_bn_raw[col] = pd.Categorical(df_bn_raw[col]).codes

        df_bn = df_bn_raw.dropna(subset=["fwd_ret_3d", "fwd_dir_3d"]).copy()

        # BankNifty uses bn_close
        close_col_bn = "bn_close"

        # Apply the same features (fill missing with 0)
        X_bn = df_bn[[f for f in FEATURES if f in df_bn.columns]].fillna(0)
        # Add missing columns as zeros
        for f in FEATURES:
            if f not in X_bn.columns:
                X_bn[f] = 0
        X_bn = X_bn[FEATURES]

        prob_up_bn  = MODEL_B.predict_proba(X_bn)[:, 1]
        prob_vol_bn = MODEL_A.predict_proba(X_bn)[:, 1]

        range_pct_median_bn_train = df_bn[df_bn["date"] <= TRAIN_END]["range_pct"].median()

        s1_bn = (df_bn["log_ret"] > 0).astype(int).values
        s2_bn = (df_bn["ju_dignity"] > 0).astype(int).values
        s3_bn = (df_bn["range_pct"] > RANGE_PCT_MEDIAN_TRAIN).astype(int).values
        s4_bn = (df_bn["combust_Mo"] == 1).astype(int).values
        signal_count_bn = s1_bn + s2_bn + s3_bn + s4_bn

        composite_score_bn = np.round(
            (prob_up_bn * 0.6 + signal_count_bn / 4 * 0.4) * 100, 1
        )
        composite_score_bn = np.clip(composite_score_bn, 0, 100)

        composite_df_bn = pd.DataFrame({
            "date":            df_bn["date"].values,
            "composite_score": composite_score_bn,
        })

        price_df_bn = df_bn_raw[["date", close_col_bn]].rename(
            columns={close_col_bn: "close"}
        )

        trade_log_bn = run_backtest(
            composite_df_bn, price_df_bn, threshold=60,
            cost_pct=0.05, label="AstroQuant-BankNifty"
        )
        print(f"  BankNifty trades: {len(trade_log_bn)}")

        if len(trade_log_bn) > 0:
            metrics_bn, _ = compute_backtest_metrics(
                trade_log_bn, price_df_bn, label="AstroQuant-BankNifty"
            )
            print(f"  Win Rate: {metrics_bn['win_rate']:.1%}")
            print(f"  Sharpe: {metrics_bn['sharpe']:.4f}")
            print(f"  Cumulative Return: {metrics_bn['cum_ret_pct']:.2f}%")
        else:
            metrics_bn = {
                "label": "AstroQuant-BankNifty", "total_trades": 0,
                "win_rate": None, "sharpe": None
            }

        # Compare vs Nifty
        nifty_win_rate = BT_SUMMARY["win_rate"].values[0] if BT_SUMMARY is not None else None
        nifty_sharpe   = BT_SUMMARY["sharpe"].values[0]   if BT_SUMMARY is not None else None

        transfer_rows = [
            {"index": "metric", "banknifty": "value", "nifty_baseline": "value"},
            {"index": "win_rate",    "banknifty": metrics_bn.get("win_rate"),    "nifty_baseline": nifty_win_rate},
            {"index": "sharpe",      "banknifty": metrics_bn.get("sharpe"),      "nifty_baseline": nifty_sharpe},
            {"index": "total_trades","banknifty": metrics_bn.get("total_trades"),"nifty_baseline": len(TRADE_LOG) if TRADE_LOG is not None else None},
            {"index": "cum_ret_pct", "banknifty": metrics_bn.get("cum_ret_pct"), "nifty_baseline": BT_SUMMARY["cum_ret_pct"].values[0] if BT_SUMMARY is not None else None},
        ]
        pd.DataFrame(transfer_rows).to_csv(
            RESULTS_DIR / "banknifty_transfer_test.csv", index=False
        )
        print(f"  Saved banknifty_transfer_test.csv")

    STEP48_OK = True
except Exception as e:
    print(f"  ERROR in 4.8: {e}")
    traceback.print_exc()
    STEP48_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# 4.9 — Forward 1-Year Planetary Calendar
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4.9] Forward 1-Year Planetary Calendar...")

try:
    # Load last known planetary positions from data.json
    with open(BASE / "data.json") as f:
        data_json = json.load(f)

    records_json = data_json["records"]
    last_rec = records_json[-1]
    last_date = pd.to_datetime(last_rec["date"])
    last_positions = last_rec["p"]   # sidereal positions from data.json
    # data.json stores tropical positions — we need to subtract ayanamsa
    last_ayan = last_rec["ayan"]["la"]  # Lahiri ayanamsa

    print(f"  Last data.json date: {last_date.date()}")
    print(f"  Last positions (tropical): {last_positions}")
    print(f"  Lahiri ayanamsa: {last_ayan:.4f}")

    # Mean daily motions (degrees/day)
    MEAN_MOTION = {
        "Su": 0.9856, "Mo": 13.176, "Me": 1.383, "Ve": 1.2,
        "Ma": 0.524,  "Ju": 0.083,  "Sa": 0.033
    }

    LAHIRI_AYAN_2026 = 24.198  # approximate for 2026-2027

    def sidereal_lon(planet, elapsed_days):
        """Compute sidereal longitude from last known position."""
        trop_lon = (last_positions[planet] + MEAN_MOTION[planet] * elapsed_days) % 360
        sid_lon  = (trop_lon - LAHIRI_AYAN_2026) % 360
        return sid_lon

    def get_sign(lon):
        """0-based sign index (0=Aries .. 11=Pisces)."""
        return int(lon // 30)

    def get_nakshatra(lon):
        """1-based nakshatra number."""
        return int(lon / (360 / 27)) + 1

    # Jupiter dignity: exaltation sign = Cancer(3), own = Sagittarius(8)/Pisces(11)
    # For simplicity: ju_dignity > 0 when in Cancer, Sagittarius, or Pisces
    def jupiter_dignity_est(ju_lon):
        sign = get_sign(ju_lon)
        if sign in [3]:       # exaltation (Cancer)
            return 3
        elif sign in [8, 11]: # own sign (Sagittarius, Pisces)
            return 2
        elif sign in [6]:     # debilitation (Capricorn)
            return -3
        else:
            return 0

    # Moon combust: Moon within 12° of Sun (sidereal)
    def moon_combust_est(su_lon, mo_lon):
        diff = abs(su_lon - mo_lon)
        if diff > 180:
            diff = 360 - diff
        return 1 if diff <= 12 else 0

    # Indian market holidays 2026-2027 (approximate fixed dates)
    HOLIDAYS_2026_2027 = set([
        "2026-01-26",  # Republic Day
        "2026-04-14",  # Dr Ambedkar Jayanti
        "2026-05-01",  # Maharashtra Day
        "2026-08-15",  # Independence Day
        "2026-10-02",  # Gandhi Jayanti
        "2026-12-25",  # Christmas
        "2027-01-26",  # Republic Day
        "2027-04-14",  # Dr Ambedkar Jayanti
        "2027-05-01",  # Maharashtra Day
        "2027-08-15",  # Independence Day
        "2027-10-02",  # Gandhi Jayanti
        "2027-12-25",  # Christmas
        # Approximate Diwali / Holi / Good Friday / Gurunanak (vary by year)
        "2026-03-20",  # Holi approx
        "2026-04-03",  # Good Friday approx
        "2026-11-09",  # Diwali Laxmi Puja approx
        "2026-11-10",  # Diwali Balipratipada approx
        "2026-11-15",  # Gurunanak Jayanti approx
        "2027-03-10",  # Holi approx
        "2027-03-26",  # Good Friday approx
        "2027-10-29",  # Diwali approx
        "2027-11-04",  # Gurunanak Jayanti approx
    ])

    # Generate 252 trading days from 2026-06-13
    start_cal = pd.Timestamp("2026-06-13")
    cal_rows  = []
    current   = start_cal
    count     = 0
    max_iter  = 500

    while count < 252 and max_iter > 0:
        max_iter -= 1
        # Skip weekends and holidays
        if current.weekday() >= 5:
            current += pd.Timedelta(days=1)
            continue
        if current.strftime("%Y-%m-%d") in HOLIDAYS_2026_2027:
            current += pd.Timedelta(days=1)
            continue

        elapsed = (current - last_date).days

        # Compute planetary positions
        su_lon = sidereal_lon("Su", elapsed)
        mo_lon = sidereal_lon("Mo", elapsed)
        ju_lon = sidereal_lon("Ju", elapsed)

        ju_dig_est   = jupiter_dignity_est(ju_lon)
        mo_comb_est  = moon_combust_est(su_lon, mo_lon)

        # Signals (use fixed prob_up=0.55 as per spec)
        # log_ret: use 0 (no price data yet, treat as neutral — signal inactive)
        # range_pct: use 0 (neutral)
        # ju_dignity > 0: use estimated dignity
        # combust_Mo == 1: use estimated combust

        s1 = 0              # log_ret unknown for future
        s2 = 1 if ju_dig_est > 0 else 0
        s3 = 0              # range_pct unknown for future
        s4 = mo_comb_est
        signal_count = s1 + s2 + s3 + s4

        prob_up_fixed = 0.55
        composite_score = round((prob_up_fixed * 0.6 + signal_count / 4 * 0.4) * 100, 1)
        composite_score = max(0, min(100, composite_score))

        # Classification
        if composite_score >= 75:
            classification = "PRIME_TRADE"
        elif composite_score >= 60:
            classification = "HIGH_VOL"
        elif composite_score >= 45:
            classification = "WATCH"
        elif composite_score >= 30:
            classification = "NEUTRAL"
        else:
            classification = "AVOID"

        cal_rows.append({
            "date":              current.strftime("%Y-%m-%d"),
            "day_of_week":       current.strftime("%A"),
            "composite_score":   composite_score,
            "classification":    classification,
            "campaign_window":   False,
            "dead_zone":         False,
            "signal_count":      signal_count,
            "ju_dignity_est":    ju_dig_est,
            "moon_combust_est":  mo_comb_est,
        })
        count  += 1
        current += pd.Timedelta(days=1)

    cal_df = pd.DataFrame(cal_rows)
    print(f"  Generated {len(cal_df)} forward trading days")

    # Mark campaign windows (3+ consecutive PRIME_TRADE or HIGH_VOL)
    active = cal_df["classification"].isin(["PRIME_TRADE", "HIGH_VOL"])
    campaign_run = 0
    campaign_starts = []
    for i, a in enumerate(active):
        if a:
            campaign_run += 1
            if campaign_run == 3:
                campaign_starts.append(i - 2)
        else:
            campaign_run = 0

    # Mark all rows in 3+ consecutive runs
    in_campaign = [False] * len(cal_df)
    campaign_run = 0
    for i, a in enumerate(active):
        if a:
            campaign_run += 1
        else:
            campaign_run = 0
        if campaign_run >= 3:
            for j in range(i - campaign_run + 1, i + 1):
                in_campaign[j] = True

    # Mark dead zones (5+ consecutive AVOID or NEUTRAL)
    dead = cal_df["classification"].isin(["AVOID", "NEUTRAL"])
    in_dead = [False] * len(cal_df)
    dead_run = 0
    for i, d in enumerate(dead):
        if d:
            dead_run += 1
        else:
            dead_run = 0
        if dead_run >= 5:
            for j in range(i - dead_run + 1, i + 1):
                in_dead[j] = True

    cal_df["campaign_window"] = in_campaign
    cal_df["dead_zone"]       = in_dead

    cal_df.to_csv(RESULTS_DIR / "forward_calendar.csv", index=False)
    print(f"  Saved forward_calendar.csv ({len(cal_df)} rows)")

    # Count by classification
    class_counts = cal_df["classification"].value_counts()
    print("  Classification counts:")
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt}")

    campaign_count = cal_df["campaign_window"].sum()
    dead_count     = cal_df["dead_zone"].sum()
    print(f"  Campaign window days: {campaign_count}")
    print(f"  Dead zone days: {dead_count}")

    FORWARD_CAL_COUNTS = class_counts.to_dict()
    N_CAMPAIGN_DAYS    = int(campaign_count)
    N_DEAD_DAYS        = int(dead_count)

    STEP49_OK = True

except Exception as e:
    print(f"  ERROR in 4.9: {e}")
    traceback.print_exc()
    STEP49_OK = False
    FORWARD_CAL_COUNTS = {}
    N_CAMPAIGN_DAYS    = 0
    N_DEAD_DAYS        = 0


# ══════════════════════════════════════════════════════════════════════════════
# 4.10 — Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4 SUMMARY")
print("=" * 70)

print(f"\nModel A (Volatility Predictor):")
print(f"  CV AUC (5-fold):   {MODEL_A_CV_AUC:.4f}" if MODEL_A_CV_AUC else "  SKIPPED")
print(f"  OOS AUC (2023+):   {MODEL_A_OOS_AUC:.4f}" if MODEL_A_OOS_AUC else "  SKIPPED")

print(f"\nModel B (Directional Predictor):")
print(f"  CV AUC (5-fold):   {MODEL_B_CV_AUC:.4f}" if MODEL_B_CV_AUC else "  SKIPPED")
print(f"  OOS Accuracy:      {MODEL_B_OOS_ACC:.4f}" if MODEL_B_OOS_ACC else "  SKIPPED")
print(f"  OOS AUC:           {MODEL_B_OOS_AUC:.4f}" if MODEL_B_OOS_AUC else "  SKIPPED")

if BT_SUMMARY is not None and len(BT_SUMMARY) > 0:
    bt = BT_SUMMARY.iloc[0]
    print(f"\nBacktest (Nifty 50, threshold=60):")
    print(f"  Total trades:      {int(bt['total_trades'])}")
    print(f"  Win rate:          {bt['win_rate']:.1%}")
    print(f"  Mean ret/trade:    {bt['mean_ret_pct']:.4f}%")
    print(f"  Sharpe ratio:      {bt['sharpe']:.4f}")
    print(f"  Max drawdown:      {bt['max_drawdown']:.4f}")
    print(f"  Cumulative return: {bt['cum_ret_pct']:.2f}%")
    print(f"  Buy-and-hold ret:  {bt['bh_return_pct']:.2f}%")
    print(f"  MC Sharpe P5/P50/P95: {bt['mc_sharpe_p5']:.4f} / {bt['mc_sharpe_p50']:.4f} / {bt['mc_sharpe_p95']:.4f}")

if STEP48_OK and BT_SUMMARY is not None:
    pass  # Already printed above

print(f"\nForward Calendar (next 252 trading days from 2026-06-13):")
for cls_name in ["PRIME_TRADE", "HIGH_VOL", "WATCH", "NEUTRAL", "AVOID"]:
    cnt = FORWARD_CAL_COUNTS.get(cls_name, 0)
    print(f"  {cls_name:15s}: {cnt}")
print(f"  Campaign window days: {N_CAMPAIGN_DAYS}")
print(f"  Dead zone days:       {N_DEAD_DAYS}")

print("\n[Output files created]:")
for f in sorted(RESULTS_DIR.iterdir()):
    try:
        rows = len(pd.read_csv(f))
        print(f"  {f.name} ({rows} rows)")
    except:
        print(f"  {f.name}")
for f in sorted(MODELS_DIR.iterdir()):
    size_kb = os.path.getsize(f) // 1024
    print(f"  models/{f.name} ({size_kb} KB)")

print("\nStep 4 complete.")
