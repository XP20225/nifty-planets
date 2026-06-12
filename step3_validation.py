"""
Step 3: Statistical Validation of Astro-Quant Signal Candidates
================================================================
Performs:
  3.1 Monte Carlo Permutation Testing (10,000 shuffles)
  3.2 Benjamini-Hochberg FDR correction at 1%
  3.3 Temporal Stability (rolling 5-year windows, threshold >= 0.70)
  3.4 Regime Robustness (BULL/BEAR/TRANSITIONAL)
  3.5 Bootstrap Confidence Intervals (5000 resamples)
  3.6 Accuracy-Selectivity Surface
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
DATA_PATH = os.path.join(BASE, "data", "nifty_enriched.csv")
SCAN_PATH = os.path.join(BASE, "results", "discovery", "unconditional_scan.csv")
OUT_DIR   = os.path.join(BASE, "results", "validation")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
rng  = np.random.default_rng(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 3 – STATISTICAL VALIDATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Enriched data: {df.shape[0]:,} rows × {df.shape[1]} columns")

scan = pd.read_csv(SCAN_PATH)
print(f"Scan results : {scan.shape[0]:,} rows")

# ── Candidate selection ───────────────────────────────────────────────────────
fwd3 = scan[scan["target"] == "fwd_ret_3d"].copy()

cat_mask  = (fwd3["feature_type"].isin(["categorical", "binary"])) & (fwd3["abs_effect_size"] > 0.02)
cont_mask = (fwd3["feature_type"] == "continuous") & (fwd3["mutual_info"].abs() > 0.01)
candidates_raw = pd.concat([fwd3[cat_mask], fwd3[cont_mask]])

# Deduplicate on feature name (keep highest abs_effect_size per feature)
candidates = (
    candidates_raw
    .sort_values("abs_effect_size", ascending=False)
    .drop_duplicates(subset=["feature"])
    .reset_index(drop=True)
)

# Limit to top 200 by abs_effect_size
candidates = candidates.head(200).copy()
print(f"\nCandidates entering Monte Carlo: {len(candidates)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.1  Monte Carlo Permutation Testing
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.1] Monte Carlo Permutation Testing (10,000 shuffles) …")

N_PERM = 10_000
target_col = "fwd_ret_3d"
# Drop rows with NaN in target
target_vals = df[target_col].values

mc_records = []

for idx, row in candidates.iterrows():
    feat = row["feature"]
    ftype = row["feature_type"]

    # Print progress every 25 candidates
    pos = len(mc_records)
    if pos % 25 == 0:
        print(f"  … {pos}/{len(candidates)} candidates done")

    # Align feature + target (drop rows where either is NaN)
    feat_s  = df[feat].values
    tgt_s   = target_vals.copy()
    mask    = ~(np.isnan(feat_s.astype(float)) | np.isnan(tgt_s))
    x_clean = feat_s[mask]
    y_clean = tgt_s[mask]

    if len(y_clean) < 50:
        mc_records.append({
            "feature": feat, "feature_type": ftype,
            "observed_metric": np.nan, "p_value": 1.0, "n_used": len(y_clean)
        })
        continue

    # Compute observed metric
    if ftype in ("categorical", "binary"):
        # Best-level mean: for binary (0/1) take the level with higher mean
        levels = np.unique(x_clean)
        level_means = {lv: y_clean[x_clean == lv].mean() for lv in levels}
        best_lv = max(level_means, key=lambda l: level_means[l])
        observed = level_means[best_lv]
        # Null distribution: shuffle labels, recompute best-level mean
        null_dist = np.empty(N_PERM)
        for p in range(N_PERM):
            xp = rng.permutation(x_clean)
            lm = {lv: y_clean[xp == lv].mean() for lv in levels if (xp == lv).sum() > 0}
            null_dist[p] = max(lm.values()) if lm else 0.0
    else:
        # Continuous: Spearman rho
        observed, _ = spearmanr(x_clean, y_clean)
        observed = abs(observed)          # use absolute value for one-sided test
        # Null distribution
        null_dist = np.empty(N_PERM)
        for p in range(N_PERM):
            yp = rng.permutation(y_clean)
            rho, _ = spearmanr(x_clean, yp)
            null_dist[p] = abs(rho)

    p_value = (null_dist >= observed).mean()

    mc_records.append({
        "feature"         : feat,
        "feature_type"    : ftype,
        "observed_metric" : observed,
        "p_value"         : p_value,
        "n_used"          : int(mask.sum()),
    })

print(f"  … {len(mc_records)}/{len(candidates)} candidates done")

mc_df = pd.DataFrame(mc_records)
mc_df.to_csv(os.path.join(OUT_DIR, "monte_carlo_results.csv"), index=False)

n_survive_raw = (mc_df["p_value"] < 0.005).sum()
print(f"\n  Candidates entering Monte Carlo : {len(mc_df)}")
print(f"  Surviving p < 0.005            : {n_survive_raw}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.2  Benjamini-Hochberg FDR at 1%
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.2] Benjamini-Hochberg FDR correction (α = 1%) …")

p_vals = mc_df["p_value"].fillna(1.0).values
reject, pvals_adj, _, _ = multipletests(p_vals, alpha=0.01, method="fdr_bh")
mc_df["p_adj_bh"] = pvals_adj
mc_df["fdr_reject"] = reject

fdr_surv = mc_df[mc_df["fdr_reject"]].copy()
fdr_surv.to_csv(os.path.join(OUT_DIR, "fdr_survivors.csv"), index=False)

print(f"  Surviving FDR correction : {len(fdr_surv)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.3  Temporal Stability (rolling 5-year windows)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.3] Temporal Stability (rolling 5-year windows, threshold ≥ 0.70) …")

TARGET_COL = "fwd_ret_3d"
WINDOW_YRS  = 5
STAB_THRESH = 0.70

df_dt = df.copy()
df_dt["year"] = df_dt["date"].dt.year
min_year = df_dt["year"].min()
max_year = df_dt["year"].max()

# Build list of windows: (start_year, end_year) where window = 5 calendar years
windows = []
sy = min_year
while sy + WINDOW_YRS - 1 <= max_year:
    windows.append((sy, sy + WINDOW_YRS - 1))
    sy += 1

stability_records = []

for _, row in fdr_surv.iterrows():
    feat  = row["feature"]
    ftype = row["feature_type"]

    # Overall direction
    feat_s  = df[feat].values.astype(float)
    tgt_s   = df[TARGET_COL].values
    mask    = ~(np.isnan(feat_s) | np.isnan(tgt_s))
    x_all   = feat_s[mask]
    y_all   = tgt_s[mask]

    if ftype in ("categorical", "binary"):
        levels = np.unique(x_all)
        lm     = {lv: y_all[x_all == lv].mean() for lv in levels}
        overall_best_lv   = max(lm, key=lambda l: lm[l])
        overall_direction = lm[overall_best_lv]   # positive: best level beats average
    else:
        overall_direction, _ = spearmanr(x_all, y_all)

    consistent = 0
    total      = 0

    for (sy, ey) in windows:
        w_mask = (df_dt["year"] >= sy) & (df_dt["year"] <= ey)
        wx = feat_s[w_mask & mask[:len(w_mask)]]
        wy = tgt_s[w_mask & mask[:len(w_mask)]]
        # Re-apply NaN mask within window
        wmask2 = ~(np.isnan(wx) | np.isnan(wy))
        wx, wy = wx[wmask2], wy[wmask2]

        if len(wy) < 30:
            continue

        if ftype in ("categorical", "binary"):
            lm_w  = {lv: wy[wx == lv].mean() for lv in levels if (wx == lv).sum() > 0}
            if not lm_w:
                continue
            best_lv_w  = max(lm_w, key=lambda l: lm_w[l])
            window_dir = lm_w[best_lv_w]
            same_dir   = (window_dir > 0) == (overall_direction > 0)
        else:
            window_rho, _ = spearmanr(wx, wy)
            same_dir = (window_rho > 0) == (overall_direction > 0)

        total += 1
        if same_dir:
            consistent += 1

    stability = consistent / total if total > 0 else 0.0

    stability_records.append({
        "feature"             : feat,
        "feature_type"        : ftype,
        "overall_direction"   : overall_direction,
        "n_windows"           : total,
        "consistent_windows"  : consistent,
        "stability_score"     : stability,
        "stable"              : stability >= STAB_THRESH,
    })

stab_df = pd.DataFrame(stability_records)
stab_df.to_csv(os.path.join(OUT_DIR, "temporal_stability.csv"), index=False)

stable_df = stab_df[stab_df["stable"]].copy()
print(f"  Surviving temporal stability : {len(stable_df)}")

# If fewer than 5 survive, use all FDR survivors for subsequent steps
if len(stable_df) < 5:
    print(f"\n  *** FINDING: Only {len(stable_df)} signal(s) passed temporal stability.")
    print(f"      Using all {len(fdr_surv)} FDR survivors for surface table. ***")
    signals_for_next = fdr_surv["feature"].tolist()
    surface_note = f"NOTE: Fewer than 5 stable signals; all {len(fdr_surv)} FDR survivors used."
else:
    signals_for_next = stable_df["feature"].tolist()
    surface_note = ""


# ══════════════════════════════════════════════════════════════════════════════
# 3.4  Regime Robustness
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.4] Regime Robustness …")

REGIMES = ["BULL", "BEAR", "TRANSITIONAL"]

regime_records = []

for feat in signals_for_next:
    ftype = candidates[candidates["feature"] == feat]["feature_type"].values
    ftype = ftype[0] if len(ftype) > 0 else (
        fdr_surv[fdr_surv["feature"] == feat]["feature_type"].values[0]
    )

    feat_s  = df[feat].values.astype(float)
    tgt_s   = df[TARGET_COL].values
    mask    = ~(np.isnan(feat_s) | np.isnan(tgt_s))

    # Overall direction
    x_all, y_all = feat_s[mask], tgt_s[mask]
    if ftype in ("categorical", "binary"):
        lvs    = np.unique(x_all)
        lm     = {lv: y_all[x_all == lv].mean() for lv in lvs}
        overall_dir = max(lm.values())
    else:
        rho, _ = spearmanr(x_all, y_all)
        overall_dir = rho

    regime_col = df["market_regime"].values
    regime_results = {}

    for rg in REGIMES:
        rg_mask = (regime_col == rg) & mask
        xr, yr  = feat_s[rg_mask], tgt_s[rg_mask]
        if len(yr) < 20:
            regime_results[rg] = None
            continue
        if ftype in ("categorical", "binary"):
            lvs_r = np.unique(xr)
            lm_r  = {lv: yr[xr == lv].mean() for lv in lvs_r if (xr == lv).sum() > 0}
            rg_dir = max(lm_r.values()) if lm_r else 0.0
        else:
            rg_rho, _ = spearmanr(xr, yr)
            rg_dir = rg_rho
        regime_results[rg] = rg_dir

    # Classify
    valid_regimes = {k: v for k, v in regime_results.items() if v is not None}
    if not valid_regimes:
        classification = "MIXED"
    else:
        same = {k: (v > 0) == (overall_dir > 0) for k, v in valid_regimes.items()}
        n_same = sum(same.values())
        n_valid = len(same)
        if n_same == n_valid:
            classification = "UNIVERSAL"
        elif n_same == 0:
            classification = "MIXED"
        else:
            # Find which regime is consistent
            consistent_regimes = [k for k, v in same.items() if v]
            if consistent_regimes == ["BULL"]:
                classification = "CONDITIONAL_BULL"
            elif consistent_regimes == ["BEAR"]:
                classification = "CONDITIONAL_BEAR"
            elif consistent_regimes == ["TRANSITIONAL"]:
                classification = "CONDITIONAL_TRANSITIONAL"
            else:
                classification = "MIXED"

    rec = {
        "feature"          : feat,
        "feature_type"     : ftype,
        "classification"   : classification,
        "overall_direction": overall_dir,
    }
    for rg in REGIMES:
        rec[f"dir_{rg}"] = regime_results.get(rg)
    regime_records.append(rec)

regime_df = pd.DataFrame(regime_records)
regime_df.to_csv(os.path.join(OUT_DIR, "regime_robustness.csv"), index=False)
print(f"  Regime classifications:")
print(regime_df["classification"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 3.5  Bootstrap Confidence Intervals
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.5] Bootstrap Confidence Intervals (5,000 resamples) …")

N_BOOT = 5_000
dir_col = "fwd_dir_3d"

boot_records = []

for _, row in regime_df.iterrows():
    feat  = row["feature"]
    ftype = row["feature_type"]

    feat_s  = df[feat].values.astype(float)
    tgt_s   = df[TARGET_COL].values
    dir_s   = df[dir_col].values
    mask    = ~(np.isnan(feat_s) | np.isnan(tgt_s) | np.isnan(dir_s))
    x_clean = feat_s[mask]
    y_clean = tgt_s[mask]
    d_clean = dir_s[mask]
    n       = len(y_clean)

    # Determine "signal fires" condition
    if ftype in ("categorical", "binary"):
        lvs = np.unique(x_clean)
        lm  = {lv: y_clean[x_clean == lv].mean() for lv in lvs}
        best_lv = max(lm, key=lambda l: lm[l])
        active  = (x_clean == best_lv)
    else:
        # Use signal direction: if positive rho, take top 30th percentile; else bottom
        rho, _ = spearmanr(x_clean, y_clean)
        thresh  = np.percentile(x_clean, 70 if rho > 0 else 30)
        active  = (x_clean >= thresh) if rho > 0 else (x_clean <= thresh)

    y_active  = y_clean[active]
    d_active  = d_clean[active]
    n_active  = len(y_active)

    if n_active < 20:
        boot_records.append({
            "feature": feat, "feature_type": ftype,
            "n_active": n_active,
            "mean_ret_obs": np.nan, "hit_rate_obs": np.nan,
            "ret_ci95_lo": np.nan, "ret_ci95_hi": np.nan,
            "ret_ci99_lo": np.nan, "ret_ci99_hi": np.nan,
            "hr_ci95_lo": np.nan, "hr_ci95_hi": np.nan,
            "hr_ci99_lo": np.nan, "hr_ci99_hi": np.nan,
            "directionally_uncertain": True,
        })
        continue

    mean_ret_obs = y_active.mean()
    hit_rate_obs = d_active.mean()

    boot_means = np.empty(N_BOOT)
    boot_hrs   = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n_active, size=n_active)
        boot_means[b] = y_active[idx].mean()
        boot_hrs[b]   = d_active[idx].mean()

    ret_ci95  = np.percentile(boot_means, [2.5, 97.5])
    ret_ci99  = np.percentile(boot_means, [0.5, 99.5])
    hr_ci95   = np.percentile(boot_hrs,   [2.5, 97.5])
    hr_ci99   = np.percentile(boot_hrs,   [0.5, 99.5])

    dir_uncertain = hr_ci99[0] < 0.50

    boot_records.append({
        "feature"              : feat,
        "feature_type"         : ftype,
        "n_active"             : n_active,
        "mean_ret_obs"         : mean_ret_obs,
        "hit_rate_obs"         : hit_rate_obs,
        "ret_ci95_lo"          : ret_ci95[0],
        "ret_ci95_hi"          : ret_ci95[1],
        "ret_ci99_lo"          : ret_ci99[0],
        "ret_ci99_hi"          : ret_ci99[1],
        "hr_ci95_lo"           : hr_ci95[0],
        "hr_ci95_hi"           : hr_ci95[1],
        "hr_ci99_lo"           : hr_ci99[0],
        "hr_ci99_hi"           : hr_ci99[1],
        "directionally_uncertain": dir_uncertain,
    })

boot_df = pd.DataFrame(boot_records)
boot_df.to_csv(os.path.join(OUT_DIR, "bootstrap_ci.csv"), index=False)

n_uncertain = boot_df["directionally_uncertain"].sum()
print(f"  Directionally uncertain signals : {n_uncertain}/{len(boot_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# Build validated_signals (pass ALL stages)
# ══════════════════════════════════════════════════════════════════════════════
# A signal passes all stages if:
#   - survived FDR
#   - stable (or fallback used)
#   - not directionally uncertain
#   - not MIXED regime (optional filter: keep all except MIXED + uncertain)

if len(stable_df) >= 5:
    valid_set = set(stable_df["feature"])
else:
    valid_set = set(fdr_surv["feature"])

validated = (
    boot_df[
        (boot_df["feature"].isin(valid_set)) &
        (~boot_df["directionally_uncertain"])
    ]
    .merge(regime_df[["feature", "classification", "overall_direction"]], on="feature", how="left")
    .merge(stab_df[["feature", "stability_score", "n_windows", "consistent_windows"]], on="feature", how="left")
    .merge(mc_df[["feature", "p_value", "p_adj_bh"]], on="feature", how="left")
)
validated.to_csv(os.path.join(OUT_DIR, "validated_signals.csv"), index=False)
print(f"\n  Validated signals (all stages) : {len(validated)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.6  Accuracy-Selectivity Surface
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3.6] Accuracy-Selectivity Surface …")

# Build active_* columns for each validated signal on the full df
working = df.copy()

# Use all signals_for_next for the surface (validated or fallback)
surface_features = signals_for_next  # already set above

# Also include signals that were directionally uncertain but still have a direction
# (surface uses all regime-classified signals, including uncertain ones, per spec)
surface_features_set = set(surface_features)

for feat in surface_features_set:
    ftype = (
        candidates[candidates["feature"] == feat]["feature_type"].values
        if feat in candidates["feature"].values
        else None
    )
    if ftype is None or len(ftype) == 0:
        row_s = fdr_surv[fdr_surv["feature"] == feat]
        ftype = row_s["feature_type"].values[0] if len(row_s) > 0 else "continuous"
    else:
        ftype = ftype[0]

    feat_s  = working[feat].values.astype(float)
    tgt_s   = working[TARGET_COL].values
    mask    = ~(np.isnan(feat_s) | np.isnan(tgt_s))
    x_c     = feat_s[mask]
    y_c     = tgt_s[mask]

    if ftype in ("categorical", "binary"):
        lvs = np.unique(x_c)
        lm  = {lv: y_c[x_c == lv].mean() for lv in lvs}
        best_lv = max(lm, key=lambda l: lm[l])
        active_col = (feat_s == best_lv).astype(int)
    else:
        rho, _ = spearmanr(x_c, y_c)
        thresh  = np.nanpercentile(feat_s, 70 if rho > 0 else 30)
        if rho > 0:
            active_col = (feat_s >= thresh).astype(int)
        else:
            active_col = (feat_s <= thresh).astype(int)
        # NaN → 0 (not active)
        active_col[np.isnan(feat_s)] = 0

    working[f"active_{feat}"] = active_col

# Sum across active columns
active_cols = [f"active_{f}" for f in surface_features_set]
working["signal_count"] = working[active_cols].sum(axis=1)

# Prepare target columns for surface
need_cols = ["fwd_dir_1d", "fwd_dir_3d", "fwd_dir_5d", "fwd_dir_10d",
             "fwd_ret_1d",  "fwd_ret_3d",  "fwd_ret_5d",  "fwd_ret_10d",
             "signal_count"]

surf_data = working[need_cols].copy()
max_sc = int(surf_data["signal_count"].max())

TRADING_DAYS = 252
N_BOOT_SURF = 5_000

surface_rows = []

def compute_surface_row(subset, label):
    n   = len(subset)
    if n == 0:
        return None
    wr_1d  = subset["fwd_dir_1d"].mean()
    wr_3d  = subset["fwd_dir_3d"].mean()
    wr_5d  = subset["fwd_dir_5d"].mean()
    wr_10d = subset["fwd_dir_10d"].mean()
    mr_3d  = subset["fwd_ret_3d"].mean()

    # Bootstrap 3d win rate CI
    d3 = subset["fwd_dir_3d"].values
    boot_hr = np.empty(N_BOOT_SURF)
    for b in range(N_BOOT_SURF):
        idx = rng.integers(0, n, size=n)
        boot_hr[b] = d3[idx].mean()
    hr95 = np.percentile(boot_hr, [2.5, 97.5])
    hr99 = np.percentile(boot_hr, [0.5, 99.5])

    # Annualised Sharpe (daily returns, 3d holding)
    rets = subset["fwd_ret_3d"].values
    sharpe = (rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else np.nan

    return {
        "threshold_label"   : label,
        "n_trades"          : n,
        "pct_of_days"       : n / len(surf_data),
        "win_rate_1d"       : wr_1d,
        "win_rate_3d"       : wr_3d,
        "win_rate_5d"       : wr_5d,
        "win_rate_10d"      : wr_10d,
        "mean_ret_3d"       : mr_3d,
        "wr3d_ci95_lo"      : hr95[0],
        "wr3d_ci95_hi"      : hr95[1],
        "wr3d_ci99_lo"      : hr99[0],
        "wr3d_ci99_hi"      : hr99[1],
        "sharpe_ann"        : sharpe,
    }

# Exact counts 1 … max_sc
for sc in range(1, max_sc + 1):
    subset = surf_data[surf_data["signal_count"] == sc].dropna(subset=["fwd_dir_3d", "fwd_ret_3d"])
    row = compute_surface_row(subset, f"exactly_{sc}")
    if row:
        surface_rows.append(row)

# Cumulative "N or more"
for sc in range(1, max_sc + 1):
    subset = surf_data[surf_data["signal_count"] >= sc].dropna(subset=["fwd_dir_3d", "fwd_ret_3d"])
    row = compute_surface_row(subset, f"{sc}_or_more")
    if row:
        surface_rows.append(row)

surf_df = pd.DataFrame(surface_rows)
surf_df.to_csv(os.path.join(OUT_DIR, "accuracy_selectivity_surface.csv"), index=False)

# Find empirically optimal threshold: max 95% CI lower bound on 3d win rate
or_more = surf_df[surf_df["threshold_label"].str.endswith("_or_more")].copy()
if len(or_more) > 0:
    best_idx    = or_more["wr3d_ci95_lo"].idxmax()
    best_row    = or_more.loc[best_idx]
    opt_thresh  = best_row["threshold_label"]
    opt_wr      = best_row["win_rate_3d"]
    opt_lo      = best_row["wr3d_ci95_lo"]
    opt_ntrades = best_row["n_trades"]
    opt_pct     = best_row["pct_of_days"]
else:
    opt_thresh  = "N/A"
    opt_wr      = np.nan
    opt_lo      = np.nan
    opt_ntrades = 0
    opt_pct     = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("PIPELINE SUMMARY")
print("=" * 65)
print(f"  Candidates entering Monte Carlo    : {len(mc_df)}")
print(f"  Surviving p < 0.005               : {n_survive_raw}")
print(f"  Surviving FDR correction (BH 1%)  : {len(fdr_surv)}")
print(f"  Surviving temporal stability ≥0.70: {len(stable_df)}")
print(f"  Validated signals (all stages)    : {len(validated)}")
if surface_note:
    print(f"  {surface_note}")
print()
print(f"  Empirically optimal threshold : {opt_thresh}")
print(f"    3d win rate (observed)      : {opt_wr:.1%}" if not np.isnan(opt_wr) else "    3d win rate: N/A")
print(f"    95% CI lower bound          : {opt_lo:.1%}" if not np.isnan(opt_lo) else "    95% CI lower bound: N/A")
print(f"    Trade frequency             : {opt_ntrades} days ({opt_pct:.1%} of dataset)")
print()

# Full accuracy-selectivity surface table
print("=" * 65)
print("ACCURACY-SELECTIVITY SURFACE")
print("=" * 65)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)
print(surf_df.to_string(index=False))

# Verify all output files exist and non-empty
print()
print("=" * 65)
print("OUTPUT FILE VERIFICATION")
print("=" * 65)
files = [
    "monte_carlo_results.csv",
    "fdr_survivors.csv",
    "temporal_stability.csv",
    "regime_robustness.csv",
    "bootstrap_ci.csv",
    "accuracy_selectivity_surface.csv",
    "validated_signals.csv",
]
all_ok = True
for f in files:
    path = os.path.join(OUT_DIR, f)
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    rows   = pd.read_csv(path).shape[0] if exists and size > 0 else 0
    status = "OK" if exists and rows > 0 else "PROBLEM"
    if status != "OK":
        all_ok = False
    print(f"  [{status}] {f:45s} {rows:>5} rows")

print()
print("All outputs verified." if all_ok else "WARNING: Some outputs missing or empty!")
print("Step 3 complete.")
