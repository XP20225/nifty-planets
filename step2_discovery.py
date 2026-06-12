"""
Step 2: Discovery Loop
Runs full statistical discovery on nifty_enriched.csv
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
DATA_PATH = f"{BASE}/data/nifty_enriched.csv"
OUT_DIR = f"{BASE}/results/discovery"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"  Loaded {len(df)} rows × {df.shape[1]} cols")

# ── feature groups ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "paksha", "moon_phase_quad", "tithi_quality", "moon_nak_name",
    "moon_nak_lord", "moon_nak_quality", "karana", "karana_lord",
    "yogi_nak", "yogi_lord", "mahadasha", "antardasha", "dasha_quality",
    "ju_sa_quad", "rahu_sign", "rahu_ketu_axis", "market_regime",
    "sarva_bucket", "dow", "month",
    "ix_nak_tithi_qual", "ix_nak_paksha", "ix_nak_regime", "ix_nak_me_retro",
    "ix_karana_moon_phase", "ix_tithiq_nakq", "ix_dasha_regime",
    "ix_gandanta_war", "ix_eclipse_moonphase", "ix_sarva_regime",
    "ix_sarva_dasha", "ix_jusa_paksha", "ix_rahu_sign_nak",
    "ix_comb_benefic_regime", "ix_yogi_paksha_tithiq",
]

CONTINUOUS_FEATURES = [
    "tithi_num", "moon_nak_num", "sun_moon_sep", "ju_sa_angle",
    "sarvashtakavarga", "ju_dignity", "sa_dignity", "moon_strength", "yogi_pt",
    "log_ret", "range_pct", "rvol_5d", "rvol_10d", "rvol_20d", "rvol_60d",
    "atr14", "dist_sma50", "dist_sma200", "mom_rank",
]

BINARY_FEATURES = [
    "me_retro_confirmed", "graha_yuddha", "eclipse_corridor", "eclipse_day",
    "gandanta_moon", "gandanta_any", "yogi_activation", "avayogi_activation",
    "combust_any_benefic", "combust_Mo", "combust_Me", "combust_Ve",
    "combust_Ma", "combust_Ju", "combust_Sa",
]

CONT_TARGETS = ["fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d"]
BIN_TARGETS  = ["fwd_dir_3d", "fwd_vol_hi"]
ALL_TARGETS  = CONT_TARGETS + BIN_TARGETS

# keep only features that exist
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in df.columns]
CONTINUOUS_FEATURES  = [c for c in CONTINUOUS_FEATURES  if c in df.columns]
BINARY_FEATURES      = [c for c in BINARY_FEATURES      if c in df.columns]
ALL_TARGETS          = [c for c in ALL_TARGETS           if c in df.columns]

print(f"  Cat features: {len(CATEGORICAL_FEATURES)}")
print(f"  Cont features: {len(CONTINUOUS_FEATURES)}")
print(f"  Binary features: {len(BINARY_FEATURES)}")
print(f"  Targets: {ALL_TARGETS}")


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def eta_squared_kw(H, k, n):
    """η² from Kruskal-Wallis H stat."""
    if n <= k:
        return 0.0
    val = (H - k + 1) / (n - k)
    return float(np.clip(val, 0, 1))


def kruskal_stats(series_feat, series_tgt):
    """Return (H, p, eta2, per_level_dict) for a categorical feature vs continuous target."""
    groups = {}
    for lv, grp in series_tgt.groupby(series_feat):
        g = grp.dropna()
        if len(g) >= 3:
            groups[lv] = g.values
    if len(groups) < 2:
        return None
    k = len(groups)
    n = sum(len(v) for v in groups.values())
    try:
        H, p = stats.kruskal(*groups.values())
    except Exception:
        return None
    eta2 = eta_squared_kw(H, k, n)
    per_level = {}
    for lv, vals in groups.items():
        per_level[lv] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "hit_rate": float(np.mean(vals > 0)),
            "count": len(vals),
        }
    return {"H": H, "p": p, "eta2": eta2, "k": k, "n": n, "per_level": per_level}


def binary_target_stats(series_feat, series_tgt):
    """Chi-square + Cramér V for categorical feature vs binary target."""
    tmp = pd.DataFrame({"f": series_feat, "t": series_tgt}).dropna()
    if len(tmp) < 10:
        return None
    ct = pd.crosstab(tmp["f"], tmp["t"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    try:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
    except Exception:
        return None
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * max(k, 1))))
    per_level = {}
    for lv in ct.index:
        row = ct.loc[lv]
        total = row.sum()
        pos = row.get(1, 0)
        per_level[lv] = {
            "mean": float(pos / total) if total > 0 else np.nan,
            "median": np.nan,
            "hit_rate": float(pos / total) if total > 0 else np.nan,
            "count": int(total),
        }
    return {"H": chi2, "p": p, "eta2": v, "k": ct.shape[0], "n": int(n), "per_level": per_level}


def mi_continuous(feat_vals, tgt_vals, is_binary_target=False):
    """Mutual information between continuous feature and target."""
    mask = ~(np.isnan(feat_vals) | np.isnan(tgt_vals))
    X = feat_vals[mask].reshape(-1, 1)
    y = tgt_vals[mask]
    if len(y) < 20:
        return np.nan
    try:
        if is_binary_target:
            return float(mutual_info_classif(X, y, discrete_features=False, random_state=42)[0])
        else:
            return float(mutual_info_regression(X, y, random_state=42)[0])
    except Exception:
        return np.nan


def mi_categorical(feat_vals, tgt_vals, is_binary_target=False):
    """Mutual information between label-encoded categorical feature and target."""
    tmp = pd.DataFrame({"f": feat_vals, "t": tgt_vals}).dropna()
    if len(tmp) < 20:
        return np.nan
    le = LabelEncoder()
    X = le.fit_transform(tmp["f"].astype(str)).reshape(-1, 1)
    y = tmp["t"].values
    try:
        if is_binary_target:
            return float(mutual_info_classif(X, y, discrete_features=True, random_state=42)[0])
        else:
            return float(mutual_info_regression(X, y, discrete_features=True, random_state=42)[0])
    except Exception:
        return np.nan


def spearman_pearson(feat_vals, tgt_vals):
    mask = ~(np.isnan(np.array(feat_vals, dtype=float)) | np.isnan(np.array(tgt_vals, dtype=float)))
    x = np.array(feat_vals, dtype=float)[mask]
    y = np.array(tgt_vals, dtype=float)[mask]
    if len(y) < 10:
        return np.nan, np.nan, np.nan, np.nan
    try:
        rho, sp = stats.spearmanr(x, y)
    except Exception:
        rho, sp = np.nan, np.nan
    try:
        r, pp = stats.pearsonr(x, y)
    except Exception:
        r, pp = np.nan, np.nan
    return float(rho), float(sp), float(r), float(pp)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(data, label="full"):
    rows = []
    n_total = len(data)

    # ── CATEGORICAL features ────────────────────────────────────────────────
    print(f"  [{label}] Categorical scan ({len(CATEGORICAL_FEATURES)} features × {len(ALL_TARGETS)} targets)...")
    for feat in CATEGORICAL_FEATURES:
        if feat not in data.columns:
            continue
        for tgt in ALL_TARGETS:
            if tgt not in data.columns:
                continue
            is_bin = tgt in BIN_TARGETS
            sub = data[[feat, tgt]].dropna()
            if len(sub) < 20:
                continue

            if is_bin:
                res = binary_target_stats(sub[feat], sub[tgt])
            else:
                res = kruskal_stats(sub[feat], sub[tgt])

            mi = mi_categorical(sub[feat], sub[tgt], is_binary_target=is_bin)

            if res is None:
                continue

            base = {
                "feature": feat,
                "feature_type": "categorical",
                "target": tgt,
                "n": res["n"],
                "k_levels": res["k"],
                "H_or_chi2": res["H"],
                "p_value": res["p"],
                "effect_size": res["eta2"],   # eta2 or cramers_v
                "abs_effect_size": abs(res["eta2"]),
                "mutual_info": mi,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
            }
            # per-level stats (flattened)
            for lv, st in res["per_level"].items():
                row = dict(base)
                row["level"] = str(lv)
                row["level_mean"] = st["mean"]
                row["level_median"] = st["median"]
                row["level_hit_rate"] = st["hit_rate"]
                row["level_count"] = st["count"]
                rows.append(row)

    # ── CONTINUOUS features ─────────────────────────────────────────────────
    print(f"  [{label}] Continuous scan ({len(CONTINUOUS_FEATURES)} features × {len(ALL_TARGETS)} targets)...")
    for feat in CONTINUOUS_FEATURES:
        if feat not in data.columns:
            continue
        feat_arr = pd.to_numeric(data[feat], errors="coerce").values
        for tgt in ALL_TARGETS:
            if tgt not in data.columns:
                continue
            is_bin = tgt in BIN_TARGETS
            tgt_arr = pd.to_numeric(data[tgt], errors="coerce").values
            mask = ~(np.isnan(feat_arr) | np.isnan(tgt_arr))
            n = mask.sum()
            if n < 20:
                continue
            rho, sp, r, pp = spearman_pearson(feat_arr[mask], tgt_arr[mask])
            mi = mi_continuous(feat_arr, tgt_arr, is_binary_target=is_bin)
            rows.append({
                "feature": feat,
                "feature_type": "continuous",
                "target": tgt,
                "n": int(n),
                "k_levels": np.nan,
                "H_or_chi2": np.nan,
                "p_value": sp,
                "effect_size": rho,
                "abs_effect_size": abs(rho) if not np.isnan(rho) else np.nan,
                "mutual_info": mi,
                "spearman_rho": rho,
                "spearman_p": sp,
                "pearson_r": r,
                "pearson_p": pp,
                "level": np.nan,
                "level_mean": np.nan,
                "level_median": np.nan,
                "level_hit_rate": np.nan,
                "level_count": np.nan,
            })

    # ── BINARY features ─────────────────────────────────────────────────────
    print(f"  [{label}] Binary scan ({len(BINARY_FEATURES)} features × {len(ALL_TARGETS)} targets)...")
    for feat in BINARY_FEATURES:
        if feat not in data.columns:
            continue
        for tgt in ALL_TARGETS:
            if tgt not in data.columns:
                continue
            is_bin = tgt in BIN_TARGETS
            sub = data[[feat, tgt]].dropna()
            if len(sub) < 20:
                continue
            feat_arr = pd.to_numeric(sub[feat], errors="coerce").values
            tgt_arr  = pd.to_numeric(sub[tgt],  errors="coerce").values
            mask = ~(np.isnan(feat_arr) | np.isnan(tgt_arr))
            if mask.sum() < 20:
                continue

            rho, sp, r, pp = spearman_pearson(feat_arr[mask], tgt_arr[mask])
            mi = mi_continuous(feat_arr, tgt_arr, is_binary_target=is_bin)

            # also per-level stats (0 vs 1)
            groups = {0: tgt_arr[mask & (feat_arr == 0)], 1: tgt_arr[mask & (feat_arr == 1)]}
            for lv, vals in groups.items():
                if len(vals) == 0:
                    continue
                rows.append({
                    "feature": feat,
                    "feature_type": "binary",
                    "target": tgt,
                    "n": int(mask.sum()),
                    "k_levels": 2,
                    "H_or_chi2": np.nan,
                    "p_value": sp,
                    "effect_size": rho,
                    "abs_effect_size": abs(rho) if not np.isnan(rho) else np.nan,
                    "mutual_info": mi,
                    "spearman_rho": rho,
                    "spearman_p": sp,
                    "pearson_r": r,
                    "pearson_p": pp,
                    "level": int(lv),
                    "level_mean": float(np.mean(vals)),
                    "level_median": float(np.median(vals)),
                    "level_hit_rate": float(np.mean(vals > 0)),
                    "level_count": len(vals),
                })

    out = pd.DataFrame(rows)
    out = out.sort_values("abs_effect_size", ascending=False).reset_index(drop=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. UNCONDITIONAL SCAN
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. Unconditional Scan ===")
unc = run_scan(df, "full")
path1 = f"{OUT_DIR}/unconditional_scan.csv"
unc.to_csv(path1, index=False)
print(f"  Saved {len(unc)} rows → {path1}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONDITIONAL SCANS (by market_regime)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Conditional Scans by market_regime ===")
for regime in ["BULL", "BEAR", "TRANSITIONAL"]:
    sub = df[df["market_regime"] == regime].copy()
    print(f"  Regime={regime}: {len(sub)} rows")
    scan = run_scan(sub, regime)
    fname = f"{OUT_DIR}/conditional_scan_{regime.lower()}.csv"
    scan.to_csv(fname, index=False)
    print(f"  Saved {len(scan)} rows → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXTREME DAY PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Extreme Day Patterns ===")

def extreme_profile(sub, full_df, label):
    """Build a profile dict for a subset of extreme days."""
    if len(sub) == 0:
        return {}

    def pct_col(col, val=1):
        if col not in sub.columns:
            return np.nan
        return float((sub[col] == val).mean() * 100)

    def top_val(col):
        if col not in sub.columns:
            return np.nan
        vc = sub[col].value_counts()
        return vc.index[0] if len(vc) > 0 else np.nan

    full_sarva = full_df["sarvashtakavarga"].mean() if "sarvashtakavarga" in full_df.columns else np.nan
    sub_sarva  = sub["sarvashtakavarga"].mean() if "sarvashtakavarga" in sub.columns else np.nan

    return {
        "label": label,
        "n": len(sub),
        "top_moon_nak": top_val("moon_nak_name"),
        "top_karana": top_val("karana"),
        "top_tithi_quality": top_val("tithi_quality"),
        "top_paksha": top_val("paksha"),
        "avg_sarva": round(sub_sarva, 3) if not np.isnan(sub_sarva) else np.nan,
        "full_avg_sarva": round(full_sarva, 3) if not np.isnan(full_sarva) else np.nan,
        "sarva_vs_full": round(sub_sarva - full_sarva, 3) if not (np.isnan(sub_sarva) or np.isnan(full_sarva)) else np.nan,
        "eclipse_corridor_pct": pct_col("eclipse_corridor"),
        "graha_yuddha_pct": pct_col("graha_yuddha"),
        "gandanta_moon_pct": pct_col("gandanta_moon"),
        "me_retro_pct": pct_col("me_retro_confirmed"),
        "top_mahadasha": top_val("mahadasha"),
        "top_ju_sa_quad": top_val("ju_sa_quad"),
        "top_market_regime": top_val("market_regime"),
    }

records = []

# prep
df_sorted_ret = df.dropna(subset=["log_ret"]).copy()
df_sorted_vol = df.dropna(subset=["rvol_5d"]).copy()

# top50 up days
top_up = df_sorted_ret.nlargest(50, "log_ret")
records.append(extreme_profile(top_up, df, "top50_up"))

# top50 down days
top_dn = df_sorted_ret.nsmallest(50, "log_ret")
records.append(extreme_profile(top_dn, df, "top50_down"))

# top50 high-vol
top_hv = df_sorted_vol.nlargest(50, "rvol_5d")
records.append(extreme_profile(top_hv, df, "top50_high_vol"))

# top50 low-vol
top_lv = df_sorted_vol.nsmallest(50, "rvol_5d")
records.append(extreme_profile(top_lv, df, "top50_low_vol"))

# top30 trend-continuation (same sign as prior day)
if "log_ret" in df.columns:
    df2 = df.copy()
    df2["prev_ret"] = df2["log_ret"].shift(1)
    df2["continuation"] = (df2["log_ret"] * df2["prev_ret"] > 0).astype(int)
    cont_sub = df2[df2["continuation"] == 1].dropna(subset=["log_ret"])
    top_cont = cont_sub.reindex(cont_sub["log_ret"].abs().nlargest(30).index)
    records.append(extreme_profile(top_cont, df, "top30_trend_continuation"))

    # top30 reversals (opposite sign to prior day, large move)
    df2["reversal"] = (df2["log_ret"] * df2["prev_ret"] < 0).astype(int)
    rev_sub = df2[df2["reversal"] == 1].dropna(subset=["log_ret"])
    top_rev = rev_sub.reindex(rev_sub["log_ret"].abs().nlargest(30).index)
    records.append(extreme_profile(top_rev, df, "top30_reversals"))

ext_df = pd.DataFrame(records)
path3 = f"{OUT_DIR}/extreme_day_patterns.csv"
ext_df.to_csv(path3, index=False)
print(f"  Saved {len(ext_df)} rows → {path3}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. COHERENCE SCORES
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. Coherence Scores ===")

cdf = df[["date"]].copy()

# component signals
def s(cond, pos_val=1, neg_val=-1, else_val=0):
    return np.where(cond > 0, pos_val, np.where(cond < 0, neg_val, else_val))

# 1. Paksha
cdf["c_paksha"] = np.where(df["paksha"] == "SHUKLA", 1,
                  np.where(df["paksha"] == "KRISHNA", -1, 0))

# 2. Tithi quality
cdf["c_tithi"] = np.where(df["tithi_quality"] == "NANDA", 1,
                 np.where(df["tithi_quality"] == "RIKTA", -1, 0))

# 3. Nak quality
cdf["c_nak_qual"] = np.where(df["moon_nak_quality"].isin(["LAGHU", "MRIDU"]), 1,
                   np.where(df["moon_nak_quality"].isin(["TIKSHNA", "UGRA"]), -1, 0))

# 4. Dasha quality
cdf["c_dasha"] = np.where(df["dasha_quality"] == "BENEVOLENT", 1,
                np.where(df["dasha_quality"] == "MALEFIC", -1, 0))

# 5. Sarvashtakavarga
sarva = pd.to_numeric(df["sarvashtakavarga"], errors="coerce")
cdf["c_sarva"] = np.where(sarva > 48, 1, np.where(sarva < 25, -1, 0))

# 6. Jupiter dignity
ju_dig = pd.to_numeric(df["ju_dignity"], errors="coerce")
cdf["c_ju_dig"] = np.where(ju_dig >= 3, 1, np.where(ju_dig <= 0, -1, 0))

# 7. Paksha + combust_Mo interaction
cdf["c_paksha_moon"] = np.where(
    (df["paksha"] == "SHUKLA") & (df["combust_Mo"] == 0), 1,
    np.where(
        (df["paksha"] == "KRISHNA") & (df["combust_Mo"] == 1), -1, 0
    )
)

# 8. Eclipse corridor
cdf["c_eclipse"] = np.where(df["eclipse_corridor"] == 1, -1, 0)

# 9. Graha yuddha
cdf["c_war"] = np.where(df["graha_yuddha"] == 1, -1, 0)

# 10. Gandanta moon
cdf["c_gandanta"] = np.where(df["gandanta_moon"] == 1, -1, 0)

# 11. Yogi / avayogi activation
cdf["c_yogi"] = np.where(df["yogi_activation"] == 1, 1,
               np.where(df["avayogi_activation"] == 1, -1, 0))

# 12. JuSa quad
cdf["c_jusa"] = np.where(df["ju_sa_quad"] == "Q1_CONJUNCTION", 1,
               np.where(df["ju_sa_quad"] == "Q3_OPPOSITION", -1, 0))

# Sum all component columns
comp_cols = [c for c in cdf.columns if c.startswith("c_")]
cdf["coherence_direction"] = cdf[comp_cols].sum(axis=1)
cdf["coherence_magnitude"] = cdf["coherence_direction"].abs()

# Forward volatility (shift rvol_5d back by 1 to get "next 5d vol")
if "rvol_5d" in df.columns:
    cdf["fwd_vol_5d"] = df["rvol_5d"].shift(-1).values

# Forward direction 3d
if "fwd_dir_3d" in df.columns:
    cdf["fwd_dir_3d"] = df["fwd_dir_3d"].values

# Correlations
print("  Coherence correlations:")
if "fwd_vol_5d" in cdf.columns:
    tmp = cdf[["coherence_magnitude", "fwd_vol_5d"]].dropna()
    rho, p = stats.spearmanr(tmp["coherence_magnitude"], tmp["fwd_vol_5d"])
    print(f"    coherence_magnitude vs fwd_vol_5d: rho={rho:.4f}, p={p:.4f}, n={len(tmp)}")
    cdf["corr_mag_vol_rho"] = rho
    cdf["corr_mag_vol_p"] = p

if "fwd_dir_3d" in cdf.columns:
    tmp2 = cdf[["coherence_direction", "fwd_dir_3d"]].dropna()
    rho2, p2 = stats.spearmanr(tmp2["coherence_direction"], tmp2["fwd_dir_3d"])
    print(f"    coherence_direction vs fwd_dir_3d: rho={rho2:.4f}, p={p2:.4f}, n={len(tmp2)}")
    cdf["corr_dir_fwddir_rho"] = rho2
    cdf["corr_dir_fwddir_p"] = p2

print(f"  Coherence direction range: [{cdf['coherence_direction'].min()}, {cdf['coherence_direction'].max()}]")
print(f"  Coherence magnitude range: [{cdf['coherence_magnitude'].min()}, {cdf['coherence_magnitude'].max()}]")

path4 = f"{OUT_DIR}/coherence_scores.csv"
cdf.to_csv(path4, index=False)
print(f"  Saved {len(cdf)} rows → {path4}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Output file check ===")
files = [
    f"{OUT_DIR}/unconditional_scan.csv",
    f"{OUT_DIR}/conditional_scan_bull.csv",
    f"{OUT_DIR}/conditional_scan_bear.csv",
    f"{OUT_DIR}/conditional_scan_transitional.csv",
    f"{OUT_DIR}/extreme_day_patterns.csv",
    f"{OUT_DIR}/coherence_scores.csv",
]
for f in files:
    if os.path.exists(f):
        sz = os.path.getsize(f)
        rows = sum(1 for _ in open(f)) - 1
        print(f"  OK  {os.path.basename(f):45s}  {rows:6d} rows  {sz:10d} bytes")
    else:
        print(f"  MISSING: {f}")

print("\nDone.")
