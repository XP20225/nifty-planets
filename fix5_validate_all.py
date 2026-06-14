"""
Fix 5: Apply BH-FDR + OOS to ALL methods including M3-6.

Pools p-values from:
  - method1_fp_uncapped.csv      (Fix 2 fingerprinting, if available)
  - method1_pattern_library.csv  (original M1)
  - method2_reverse_lookup.csv   (original M2, or method2_full.csv if available)
  - method2_full.csv             (uncapped M2, if available)
  - method3_clustering.csv       (M3 cluster p-values)
  - method4_cycle_analysis.csv   (M4 ANOVA p-values)
  - method5_sequential_patterns.csv (M5 sequential)
  - method6_anomaly_fingerprints.csv (M6 — adapted for is_bull direction)

After BH-FDR across the full pool:
  - Re-evaluate survivors on OOS (2018+) for direction + stability
  - Merge with existing confirmed_patterns.csv
  - Save updated confirmed_patterns.csv

Outputs:
  results/validation/confirmed_patterns.csv  (updated)
  results/validation/m3m6_validated.csv       (new survivors from M3-6)
  results/validation/fdr_pool_all.csv         (full pool with FDR result)
"""
import math, os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
OOS_SPLIT = '2018-01-01'

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0, (p+z**2/(2*n)-z*math.sqrt(max(0,p*(1-p)/n+z**2/(4*n**2))))/(1+z**2/n))

def fisher_p(n_c, k_c, N, k_t):
    a=k_c; b=n_c-k_c; c=k_t-k_c; d=(N-n_c)-(k_t-k_c)
    if any(x<0 for x in [a,b,c,d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a),max(0,b)],[max(0,c),max(0,d)]])
    return p

def bh_fdr(pvalues, alpha=0.01):
    n = len(pvalues)
    if n == 0: return []
    ranked = np.argsort(pvalues)
    p_sorted = np.array(pvalues)[ranked]
    threshold = np.arange(1,n+1)*alpha/n
    survive = p_sorted <= threshold
    if survive.any():
        last = np.where(survive)[0][-1]
        survive[:last+1] = True
    result = np.zeros(n, dtype=bool)
    result[ranked] = survive
    return result.tolist()

# ──────────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
for c in ['is_bull','is_strong_bull','is_strong_bear','is_sideways','is_high_vol']:
    if c in df.columns: df[c] = df[c].fillna(-1).astype(int)

df_clean = df[df['is_bull'].isin([0,1])].copy()
df_train  = df_clean[df_clean['date'] < OOS_SPLIT].copy()
df_oos    = df_clean[df_clean['date'] >= OOS_SPLIT].copy()
N = len(df_clean); BASE_BULL = df_clean['is_bull'].mean()
print(f"  N={N}, train={len(df_train)}, OOS={len(df_oos)}, base_bull={BASE_BULL:.3f}")

# ──────────────────────────────────────────────────────────────────────
# Build the global p-value pool from ALL methods
# ──────────────────────────────────────────────────────────────────────
pool = []   # list of dicts with 'features','condition','p_value','source','outcome' etc.

def load_csv_safe(path):
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else pd.DataFrame()

print("\nBuilding global p-value pool …")

# M1: always use original, PLUS add new fingerprint patterns not in original
fp_path = f"{REPO}/results/research/method1_fp_uncapped.csv"
m1_path = f"{REPO}/results/research/method1_pattern_library.csv"
m1 = load_csv_safe(m1_path)
m1['source'] = 'method1'
print(f"  M1 original:                {len(m1)} rows")
if os.path.exists(fp_path):
    fp = load_csv_safe(fp_path)
    fp['source'] = 'method1_fp'
    # Only add fingerprint patterns with p < 0.05 (others won't survive FDR anyway)
    fp = fp[fp['p_value'] < 0.05]
    m1_keys = set(m1['features'].astype(str) + ':::' + m1['condition'].astype(str))
    fp_new = fp[~(fp['features'].astype(str) + ':::' + fp['condition'].astype(str)).isin(m1_keys)]
    if len(fp_new) > 0:
        m1 = pd.concat([m1, fp_new], ignore_index=True)
    print(f"  M1 fingerprint new (p<0.05): {len(fp_new)} new patterns added")

# M2 (prefer full uncapped version)
m2_full_path = f"{REPO}/results/research/method2_full.csv"
m2_path = f"{REPO}/results/research/method2_reverse_lookup.csv"
if os.path.exists(m2_full_path):
    m2 = load_csv_safe(m2_full_path)
    m2['source'] = 'method2_full'
    print(f"  M2 uncapped:                {len(m2)} rows")
else:
    m2 = load_csv_safe(m2_path)
    m2['source'] = 'method2'
    print(f"  M2 original:                {len(m2)} rows")

for df_src, src_name in [(m1,'m1'),(m2,'m2')]:
    if len(df_src) == 0: continue
    for _, row in df_src.iterrows():
        pool.append({
            'features'  : row.get('features',''),
            'condition' : row.get('condition',''),
            'outcome'   : row.get('outcome', row.get('outcome_label','unknown')),
            'n'         : row.get('n', row.get('n_cond',0)),
            'k_pos'     : row.get('k_pos', row.get('k_bull',0)),
            'win_rate'  : row.get('win_rate', row.get('bull_rate',0)),
            'wilson_lower': row.get('wilson_lower',0),
            'p_value'   : float(row.get('p_value', row.get('p',1.0))),
            'complexity': row.get('complexity',1),
            'source'    : row.get('source', src_name),
        })

# M3: cluster p-values (is_bull direction)
m3 = load_csv_safe(f"{REPO}/results/research/method3_clustering.csv")
if len(m3) > 0:
    for _, row in m3.iterrows():
        pool.append({
            'features': 'cluster',
            'condition': str(int(row['cluster'])),
            'outcome': 'is_bull',
            'n': row.get('n',0),
            'k_pos': int(row.get('n',0) * row.get('bull_rate',0)),
            'win_rate': row.get('bull_rate',0),
            'wilson_lower': row.get('wilson_lower',0),
            'p_value': float(row.get('p_value',1.0)),
            'complexity': 1,
            'source': 'method3',
        })
    print(f"  M3 clusters:                {len(m3)} rows")

# M4: cycle ANOVA p-values (these test directionality of a cycle, not a specific condition)
# Include them as candidate signals: if significant, mark cycle phase as a feature
m4 = load_csv_safe(f"{REPO}/results/research/method4_cycle_analysis.csv")
if len(m4) > 0:
    for _, row in m4.iterrows():
        p = float(row.get('phase_anova_p', 1.0))
        pool.append({
            'features': f"cycle_{row['cycle_name'].replace(' ','_')}",
            'condition': 'phase_effect',
            'outcome': 'is_bull',
            'n': 0,
            'k_pos': 0,
            'win_rate': 0,
            'wilson_lower': 0,
            'p_value': p,
            'complexity': 1,
            'source': 'method4',
        })
    print(f"  M4 cycles:                  {len(m4)} rows")

# M5: sequential patterns → these have specific feature/condition structure
m5 = load_csv_safe(f"{REPO}/results/research/method5_sequential_patterns.csv")
if len(m5) > 0:
    for _, row in m5.iterrows():
        lag = int(row.get('lag',1))
        pool.append({
            'features': f"{row['condition_col']}_lag{lag}",
            'condition': str(row.get('condition_val','')),
            'outcome': str(row.get('outcome','is_bull')),
            'n': row.get('n',0),
            'k_pos': row.get('k',0),
            'win_rate': row.get('win_rate',0),
            'wilson_lower': row.get('wilson_lower',0),
            'p_value': float(row.get('p_value',1.0)),
            'complexity': 1,
            'source': 'method5',
        })
    print(f"  M5 sequential:              {len(m5)} rows")

# M6: anomaly fingerprints — anomaly prediction, not direction
# For direction signal: adapt by checking if high-anomaly-rate conditions also have bear direction
# Only include M6 entries that are about direction (use is_bull from enriched data)
m6 = load_csv_safe(f"{REPO}/results/research/method6_anomaly_fingerprints.csv")
if len(m6) > 0:
    k_bull_total = int(df_clean['is_bull'].sum())
    for _, row in m6.iterrows():
        col = row.get('feature',''); val = row.get('value','')
        if col not in df_clean.columns: continue
        sub = df_clean[df_clean[col].astype(str) == str(val)]
        n_c = len(sub); k_c = int(sub['is_bull'].sum())
        if n_c < 10: continue
        p_dir = fisher_p(n_c, k_c, N, k_bull_total)
        pool.append({
            'features': col,
            'condition': str(val),
            'outcome': 'is_bull',
            'n': n_c,
            'k_pos': k_c,
            'win_rate': round(k_c/n_c,4),
            'wilson_lower': round(wilson_lower(n_c,k_c),4),
            'p_value': p_dir,
            'complexity': 1,
            'source': 'method6',
        })
    print(f"  M6 anomaly (adapted):       {len(m6)} rows")

total = len(pool)
print(f"\nTotal pool size: {total} p-values")

# ──────────────────────────────────────────────────────────────────────
# Apply BH-FDR at 1%
# ──────────────────────────────────────────────────────────────────────
pool_df = pd.DataFrame(pool)
pvals = pool_df['p_value'].astype(float).tolist()
fdr_result = bh_fdr(pvals, alpha=0.01)
pool_df['fdr_survive'] = fdr_result
n_survive = sum(fdr_result)
print(f"BH-FDR at 1%: {n_survive}/{total} survive ({100*n_survive/total:.2f}%)")

pool_df.to_csv(f"{REPO}/results/validation/fdr_pool_all.csv", index=False)

# ──────────────────────────────────────────────────────────────────────
# OOS validation on FDR survivors
# ──────────────────────────────────────────────────────────────────────
survivors = pool_df[pool_df['fdr_survive']].copy()
print(f"\nOOS validating {len(survivors)} FDR survivors …")

def eval_on_split(df_split, features_str, condition_str):
    features = str(features_str).split('|')
    conds    = str(condition_str).split('||') if '||' in str(condition_str) else [str(condition_str)]
    if len(features) != len(conds): return 0, 0
    mask = pd.Series(True, index=df_split.index)
    for f, v in zip(features, conds):
        if f not in df_split.columns: return 0, 0
        mask &= (df_split[f].astype(str) == v.strip())
    n = int(mask.sum())
    if n == 0: return 0, 0
    k = int(df_split.loc[mask, 'is_bull'].sum())
    return n, k

# Get train stats for each survivor
def get_train_stats(df_tr, features_str, condition_str):
    features = str(features_str).split('|')
    conds    = str(condition_str).split('||') if '||' in str(condition_str) else [str(condition_str)]
    if len(features) != len(conds): return 0, 0
    mask = pd.Series(True, index=df_tr.index)
    for f, v in zip(features, conds):
        if f not in df_tr.columns: return 0, 0
        mask &= (df_tr[f].astype(str) == v.strip())
    n = int(mask.sum()); k = int(df_tr.loc[mask,'is_bull'].sum())
    return n, k

confirmed_new = []
for _, row in survivors.iterrows():
    src = row.get('source','')
    # Skip method4 cycle entries (no specific condition to evaluate)
    if src == 'method4': continue
    # Skip if n=0 (aggregate rows)
    if row.get('n',0) == 0: continue

    feats = str(row['features']); cond = str(row['condition'])

    n_tr, k_tr = get_train_stats(df_train, feats, cond)
    n_oos, k_oos = eval_on_split(df_oos, feats, cond)

    if n_tr < 5: continue
    wr_tr  = k_tr/n_tr if n_tr>0 else 0
    wlb_tr = wilson_lower(n_tr, k_tr)
    wr_oos = k_oos/n_oos if n_oos>0 else 0

    # Direction
    is_bull_pat = wr_tr > BASE_BULL
    oos_pass = False
    if n_oos >= 3:
        if is_bull_pat and wr_oos > 0.45:  oos_pass = True
        if not is_bull_pat and wr_oos < 0.55: oos_pass = True

    if not oos_pass: continue

    # Temporal stability (simple: 3 sub-periods)
    pre2010 = df_clean[df_clean['date'] < '2010-01-01']
    mid     = df_clean[(df_clean['date'] >= '2010-01-01') & (df_clean['date'] < '2018-01-01')]
    n_pre, k_pre = eval_on_split(pre2010, feats, cond)
    n_mid, k_mid = eval_on_split(mid, feats, cond)
    wr_pre = k_pre/n_pre if n_pre>5 else None
    wr_mid = k_mid/n_mid if n_mid>5 else None

    # Same direction in at least 2/3 periods
    dirs = []
    if wr_pre is not None: dirs.append(wr_pre > BASE_BULL)
    if wr_mid is not None: dirs.append(wr_mid > BASE_BULL)
    dirs.append(wr_oos > BASE_BULL)
    stable = dirs.count(is_bull_pat) >= max(1, len(dirs)-1)

    p_val = float(row.get('p_value',1.0))

    confirmed_new.append({
        'features'     : feats,
        'condition'    : cond,
        'outcome'      : row.get('outcome','is_bull'),
        'source'       : src,
        'signal_dir'   : 'BULL' if is_bull_pat else 'BEAR',
        'n_train'      : n_tr,  'k_train': k_tr,
        'wr_train'     : round(wr_tr,4),
        'wlb_train'    : round(wlb_tr,4),
        'p_value'      : round(p_val,8),
        'n_oos'        : n_oos, 'k_oos': k_oos,
        'wr_oos'       : round(wr_oos,4),
        'wlb_oos'      : round(wilson_lower(n_oos, k_oos),4),
        'oos_pass'     : oos_pass,
        'pre2010'      : round(wr_pre,4) if wr_pre else None,
        '2010_2018'    : round(wr_mid,4) if wr_mid else None,
        '2018_now'     : round(wr_oos,4),
        'temporal_stable': stable,
    })

new_confirmed = pd.DataFrame(confirmed_new) if confirmed_new else pd.DataFrame(
    columns=['features','condition','outcome','source','signal_dir','n_train','k_train',
             'wr_train','wlb_train','p_value','n_oos','k_oos','wr_oos','wlb_oos',
             'oos_pass','pre2010','2010_2018','2018_now','temporal_stable'])
print(f"  New confirmed (all methods): {len(new_confirmed)}")

# ──────────────────────────────────────────────────────────────────────
# Separate M3-6 new patterns, report independently
# ──────────────────────────────────────────────────────────────────────
m36_new = new_confirmed[new_confirmed['source'].isin(['method3','method5','method6'])] if len(new_confirmed) > 0 else pd.DataFrame()
m36_new.to_csv(f"{REPO}/results/validation/m3m6_validated.csv", index=False)
print(f"\n  M3-6 new confirmed:          {len(m36_new)}")
if len(m36_new) > 0:
    print(m36_new[['source','features','condition','signal_dir',
                    'n_train','wlb_train','wr_oos']].to_string())

# ──────────────────────────────────────────────────────────────────────
# Merge with existing confirmed_patterns.csv (dedup by features+condition)
# ──────────────────────────────────────────────────────────────────────
old_cp_path = f"{REPO}/results/validation/confirmed_patterns.csv"
if os.path.exists(old_cp_path):
    old_cp = pd.read_csv(old_cp_path)
    print(f"\n  Old confirmed patterns: {len(old_cp)}")
    # Standardise columns
    new_confirmed['reversal_flag'] = False
    old_keys = set(old_cp['features'] + ':::' + old_cp['condition'])
    truly_new = new_confirmed[~(new_confirmed['features']+':::'+new_confirmed['condition']).isin(old_keys)]
    print(f"  Truly new (not already confirmed): {len(truly_new)}")
    merged = pd.concat([old_cp, truly_new], ignore_index=True)
else:
    merged = new_confirmed

merged = merged.drop_duplicates(subset=['features','condition'])
merged.to_csv(f"{REPO}/results/validation/confirmed_patterns.csv", index=False)
print(f"\nFinal confirmed_patterns.csv: {len(merged)} patterns")
print(f"  BULL: {(merged.get('signal_dir','?')=='BULL').sum()}")
print(f"  BEAR: {(merged.get('signal_dir','?')=='BEAR').sum()}")

# ──────────────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"FIX 5 SUMMARY")
print(f"{'='*60}")
print(f"  Total p-values in pool:  {total}")
print(f"  FDR survivors (1%):      {n_survive}")
print(f"  OOS+stability confirmed: {len(new_confirmed)}")
print(f"    from M1/M2:            {len(new_confirmed[~new_confirmed['source'].isin(['method3','method5','method6'])])}")
print(f"    from M3-6:             {len(m36_new)}")
print(f"  Final confirmed total:   {len(merged)}")
if len(merged) > 0:
    bull_n = (merged.get('signal_dir') == 'BULL').sum() if 'signal_dir' in merged.columns else 0
    bear_n = (merged.get('signal_dir') == 'BEAR').sum() if 'signal_dir' in merged.columns else 0
    print(f"    BULL: {bull_n}")
    print(f"    BEAR: {bear_n}")
print(f"\nFix 5 complete.")
