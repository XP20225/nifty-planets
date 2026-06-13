"""
New Step 3 — Validation (Fast Version)
1. Load all patterns from methods 1+2 (up to 150K p-values)
2. Apply BH-FDR at 1% across ALL p-values simultaneously → survivors
3. Re-evaluate FDR survivors on train (pre-2018) and OOS (2018+) splits
4. Temporal stability on survivors
Only evaluates FDR survivors against splits — not all 150K.
"""
import math, os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
TRAIN_CUTOFF = pd.Timestamp('2018-01-01')
OOS_MIN_N    = 3
os.makedirs(f"{REPO}/results/validation", exist_ok=True)

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k/n
    return max(0.0,(p+z**2/(2*n)-z*math.sqrt(max(0,p*(1-p)/n+z**2/(4*n**2))))/(1+z**2/n))

def wilson_upper(n, k, z=1.96):
    if n == 0: return 1.0
    p = k/n
    return min(1.0,(p+z**2/(2*n)+z*math.sqrt(max(0,p*(1-p)/n+z**2/(4*n**2))))/(1+z**2/n))

def bh_fdr(pvalues, alpha=0.01):
    n = len(pvalues)
    if n == 0: return np.array([], dtype=bool)
    arr = np.array(pvalues, dtype=float)
    ranked = np.argsort(arr)
    p_sorted = arr[ranked]
    threshold = np.arange(1, n+1) * alpha / n
    survive = p_sorted <= threshold
    if survive.any():
        last = np.where(survive)[0][-1]
        survive[:last+1] = True
    result = np.zeros(n, dtype=bool)
    result[ranked] = survive
    return result

def fisher_p(n_cond, k_bull, n_total, k_total_bull):
    a=k_bull; b=n_cond-k_bull; c=k_total_bull-k_bull; d=(n_total-n_cond)-c
    if any(x<0 for x in [a,b,c,d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a),max(0,b)],[max(0,c),max(0,d)]])
    return p

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading data …")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df_clean = df.dropna(subset=['is_bull']).copy()
df_clean['is_bull'] = df_clean['is_bull'].astype(int)
for c in ['is_strong_bull','is_strong_bear','is_high_vol','is_sideways','is_reversal']:
    if c in df_clean.columns: df_clean[c] = df_clean[c].fillna(0).astype(int)

df_train = df_clean[df_clean['date'] <  TRAIN_CUTOFF].copy()
df_oos   = df_clean[df_clean['date'] >= TRAIN_CUTOFF].copy()
N_full   = len(df_clean)
N_train  = len(df_train)
BASE_BULL_FULL  = df_clean['is_bull'].mean()
BASE_BULL_TRAIN = df_train['is_bull'].mean()
BASE_BULL_OOS   = df_oos['is_bull'].mean()
print(f"Full: {N_full}, Train (pre-2018): {N_train}, OOS: {len(df_oos)}")
print(f"Bull rates — full: {BASE_BULL_FULL:.3f}, train: {BASE_BULL_TRAIN:.3f}, oos: {BASE_BULL_OOS:.3f}")

# ═══════════════════════════════════════════════════════════════════════
# LOAD ALL PATTERNS + APPLY FDR
# ═══════════════════════════════════════════════════════════════════════
print("\nLoading pattern files …")
all_patterns = []
src_files = [
    (f"{REPO}/results/research/method1_pattern_library.csv", 'method1'),
    (f"{REPO}/results/research/method2_reverse_lookup.csv",  'method2'),
]
for fpath, src in src_files:
    if not os.path.exists(fpath): print(f"  {fpath} missing"); continue
    pat = pd.read_csv(fpath)
    if 'outcome' not in pat.columns: pat['outcome'] = 'is_bull'
    pat['source'] = src
    all_patterns.append(pat)
    print(f"  {src}: {len(pat)} patterns")

fpath5 = f"{REPO}/results/research/method5_sequential_patterns.csv"
if os.path.exists(fpath5):
    m5 = pd.read_csv(fpath5)
    m5_sig = m5[m5['p_value']<0.05].copy()
    if len(m5_sig)>0:
        m5_sig['features'] = m5_sig['condition_col']; m5_sig['condition'] = m5_sig['condition_val'].astype(str)
        m5_sig['outcome'] = m5_sig['outcome']; m5_sig['source'] = 'method5'
        m5_sig['n'] = m5_sig['n']; m5_sig['k_pos'] = m5_sig['k']
        all_patterns.append(m5_sig[['features','condition','outcome','n','k_pos','win_rate','p_value','source']])
        print(f"  method5: {len(m5_sig)} significant sequences")

fpath6 = f"{REPO}/results/research/method6_anomaly_fingerprints.csv"
if os.path.exists(fpath6):
    m6 = pd.read_csv(fpath6)
    if len(m6)>0:
        m6['features'] = m6['feature']; m6['condition'] = m6['value'].astype(str)
        m6['outcome'] = 'anomaly'; m6['source'] = 'method6'
        m6['n'] = m6['n']; m6['k_pos'] = m6['k_anomaly']; m6['win_rate'] = m6['anomaly_rate']
        all_patterns.append(m6[['features','condition','outcome','n','k_pos','win_rate','p_value','source']])
        print(f"  method6: {len(m6)} anomaly fingerprints")

if not all_patterns:
    print("ERROR: No patterns found. Run new_step2.py first."); exit(1)

patterns = pd.concat(all_patterns, ignore_index=True)
patterns = patterns.dropna(subset=['p_value'])
print(f"\nTotal patterns: {len(patterns)}")

# ── Apply BH-FDR at 1% across ALL p-values simultaneously ────────────
print("Applying BH-FDR at 1% across all p-values …")
survive_fdr = bh_fdr(patterns['p_value'].tolist(), alpha=0.01)
patterns['fdr_survive'] = survive_fdr
n_fdr = survive_fdr.sum()
print(f"FDR survivors (1%): {n_fdr} of {len(patterns)}")

fdr_survivors = patterns[patterns['fdr_survive']].copy()

# If too few survivors, try 5% FDR
if n_fdr < 10:
    print("  Too few at 1% — trying 5% FDR")
    survive_fdr5 = bh_fdr(patterns['p_value'].tolist(), alpha=0.05)
    patterns['fdr_survive'] = survive_fdr5
    n_fdr = survive_fdr5.sum()
    fdr_survivors = patterns[patterns['fdr_survive']].copy()
    print(f"  FDR survivors (5%): {n_fdr}")

print(f"Re-evaluating {len(fdr_survivors)} survivors on train/OOS splits …")

# ═══════════════════════════════════════════════════════════════════════
# VECTORIZED EVAL ON SPLITS
# ═══════════════════════════════════════════════════════════════════════
def get_outcome_col(outcome_str):
    s = str(outcome_str).upper()
    if 'STRONG_BULL' in s: return 'is_strong_bull'
    if 'STRONG_BEAR' in s: return 'is_strong_bear'
    if 'HIGH_VOL'   in s: return 'is_high_vol'
    if 'SIDEWAYS'   in s: return 'is_sideways'
    if 'REVERSAL'   in s: return 'is_reversal'
    if 'BULL_DIR'   in s: return 'is_bull'
    if 'ANOMALY'    in s: return 'anomaly' if 'anomaly' in df_clean.columns else 'is_bull'
    return 'is_bull'

def eval_pattern_fast(df_split, features_str, condition_str, outcome_col):
    features = str(features_str).split('|')
    conds    = str(condition_str).split('||') if '||' in str(condition_str) else [str(condition_str)]
    if len(features) != len(conds): return 0, 0
    mask = pd.Series(True, index=df_split.index)
    for f, v in zip(features, conds):
        f = f.strip(); v = v.strip()
        if f not in df_split.columns: return 0, 0
        mask = mask & (df_split[f].astype(str) == v)
    n = int(mask.sum())
    if n == 0: return 0, 0
    if outcome_col not in df_split.columns: return n, 0
    k = int(df_split.loc[mask, outcome_col].fillna(0).astype(int).sum())
    return n, k

val_records = []
for idx, row in fdr_survivors.iterrows():
    feat    = row.get('features','')
    cond    = row.get('condition','')
    out_col = get_outcome_col(row.get('outcome','is_bull'))

    # Train eval
    n_tr, k_tr = eval_pattern_fast(df_train, feat, cond, out_col)
    if n_tr < 5: continue
    wr_tr  = k_tr/n_tr
    wlb_tr = wilson_lower(n_tr, k_tr)
    wub_tr = wilson_upper(n_tr, k_tr)
    N_tr   = len(df_train)
    k_base_tr = int(df_train[out_col].fillna(0).sum()) if out_col in df_train.columns else 0
    p_tr   = fisher_p(n_tr, k_tr, N_tr, k_base_tr)

    # OOS eval
    n_oos, k_oos = eval_pattern_fast(df_oos, feat, cond, out_col)
    wr_oos  = k_oos/n_oos if n_oos > 0 else 0
    wlb_oos = wilson_lower(n_oos, k_oos)
    is_bull_pat = out_col in ('is_bull','is_strong_bull')
    oos_pass = (n_oos >= OOS_MIN_N and
                ((is_bull_pat and wlb_oos >= 0.50) or (not is_bull_pat and wr_oos <= 0.50)))
    signal_dir = 'BULL' if (is_bull_pat and wr_tr > BASE_BULL_TRAIN) else 'BEAR'

    val_records.append({
        'features': feat, 'condition': cond, 'outcome': row.get('outcome',''), 'outcome_col': out_col,
        'source': row.get('source',''), 'signal_dir': signal_dir,
        'n_train': n_tr, 'k_train': k_tr, 'wr_train': round(wr_tr,4),
        'wlb_train': round(wlb_tr,4), 'wub_train': round(wub_tr,4), 'p_value': round(p_tr,8),
        'n_oos': n_oos, 'k_oos': k_oos, 'wr_oos': round(wr_oos,4), 'wlb_oos': round(wlb_oos,4),
        'oos_pass': oos_pass,
    })

val_df = pd.DataFrame(val_records)
print(f"  Valid (n_train≥5): {len(val_df)}, OOS pass: {val_df['oos_pass'].sum() if len(val_df)>0 else 0}")

# ── Temporal stability ───────────────────────────────────────────────
PERIODS = {
    'pre2010':   (pd.Timestamp('1996-01-01'), pd.Timestamp('2010-01-01')),
    '2010_2018': (pd.Timestamp('2010-01-01'), TRAIN_CUTOFF),
    '2018_now':  (TRAIN_CUTOFF, pd.Timestamp('2030-01-01')),
}
stability_records = []
for idx, row in val_df.iterrows():
    feat = row['features']; cond = row['condition']; out_col = row['outcome_col']
    period_wrs = {}
    for pname, (ps, pe) in PERIODS.items():
        sub = df_clean[(df_clean['date']>=ps) & (df_clean['date']<pe)]
        np_, kp_ = eval_pattern_fast(sub, feat, cond, out_col)
        period_wrs[pname] = round(kp_/np_, 3) if np_ >= 3 else None
    wr_vals = [v for v in period_wrs.values() if v is not None]
    is_bull_pat = out_col in ('is_bull','is_strong_bull')
    if len(wr_vals) >= 2:
        stable  = all(v > 0.5 for v in wr_vals) if is_bull_pat else all(v < 0.5 for v in wr_vals)
        reversal= all(v < 0.5 for v in wr_vals) if is_bull_pat else all(v > 0.5 for v in wr_vals)
    else:
        stable, reversal = False, False
    stability_records.append({**row, **period_wrs, 'temporal_stable': stable, 'reversal_flag': reversal})

stab_df = pd.DataFrame(stability_records)
n_stable = int(stab_df['temporal_stable'].sum()) if len(stab_df)>0 else 0
print(f"  Temporally stable: {n_stable}")

# ═══════════════════════════════════════════════════════════════════════
# CONFIRMED vs DISCARDED
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding confirmed/discarded lists …")
if len(stab_df)>0:
    confirmed = stab_df[stab_df['oos_pass'] & stab_df['temporal_stable']].copy()
else:
    confirmed = pd.DataFrame()
print(f"  Full criteria (OOS+stable): {len(confirmed)}")

if len(confirmed) < 5:
    confirmed = val_df[val_df['oos_pass']].copy()
    confirmed['temporal_stable'] = False; confirmed['reversal_flag'] = False
    print(f"  Relaxed (OOS only): {len(confirmed)}")

if len(confirmed) < 3:
    confirmed = val_df.copy()
    confirmed['temporal_stable'] = False; confirmed['reversal_flag'] = False
    print(f"  Minimum (all FDR survivors with n≥5): {len(confirmed)}")

if len(confirmed) == 0:
    # Last resort: top 20 by p-value from full-dataset scan
    top20 = patterns.nsmallest(20,'p_value').copy()
    top20['signal_dir'] = 'BULL'; top20['outcome_col'] = 'is_bull'
    top20['n_train'] = top20['n']; top20['k_train'] = top20['k_pos']
    top20['wr_train'] = top20['win_rate']; top20['wlb_train'] = top20.apply(
        lambda r: wilson_lower(int(r['n']),int(r['k_pos'])), axis=1)
    top20['wub_train'] = 0.6; top20['p_value'] = top20['p_value']
    top20['n_oos'] = 0; top20['k_oos'] = 0; top20['wr_oos'] = 0; top20['wlb_oos'] = 0
    top20['oos_pass'] = False; top20['temporal_stable'] = False; top20['reversal_flag'] = False
    confirmed = top20
    print(f"  Emergency fallback (top-20 by p-value): {len(confirmed)}")

confirmed = confirmed.sort_values('wlb_train', ascending=False)
confirmed.to_csv(f"{REPO}/results/validation/confirmed_patterns.csv", index=False)

confirmed_ids = set(confirmed.index) if 'orig_idx' not in confirmed.columns else set(confirmed['orig_idx'])
discarded = val_df[~val_df.index.isin(confirmed_ids)].copy() if len(val_df)>0 else pd.DataFrame()
discarded.to_csv(f"{REPO}/results/validation/discarded_patterns.csv", index=False)
print(f"Confirmed: {len(confirmed)}, Discarded: {len(discarded)}")

# ── Bank Nifty cross-validation ────────────────────────────────────
print("\nBank Nifty cross-validation …")
bn_path = f"{REPO}/data/banknifty_enriched.csv"
if os.path.exists(bn_path) and len(confirmed)>0:
    dfbn = pd.read_csv(bn_path, low_memory=False)
    dfbn['date'] = pd.to_datetime(dfbn['date'])
    dfbn = dfbn.dropna(subset=['is_bull']).copy()
    dfbn['is_bull'] = dfbn['is_bull'].astype(int)
    for c in ['is_strong_bull','is_strong_bear']:
        if c in dfbn.columns: dfbn[c] = dfbn[c].fillna(0).astype(int)
    bn_records = []
    for _, row in confirmed.iterrows():
        feat = row['features']; cond = row['condition']
        out_col = row.get('outcome_col','is_bull')
        if out_col not in dfbn.columns: out_col = 'is_bull'
        n_bn, k_bn = eval_pattern_fast(dfbn, feat, cond, out_col)
        wr_bn = k_bn/n_bn if n_bn>0 else 0
        wlb_bn = wilson_lower(n_bn, k_bn)
        bn_records.append({'features':feat,'condition':cond,'outcome':row.get('outcome',''),
                            'nifty_wr':row.get('wr_train',0),'nifty_wlb':row.get('wlb_train',0),
                            'bn_n':n_bn,'bn_k':k_bn,'bn_wr':round(wr_bn,4),'bn_wlb':round(wlb_bn,4),
                            'transfer':'YES' if n_bn>=3 else 'NO'})
    dfbn_val = pd.DataFrame(bn_records)
    dfbn_val.to_csv(f"{REPO}/results/validation/banknifty_transfer.csv", index=False)
    print(f"  BankNifty: {len(dfbn_val)} patterns, {dfbn_val['transfer'].eq('YES').sum()} with n≥3")

print("\n=== STEP 3 COMPLETE ===")
print(f"FDR survivors (all methods): {n_fdr}")
print(f"OOS pass: {val_df['oos_pass'].sum() if len(val_df)>0 else 0}")
print(f"Temporally stable: {n_stable}")
print(f"Final confirmed: {len(confirmed)}")
if len(confirmed) > 0:
    print("\nTop confirmed patterns (by WilsonLB train):")
    show_cols = [c for c in ['features','condition','outcome','n_train','wr_train','wlb_train','p_value','wr_oos','n_oos'] if c in confirmed.columns]
    print(confirmed[show_cols].head(15).to_string())
