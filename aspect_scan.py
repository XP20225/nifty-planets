"""
Targeted k=1,2 scan for inter-planetary aspect features only.
The full M2 scan (330 sig_cols, k=3) is computationally infeasible.
This script scans ONLY the 350+ aspect columns added by fix6_aspects.py
at k=1 and k=2, finding simple generalizable aspect patterns.
Runtime: ~2-5 minutes.
"""
import math, os, time, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"

MIN_N = 10
SIG_THRESH = 0.05
K2_THRESH  = 0.15   # column eligibility for k=2

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0, (p + z**2/(2*n) - z*math.sqrt(max(0, p*(1-p)/n + z**2/(4*n**2)))) /
               (1 + z**2/n))

def fisher_p(n_c, k_c, N, k_t):
    a = k_c; b = n_c - k_c; c = k_t - k_c; d = (N - n_c) - (k_t - k_c)
    if any(x < 0 for x in [a,b,c,d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a),max(0,b)],[max(0,c),max(0,d)]])
    return p

print("Loading nifty_enriched.csv …")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

OUTCOME_COLS = ['is_strong_bull','is_strong_bear','is_sideways','is_high_vol','is_bull',
                'is_continuation','is_reversal']
df_clean = df.dropna(subset=[c for c in OUTCOME_COLS if c in df.columns]).copy()
for c in OUTCOME_COLS:
    if c in df_clean.columns:
        df_clean[c] = df_clean[c].astype(int)

N = len(df_clean)
print(f"  Clean rows: {N}, columns: {len(df_clean.columns)}")

# ── Discover aspect columns ─────────────────────────────────────────────────
ASP_PREFIXES = (
    'asp3_','asp4_','asp5_','asp7_','asp8_','asp9_','asp10_',
    'asp_',          # lord-domain chains: asp_Ju_dom_Sa
    'deg_',          # degree-based: deg_conj_Ju_Sa
    'ex_',           # exact aspects ≤3°: ex_conj_Ju_Sa
    'n_asp_',        # count under slow planet
    'mut_asp_',      # mutual aspects
    'any_',          # any-aspect aggregates
    'natal_asp_',    # natal Moon aspects
)
# These overlap with non-aspect cols, so exclude known non-aspect
EXCLUDE_EXACT = {'any_sig_feat', 'deg_in_sign_Su', 'deg_in_sign_Mo',  # just in case
                 'aspected_sign_Su_7'}  # already covered via asp7 cols

asp_cols_raw = []
for col in df_clean.columns:
    if col.startswith(ASP_PREFIXES) and col not in EXCLUDE_EXACT:
        asp_cols_raw.append(col)
# Also grab aspected_sign_* and aspected_lord_* (categorical)
for col in df_clean.columns:
    if col.startswith('aspected_sign_') or col.startswith('aspected_lord_'):
        asp_cols_raw.append(col)

asp_cols_raw = list(dict.fromkeys(asp_cols_raw))  # deduplicate, preserve order
print(f"  Raw aspect columns found: {len(asp_cols_raw)}")

# Split into binary (0/1) and categorical (multi-value)
bin_cols = []
cat_cols = []
for col in asp_cols_raw:
    if col not in df_clean.columns: continue
    s = df_clean[col].dropna()
    vals = set(s.unique())
    if vals <= {0, 1, 0.0, 1.0}:
        bin_cols.append(col)
    elif s.dtype == object or len(vals) <= 15:
        cat_cols.append(col)
    # skip high-cardinality numeric (raw degrees)

print(f"  Binary: {len(bin_cols)},  Categorical: {len(cat_cols)}")

# For binary cols, only test value=1 at k=1 (value=0 is complement)
# For cat cols, test all values with n >= MIN_N

def scan_k1(df, outcome_col):
    N = len(df); k_t = int(df[outcome_col].sum())
    if k_t == 0 or k_t == N: return [], []
    results = []
    sig_cols = []   # columns with p < K2_THRESH for any value

    # Binary: only test =1
    for col in bin_cols:
        if col not in df.columns: continue
        grp1 = df[df[col] == 1]
        n_c = len(grp1); k_c = int(grp1[outcome_col].sum())
        if n_c < MIN_N: continue
        p = fisher_p(n_c, k_c, N, k_t)
        if p < SIG_THRESH:
            results.append({'features': col, 'condition': '1',
                            'n': n_c, 'k_pos': k_c,
                            'win_rate': round(k_c/n_c,4),
                            'wilson_lower': round(wilson_lower(n_c, k_c),4),
                            'p_value': round(p,8), 'complexity': 1})
        if p < K2_THRESH and col not in sig_cols:
            sig_cols.append(col)

    # Categorical: test all values
    for col in cat_cols:
        if col not in df.columns: continue
        grpd = df.groupby(df[col].astype(str))[outcome_col]
        best_p = 1.0
        for val, grp in grpd:
            n_c = len(grp); k_c = int(grp.sum())
            if n_c < MIN_N: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < SIG_THRESH:
                results.append({'features': col, 'condition': val,
                                'n': n_c, 'k_pos': k_c,
                                'win_rate': round(k_c/n_c,4),
                                'wilson_lower': round(wilson_lower(n_c, k_c),4),
                                'p_value': round(p,8), 'complexity': 1})
            best_p = min(best_p, p)
        if best_p < K2_THRESH and col not in sig_cols:
            sig_cols.append(col)

    return results, sig_cols


def scan_k2(df, outcome_col, sig_cols):
    N = len(df); k_t = int(df[outcome_col].sum())
    if k_t == 0 or k_t == N: return []
    results = []
    for c1, c2 in itertools.combinations(sig_cols, 2):
        key = df[c1].astype(str) + '||' + df[c2].astype(str)
        for kv, idx in df.groupby(key).groups.items():
            n_c = len(idx); k_c = int(df.loc[idx, outcome_col].sum())
            if n_c < MIN_N: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < SIG_THRESH:
                results.append({'features': f'{c1}|{c2}', 'condition': kv,
                                'n': n_c, 'k_pos': k_c,
                                'win_rate': round(k_c/n_c,4),
                                'wilson_lower': round(wilson_lower(n_c, k_c),4),
                                'p_value': round(p,8), 'complexity': 2})
    return results


all_results = []
OUTCOMES = [
    ('is_strong_bull', 'STRONG_BULL'),
    ('is_strong_bear', 'STRONG_BEAR'),
    ('is_sideways',    'SIDEWAYS'),
    ('is_high_vol',    'HIGH_VOL'),
    ('is_bull',        'BULL_DIR'),
]

t0 = time.time()
for out_col, out_label in OUTCOMES:
    if out_col not in df_clean.columns: continue
    t1 = time.time()
    print(f"\n[{out_label}] N={N}, pos={int(df_clean[out_col].sum())}")

    k1_res, sig_cols = scan_k1(df_clean, out_col)
    print(f"  k=1: {len(k1_res)} patterns, {len(sig_cols)} sig cols for k=2")

    k2_res = scan_k2(df_clean, out_col, sig_cols)
    print(f"  k=2: {len(k2_res)} patterns  [{time.time()-t1:.1f}s]")

    for r in k1_res + k2_res:
        r['outcome'] = out_col
        r['outcome_label'] = out_label
        r['source'] = 'method1_asp'
    all_results.extend(k1_res + k2_res)

asp_df = pd.DataFrame(all_results)
if len(asp_df) > 0:
    asp_df = asp_df.drop_duplicates(subset=['features','condition','outcome'])
    asp_df = asp_df.sort_values('wilson_lower', ascending=False)

out_path = f"{REPO}/results/research/method1_asp_scan.csv"
asp_df.to_csv(out_path, index=False)
print(f"\n{'='*60}")
print(f"Aspect scan complete: {len(asp_df)} patterns in {time.time()-t0:.1f}s")
print(f"Saved: {out_path}")
if len(asp_df) > 0:
    print(f"\nTop 10 by Wilson lower bound:")
    print(asp_df[['features','condition','outcome','n','win_rate','wilson_lower','p_value']].head(10).to_string())
