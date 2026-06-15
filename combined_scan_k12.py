"""
Combined k=1,2 scan on ALL 668 astrological features (dignities, dashas,
nakshatras, panchanga, AND the 398 inter-planetary aspects from fix6).

Aspects are NOT treated separately. They are just more ingredients.
Patterns like "Saturn aspects Venus's sign AND Moon in enemy dignity"
emerge naturally from the same search that finds all other patterns.

k=3 is skipped — computationally infeasible with 300+ sig_cols.
k=1,2 with ~330 sig_cols takes ~30-60 minutes.

Output: results/research/method1_combined_k12.csv
"""
import math, os, time, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
os.makedirs(f"{REPO}/results/research", exist_ok=True)

MIN_N       = 10
SIG_THRESH  = 0.05
K2_ELIGIBLE = 0.10   # column must have p < 0.10 somewhere to qualify for k=2

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0, (p + z**2/(2*n) - z*math.sqrt(max(0, p*(1-p)/n + z**2/(4*n**2)))) /
               (1 + z**2/n))

def fisher_p(n_c, k_c, N, k_t):
    a = k_c; b = n_c - k_c; c = k_t - k_c; d = (N - n_c) - (k_t - k_c)
    if any(x < 0 for x in [a, b, c, d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a), max(0,b)], [max(0,c), max(0,d)]])
    return p

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading nifty_enriched.csv …")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

OUTCOME_COLS = ['is_strong_bull', 'is_strong_bear', 'is_sideways',
                'is_high_vol', 'is_bull']
df_clean = df.dropna(subset=[c for c in OUTCOME_COLS if c in df.columns]).copy()
for c in OUTCOME_COLS:
    if c in df_clean.columns:
        df_clean[c] = df_clean[c].astype(int)

N = len(df_clean)
print(f"  Rows: {N}, columns: {len(df_clean.columns)}")

# ── Auto-discover ALL astrological features (same logic as fix2) ─────────────
EXCLUDE_PREFIXES = ('open','high','low','close','volume','fwd_','ret_',
                    'is_','log_','range_','atr','date','sid_','spd_',
                    'sign_','nak_Su','nak_Mo','nak_Ma','nak_Me','nak_Ju',
                    'nak_Ve','nak_Sa','nak_Ra','nak_Ke','prior_ret')
EXCLUDE_EXACT = {'index','oc','signal','outcome_3d'}

CAT_FEATURES = []
BIN_FEATURES = []
for col in df_clean.columns:
    if col.startswith(EXCLUDE_PREFIXES) or col in EXCLUDE_EXACT:
        continue
    s = df_clean[col]
    if s.dtype == object:
        CAT_FEATURES.append(col)
    else:
        vals = set(s.dropna().unique())
        if vals <= {0, 1, 0.0, 1.0}:
            BIN_FEATURES.append(col)
        elif len(vals) <= 15:
            CAT_FEATURES.append(col)

# Convert binary to string columns
for col in BIN_FEATURES:
    sname = col + '_s'
    if sname not in df_clean.columns:
        df_clean[sname] = col + '=' + df_clean[col].astype(str)

BIN_STR = [c + '_s' for c in BIN_FEATURES if (c + '_s') in df_clean.columns]
ALL_FEAT = CAT_FEATURES + BIN_STR

print(f"  Feature pool: {len(ALL_FEAT)} ({len(CAT_FEATURES)} cat + {len(BIN_FEATURES)} bin)")
asp_feat = [f for f in ALL_FEAT if any(f.startswith(p) for p in
            ('asp3_','asp4_','asp5_','asp7_','asp8_','asp9_','asp10_','asp_',
             'deg_','ex_','n_asp','mut_','any_','natal_','aspected_'))]
print(f"  Of which aspect features: {len(asp_feat)}")

# ── k=1,2 scan ───────────────────────────────────────────────────────────────
OUTCOMES = [
    ('is_strong_bull', 'STRONG_BULL'),
    ('is_strong_bear', 'STRONG_BEAR'),
    ('is_sideways',    'SIDEWAYS'),
    ('is_high_vol',    'HIGH_VOL'),
    ('is_bull',        'BULL_DIR'),
]

all_results = []
t0 = time.time()

for out_col, out_label in OUTCOMES:
    if out_col not in df_clean.columns:
        continue
    t1 = time.time()
    N_oc = len(df_clean)
    k_t  = int(df_clean[out_col].sum())
    print(f"\n[{out_label}] pos={k_t}/{N_oc} ({100*k_t/N_oc:.1f}%)")

    # k=1 scan
    k1_results = []
    sig_cols   = []   # eligible for k=2 (p < K2_ELIGIBLE somewhere)

    present = [c for c in ALL_FEAT if c in df_clean.columns]
    for col in present:
        grpd    = df_clean.groupby(df_clean[col].astype(str))[out_col]
        best_p  = 1.0
        for val, grp in grpd:
            n_c = len(grp); k_c = int(grp.sum())
            if n_c < MIN_N: continue
            p = fisher_p(n_c, k_c, N_oc, k_t)
            if p < SIG_THRESH:
                k1_results.append({
                    'features': col, 'condition': val, 'outcome': out_col,
                    'outcome_label': out_label,
                    'n': n_c, 'k_pos': k_c,
                    'win_rate':     round(k_c / n_c, 4),
                    'wilson_lower': round(wilson_lower(n_c, k_c), 4),
                    'p_value':      round(p, 8),
                    'complexity':   1,
                    'source':       'method1_combined',
                })
            best_p = min(best_p, p)
        if best_p < K2_ELIGIBLE:
            sig_cols.append(col)

    print(f"  k=1: {len(k1_results)} patterns, {len(sig_cols)} sig cols for k=2")
    all_results.extend(k1_results)

    # k=2 scan — combinations of all sig_cols (mix of dignities + aspects + dashas)
    k2_count = 0
    n_combos  = len(sig_cols) * (len(sig_cols) - 1) // 2
    print(f"  k=2: {n_combos} combinations …", end='', flush=True)

    for c1, c2 in itertools.combinations(sig_cols, 2):
        key = df_clean[c1].astype(str) + '||' + df_clean[c2].astype(str)
        for kv, idx in df_clean.groupby(key).groups.items():
            n_c = len(idx); k_c = int(df_clean.loc[idx, out_col].sum())
            if n_c < MIN_N: continue
            p = fisher_p(n_c, k_c, N_oc, k_t)
            if p < SIG_THRESH:
                f1, f2 = c1, c2
                v1, v2 = kv.split('||', 1)
                all_results.append({
                    'features':     f'{f1}|{f2}',
                    'condition':    kv,
                    'outcome':      out_col,
                    'outcome_label': out_label,
                    'n':            n_c,
                    'k_pos':        k_c,
                    'win_rate':     round(k_c / n_c, 4),
                    'wilson_lower': round(wilson_lower(n_c, k_c), 4),
                    'p_value':      round(p, 8),
                    'complexity':   2,
                    'source':       'method1_combined',
                })
                k2_count += 1

    elapsed = time.time() - t1
    print(f" {k2_count} patterns  [{elapsed:.0f}s]")

# ── Save ─────────────────────────────────────────────────────────────────────
result_df = pd.DataFrame(all_results)
if len(result_df) > 0:
    result_df = result_df.drop_duplicates(subset=['features', 'condition', 'outcome'])
    result_df = result_df.sort_values('wilson_lower', ascending=False)

out_path = f"{REPO}/results/research/method1_combined_k12.csv"
result_df.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print(f"Combined k=1,2 scan complete in {time.time()-t0:.0f}s")
print(f"Total patterns: {len(result_df)}")
if len(result_df) > 0:
    asp_pats = result_df[result_df['features'].str.contains(
        'asp|deg_|ex_|dom_|natal', regex=True, na=False)]
    print(f"Patterns using aspect features: {len(asp_pats)}")
    mixed = asp_pats[asp_pats['features'].str.contains('\|', regex=True, na=False)]
    print(f"  Mixed (aspect + other feature): {len(mixed)}")
    print(f"\nTop 10 by Wilson LB:")
    print(result_df[['features','condition','outcome','n','win_rate',
                      'wilson_lower','p_value']].head(10).to_string())
print(f"\nSaved: {out_path}")
