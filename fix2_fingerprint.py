"""
Fix 2: Correct fingerprint relaxation — no k=3 cap, no pre-screening to top-20.

Optimization: precompute numpy boolean arrays for every (col, val) pair so the
inner while-loop runs bitwise AND on numpy arrays (~100x faster than rebuilding
pandas Series per iteration).
"""
import math, os, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
import itertools

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
os.makedirs(f"{REPO}/results/research", exist_ok=True)

MIN_N    = 5
SCREEN_P = 0.35

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0, (p + z**2/(2*n) - z*math.sqrt(max(0, p*(1-p)/n + z**2/(4*n**2)))) / (1 + z**2/n))

def fisher_p(n_c, k_c, N, k_t):
    a = k_c; b = n_c - k_c; c = k_t - k_c; d = (N - n_c) - (k_t - k_c)
    if any(x < 0 for x in [a,b,c,d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a),max(0,b)],[max(0,c),max(0,d)]])
    return p

print("Loading nifty_enriched.csv …")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

outcome_cols = ['is_strong_bull','is_strong_bear','is_sideways','is_high_vol',
                'is_reversal','is_bull','is_continuation']
df_clean = df.dropna(subset=[c for c in outcome_cols if c in df.columns]).copy()
for c in outcome_cols:
    if c in df_clean.columns:
        df_clean[c] = df_clean[c].astype(int)

N = len(df_clean)
print(f"  Clean rows: {N}")

CAT_FEATURES = [c for c in [
    'paksha','tithi_quality','moon_phase','nak_mo_name','nak_mo_qual',
    'yoga_quality','karana_quality','vara_lord','vara_dig',
    'dig_Ju','dig_Sa','dig_Mo','dig_Ma','dig_Ve','dig_Me','dig_Su',
    'dig_Ra','dig_Ke',
    'elem_Mo','mod_Mo','tara_quality','tara_name','mahadasha','antardasha',
    'sade_sati_phase','nakl_dig','nakl_spd','choghadiya_quality','hora_at_open',
    'ix_paksha_ju_dig','ix_paksha_nak','ix_paksha_moon_sign',
    'ix_tithi_nak','ix_ju_dig_moon_sign','ix_vara_paksha',
    'ix_ju_speed_dig','ix_sa_speed_dig','ix_own_nak_ju_paksha','ix_argala_paksha',
    'true_node_cat','cheshta_cat_Ju','cheshta_cat_Sa','cheshta_cat_Mo',
    'cheshta_cat_Me','cheshta_cat_Ve',
    'atmakaraka','amatyakaraka','graha_yuddha_pair',
] if c in df_clean.columns]

BIN_FEATURES = [c for c in [
    'gajakesari','kemadruma','papakartari','shubhakartari',
    'chandra_mangala','sakata','neecha_bhanga','argala_positive',
    'argala_obstruct','vipareeta_raja',
    'parivartana_any','graha_yuddha',
    'comb_Mo','comb_Me','comb_Ve','comb_Ma','comb_Ju',
    'retro_Me','retro_Ju','retro_Sa','retro_Ma','retro_Ve',
    'gand_Mo','gand_any','sandhi_mo','nak_transition','panchaka',
    'sade_sati','ashtama_shani','ju_asp_mo','sa_asp_mo','ma_asp_mo',
    'own_nak_Ju','own_nak_Mo','own_nak_Sa','own_nak_Me','own_nak_Ve',
    'own_nak_Ma','own_nak_Su',
    'digbala_Ju','digbala_Sa','digbala_Mo',
] if c in df_clean.columns]

for col in BIN_FEATURES:
    if (col + '_s') not in df_clean.columns:
        df_clean[col + '_s'] = col + '=' + df_clean[col].astype(str)

BIN_STR_FEAT = [c + '_s' for c in BIN_FEATURES if (c + '_s') in df_clean.columns]
ALL_FEAT = CAT_FEATURES + BIN_STR_FEAT
print(f"  Feature pool: {len(ALL_FEAT)} columns")


def fingerprint_relaxation(df_clean, outcome_col, all_feat, min_n=MIN_N, screen_p=SCREEN_P):
    N      = len(df_clean)
    k_total = int(df_clean[outcome_col].sum())
    base_rate = k_total / N
    print(f"\n  [{outcome_col}] N={N}, positive={k_total}, base={base_rate:.3f}")

    # Step 1: p-value cache for every (col, val) pair
    print("  Pre-computing p-values …", end='', flush=True)
    pval_cache = {}
    present_cols = [c for c in all_feat if c in df_clean.columns]
    for col in present_cols:
        grpd = df_clean.groupby(df_clean[col].astype(str))[outcome_col]
        for val, grp in grpd:
            n_c = len(grp); k_c = int(grp.sum())
            pval_cache[(col, val)] = fisher_p(n_c, k_c, N, k_total) if n_c >= 5 else 1.0
    print(f" {len(pval_cache)} pairs cached")

    # Step 2: precompute numpy bool arrays for every (col, val) — KEY OPTIMIZATION
    # df_pos is the subset of positive-outcome rows (used for n_match checks)
    df_pos  = df_clean[df_clean[outcome_col] == 1].copy()
    n_pos   = len(df_pos)
    pos_idx = df_pos.index  # original index positions in df_clean

    print(f"  Building numpy bitmasks …", end='', flush=True)
    # Precompute string values for each column (once)
    pos_str   = {c: df_pos[c].astype(str).values   for c in present_cols}
    clean_str = {c: df_clean[c].astype(str).values for c in present_cols}

    pos_bool   = {}  # (col,val) -> numpy bool[n_pos]
    clean_bool = {}  # (col,val) -> numpy bool[N]
    for (col, val) in pval_cache:
        if col in pos_str:
            pos_bool[(col, val)]   = (pos_str[col]   == val)
            clean_bool[(col, val)] = (clean_str[col]  == val)
    print(f" done ({len(pos_bool)} arrays)", flush=True)

    # Step 3: fingerprint relaxation on each unexplained positive day
    print(f"  Positive days: {n_pos}", flush=True)
    patterns  = []
    explained = set()   # set of integer positions in df_pos (0..n_pos-1)

    for i in range(n_pos):
        if i in explained:
            continue

        if i % 500 == 0 and i > 0:
            print(f"    … {i}/{n_pos} days done, {len(patterns)} patterns so far", flush=True)

        # Fast: use precomputed str arrays (851× faster than df_pos.iloc[i][c])
        row_vals = {c: pos_str[c][i] for c in present_cols}

        # Build active feature list: features with p < screen_p for this row's values
        active = []
        for col in present_cols:
            val = row_vals[col]
            p   = pval_cache.get((col, val), 1.0)
            if p < screen_p:
                active.append((col, val, p))
        active.sort(key=lambda x: x[2])  # most significant first

        while len(active) >= 1:
            # Fast numpy bitmask intersection
            mask = np.ones(n_pos, dtype=bool)
            for col, val, _ in active:
                arr = pos_bool.get((col, val))
                if arr is not None:
                    mask &= arr
                else:
                    mask[:] = False; break
            n_match = int(mask.sum())

            if n_match >= min_n:
                # Stats on full df_clean using precomputed clean_bool arrays
                cmask = np.ones(N, dtype=bool)
                for col, val, _ in active:
                    arr = clean_bool.get((col, val))
                    if arr is not None:
                        cmask &= arr
                    else:
                        cmask[:] = False; break
                n_c = int(cmask.sum())
                k_c = int(df_clean.loc[df_clean.index[cmask], outcome_col].sum())

                matched_positions = np.where(mask)[0].tolist()
                explained.update(matched_positions)

                patterns.append({
                    'features'    : '|'.join(c for c,v,_ in active),
                    'condition'   : '||'.join(v for c,v,_ in active),
                    'outcome'     : outcome_col,
                    'n'           : n_c,
                    'k_pos'       : k_c,
                    'win_rate'    : round(k_c/n_c, 4) if n_c > 0 else 0,
                    'wilson_lower': round(wilson_lower(n_c, k_c), 4),
                    'p_value'     : round(fisher_p(n_c, k_c, N, k_total), 8),
                    'complexity'  : len(active),
                    'base_rate'   : round(base_rate, 4),
                })
                break

            active = active[:-1]  # drop least significant, retry

    pct = 100 * len(explained) / n_pos if n_pos > 0 else 0
    print(f"  → {len(patterns)} patterns, {len(explained)}/{n_pos} ({pct:.0f}%) days explained", flush=True)
    return pd.DataFrame(patterns)


def full_scan(df, outcome_col, all_feat, min_n=10, pval_thresh=0.05, max_k=3):
    N   = len(df)
    k_t = int(df[outcome_col].sum())
    results = []
    sig_cols = []

    present = [c for c in all_feat if c in df.columns]
    for col in present:
        grouped = df.groupby(df[col].astype(str))[outcome_col]
        for val, grp in grouped:
            n_c = len(grp); k_c = int(grp.sum())
            if n_c < min_n: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                results.append({'features': col, 'condition': val,
                                 'n': n_c, 'k_pos': k_c,
                                 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wilson_lower(n_c, k_c),4),
                                 'p_value': round(p,8), 'complexity': 1})
            if p < 0.15 and col not in sig_cols:
                sig_cols.append(col)

    print(f"    k=1: {len(results)} patterns, {len(sig_cols)} sig features")

    for c1, c2 in itertools.combinations(sig_cols, 2):
        key = df[c1].astype(str) + '||' + df[c2].astype(str)
        for kv, idx in df.groupby(key).groups.items():
            n_c = len(idx); k_c = int(df.loc[idx, outcome_col].sum())
            if n_c < min_n: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                results.append({'features': f'{c1}|{c2}', 'condition': kv,
                                 'n': n_c, 'k_pos': k_c,
                                 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wilson_lower(n_c, k_c),4),
                                 'p_value': round(p,8), 'complexity': 2})
    print(f"    k=2: {len(results)} total")

    for c1, c2, c3 in itertools.combinations(sig_cols, 3):
        key = df[c1].astype(str) + '||' + df[c2].astype(str) + '||' + df[c3].astype(str)
        for kv, idx in df.groupby(key).groups.items():
            n_c = len(idx); k_c = int(df.loc[idx, outcome_col].sum())
            if n_c < min_n: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                results.append({'features': f'{c1}|{c2}|{c3}', 'condition': kv,
                                 'n': n_c, 'k_pos': k_c,
                                 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wilson_lower(n_c, k_c),4),
                                 'p_value': round(p,8), 'complexity': 3})
    print(f"    k=3: {len(results)} total")
    return pd.DataFrame(results)


OUTCOMES = [
    ('is_strong_bull', 'STRONG_BULL'),
    ('is_strong_bear', 'STRONG_BEAR'),
    ('is_sideways',    'SIDEWAYS'),
    ('is_high_vol',    'HIGH_VOL'),
    ('is_bull',        'BULL_DIR'),
]

t0 = time.time()
all_results = []

for out_col, out_label in OUTCOMES:
    if out_col not in df_clean.columns:
        print(f"  Skipping {out_col} (not in data)")
        continue
    t1 = time.time()
    pats = fingerprint_relaxation(df_clean, out_col, ALL_FEAT)
    if len(pats) > 0:
        pats['outcome_label'] = out_label
        all_results.append(pats)
    elapsed = time.time() - t1
    print(f"  {out_label}: {len(pats)} patterns  [{elapsed:.1f}s]")

    # Checkpoint save after each outcome
    if all_results:
        chk = pd.concat(all_results, ignore_index=True)
        chk.to_csv(f"{REPO}/results/research/method1_fp_checkpoint.csv", index=False)

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.drop_duplicates(subset=['features','condition'])
    combined = combined.sort_values('wilson_lower', ascending=False)
else:
    combined = pd.DataFrame()

print(f"\nFingerprint relaxation: {len(combined)} total patterns  [{time.time()-t0:.1f}s]")

# Method 2: uncapped k=1,2,3 scan
print("\nRunning uncapped Method 2 scan (k=1,2,3) …")
m2_results = []
for out_col, out_label in OUTCOMES:
    if out_col not in df_clean.columns: continue
    t1 = time.time()
    print(f"  M2 scan: {out_label}")
    pats = full_scan(df_clean, out_col, ALL_FEAT)
    if len(pats) > 0:
        pats['outcome'] = out_col
        pats['outcome_label'] = out_label
        m2_results.append(pats)
    print(f"    → {len(pats)} [{time.time()-t1:.1f}s]")

m2_combined = pd.concat(m2_results, ignore_index=True).drop_duplicates(
    subset=['features','condition']) if m2_results else pd.DataFrame()

print(f"Method 2 uncapped: {len(m2_combined)} patterns")

combined['source'] = 'method1_fp'
m2_combined['source'] = 'method2_full'

combined.to_csv(f"{REPO}/results/research/method1_fp_uncapped.csv", index=False)
m2_combined.to_csv(f"{REPO}/results/research/method2_full.csv", index=False)

print(f"\nSaved method1_fp_uncapped.csv  ({len(combined)} patterns)")
print(f"Saved method2_full.csv          ({len(m2_combined)} patterns)")
print(f"\nFix 2 complete in {time.time()-t0:.1f}s")
