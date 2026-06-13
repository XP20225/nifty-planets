"""
Fix 2: Correct fingerprint relaxation — no k=3 cap, no pre-screening to top-20.

Algorithm per STRONG_BULL day:
  1. Collect all features where the day's specific value is significantly
     associated with the outcome (Fisher p < SCREEN_PVAL).
  2. Check how many other outcome days share the FULL combination.
  3. If n < MIN_N, drop the least significant variable (highest p-value).
  4. Repeat until n >= MIN_N or no features remain.
  5. Record the pattern; mark all matching outcome days as "explained".
  6. Move to next unexplained outcome day.

Runs for: STRONG_BULL, STRONG_BEAR, SIDEWAYS, HIGH_VOL, is_bull.
Saves to: results/research/method1_fp_uncapped.csv
"""
import math, os, warnings, time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
os.makedirs(f"{REPO}/results/research", exist_ok=True)

MIN_N      = 5      # minimum outcome days that must share the combination
SCREEN_P   = 0.35   # loose pre-screen: include feature if p < this for the specific value
MIN_WLB    = 0.0    # no hard floor on Wilson LB during fingerprinting (applied later in step3)

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0, (p + z**2/(2*n) - z*math.sqrt(max(0, p*(1-p)/n + z**2/(4*n**2)))) / (1 + z**2/n))

def fisher_p(n_c, k_c, N, k_t):
    a = k_c; b = n_c - k_c; c = k_t - k_c; d = (N - n_c) - (k_t - k_c)
    if any(x < 0 for x in [a,b,c,d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a),max(0,b)],[max(0,c),max(0,d)]])
    return p

# ──────────────────────────────────────────────────────────────────────
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

# Feature columns (pure astrological — NO market data, NO forward returns)
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

# ──────────────────────────────────────────────────────────────────────
def fingerprint_relaxation(df_clean, outcome_col, all_feat, min_n=MIN_N, screen_p=SCREEN_P):
    """Full fingerprint relaxation — no k cap, no top-N pre-screen."""
    N = len(df_clean)
    k_total = int(df_clean[outcome_col].sum())
    base_rate = k_total / N
    print(f"\n  [{outcome_col}] N={N}, positive={k_total}, base={base_rate:.3f}")

    # Pre-compute p-value of every (col, val) pair against this outcome
    print(f"  Pre-computing individual p-values …", end='', flush=True)
    pval_cache = {}
    for col in all_feat:
        if col not in df_clean.columns:
            continue
        grpd = df_clean.groupby(df_clean[col].astype(str))[outcome_col]
        for val, grp in grpd:
            n_c = len(grp); k_c = int(grp.sum())
            if n_c < 5:
                pval_cache[(col, val)] = 1.0
            else:
                pval_cache[(col, val)] = fisher_p(n_c, k_c, N, k_total)
    print(f" {len(pval_cache)} pairs cached")

    # Working set: only positive-outcome days
    df_pos = df_clean[df_clean[outcome_col] == 1].copy()
    n_pos  = len(df_pos)
    print(f"  Positive days: {n_pos}")

    patterns  = []
    explained = set()

    for i, idx in enumerate(df_pos.index):
        if idx in explained:
            continue

        row = df_pos.loc[idx]

        # Build initial active feature list: features where this row's value is < screen_p
        active = []
        for col in all_feat:
            if col not in row.index:
                continue
            val = str(row[col])
            p   = pval_cache.get((col, val), 1.0)
            if p < screen_p:
                active.append((col, val, p))

        # Sort: most significant (lowest p) first, least significant last
        active.sort(key=lambda x: x[2])

        found = False
        while len(active) >= 1:
            # Build mask on ALL positive-outcome days
            mask = pd.Series(True, index=df_pos.index)
            for col, val, _ in active:
                mask &= (df_pos[col].astype(str) == val)
            n_match = int(mask.sum())

            if n_match >= min_n:
                features_str = '|'.join(c for c,v,_ in active)
                cond_str     = '||'.join(v for c,v,_ in active)

                # Stats on full df_clean
                full_mask = pd.Series(True, index=df_clean.index)
                for col, val, _ in active:
                    full_mask &= (df_clean[col].astype(str) == val)
                n_c = int(full_mask.sum())
                k_c = int(df_clean.loc[full_mask, outcome_col].sum())

                matched = df_pos.index[mask].tolist()
                explained.update(matched)

                patterns.append({
                    'features'    : features_str,
                    'condition'   : cond_str,
                    'outcome'     : outcome_col,
                    'n'           : n_c,
                    'k_pos'       : k_c,
                    'win_rate'    : round(k_c/n_c, 4) if n_c > 0 else 0,
                    'wilson_lower': round(wilson_lower(n_c, k_c), 4),
                    'p_value'     : round(fisher_p(n_c, k_c, N, k_total), 8),
                    'complexity'  : len(active),
                    'base_rate'   : round(base_rate, 4),
                })
                found = True
                break

            # Drop least significant (last in sorted list)
            active = active[:-1]

        # If we exhausted all features without finding n_match>=5, this day is unexplained

    pct = 100 * len(explained) / n_pos if n_pos > 0 else 0
    print(f"  → {len(patterns)} patterns, {len(explained)}/{n_pos} ({pct:.0f}%) days explained")
    return pd.DataFrame(patterns)


# ──────────────────────────────────────────────────────────────────────
# Run for each outcome
t0 = time.time()
all_results = []

OUTCOMES = [
    ('is_strong_bull', 'STRONG_BULL'),
    ('is_strong_bear', 'STRONG_BEAR'),
    ('is_sideways',    'SIDEWAYS'),
    ('is_high_vol',    'HIGH_VOL'),
    ('is_bull',        'BULL_DIR'),
]

for out_col, out_label in OUTCOMES:
    if out_col not in df_clean.columns:
        print(f"  Skipping {out_col} (not in data)")
        continue
    t1 = time.time()
    pats = fingerprint_relaxation(df_clean, out_col, ALL_FEAT)
    if len(pats) > 0:
        pats['outcome_label'] = out_label
        all_results.append(pats)
    print(f"  {out_label}: {len(pats)} patterns  [{time.time()-t1:.1f}s]")

# ──────────────────────────────────────────────────────────────────────
# Combine and deduplicate
if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    # Remove exact duplicates on features+condition
    combined = combined.drop_duplicates(subset=['features','condition'])
    combined = combined.sort_values('wilson_lower', ascending=False)
else:
    combined = pd.DataFrame()

print(f"\nFingerprint relaxation complete: {len(combined)} total patterns  [{time.time()-t0:.1f}s]")

# Also run the old fast_scan (k=1,2,3 with full sig_cols, not capped at 20) for Method 2
# so we don't lose those patterns
print("\nRunning uncapped Method 2 scan (k=1,2,3, all sig features) …")
import itertools

def full_scan(df, outcome_col, min_n=10, pval_thresh=0.05, max_k=3):
    N   = len(df)
    k_t = int(df[outcome_col].sum())
    results = []
    sig_cols = []

    for col in ALL_FEAT:
        if col not in df.columns: continue
        grouped = df.groupby(df[col].astype(str))[outcome_col]
        for val, grp in grouped:
            n_c = len(grp); k_c = int(grp.sum())
            if n_c < min_n: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                wlb = wilson_lower(n_c, k_c)
                results.append({'features': col, 'condition': val,
                                 'n': n_c, 'k_pos': k_c,
                                 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wlb,4),
                                 'p_value': round(p,8), 'complexity': 1})
            if p < 0.15 and col not in sig_cols:
                sig_cols.append(col)

    print(f"    k=1 done: {len(results)} patterns, {len(sig_cols)} sig features")

    for c1, c2 in itertools.combinations(sig_cols, 2):
        key = df[c1].astype(str) + '||' + df[c2].astype(str)
        for kv, idx in df.groupby(key).groups.items():
            n_c = len(idx); k_c = int(df.loc[idx, outcome_col].sum())
            if n_c < min_n: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                wlb = wilson_lower(n_c, k_c)
                results.append({'features': f'{c1}|{c2}', 'condition': kv,
                                 'n': n_c, 'k_pos': k_c,
                                 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wlb,4),
                                 'p_value': round(p,8), 'complexity': 2})
    print(f"    k=2 done: {len(results)} total")

    for c1, c2, c3 in itertools.combinations(sig_cols, 3):
        key = (df[c1].astype(str) + '||' + df[c2].astype(str) + '||' + df[c3].astype(str))
        for kv, idx in df.groupby(key).groups.items():
            n_c = len(idx); k_c = int(df.loc[idx, outcome_col].sum())
            if n_c < min_n: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                wlb = wilson_lower(n_c, k_c)
                results.append({'features': f'{c1}|{c2}|{c3}', 'condition': kv,
                                 'n': n_c, 'k_pos': k_c,
                                 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wlb,4),
                                 'p_value': round(p,8), 'complexity': 3})
    print(f"    k=3 done: {len(results)} total")
    return pd.DataFrame(results)

m2_results = []
for out_col, out_label in OUTCOMES:
    if out_col not in df_clean.columns: continue
    t1 = time.time()
    print(f"  M2 scan: {out_label}")
    pats = full_scan(df_clean, out_col)
    if len(pats) > 0:
        pats['outcome'] = out_label
        pats['outcome_label'] = out_label
        m2_results.append(pats)
    print(f"    → {len(pats)} [{time.time()-t1:.1f}s]")

if m2_results:
    m2_combined = pd.concat(m2_results, ignore_index=True)
    m2_combined = m2_combined.drop_duplicates(subset=['features','condition'])
else:
    m2_combined = pd.DataFrame()

print(f"Method 2 uncapped: {len(m2_combined)} patterns")

# Save everything to method1_fp_uncapped.csv (fingerprint) and
# method2_full.csv (uncapped k=1,2,3)
combined['source'] = 'method1_fp'
m2_combined['source'] = 'method2_full'

final_m1 = combined.copy()
final_m2 = m2_combined.copy()

final_m1.to_csv(f"{REPO}/results/research/method1_fp_uncapped.csv", index=False)
final_m2.to_csv(f"{REPO}/results/research/method2_full.csv", index=False)

print(f"\nSaved method1_fp_uncapped.csv  ({len(final_m1)} patterns)")
print(f"Saved method2_full.csv          ({len(final_m2)} patterns)")
print(f"\nFix 2 complete in {time.time()-t0:.1f}s")
