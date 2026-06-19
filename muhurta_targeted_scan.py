"""
Targeted scan for Muhurta features (hora, choghadiya, rahu_kalam, gulika_kalam)
against all combinations with top astrological features.
Fast — completes in 2-3 minutes.
"""
import math, os, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"

MIN_N = 10
SIG_THRESH = 0.05

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

df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df = df[df['outcome_3d'].isin(['STRONG_BULL','STRONG_BEAR'])].copy()
df['is_bull'] = (df['outcome_3d'] == 'STRONG_BULL').astype(int)
N = len(df)
k_tot = df['is_bull'].sum()
print(f"Dataset: {N} rows, {k_tot} bull ({k_tot/N*100:.1f}%)")

# Muhurta features to scan
MUHURTA_FEATS = ['hora_at_open', 'choghadiya', 'choghadiya_quality',
                 'rahu_kalam_open', 'gulika_kalam_open']

# Top partner features to combine with
PARTNER_FEATS = ['paksha', 'dig_Ju', 'dig_Sa', 'dig_Mo', 'dig_Ma',
                 'nak_mo', 'sign_Mo', 'sign_Ju', 'sign_Sa',
                 'vara_lord', 'tithi_num', 'yoga_quality',
                 'mahadasha', 'choghadiya_quality', 'hora_at_open']

def get_sig_cols(feat_list):
    cols = []
    for f in feat_list:
        if f not in df.columns:
            continue
        for val in df[f].dropna().unique():
            mask = df[f] == val
            n_c = mask.sum()
            if n_c < MIN_N:
                continue
            k_c = df.loc[mask, 'is_bull'].sum()
            p = fisher_p(n_c, k_c, N, k_tot)
            if p < 0.10:
                cols.append((f, val, n_c, k_c, p))
    return cols

print("Scanning Muhurta k=1 patterns …")
rows = []

for f in MUHURTA_FEATS:
    if f not in df.columns:
        print(f"  MISSING: {f}")
        continue
    for val in sorted(df[f].dropna().unique()):
        mask = df[f] == val
        n_c = int(mask.sum())
        if n_c < MIN_N:
            continue
        k_c = int(df.loc[mask, 'is_bull'].sum())
        p = fisher_p(n_c, k_c, N, k_tot)
        wr = k_c / n_c
        wlb = wilson_lower(n_c, k_c)
        rows.append({
            'features': f, 'condition': f'{f}=={val}', 'outcome': 'BULL' if wr > 0.5 else 'BEAR',
            'n': n_c, 'win_rate': round(wr, 4), 'wilson_lower': round(wlb, 4), 'p_value': round(p, 6),
            'k': 1
        })

print(f"  k=1 candidates: {len(rows)}")

print("Scanning Muhurta × partner k=2 patterns …")
muhurta_sigs = []
for f in MUHURTA_FEATS:
    if f not in df.columns: continue
    for val in df[f].dropna().unique():
        mask = df[f] == val
        n_c = mask.sum()
        if n_c < MIN_N: continue
        k_c = df.loc[mask, 'is_bull'].sum()
        p = fisher_p(n_c, k_c, N, k_tot)
        if p < 0.20:
            muhurta_sigs.append((f, val, mask))

for mf, mv, mmask in muhurta_sigs:
    for pf in PARTNER_FEATS:
        if pf == mf or pf not in df.columns: continue
        for pv in df[pf].dropna().unique():
            pmask = df[pf] == pv
            combo = mmask & pmask
            n_c = int(combo.sum())
            if n_c < MIN_N: continue
            k_c = int(df.loc[combo, 'is_bull'].sum())
            p = fisher_p(n_c, k_c, N, k_tot)
            if p >= SIG_THRESH: continue
            wr = k_c / n_c
            wlb = wilson_lower(n_c, k_c)
            feats_str = f"{mf}|{pf}"
            cond_str  = f"{mf}=={mv}&{pf}=={pv}"
            rows.append({
                'features': feats_str, 'condition': cond_str,
                'outcome': 'BULL' if wr > 0.5 else 'BEAR',
                'n': n_c, 'win_rate': round(wr, 4), 'wilson_lower': round(wlb, 4),
                'p_value': round(p, 6), 'k': 2
            })

result = pd.DataFrame(rows)
result = result.sort_values('p_value')
out_path = f"{REPO}/results/research/muhurta_targeted.csv"
result.to_csv(out_path, index=False)
print(f"Saved {len(result)} patterns → {out_path}")

sig = result[result['p_value'] < SIG_THRESH]
bull = sig[sig['win_rate'] > 0.5]
bear = sig[sig['win_rate'] <= 0.5]
print(f"\nSignificant (p<0.05): {len(sig)} — {len(bull)} bull, {len(bear)} bear")
print(f"\nTop 20 by p-value:")
print(sig.head(20)[['features','condition','n','win_rate','wilson_lower','p_value','k']].to_string())

# Specifically check gulika_kalam_open and rahu_kalam_open
for feat in ['gulika_kalam_open', 'rahu_kalam_open']:
    subset = sig[sig['features'].str.contains(feat, na=False)]
    print(f"\n{feat} patterns ({len(subset)}):")
    if len(subset) > 0:
        print(subset[['condition','n','win_rate','wilson_lower','p_value']].head(10).to_string())
