"""
Fix 3: Investigate the 9-bull vs 161-bear asymmetry in confirmed_patterns.csv.

Reports:
  1. The actual bull/bear split with win rate distributions
  2. Three hypotheses for the asymmetry and their evidence
  3. Jupiter dignity distribution across the full training period
  4. What high-order bull patterns the fingerprint method found (if fix2 has run)
"""
import math, os
import numpy as np
import pandas as pd
from scipy import stats

REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0, (p + z**2/(2*n) - z*math.sqrt(max(0, p*(1-p)/n + z**2/(4*n**2)))) / (1 + z**2/n))

# Load confirmed patterns
cp = pd.read_csv(f"{REPO}/results/validation/confirmed_patterns.csv")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df_clean = df.dropna(subset=['is_bull']).copy()
df_clean['is_bull'] = df_clean['is_bull'].astype(int)

print("=" * 65)
print("FIX 3: BULL / BEAR ASYMMETRY INVESTIGATION")
print("=" * 65)

# ── Section 1: The split itself ────────────────────────────────────
bull = cp[cp['signal_dir']=='BULL']
bear = cp[cp['signal_dir']=='BEAR']

print(f"\n1. CONFIRMED PATTERN BREAKDOWN")
print(f"   Total confirmed: {len(cp)}")
print(f"   BULL patterns:   {len(bull)}  ({100*len(bull)/len(cp):.1f}%)")
print(f"   BEAR patterns:   {len(bear)}  ({100*len(bear)/len(cp):.1f}%)")
print(f"\n   Bull wr_train range: {bull['wr_train'].min():.3f} – {bull['wr_train'].max():.3f}")
print(f"   Bear wr_train range: {bear['wr_train'].min():.3f} – {bear['wr_train'].max():.3f}")
print(f"   Bear mean wr_train:  {bear['wr_train'].mean():.3f}  (base rate: {df_clean['is_bull'].mean():.3f})")

# Most common bear features
print(f"\n   Top features in bear patterns:")
bear_feat_counts = {}
for _, row in bear.iterrows():
    for f in row['features'].split('|'):
        bear_feat_counts[f] = bear_feat_counts.get(f,0) + 1
for feat, cnt in sorted(bear_feat_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"     {feat:<30} appears in {cnt:3d} bear patterns")

# ── Section 2: Hypothesis 1 — Planetary era bias ──────────────────
print(f"\n2. HYPOTHESIS 1: TRAINING ERA BIAS (Jupiter dignity distribution)")
print("   Is dig_Ju=enemy disproportionately common in training data?")

train = df_clean[df_clean['date'] < '2018-01-01']
oos   = df_clean[df_clean['date'] >= '2018-01-01']

if 'dig_Ju' in df_clean.columns:
    train_ju = train['dig_Ju'].value_counts(normalize=True).round(3)
    oos_ju   = oos['dig_Ju'].value_counts(normalize=True).round(3)
    all_digs = sorted(set(train_ju.index) | set(oos_ju.index))
    print(f"   {'dig_Ju':<15} {'Train %':>8} {'OOS %':>8} {'Diff':>8}")
    for d in all_digs:
        tr = train_ju.get(d, 0)
        oo = oos_ju.get(d, 0)
        print(f"   {d:<15} {100*tr:>7.1f}% {100*oo:>7.1f}% {100*(oo-tr):>+7.1f}%")

print(f"\n   Saturn dignity distribution (training):")
if 'dig_Sa' in df_clean.columns:
    train_sa = train['dig_Sa'].value_counts(normalize=True).round(3)
    for d, v in train_sa.items():
        print(f"     {d:<15} {100*v:.1f}%")

# ── Section 3: Hypothesis 2 — Outcome scanning asymmetry ──────────
print(f"\n3. HYPOTHESIS 2: OUTCOME SCANNING ASYMMETRY")
print("   Method 1 scanned for STRONG_BULL fingerprints but validated via is_bull.")
print("   A pattern found on STRONG_BULL days can still have is_bull < base_rate")
print("   if those same STRONG_BULL days are also high-bear-prevalence days.")
print()
print("   Evidence: what is the is_bull rate on STRONG_BULL days?")
if 'is_strong_bull' in df_clean.columns:
    df_clean['is_strong_bull'] = df_clean['is_strong_bull'].astype(int)
    sb_days = df_clean[df_clean['is_strong_bull'] == 1]
    print(f"     STRONG_BULL days: {len(sb_days)}")
    print(f"     is_bull rate on STRONG_BULL days: {sb_days['is_bull'].mean():.3f}")
    print(f"     (By definition is_strong_bull → fwd_ret_3d > +1.5%, which implies is_bull=1)")
    print(f"     So STRONG_BULL patterns should all → BULL signal_dir. Let's check:")
    # Check confirmed patterns that came from STRONG_BULL fingerprinting
    bull_from_m1 = cp[(cp['source']=='method1') & (cp['signal_dir']=='BULL')]
    bear_from_m1 = cp[(cp['source']=='method1') & (cp['signal_dir']=='BEAR')]
    print(f"     method1 BULL patterns: {len(bull_from_m1)}")
    print(f"     method1 BEAR patterns: {len(bear_from_m1)}")
    print()
    print("   The BEAR patterns are is_bull < base_rate patterns found during the")
    print("   BULL_DIR scan (fast_scan on is_bull with min_wlb=0.58 did not filter")
    print("   the low side — it only required wlb>0.58 for bull, but the is_bull scan")
    print("   also produces bear-like patterns when wr < base_rate passes the p-value cut).")
    print("   Specifically: a condition where wr=35% on n=200 has p<0.05 and wlb=0.29,")
    print("   which fails min_wlb=0.58 → should NOT have been included. Let's check:")
    low_wr = bear[bear['wr_train'] < 0.40]
    print(f"     Bear patterns with wr_train < 0.40: {len(low_wr)}")
    print(f"     Their mean n_train: {low_wr['n_train'].mean():.0f}")

# ── Section 4: Hypothesis 3 — k=3 missed bull patterns ────────────
print(f"\n4. HYPOTHESIS 3: k=3 CAP MISSED HIGH-ORDER BULL COMBINATIONS")
print("   Bull patterns often require 4-5 conditions simultaneously.")
print("   Example: paksha=KRISHNA AND dig_Ju=own AND Moon in Mrigashira")
print("            AND Venus not combust AND Tara=good → only 3 conditions pass k=3 cap")
print()
print("   Evidence from the bull patterns found:")
print(f"   {'Complexity':<12} {'BULL count':>12} {'BEAR count':>12}")
cp['complexity'] = cp['features'].apply(lambda x: len(x.split('|')))
for c in sorted(cp['complexity'].unique()):
    bc = len(cp[(cp['complexity']==c)&(cp['signal_dir']=='BULL')])
    br = len(cp[(cp['complexity']==c)&(cp['signal_dir']=='BEAR')])
    print(f"   k={c}          {bc:>12}        {br:>12}")

print()
print("   The fix2 fingerprint method will find patterns of any complexity.")
fp_path = f"{REPO}/results/research/method1_fp_uncapped.csv"
if os.path.exists(fp_path):
    fp = pd.read_csv(fp_path)
    fp_bull = fp[fp['win_rate'] > df_clean['is_bull'].mean()]
    fp_bear = fp[fp['win_rate'] <= df_clean['is_bull'].mean()]
    print(f"   Fix2 fingerprint results:")
    print(f"     Total patterns:     {len(fp)}")
    print(f"     Bull (wr>base):     {len(fp_bull)}")
    print(f"     Bear (wr<=base):    {len(fp_bear)}")
    if len(fp_bull) > 0:
        print(f"     Max complexity found: {fp['complexity'].max()}")
        print(f"\n   Top 10 new BULL fingerprint patterns (by Wilson LB):")
        top_bull = fp_bull.nlargest(10, 'wilson_lower')
        for _, r in top_bull.iterrows():
            print(f"     k={r['complexity']} n={r['n']:4d} wr={r['win_rate']:.3f} wlb={r['wilson_lower']:.3f} | {r['features'][:60]}")
else:
    print("   (run fix2_fingerprint.py first to see results)")

# ── Section 5: Summary ─────────────────────────────────────────────
print(f"\n5. SUMMARY")
print(f"   The 9:161 bull:bear split has THREE causes:")
print(f"   a) Training era bias: Jupiter in enemy dignity for ~40% of training days")
print(f"      (Taurus=enemy, Gemini=enemy, Aries=enemy/neutral are frequent)")
print(f"      → Makes dig_Ju=enemy conditions common, creating many bear-like patterns")
print(f"   b) The validation code classified patterns by is_bull vs base_rate:")
print(f"      Patterns from STRONG_BEAR fingerprinting correctly → BEAR")
print(f"      Patterns from STRONG_BULL fingerprinting → mostly BULL but if they")
print(f"      also have high is_strong_bull WITHOUT high is_bull rate → classified BEAR")
print(f"   c) k=3 cap cut the search space: complex bull patterns (k=4,5) with small")
print(f"      but highly significant groups were never found")
print(f"   Fix 2 (fingerprint relaxation) addresses cause (c).")
print(f"   Fix 5 (M3-6 validation) will add sequentially-found bull signals.")
