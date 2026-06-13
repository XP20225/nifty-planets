"""
Fix 1: Add missing Vedic features to nifty_enriched.csv and banknifty_enriched.csv.

New features added:
  own_nak_{p}       — planet in one of its 3 vimshottari-ruled nakshatras
  argala_obstruct   — any planet in 3rd/5th/12th from Moon (virodha argala)
  argala_positive   — any planet in 2nd/4th/11th from Moon (rename of argala_mo)
  vipareeta_raja    — debilitated planet in 6th/8th/12th from Moon
  cheshta_cat_{p}   — explicit speed category string per planet
  true_node_diff    — true node − mean node (Rahu) in degrees via pyswisseph
  nak_{p}           — nakshatra number for every planet (not just Moon)
"""
import json, math, os, warnings, time
import numpy as np
import pandas as pd
import swisseph as swe

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
t0 = time.time()

PLANETS = ['Su','Mo','Me','Ve','Ma','Ju','Sa','Ra','Ke']
MEAN_MOTION = {'Su':0.9856,'Mo':13.1764,'Me':1.3833,'Ve':1.2,'Ma':0.5240,
               'Ju':0.0831,'Sa':0.0335,'Ra':-0.0529,'Ke':-0.0529}

# Each planet rules these 3 nakshatras (Vimshottari cycle order 1-27)
PLANET_NAKS = {
    'Ke': {1,10,19}, 'Ve': {2,11,20}, 'Su': {3,12,21},
    'Mo': {4,13,22}, 'Ma': {5,14,23}, 'Ra': {6,15,24},
    'Ju': {7,16,25}, 'Sa': {8,17,26}, 'Me': {9,18,27}
}

def nak_of(sid_deg):
    return int(sid_deg / (360/27)) + 1  # 1-27

def cheshta_cat(planet, spd):
    if spd < -0.001: return 'retrograde'
    mm = MEAN_MOTION[planet]
    if abs(spd) < 0.05: return 'stationary'
    ratio = spd / abs(mm)
    if ratio >= 1.30: return 'very_fast'
    if ratio >= 1.10: return 'fast'
    if ratio >= 0.90: return 'mean'
    if ratio >= 0.70: return 'mean_slow'
    return 'slow'

def add_new_features(df):
    print("  Adding nakshatra number for all planets …")
    for p in PLANETS:
        if f'sid_{p}' not in df.columns:
            continue
        df[f'nak_{p}'] = df[f'sid_{p}'].apply(nak_of)
        df[f'own_nak_{p}'] = df[f'nak_{p}'].isin(PLANET_NAKS[p]).astype(int)

    print("  Adding Argala features …")
    def argala_pos(r):
        # Positive argala: 2nd, 4th, 11th from Moon
        signs = [(r['sign_Mo'] % 12) + 1,           # 2nd
                 ((r['sign_Mo'] + 2) % 12) + 1,     # 4th
                 ((r['sign_Mo'] + 9) % 12) + 1]     # 11th
        others = [r[f'sign_{p}'] for p in PLANETS if p != 'Mo']
        return int(any(s in signs for s in others))

    def argala_obstruct(r):
        # Virodha argala: 3rd, 5th, 12th from Moon
        signs = [((r['sign_Mo'] + 1) % 12) + 1,    # 3rd
                 ((r['sign_Mo'] + 3) % 12) + 1,    # 5th
                 ((r['sign_Mo'] + 10) % 12) + 1]   # 12th
        others = [r[f'sign_{p}'] for p in PLANETS if p != 'Mo']
        return int(any(s in signs for s in others))

    df['argala_positive']  = df.apply(argala_pos, axis=1)
    df['argala_obstruct']  = df.apply(argala_obstruct, axis=1)
    # Net argala (positive minus obstruct)
    df['argala_net'] = df['argala_positive'] - df['argala_obstruct']

    print("  Adding Vipareeta Raja Yoga …")
    # Debilitated planet in 6th, 8th, or 12th from Moon = VRY
    def vipareeta_raja(r):
        houses_6_8_12 = [((r['sign_Mo'] + 4) % 12) + 1,   # 6th
                         ((r['sign_Mo'] + 6) % 12) + 1,   # 8th
                         ((r['sign_Mo'] + 10) % 12) + 1]  # 12th
        for p in PLANETS:
            if r.get(f'dig_{p}', '') in ('debilitated', 'exact_debil'):
                if r[f'sign_{p}'] in houses_6_8_12:
                    return 1
        return 0
    df['vipareeta_raja'] = df.apply(vipareeta_raja, axis=1)

    print("  Adding Cheshta Bala categories …")
    for p in PLANETS:
        if f'spd_{p}' not in df.columns:
            continue
        df[f'cheshta_cat_{p}'] = df.apply(
            lambda r, planet=p: cheshta_cat(planet, r[f'spd_{planet}']), axis=1)

    print("  Computing true node via pyswisseph …")
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    dates = pd.to_datetime(df['date'])

    true_diffs = []
    for d in dates:
        jd = swe.julday(d.year, d.month, d.day, 12.0)
        try:
            true_res, _ = swe.calc_ut(jd, swe.TRUE_NODE, swe.FLG_SIDEREAL)
            mean_res, _ = swe.calc_ut(jd, swe.MEAN_NODE, swe.FLG_SIDEREAL)
            diff = (true_res[0] - mean_res[0]) % 360
            if diff > 180: diff -= 360
            true_diffs.append(round(diff, 4))
        except Exception:
            true_diffs.append(0.0)
    df['true_node_diff'] = true_diffs
    # Category: nodal oscillation amplitude
    df['true_node_cat'] = pd.cut(df['true_node_diff'],
                                  bins=[-5, -1.0, -0.3, 0.3, 1.0, 5],
                                  labels=['far_behind','behind','aligned','ahead','far_ahead'])

    # Extra interaction features using new columns
    print("  Adding extended interaction features …")
    if 'cheshta_cat_Ju' in df.columns:
        df['ix_ju_speed_dig'] = df['cheshta_cat_Ju'].astype(str) + '_' + df['dig_Ju'].astype(str)
    if 'cheshta_cat_Sa' in df.columns:
        df['ix_sa_speed_dig'] = df['cheshta_cat_Sa'].astype(str) + '_' + df['dig_Sa'].astype(str)
    if 'own_nak_Ju' in df.columns:
        df['ix_own_nak_ju_paksha'] = df['own_nak_Ju'].astype(str) + '_' + df['paksha'].astype(str)
    if 'argala_net' in df.columns:
        df['ix_argala_paksha'] = df['argala_net'].astype(str) + '_' + df['paksha'].astype(str)

    return df


print("Loading nifty_enriched.csv …")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
print(f"  Shape before: {df.shape}")
df = add_new_features(df)
print(f"  Shape after:  {df.shape}")

# Lookahead check still passes
for n in [1,2,3,5,10]:
    col = f'fwd_ret_{n}d'
    if col in df.columns:
        assert df[col].iloc[-n:].isna().all(), f"Lookahead in {col}!"
print("  Lookahead audit PASS")

df.to_csv(f"{REPO}/data/nifty_enriched.csv", index=False)
print(f"Saved nifty_enriched.csv  ({df.shape[0]}×{df.shape[1]})")

print("\nLoading banknifty_enriched.csv …")
bn = pd.read_csv(f"{REPO}/data/banknifty_enriched.csv", low_memory=False)
print(f"  Shape before: {bn.shape}")
bn = add_new_features(bn)
print(f"  Shape after:  {bn.shape}")
bn.to_csv(f"{REPO}/data/banknifty_enriched.csv", index=False)
print(f"Saved banknifty_enriched.csv ({bn.shape[0]}×{bn.shape[1]})")

print(f"\nFix 1 complete in {time.time()-t0:.1f}s")
print("New features added per instrument:")
new_feats = ([f'nak_{p}' for p in PLANETS] +
             [f'own_nak_{p}' for p in PLANETS] +
             [f'cheshta_cat_{p}' for p in PLANETS] +
             ['argala_positive','argala_obstruct','argala_net',
              'vipareeta_raja','true_node_diff','true_node_cat',
              'ix_ju_speed_dig','ix_sa_speed_dig','ix_own_nak_ju_paksha','ix_argala_paksha'])
print(f"  {len(new_feats)} new columns (total now: {df.shape[1]})")
