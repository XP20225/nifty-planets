"""
New Step 2 — Six Research Methods (Optimized)
All six methods run on astrological features only.
Uses vectorized string concatenation for fast combination scanning.
"""
import math, os, warnings, itertools, time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
os.makedirs(f"{REPO}/results/research", exist_ok=True)

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0,(p+z**2/(2*n)-z*math.sqrt(max(0,p*(1-p)/n+z**2/(4*n**2))))/(1+z**2/n))

def wilson_upper(n, k, z=1.96):
    if n == 0: return 1.0
    p = k / n
    return min(1.0,(p+z**2/(2*n)+z*math.sqrt(max(0,p*(1-p)/n+z**2/(4*n**2))))/(1+z**2/n))

def fisher_p(n_cond, k_bull, n_total, k_total_bull):
    a=k_bull; b=n_cond-k_bull; c=k_total_bull-k_bull; d=(n_total-n_cond)-(k_total_bull-k_bull)
    if any(x<0 for x in [a,b,c,d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a),max(0,b)],[max(0,c),max(0,d)]])
    return p

# ═══════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════
print("Loading enriched data …")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
drop_cols = ['is_strong_bull','is_strong_bear','is_sideways','is_high_vol','is_reversal','is_continuation','is_bull']
df_clean = df.dropna(subset=[c for c in drop_cols if c in df.columns]).copy()
for col in drop_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(int)

N = len(df_clean)
BASE_BULL = df_clean['is_bull'].mean()
print(f"Rows: {N}, base bull rate: {BASE_BULL:.3f}")

# ── Feature columns (pure astrological) ──────────────────────────────
CAT_FEATURES = [c for c in [
    'paksha','tithi_quality','moon_phase','nak_mo_name','nak_mo_qual',
    'yoga_quality','karana_quality','vara_lord','vara_dig',
    'dig_Ju','dig_Sa','dig_Mo','dig_Ma','dig_Ve','dig_Me',
    'elem_Mo','mod_Mo','tara_quality','mahadasha','sade_sati_phase',
    'nakl_dig','nakl_spd','choghadiya_quality','hora_at_open',
    'ix_paksha_ju_dig','ix_paksha_nak',
] if c in df_clean.columns]

BIN_FEATURES = [c for c in [
    'gajakesari','kemadruma','papakartari','shubhakartari',
    'chandra_mangala','sakata','neecha_bhanga','argala_mo',
    'parivartana_any','graha_yuddha',
    'comb_Mo','comb_Me','comb_Ve','comb_Ma','comb_Ju',
    'retro_Me','retro_Ju','retro_Sa','retro_Ma','retro_Ve',
    'gand_Mo','gand_any','sandhi_mo','nak_transition','panchaka',
    'sade_sati','ashtama_shani','ju_asp_mo','sa_asp_mo','ma_asp_mo',
    'garv_Ju','kshu_Ju','garv_Mo','kshu_Mo',
] if c in df_clean.columns]

for col in BIN_FEATURES:
    df_clean[col + '_s'] = col + '=' + df_clean[col].astype(str)

BIN_STR_FEAT = [c+'_s' for c in BIN_FEATURES if (c+'_s') in df_clean.columns]
ALL_FEAT     = CAT_FEATURES + BIN_STR_FEAT

def fast_scan(df, outcome_col, min_n=5, min_wlb=0.30, max_k=3, pval_thresh=0.05):
    """
    Fast vectorized combination scan.
    1. Screen individual features first (p < 0.20 or notable effect)
    2. Build 2-way then 3-way from screened features only
    """
    N   = len(df)
    k_t = df[outcome_col].sum()
    results = []

    # k=1: individual features
    sig_cols = []  # features with at least one value that shows effect
    for col in ALL_FEAT:
        if col not in df.columns: continue
        grouped = df.groupby(df[col].astype(str))[outcome_col]
        for val, grp in grouped:
            n_c = len(grp); k_c = grp.sum()
            if n_c < min_n: continue
            wlb = wilson_lower(n_c, k_c)
            if wlb < min_wlb: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                results.append({'features': col, 'condition': val,
                                 'n': n_c, 'k_pos': k_c, 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wlb,4), 'p_value': round(p,8),
                                 'complexity': 1})
                if col not in sig_cols: sig_cols.append(col)
        # Screened at looser threshold for higher-k consideration
        for val, grp in grouped:
            n_c = len(grp); k_c = grp.sum()
            if n_c < min_n or k_c == 0: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < 0.15 and col not in sig_cols: sig_cols.append(col)

    print(f"  k=1 done: {len(results)} patterns, {len(sig_cols)} sig features")

    if max_k < 2 or len(sig_cols) < 2:
        return pd.DataFrame(results)

    # k=2: vectorized
    cols_2 = sig_cols[:30]  # limit to top 30 significant features
    for c1, c2 in itertools.combinations(cols_2, 2):
        # Vectorized key
        key = df[c1].astype(str) + '||' + df[c2].astype(str)
        for kv, idx in df.groupby(key).groups.items():
            n_c = len(idx); k_c = df.loc[idx, outcome_col].sum()
            if n_c < min_n: continue
            wlb = wilson_lower(n_c, k_c)
            if wlb < min_wlb: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                parts = kv.split('||', 1)
                results.append({'features': f'{c1}|{c2}', 'condition': kv,
                                 'n': n_c, 'k_pos': k_c, 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wlb,4), 'p_value': round(p,8),
                                 'complexity': 2})

    print(f"  k=2 done: {len(results)} total patterns")

    if max_k < 3 or len(sig_cols) < 3:
        return pd.DataFrame(results)

    # k=3: only among sig_cols, limit to top 20
    cols_3 = sig_cols[:20]
    for c1, c2, c3 in itertools.combinations(cols_3, 3):
        key = df[c1].astype(str) + '||' + df[c2].astype(str) + '||' + df[c3].astype(str)
        for kv, idx in df.groupby(key).groups.items():
            n_c = len(idx); k_c = df.loc[idx, outcome_col].sum()
            if n_c < min_n: continue
            wlb = wilson_lower(n_c, k_c)
            if wlb < min_wlb: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                results.append({'features': f'{c1}|{c2}|{c3}', 'condition': kv,
                                 'n': n_c, 'k_pos': k_c, 'win_rate': round(k_c/n_c,4),
                                 'wilson_lower': round(wlb,4), 'p_value': round(p,8),
                                 'complexity': 3})

    print(f"  k=3 done: {len(results)} total patterns")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════
# METHOD 1: OUTCOME FINGERPRINT MATCHING
# ═══════════════════════════════════════════════════════════════════════
print("\n=== METHOD 1: Outcome Fingerprint Matching ===")
t0 = time.time()

outcomes = {
    'STRONG_BULL': ('is_strong_bull', 0.30),
    'STRONG_BEAR': ('is_strong_bear', 0.15),
    'SIDEWAYS':    ('is_sideways',    0.20),
    'HIGH_VOL':    ('is_high_vol',    0.15),
    'REVERSAL':    ('is_reversal',    0.10),
}

all_m1 = []
for out_name, (out_col, min_wlb) in outcomes.items():
    if out_col not in df_clean.columns: continue
    print(f"\n  {out_name} (base={df_clean[out_col].mean():.3f})")
    sub_df = df_clean.copy()
    pats = fast_scan(sub_df, out_col, min_n=5, min_wlb=min_wlb, max_k=3, pval_thresh=0.05)
    if len(pats) > 0:
        pats['outcome'] = out_name
        all_m1.append(pats)
    print(f"  → {len(pats)} patterns for {out_name} in {time.time()-t0:.1f}s")

# Also run for is_bull (base directional)
print(f"\n  IS_BULL base direction (base={BASE_BULL:.3f})")
pats_bull = fast_scan(df_clean, 'is_bull', min_n=10, min_wlb=0.58, max_k=3, pval_thresh=0.05)
if len(pats_bull) > 0:
    pats_bull['outcome'] = 'BULL_DIR'
    all_m1.append(pats_bull)
print(f"  → {len(pats_bull)} bull direction patterns")

m1_all = pd.concat(all_m1, ignore_index=True) if all_m1 else pd.DataFrame()
m1_all = m1_all.sort_values('wilson_lower', ascending=False) if len(m1_all)>0 else m1_all
m1_all.to_csv(f"{REPO}/results/research/method1_pattern_library.csv", index=False)
print(f"\nMethod 1 complete: {len(m1_all)} patterns in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# METHOD 2: REVERSE CONDITION LOOKUP
# ═══════════════════════════════════════════════════════════════════════
print("\n=== METHOD 2: Reverse Condition Lookup ===")
t1 = time.time()

# Use interaction features as proxies for 4-5-variable combinations
# e.g. ix_paksha_ju_dig IS paksha × jupiter_dignity (2-variable already encoded)
CORE_CATS = [c for c in [
    'paksha','tithi_quality','nak_mo_name','vara_lord','dig_Ju','dig_Sa',
    'elem_Mo','mod_Mo','tara_quality','sade_sati_phase',
    'ix_paksha_ju_dig','ix_paksha_nak','ix_ju_dig_moon_sign',
] if c in df_clean.columns]
CORE_BINS = [c+'_s' for c in [
    'gajakesari','kemadruma','comb_Mo','retro_Me','retro_Ju',
    'gand_Mo','sade_sati','ju_asp_mo','sa_asp_mo','papakartari',
] if (c+'_s') in df_clean.columns]
CORE_FEAT = CORE_CATS + CORE_BINS  # max ~23 features

m2_results = []
N   = len(df_clean); k_t = df_clean['is_bull'].sum()
# k=4,5 are implicitly covered by the interaction features ix_*
min_n_map = {1:30, 2:30, 3:20}

for k, min_n in min_n_map.items():
    n_tested = 0
    for combo in itertools.combinations(CORE_FEAT, k):
        n_tested += 1
        if k == 1:
            key_series = df_clean[combo[0]].astype(str)
        else:
            key_series = df_clean[combo[0]].astype(str)
            for c in combo[1:]: key_series = key_series + '||' + df_clean[c].astype(str)

        for kv, idx in df_clean.groupby(key_series).groups.items():
            n_c = len(idx)
            if n_c < min_n: continue
            k_c = df_clean.loc[idx,'is_bull'].sum()
            wr  = k_c/n_c; wlb = wilson_lower(n_c,k_c); wub = wilson_upper(n_c,k_c)
            p   = fisher_p(n_c, k_c, N, k_t)
            m2_results.append({'features':'|'.join(combo),'condition':kv,'n':n_c,'k_pos':k_c,
                                'win_rate':round(wr,4),'wilson_lower':round(wlb,4),
                                'wilson_upper':round(wub,4),'p_value':round(p,8),'complexity':k})
    print(f"  k={k} ({n_tested} combos) done")

m2 = pd.DataFrame(m2_results).sort_values('p_value') if m2_results else pd.DataFrame()
m2.to_csv(f"{REPO}/results/research/method2_reverse_lookup.csv", index=False)
print(f"Method 2 complete: {len(m2)} conditions in {time.time()-t1:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# METHOD 3: ASTROLOGICAL SIMILARITY CLUSTERING
# ═══════════════════════════════════════════════════════════════════════
print("\n=== METHOD 3: Astrological Similarity Clustering ===")

cluster_cats = [c for c in ['paksha','tithi_quality','nak_mo_name','yoga_quality','vara_lord',
                              'dig_Ju','dig_Sa','dig_Mo','elem_Mo','mod_Mo','tara_quality',
                              'sade_sati_phase','nakl_dig','hora_at_open','choghadiya_quality']
               if c in df_clean.columns]
cluster_bins = [c for c in ['gajakesari','kemadruma','papakartari','comb_Mo','comb_Ve',
                              'retro_Me','retro_Ju','gand_Mo','sade_sati','ju_asp_mo','sa_asp_mo']
               if c in df_clean.columns]

encoded = []
for col in cluster_cats:
    le = LabelEncoder()
    encoded.append(le.fit_transform(df_clean[col].astype(str)))
for col in cluster_bins:
    encoded.append(df_clean[col].fillna(0).astype(float).values)

X = np.column_stack(encoded)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
import os as _os
_os.environ['OMP_NUM_THREADS'] = '1'
km = KMeans(n_clusters=8, random_state=42, n_init=10, max_iter=300, algorithm='lloyd')
df_clean['cluster'] = km.fit_predict(X_scaled)

cluster_stats = []
for c in range(8):
    sub = df_clean[df_clean['cluster']==c]
    n = len(sub); k = sub['is_bull'].sum()
    wr = k/n; wlb = wilson_lower(n,k)
    p = fisher_p(n, k, len(df_clean), int(df_clean['is_bull'].sum()))
    top_pak = sub['paksha'].value_counts().index[0] if 'paksha' in sub else ''
    top_nak = sub['nak_mo_name'].value_counts().index[0] if 'nak_mo_name' in sub else ''
    top_ju  = sub['dig_Ju'].value_counts().index[0] if 'dig_Ju' in sub else ''
    cluster_stats.append({
        'cluster':c,'n':n,'bull_rate':round(wr,4),'wilson_lower':round(wlb,4),
        'strong_bull_pct':round(sub['is_strong_bull'].sum()/n,4),
        'strong_bear_pct':round(sub['is_strong_bear'].sum()/n,4),
        'high_vol_pct':round(sub['is_high_vol'].sum()/n,4),
        'p_value':round(p,6),'dominant_paksha':top_pak,'dominant_nak':top_nak,'dominant_ju_dig':top_ju,
        'character':('BULL' if wr>BASE_BULL+0.05 else 'BEAR' if wr<BASE_BULL-0.05 else 'NEUTRAL')
    })

m3 = pd.DataFrame(cluster_stats).sort_values('bull_rate', ascending=False)
m3.to_csv(f"{REPO}/results/research/method3_clustering.csv", index=False)
print(f"Method 3: 8 clusters")
print(m3[['cluster','n','bull_rate','character','dominant_paksha','dominant_nak','dominant_ju_dig']].to_string())

# ═══════════════════════════════════════════════════════════════════════
# METHOD 4: PLANETARY CYCLE PERIOD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("\n=== METHOD 4: Planetary Cycle Period Analysis ===")
from statsmodels.tsa.stattools import acf
from scipy.fft import rfft, rfftfreq

ret = df_clean['fwd_ret_1d'].dropna().values
N_r = len(ret)
acf_vals, acf_ci = acf(ret, nlags=800, alpha=0.05)
freqs = rfftfreq(N_r, d=1.0)
power = np.abs(rfft(ret - ret.mean()))**2

CYCLES = {
    'Moon synodic(td)': 21.1,'Moon monthly': 29.5,'Mercury retro(td)': 83,
    'Mercury synodic': 116,'Venus synodic(td)': 161,'Mars synodic(td)': 557,
    'Jupiter sign(td)': 22,'Saturn sign(td)': 630,'Rahu sign(td)': 390,
}

cycle_results = []
for name, ptd in CYCLES.items():
    ai = int(round(ptd))
    acf_val = acf_vals[ai] if 0 < ai < len(acf_vals) else 0
    acf_sig = abs(acf_val) > 2/math.sqrt(N_r)
    tf = 1.0/ptd; fi = np.argmin(np.abs(freqs-tf))
    pwr = float(power[fi]/power.mean()) if power.mean()>0 else 0
    day_nums = np.arange(len(ret))
    phase = day_nums % int(round(ptd))
    q1 = ret[phase < ptd/4]; q2 = ret[(phase>=ptd/4)&(phase<ptd/2)]
    q3 = ret[(phase>=ptd/2)&(phase<3*ptd/4)]; q4 = ret[phase>=3*ptd/4]
    if all(len(q)>10 for q in [q1,q2,q3,q4]):
        f_stat, f_p = stats.f_oneway(q1,q2,q3,q4)
    else: f_stat, f_p = 0, 1
    cycle_results.append({'cycle_name':name,'period_td':round(ptd,1),'acf_value':round(acf_val,4),
                           'acf_significant':acf_sig,'fft_power_ratio':round(pwr,2),
                           'phase_anova_f':round(f_stat,3),'phase_anova_p':round(f_p,6),
                           'evidence':'YES' if (acf_sig or pwr>3 or f_p<0.05) else 'NO'})

m4 = pd.DataFrame(cycle_results)
m4.to_csv(f"{REPO}/results/research/method4_cycle_analysis.csv", index=False)
print(m4[['cycle_name','period_td','acf_significant','fft_power_ratio','phase_anova_p','evidence']].to_string())

# ═══════════════════════════════════════════════════════════════════════
# METHOD 5: SEQUENTIAL PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════
print("\n=== METHOD 5: Sequential Pattern Detection ===")

df_clean['is_new_moon']  = (df_clean['tithi_num'] == 30).astype(int)
df_clean['is_full_moon'] = (df_clean['tithi_num'] == 15).astype(int)

def test_seq(df, cond_col, cond_val, out_col, lags, min_n=10):
    mask = df[cond_col].astype(str) == str(cond_val)
    idxs = df.index[mask].tolist()
    results = []
    N = len(df); k_base = df[out_col].sum()
    for lag in lags:
        oidxs = [i+lag for i in idxs if i+lag < len(df)]
        if len(oidxs) < min_n: continue
        k = df.loc[oidxs, out_col].sum()
        n = len(oidxs); wr = k/n
        wlb = wilson_lower(n,k); p = fisher_p(n,k,N,k_base)
        results.append({'condition_col':cond_col,'condition_val':cond_val,'outcome':out_col,
                         'lag':lag,'n':n,'k':k,'win_rate':round(wr,4),'wilson_lower':round(wlb,4),'p_value':round(p,6)})
    return results

seq_results = []
SEQUENCES = [
    ('graha_yuddha',1,'is_bull',[1,2,3,4,5]),
    ('gand_Mo',1,'is_bull',[1,2,3]),
    ('nak_transition',1,'is_bull',[1,2]),
    ('sade_sati',1,'is_bull',[1,2,3,4,5]),
    ('ashtama_shani',1,'is_bull',[1,2,3,4,5]),
    ('kemadruma',1,'is_bull',[1,2,3]),
    ('gajakesari',1,'is_bull',[1,2,3]),
    ('papakartari',1,'is_bull',[1,2,3]),
    ('is_new_moon',1,'is_bull',list(range(1,16))),
    ('is_full_moon',1,'is_bull',list(range(1,16))),
    ('is_new_moon',1,'is_high_vol',[1,2,3,4,5]),
    ('retro_Me',1,'is_bull',[1,2,3,5,7,10,14,21]),
    ('retro_Me',1,'is_high_vol',[1,2,3,5,7,10,14,21]),
    ('ingress_Ju',1,'is_bull',list(range(1,31,2))) if 'ingress_Ju' in df_clean.columns else None,
    ('ingress_Sa',1,'is_bull',list(range(1,91,5))) if 'ingress_Sa' in df_clean.columns else None,
]
for seq in SEQUENCES:
    if seq is None: continue
    cond_col,cond_val,out_col,lags = seq
    if cond_col not in df_clean.columns: continue
    seq_results.extend(test_seq(df_clean, cond_col, cond_val, out_col, lags))

# Nakshatra lag patterns
for nak in range(1,28):
    df_clean[f'nak_{nak}'] = (df_clean['nak_mo']==nak).astype(int)
    seq_results.extend(test_seq(df_clean, f'nak_{nak}', 1, 'is_bull', [1,2,3], min_n=10))

# Tithi lag patterns
for t in [1,8,14,15,23,29,30]:
    df_clean[f'tit_{t}'] = (df_clean['tithi_num']==t).astype(int)
    seq_results.extend(test_seq(df_clean, f'tit_{t}', 1, 'is_bull', [1,2,3]))

m5 = pd.DataFrame(seq_results).drop_duplicates()
if len(m5)>0: m5 = m5.sort_values('p_value')
m5.to_csv(f"{REPO}/results/research/method5_sequential_patterns.csv", index=False)
sig5 = m5[m5['p_value']<0.05] if len(m5)>0 else m5
print(f"Method 5: {len(m5)} tests, {len(sig5)} significant (p<0.05)")

# ═══════════════════════════════════════════════════════════════════════
# METHOD 6: ANOMALY FINGERPRINTING
# ═══════════════════════════════════════════════════════════════════════
print("\n=== METHOD 6: Anomaly Fingerprinting ===")
ret_s = df_clean['fwd_ret_1d'].fillna(0)
rm = ret_s.rolling(20).mean().shift(1); rs = ret_s.rolling(20).std().shift(1)
z = (ret_s - rm) / (rs + 1e-8)
df_clean['anomaly'] = (z.abs() > 2.0).astype(int)
n_anom = df_clean['anomaly'].sum()
print(f"  Anomaly days (|z|>2): {n_anom}")

anom_results = []
N_a = len(df_clean); k_a = int(df_clean['anomaly'].sum())
for col in CAT_FEATURES[:20] + BIN_STR_FEAT[:15]:
    if col not in df_clean.columns: continue
    for val, idx in df_clean.groupby(df_clean[col].astype(str)).groups.items():
        n_c = len(idx)
        if n_c < 10: continue
        k_c = df_clean.loc[idx,'anomaly'].sum()
        wr = k_c/n_c; wlb = wilson_lower(n_c,k_c)
        p = fisher_p(n_c, k_c, N_a, k_a)
        if p < 0.10:
            anom_results.append({'feature':col,'value':val,'n':n_c,'k_anomaly':k_c,
                                  'anomaly_rate':round(wr,4),'wilson_lower':round(wlb,4),'p_value':round(p,6)})

m6 = pd.DataFrame(anom_results).sort_values('p_value') if anom_results else pd.DataFrame()
m6.to_csv(f"{REPO}/results/research/method6_anomaly_fingerprints.csv", index=False)
print(f"Method 6: {len(m6)} anomaly fingerprints")

print("\n=== STEP 2 COMPLETE ===")
print(f"Method 1: {len(m1_all)} patterns")
print(f"Method 2: {len(m2)} conditions")
print(f"Method 3: 8 clusters")
print(f"Method 4: {len(m4)} cycle tests")
print(f"Method 5: {len(m5)} sequences, {len(sig5)} significant")
print(f"Method 6: {len(m6)} anomaly fingerprints")
