"""
Step 2 continued — Methods 3-6 only (Methods 1-2 already complete)
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import math, warnings, itertools, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.stattools import acf

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
os.makedirs(f"{REPO}/results/research", exist_ok=True)

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0,(p+z**2/(2*n)-z*math.sqrt(max(0,p*(1-p)/n+z**2/(4*n**2))))/(1+z**2/n))

def fisher_p(n_cond, k_bull, n_total, k_total_bull):
    a=k_bull; b=n_cond-k_bull; c=k_total_bull-k_bull; d=(n_total-n_cond)-(k_total_bull-k_bull)
    if any(x<0 for x in [a,b,c,d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a),max(0,b)],[max(0,c),max(0,d)]])
    return p

print("Loading data …")
df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df_clean = df.dropna(subset=['is_bull']).copy()
df_clean['is_bull'] = df_clean['is_bull'].astype(int)
for c in ['is_strong_bull','is_strong_bear','is_high_vol','is_reversal']:
    if c in df_clean.columns:
        df_clean[c] = df_clean[c].fillna(0).astype(int)
N = len(df_clean)
BASE_BULL = df_clean['is_bull'].mean()
print(f"Rows: {N}")

# ═══════════════════════════════════════════════════════════════════════
# METHOD 3: CLUSTERING (using scipy instead of sklearn to avoid threadpool bug)
# ═══════════════════════════════════════════════════════════════════════
print("\n=== METHOD 3: Clustering ===")
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

X = np.column_stack(encoded).astype(np.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use scipy's vq (no threadpool issue)
from scipy.cluster.vq import kmeans2, whiten
X_w = whiten(X_scaled)
_, labels = kmeans2(X_w, 8, seed=42, minit='points', iter=100)
df_clean['cluster'] = labels

cluster_stats = []
for c in range(8):
    sub = df_clean[df_clean['cluster']==c]
    n = len(sub); k = sub['is_bull'].sum()
    wr = k/n if n>0 else 0; wlb = wilson_lower(n,k)
    p = fisher_p(n, k, N, int(df_clean['is_bull'].sum()))
    top_pak = sub['paksha'].value_counts().index[0] if 'paksha' in sub.columns and len(sub)>0 else ''
    top_nak = sub['nak_mo_name'].value_counts().index[0] if 'nak_mo_name' in sub.columns and len(sub)>0 else ''
    top_ju  = sub['dig_Ju'].value_counts().index[0] if 'dig_Ju' in sub.columns and len(sub)>0 else ''
    cluster_stats.append({
        'cluster':c,'n':n,'bull_rate':round(wr,4),'wilson_lower':round(wlb,4),
        'strong_bull_pct':round(sub.get('is_strong_bull',pd.Series([0]*n)).sum()/n,4) if 'is_strong_bull' in sub else 0,
        'strong_bear_pct':round(sub.get('is_strong_bear',pd.Series([0]*n)).sum()/n,4) if 'is_strong_bear' in sub else 0,
        'high_vol_pct':round(sub.get('is_high_vol',pd.Series([0]*n)).sum()/n,4) if 'is_high_vol' in sub else 0,
        'p_value':round(p,6),'dominant_paksha':top_pak,'dominant_nak':top_nak,'dominant_ju_dig':top_ju,
        'character':('BULL' if wr>BASE_BULL+0.05 else 'BEAR' if wr<BASE_BULL-0.05 else 'NEUTRAL')
    })

m3 = pd.DataFrame(cluster_stats).sort_values('bull_rate', ascending=False)
m3.to_csv(f"{REPO}/results/research/method3_clustering.csv", index=False)
print(f"Method 3 complete: 8 clusters")
print(m3[['cluster','n','bull_rate','character','dominant_paksha','dominant_nak','dominant_ju_dig']].to_string())

# ═══════════════════════════════════════════════════════════════════════
# METHOD 4: PLANETARY CYCLE PERIOD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("\n=== METHOD 4: Planetary Cycle Period Analysis ===")
from scipy.fft import rfft, rfftfreq

ret = df_clean['fwd_ret_1d'].dropna().values
N_r = len(ret)
try:
    acf_vals, acf_ci = acf(ret, nlags=800, alpha=0.05)
except: acf_vals = np.zeros(801); acf_ci = np.zeros((801,2))
freqs = rfftfreq(N_r, d=1.0)
power = np.abs(rfft(ret - ret.mean()))**2

CYCLES = {
    'Moon synodic(td)': 21.1, 'Moon monthly': 29.5, 'Mercury retro(td)': 83,
    'Mercury synodic': 116, 'Venus synodic(td)': 161, 'Mars synodic(td)': 557,
    'Jupiter sign(td)': 22, 'Saturn sign(td)': 630, 'Rahu sign(td)': 390,
}
cycle_results = []
for name, ptd in CYCLES.items():
    ai = int(round(ptd))
    acf_val = float(acf_vals[ai]) if 0 < ai < len(acf_vals) else 0
    acf_sig = abs(acf_val) > 2/math.sqrt(N_r)
    tf = 1.0/ptd; fi = np.argmin(np.abs(freqs-tf))
    pwr = float(power[fi]/power.mean()) if power.mean()>0 else 0
    day_nums = np.arange(len(ret)); phase = day_nums % int(round(ptd))
    q1=ret[phase<ptd/4]; q2=ret[(phase>=ptd/4)&(phase<ptd/2)]
    q3=ret[(phase>=ptd/2)&(phase<3*ptd/4)]; q4=ret[phase>=3*ptd/4]
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
    if cond_col not in df.columns or out_col not in df.columns: return []
    mask = df[cond_col].astype(str) == str(cond_val)
    idxs = df.index[mask].tolist()
    N = len(df); k_base = df[out_col].sum()
    results = []
    for lag in lags:
        oidxs = [i+lag for i in idxs if i+lag < len(df)]
        if len(oidxs) < min_n: continue
        k = df.loc[oidxs, out_col].sum(); n = len(oidxs)
        wlb = wilson_lower(n,k); p = fisher_p(n,k,N,k_base)
        results.append({'condition_col':cond_col,'condition_val':cond_val,'outcome':out_col,
                         'lag':lag,'n':n,'k':k,'win_rate':round(k/n,4),
                         'wilson_lower':round(wlb,4),'p_value':round(p,6)})
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
]
if 'ingress_Ju' in df_clean.columns:
    SEQUENCES.append(('ingress_Ju',1,'is_bull',list(range(1,31,2))))
if 'ingress_Sa' in df_clean.columns:
    SEQUENCES.append(('ingress_Sa',1,'is_bull',list(range(1,91,5))))

for cond_col,cond_val,out_col,lags in SEQUENCES:
    seq_results.extend(test_seq(df_clean, cond_col, cond_val, out_col, lags))

for nak in range(1,28):
    df_clean[f'nak_{nak}'] = (df_clean['nak_mo']==nak).astype(int)
    seq_results.extend(test_seq(df_clean, f'nak_{nak}', 1, 'is_bull', [1,2,3], min_n=10))

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
z = (ret_s - rm)/(rs+1e-8)
df_clean['anomaly'] = (z.abs()>2.0).astype(int)
n_anom = df_clean['anomaly'].sum()
print(f"  Anomaly days: {n_anom}")

CAT_FEAT_6 = [c for c in ['paksha','tithi_quality','nak_mo_name','vara_lord','dig_Ju',
                            'dig_Sa','elem_Mo','mod_Mo','tara_quality','yoga_quality',
                            'ix_paksha_ju_dig','ix_paksha_nak','sade_sati_phase']
              if c in df_clean.columns]
BIN_FEAT_6 = [c+'_s' for c in ['gajakesari','comb_Mo','retro_Me','gand_Mo','sade_sati','papakartari']
              if (c+'_s') in df_clean.columns]
for c in BIN_FEAT_6:
    if c not in df_clean.columns:
        col = c.replace('_s','')
        if col in df_clean.columns: df_clean[c] = col+'='+df_clean[col].astype(str)

anom_results = []
N_a = N; k_a = int(df_clean['anomaly'].sum())
for col in CAT_FEAT_6 + BIN_FEAT_6:
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

print("\n=== STEP 2b COMPLETE ===")
print(f"Method 3: 8 clusters, saved")
print(f"Method 4: {len(m4)} cycle tests, saved")
print(f"Method 5: {len(m5)} sequence tests, saved")
print(f"Method 6: {len(m6)} anomaly fingerprints, saved")
