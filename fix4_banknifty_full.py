"""
Fix 4: Full six-method research on Bank Nifty independently.

Runs the complete pipeline on banknifty_enriched.csv.
Then compares: universal patterns (Nifty + BankNifty), Nifty-only, BankNifty-only.

Output files:
  results/research/bnk_method1_fp.csv
  results/research/bnk_method2_conditions.csv
  results/research/bnk_method3_clustering.csv
  results/research/bnk_method4_cycles.csv
  results/research/bnk_method5_sequential.csv
  results/research/bnk_method6_anomaly.csv
  results/validation/bnk_confirmed_patterns.csv
  results/validation/cross_instrument_comparison.csv
"""
import math, os, warnings, itertools, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.vq import kmeans2, whiten
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.stattools import acf

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
os.makedirs(f"{REPO}/results/research", exist_ok=True)
os.makedirs(f"{REPO}/results/validation", exist_ok=True)

PLANETS = ['Su','Mo','Me','Ve','Ma','Ju','Sa','Ra','Ke']
OOS_SPLIT = '2018-01-01'

# ──────────────────────────────────────────────────────────────────────
def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0, (p + z**2/(2*n) - z*math.sqrt(max(0, p*(1-p)/n + z**2/(4*n**2)))) / (1 + z**2/n))

def fisher_p(n_c, k_c, N, k_t):
    a = k_c; b = n_c - k_c; c = k_t - k_c; d = (N - n_c) - (k_t - k_c)
    if any(x < 0 for x in [a,b,c,d]): return 1.0
    _, p = stats.fisher_exact([[max(0,a),max(0,b)],[max(0,c),max(0,d)]])
    return p

def bh_fdr(pvalues, alpha=0.01):
    n = len(pvalues)
    if n == 0: return []
    ranked = np.argsort(pvalues)
    p_sorted = np.array(pvalues)[ranked]
    threshold = np.arange(1, n+1) * alpha / n
    survive = p_sorted <= threshold
    if survive.any():
        last = np.where(survive)[0][-1]
        survive[:last+1] = True
    result = np.zeros(n, dtype=bool)
    result[ranked] = survive
    return result.tolist()

# ──────────────────────────────────────────────────────────────────────
print("Loading banknifty_enriched.csv …")
df = pd.read_csv(f"{REPO}/data/banknifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

outcome_cols = ['is_bull','is_strong_bull','is_strong_bear','is_sideways','is_high_vol','is_reversal']
df_clean = df.dropna(subset=[c for c in outcome_cols if c in df.columns]).copy()
for c in outcome_cols:
    if c in df_clean.columns:
        df_clean[c] = df_clean[c].astype(int)

df_train = df_clean[df_clean['date'] < OOS_SPLIT].copy()
df_oos   = df_clean[df_clean['date'] >= OOS_SPLIT].copy()

N = len(df_clean); N_tr = len(df_train); N_oos = len(df_oos)
BASE_BULL = df_clean['is_bull'].mean()
print(f"  Rows: {N} (train={N_tr}, OOS={N_oos}), base_bull={BASE_BULL:.3f}")

# Feature pool (same as Nifty)
CAT_FEATURES = [c for c in [
    'paksha','tithi_quality','moon_phase','nak_mo_name','nak_mo_qual',
    'yoga_quality','karana_quality','vara_lord','vara_dig',
    'dig_Ju','dig_Sa','dig_Mo','dig_Ma','dig_Ve','dig_Me',
    'elem_Mo','mod_Mo','tara_quality','mahadasha','sade_sati_phase',
    'nakl_dig','nakl_spd','choghadiya_quality','hora_at_open',
    'ix_paksha_ju_dig','ix_paksha_nak','ix_paksha_moon_sign',
    'ix_tithi_nak','ix_ju_dig_moon_sign','ix_vara_paksha',
    'true_node_cat','cheshta_cat_Ju','cheshta_cat_Sa',
    'ix_ju_speed_dig','ix_sa_speed_dig',
] if c in df_clean.columns]

BIN_FEATURES = [c for c in [
    'gajakesari','kemadruma','papakartari','shubhakartari',
    'chandra_mangala','sakata','neecha_bhanga',
    'argala_positive','argala_obstruct','vipareeta_raja',
    'parivartana_any','graha_yuddha',
    'comb_Mo','comb_Me','comb_Ve','comb_Ma','comb_Ju',
    'retro_Me','retro_Ju','retro_Sa',
    'gand_Mo','gand_any','sandhi_mo','nak_transition','panchaka',
    'sade_sati','ashtama_shani','ju_asp_mo','sa_asp_mo','ma_asp_mo',
    'own_nak_Ju','own_nak_Mo','own_nak_Sa',
] if c in df_clean.columns]

for col in BIN_FEATURES:
    if (col + '_s') not in df_clean.columns:
        df_clean[col + '_s'] = col + '=' + df_clean[col].astype(str)
        df_train[col + '_s'] = col + '=' + df_train[col].astype(str)
        df_oos[col + '_s']   = col + '=' + df_oos[col].astype(str)

BIN_STR = [c + '_s' for c in BIN_FEATURES if (c + '_s') in df_clean.columns]
ALL_FEAT = CAT_FEATURES + BIN_STR
print(f"  Feature pool: {len(ALL_FEAT)} columns")

# ──────────────────────────────────────────────────────────────────────
# METHOD 1 (Fingerprint) + METHOD 2 (Conditions) on Bank Nifty train
# ──────────────────────────────────────────────────────────────────────
print("\n=== METHOD 1+2: Pattern Scanning (training data only) ===")
all_pats = []

def scan_outcome(df_tr, outcome_col, label, min_n=5, pval_thresh=0.05):
    N = len(df_tr); k_t = int(df_tr[outcome_col].sum())
    result = []
    sig_cols = []
    for col in ALL_FEAT:
        if col not in df_tr.columns: continue
        for val, grp in df_tr.groupby(df_tr[col].astype(str))[outcome_col]:
            n_c = len(grp); k_c = int(grp.sum())
            if n_c < min_n: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                wlb = wilson_lower(n_c, k_c)
                result.append({'features':col,'condition':val,'outcome':label,
                                'n':n_c,'k_pos':k_c,'win_rate':round(k_c/n_c,4),
                                'wilson_lower':round(wlb,4),'p_value':round(p,8),
                                'complexity':1})
            if p < 0.15 and col not in sig_cols: sig_cols.append(col)
    for c1,c2 in itertools.combinations(sig_cols, 2):
        key = df_tr[c1].astype(str)+'||'+df_tr[c2].astype(str)
        for kv, idx in df_tr.groupby(key).groups.items():
            n_c=len(idx); k_c=int(df_tr.loc[idx,outcome_col].sum())
            if n_c < min_n: continue
            p = fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                wlb=wilson_lower(n_c,k_c)
                result.append({'features':f'{c1}|{c2}','condition':kv,'outcome':label,
                                'n':n_c,'k_pos':k_c,'win_rate':round(k_c/n_c,4),
                                'wilson_lower':round(wlb,4),'p_value':round(p,8),
                                'complexity':2})
    for c1,c2,c3 in itertools.combinations(sig_cols[:30], 3):
        key=(df_tr[c1].astype(str)+'||'+df_tr[c2].astype(str)+'||'+df_tr[c3].astype(str))
        for kv, idx in df_tr.groupby(key).groups.items():
            n_c=len(idx); k_c=int(df_tr.loc[idx,outcome_col].sum())
            if n_c < min_n: continue
            p=fisher_p(n_c, k_c, N, k_t)
            if p < pval_thresh:
                wlb=wilson_lower(n_c,k_c)
                result.append({'features':f'{c1}|{c2}|{c3}','condition':kv,'outcome':label,
                                'n':n_c,'k_pos':k_c,'win_rate':round(k_c/n_c,4),
                                'wilson_lower':round(wlb,4),'p_value':round(p,8),
                                'complexity':3})
    print(f"  {label}: {len(result)} patterns (sig_cols={len(sig_cols)})")
    return result

for out_col, label in [('is_bull','BULL_DIR'),('is_strong_bull','STRONG_BULL'),
                        ('is_strong_bear','STRONG_BEAR'),('is_high_vol','HIGH_VOL')]:
    if out_col in df_train.columns:
        all_pats.extend(scan_outcome(df_train, out_col, label))

m12 = pd.DataFrame(all_pats).drop_duplicates(subset=['features','condition'])
m12.to_csv(f"{REPO}/results/research/bnk_method1_fp.csv", index=False)
print(f"  M1+M2 total: {len(m12)} patterns")

# ──────────────────────────────────────────────────────────────────────
# METHODS 3-6
# ──────────────────────────────────────────────────────────────────────
print("\n=== METHOD 3: Clustering ===")
cluster_cats = [c for c in ['paksha','tithi_quality','nak_mo_name','yoga_quality','vara_lord',
                              'dig_Ju','dig_Sa','dig_Mo','elem_Mo','mod_Mo','tara_quality',
                              'sade_sati_phase','nakl_dig','hora_at_open','choghadiya_quality']
               if c in df_clean.columns]
cluster_bins = [c for c in ['gajakesari','kemadruma','papakartari','comb_Mo','comb_Ve',
                              'retro_Me','retro_Ju','gand_Mo','sade_sati','ju_asp_mo','sa_asp_mo']
               if c in df_clean.columns]
enc = []
for col in cluster_cats:
    le = LabelEncoder()
    enc.append(le.fit_transform(df_clean[col].astype(str)))
for col in cluster_bins:
    enc.append(df_clean[col].fillna(0).astype(float).values)
X = np.column_stack(enc).astype(np.float32)
Xw = whiten(StandardScaler().fit_transform(X))
_, labels = kmeans2(Xw, 8, seed=42, minit='points', iter=100)
df_clean['cluster'] = labels
N_all = len(df_clean); k_base = int(df_clean['is_bull'].sum())
m3_rows = []
for c in range(8):
    sub = df_clean[df_clean['cluster']==c]
    n=len(sub); k=int(sub['is_bull'].sum())
    wr=k/n if n>0 else 0; wlb=wilson_lower(n,k)
    p=fisher_p(n,k,N_all,k_base)
    char = 'BULL' if wr>BASE_BULL+0.05 else ('BEAR' if wr<BASE_BULL-0.05 else 'NEUTRAL')
    m3_rows.append({'cluster':c,'n':n,'bull_rate':round(wr,4),'wilson_lower':round(wlb,4),
                    'p_value':round(p,6),'character':char,
                    'top_paksha': sub['paksha'].mode()[0] if 'paksha' in sub else '',
                    'top_nak': sub['nak_mo_name'].mode()[0] if 'nak_mo_name' in sub else '',
                    'top_ju': sub['dig_Ju'].mode()[0] if 'dig_Ju' in sub else ''})
m3 = pd.DataFrame(m3_rows)
m3.to_csv(f"{REPO}/results/research/bnk_method3_clustering.csv", index=False)
print(m3[['cluster','n','bull_rate','character','top_paksha','top_nak','top_ju']].to_string())

print("\n=== METHOD 4: Cycle Analysis ===")
ret = df_clean['fwd_ret_1d'].dropna().values; N_r = len(ret)
try: acf_vals, _ = acf(ret, nlags=800, alpha=0.05)
except: acf_vals = np.zeros(801)
freqs = rfftfreq(N_r); power = np.abs(rfft(ret - ret.mean()))**2
CYCLES = {'Moon_monthly':29.5,'Mercury_synodic':116,'Venus_synodic':161,'Rahu_sign':390}
m4_rows = []
for name, ptd in CYCLES.items():
    ai = int(round(ptd)); acf_val = float(acf_vals[ai]) if 0 < ai < len(acf_vals) else 0
    acf_sig = abs(acf_val) > 2/math.sqrt(N_r)
    tf=1/ptd; fi=np.argmin(np.abs(freqs-tf)); pwr=float(power[fi]/power.mean()) if power.mean()>0 else 0
    day_nums=np.arange(len(ret)); phase=day_nums%int(round(ptd))
    q1=ret[phase<ptd/4]; q2=ret[(phase>=ptd/4)&(phase<ptd/2)]
    q3=ret[(phase>=ptd/2)&(phase<3*ptd/4)]; q4=ret[phase>=3*ptd/4]
    f_stat,f_p = (stats.f_oneway(q1,q2,q3,q4) if all(len(q)>10 for q in [q1,q2,q3,q4]) else (0,1))
    m4_rows.append({'cycle_name':name,'period_td':round(ptd,1),'acf_value':round(acf_val,4),
                    'acf_significant':acf_sig,'fft_power_ratio':round(pwr,2),
                    'phase_anova_p':round(f_p,6),'evidence':'YES' if (acf_sig or pwr>3 or f_p<0.05) else 'NO'})
m4 = pd.DataFrame(m4_rows)
m4.to_csv(f"{REPO}/results/research/bnk_method4_cycles.csv", index=False)
print(m4[['cycle_name','acf_significant','fft_power_ratio','phase_anova_p','evidence']].to_string())

print("\n=== METHOD 5: Sequential Patterns ===")
df_clean['is_new_moon']  = (df_clean['tithi_num'] == 30).astype(int)
df_clean['is_full_moon'] = (df_clean['tithi_num'] == 15).astype(int)
def test_seq(df, cond_col, cond_val, out_col, lags, min_n=10):
    if cond_col not in df.columns or out_col not in df.columns: return []
    mask = df[cond_col].astype(str) == str(cond_val)
    idxs = df.index[mask].tolist(); N=len(df); k_base=df[out_col].sum()
    results = []
    for lag in lags:
        oidxs = [i+lag for i in idxs if i+lag < len(df)]
        if len(oidxs) < min_n: continue
        k=df.loc[oidxs,out_col].sum(); n=len(oidxs)
        wlb=wilson_lower(n,k); p=fisher_p(n,k,N,k_base)
        results.append({'condition_col':cond_col,'condition_val':cond_val,'outcome':out_col,
                         'lag':lag,'n':n,'k':k,'win_rate':round(k/n,4),
                         'wilson_lower':round(wlb,4),'p_value':round(p,6)})
    return results
m5_rows = []
for cond_col,cond_val,out_col,lags in [
    ('is_new_moon',1,'is_bull',list(range(1,16))),
    ('is_full_moon',1,'is_bull',list(range(1,16))),
    ('sade_sati',1,'is_bull',[1,2,3,4,5]),
    ('retro_Me',1,'is_bull',[1,2,3,5,7,10,14]),
    ('gand_Mo',1,'is_bull',[1,2,3]),
    ('graha_yuddha',1,'is_bull',[1,2,3,4,5]),
]:
    if cond_col in df_clean.columns:
        m5_rows.extend(test_seq(df_clean,cond_col,cond_val,out_col,lags))
m5 = pd.DataFrame(m5_rows).sort_values('p_value') if m5_rows else pd.DataFrame()
m5.to_csv(f"{REPO}/results/research/bnk_method5_sequential.csv", index=False)
print(f"Method 5: {len(m5)} tests, {(m5['p_value']<0.05).sum() if len(m5)>0 else 0} significant (p<0.05)")

print("\n=== METHOD 6: Anomaly Fingerprinting ===")
ret_s = df_clean['fwd_ret_1d'].fillna(0)
rm=ret_s.rolling(20).mean().shift(1); rs=ret_s.rolling(20).std().shift(1)
z=(ret_s-rm)/(rs+1e-8)
df_clean['anomaly']=(z.abs()>2.0).astype(int)
N_a=len(df_clean); k_a=int(df_clean['anomaly'].sum())
m6_rows=[]
for col in CAT_FEATURES[:15]:
    if col not in df_clean.columns: continue
    for val, idx in df_clean.groupby(df_clean[col].astype(str)).groups.items():
        n_c=len(idx)
        if n_c<10: continue
        k_c=int(df_clean.loc[idx,'anomaly'].sum()); wr=k_c/n_c
        wlb=wilson_lower(n_c,k_c); p=fisher_p(n_c,k_c,N_a,k_a)
        if p<0.10:
            m6_rows.append({'feature':col,'value':val,'n':n_c,'k_anomaly':k_c,
                            'anomaly_rate':round(wr,4),'wilson_lower':round(wlb,4),'p_value':round(p,6)})
m6=pd.DataFrame(m6_rows).sort_values('p_value') if m6_rows else pd.DataFrame()
m6.to_csv(f"{REPO}/results/research/bnk_method6_anomaly.csv", index=False)
print(f"Method 6: {len(m6)} anomaly fingerprints")

# ──────────────────────────────────────────────────────────────────────
# VALIDATION: BH-FDR + OOS on Bank Nifty patterns
# ──────────────────────────────────────────────────────────────────────
print("\n=== VALIDATION: BH-FDR + OOS on Bank Nifty ===")
all_pvals = m12['p_value'].tolist()
if len(m5) > 0: all_pvals += m5['p_value'].tolist()
if len(m6) > 0: all_pvals += m6['p_value'].tolist()

survive = bh_fdr(all_pvals, alpha=0.01)
n_fdr = sum(survive)
print(f"  Total p-values: {len(all_pvals)}, FDR survivors: {n_fdr}")

# Apply to m12
m12_fdr = m12[bh_fdr(m12['p_value'].tolist(), alpha=0.01)].copy()
print(f"  M1+M2 FDR survivors: {len(m12_fdr)}")

# OOS test
def eval_oos(df_oos, features_str, condition_str, outcome_col='is_bull'):
    features = features_str.split('|')
    conds = condition_str.split('||') if '||' in str(condition_str) else [str(condition_str)]
    if len(features) != len(conds): return 0, 0
    mask = pd.Series(True, index=df_oos.index)
    for f, v in zip(features, conds):
        if f not in df_oos.columns: return 0, 0
        mask &= (df_oos[f].astype(str) == v.strip())
    n = int(mask.sum())
    if n < 3: return n, 0
    k = int(df_oos.loc[mask, outcome_col].sum() if outcome_col in df_oos.columns else 0)
    return n, k

BASE_BULL_OOS = df_oos['is_bull'].mean()
confirmed = []
for _, pat in m12_fdr.iterrows():
    # Determine outcome column
    oc = 'is_bull'
    if pat.get('outcome') == 'STRONG_BULL': oc = 'is_bull'
    elif pat.get('outcome') == 'STRONG_BEAR': oc = 'is_bull'
    elif pat.get('outcome') == 'HIGH_VOL': oc = 'is_high_vol'
    else: oc = 'is_bull'
    if oc not in df_oos.columns: oc = 'is_bull'

    n_oos, k_oos = eval_oos(df_oos, pat['features'], pat['condition'], oc)
    wr_oos = k_oos/n_oos if n_oos > 0 else 0

    # OOS pass: must maintain direction
    oos_pass = False
    if n_oos >= 3:
        if pat['win_rate'] > BASE_BULL and wr_oos > 0.45: oos_pass = True
        if pat['win_rate'] <= BASE_BULL and wr_oos < 0.55: oos_pass = True

    if oos_pass:
        signal_dir = 'BULL' if pat['win_rate'] > BASE_BULL else 'BEAR'
        confirmed.append({**pat.to_dict(),
                          'n_oos': n_oos, 'k_oos': k_oos, 'wr_oos': round(wr_oos,4),
                          'oos_pass': oos_pass, 'signal_dir': signal_dir,
                          'instrument': 'BANKNIFTY', 'source': 'bnk_method1'})

bnk_confirmed = pd.DataFrame(confirmed)
bnk_confirmed.to_csv(f"{REPO}/results/validation/bnk_confirmed_patterns.csv", index=False)
print(f"  Bank Nifty confirmed: {len(bnk_confirmed)} patterns")
if len(bnk_confirmed) > 0:
    print(f"    BULL: {(bnk_confirmed['signal_dir']=='BULL').sum()}")
    print(f"    BEAR: {(bnk_confirmed['signal_dir']=='BEAR').sum()}")

# ──────────────────────────────────────────────────────────────────────
# CROSS-INSTRUMENT COMPARISON
# ──────────────────────────────────────────────────────────────────────
print("\n=== CROSS-INSTRUMENT COMPARISON ===")
nifty_cp_path = f"{REPO}/results/validation/confirmed_patterns.csv"
if os.path.exists(nifty_cp_path) and len(bnk_confirmed) > 0:
    nifty_cp = pd.read_csv(nifty_cp_path)
    nifty_keys = set(nifty_cp['features'] + ':::' + nifty_cp['condition'])
    bnk_keys   = set(bnk_confirmed['features'] + ':::' + bnk_confirmed['condition'])

    universal   = nifty_keys & bnk_keys
    nifty_only  = nifty_keys - bnk_keys
    bnk_only    = bnk_keys - nifty_keys

    print(f"  Universal patterns (both instruments): {len(universal)}")
    print(f"  Nifty-only patterns:                   {len(nifty_only)}")
    print(f"  BankNifty-only patterns:               {len(bnk_only)}")

    comp_rows = []
    for key in universal:
        feat, cond = key.split(':::')
        nr = nifty_cp[(nifty_cp['features']==feat)&(nifty_cp['condition']==cond)].iloc[0]
        br = bnk_confirmed[(bnk_confirmed['features']==feat)&(bnk_confirmed['condition']==cond)].iloc[0]
        comp_rows.append({'features':feat,'condition':cond,
                          'nifty_wlb':nr.get('wlb_train',0),'nifty_wr':nr.get('wr_train',0),
                          'bnk_wlb':br.get('wilson_lower',0),'bnk_wr':br.get('win_rate',0),
                          'universal':True,'signal_dir':nr.get('signal_dir','?')})
    for key in nifty_only:
        feat,cond=key.split(':::')
        nr=nifty_cp[(nifty_cp['features']==feat)&(nifty_cp['condition']==cond)].iloc[0]
        comp_rows.append({'features':feat,'condition':cond,
                          'nifty_wlb':nr.get('wlb_train',0),'nifty_wr':nr.get('wr_train',0),
                          'bnk_wlb':None,'bnk_wr':None,'universal':False,'signal_dir':nr.get('signal_dir','?')})
    for key in bnk_only:
        feat,cond=key.split(':::')
        br=bnk_confirmed[(bnk_confirmed['features']==feat)&(bnk_confirmed['condition']==cond)].iloc[0]
        comp_rows.append({'features':feat,'condition':cond,
                          'nifty_wlb':None,'nifty_wr':None,
                          'bnk_wlb':br.get('wilson_lower',0),'bnk_wr':br.get('win_rate',0),
                          'universal':False,'signal_dir':br.get('signal_dir','?')})

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(f"{REPO}/results/validation/cross_instrument_comparison.csv", index=False)
    print(f"  Saved cross_instrument_comparison.csv ({len(comp_df)} rows)")

    if len(universal) > 0:
        print(f"\n  Top universal patterns (both instruments):")
        uni_rows = comp_df[comp_df['universal']==True].nlargest(10, 'nifty_wlb')
        for _, r in uni_rows.iterrows():
            print(f"    [{r['signal_dir']}] nifty_wlb={r['nifty_wlb']:.3f} bnk_wlb={r['bnk_wlb']:.3f} | {r['features'][:50]}")

print("\nFix 4 complete.")
