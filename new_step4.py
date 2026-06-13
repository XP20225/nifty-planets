"""
New Step 4 — System Build
Composite score, backtest, forward calendar using pyswisseph + confirmed patterns only.
No market data used in forward-looking signals.
"""
import json, math, os, warnings
from datetime import date, timedelta
import numpy as np
import pandas as pd
import swisseph as swe

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
os.makedirs(f"{REPO}/results/forward_calendar", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# HELPERS (mirror of step1 — needed for forward calendar computation)
# ═══════════════════════════════════════════════════════════════════════

def wilson_lower(n, k, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return max(0.0, (p + z**2/(2*n) - z*math.sqrt(max(0, p*(1-p)/n + z**2/(4*n**2)))) /
               (1 + z**2/n))

SIGN_ELEMENT  = {1:'F',2:'E',3:'A',4:'W',5:'F',6:'E',7:'A',8:'W',9:'F',10:'E',11:'A',12:'W'}
SIGN_MODALITY = {1:'M',2:'X',3:'D',4:'M',5:'X',6:'D',7:'M',8:'X',9:'D',10:'M',11:'X',12:'D'}
NAK_LORDS     = ['Ke','Ve','Su','Mo','Ma','Ra','Ju','Sa','Me'] * 3
NAK_NAMES     = ['Ashwini','Bharani','Krittika','Rohini','Mrigashira','Ardra','Punarvasu',
                 'Pushya','Ashlesha','Magha','PurvaPhalguni','UttaraPhalguni','Hasta',
                 'Chitra','Swati','Vishakha','Anuradha','Jyeshtha','Mula','PurvaAshadha',
                 'UttaraAshadha','Shravana','Dhanishtha','Shatabhisha','PurvaBhadrapada',
                 'UttaraBhadrapada','Revati']
NAK_QUALITY   = ['Laghu','Ugra','Mishra','Dhruva','Mridu','Tikshna','Chara','Laghu','Tikshna',
                 'Ugra','Ugra','Dhruva','Laghu','Mridu','Chara','Mishra','Mridu','Tikshna',
                 'Tikshna','Ugra','Dhruva','Chara','Chara','Chara','Ugra','Dhruva','Mridu']
YOGA_NAMES    = ['Vishkambha','Priti','Ayushman','Saubhagya','Shobhana','Atiganda','Sukarma',
                 'Dhriti','Shoola','Ganda','Vriddhi','Dhruva','Vyaghata','Harshana','Vajra',
                 'Siddhi','Vyatipata','Variyan','Parigha','Shiva','Siddha','Sadhya','Shubha',
                 'Shukla','Brahma','Indra','Vaidhriti']
YOGA_INAUSPICIOUS = {1,6,9,10,13,15,17,19,27}
EXALT   = {'Su':(1,10),'Mo':(2,3),'Ma':(10,28),'Me':(6,15),'Ju':(4,5),'Ve':(12,27),'Sa':(7,20),'Ra':(2,20),'Ke':(8,20)}
DEBIL   = {'Su':(7,10),'Mo':(8,3),'Ma':(4,28),'Me':(12,15),'Ju':(10,5),'Ve':(6,27),'Sa':(1,20),'Ra':(8,20),'Ke':(2,20)}
MOOLA   = {'Su':(5,0,20),'Mo':(2,4,30),'Ma':(1,0,12),'Me':(6,16,20),'Ju':(9,0,10),'Ve':(7,0,15),'Sa':(11,0,20)}
OWN     = {'Su':[5],'Mo':[4],'Ma':[1,8],'Me':[3,6],'Ju':[9,12],'Ve':[2,7],'Sa':[10,11],'Ra':[],'Ke':[]}
FRIEND  = {'Su':['Mo','Ma','Ju'],'Mo':['Su','Me'],'Ma':['Su','Mo','Ju'],'Me':['Su','Ve'],
           'Ju':['Su','Mo','Ma'],'Ve':['Me','Sa'],'Sa':['Me','Ve'],'Ra':['Me','Ve','Sa'],'Ke':['Me','Ve','Sa']}
ENEMY   = {'Su':['Ve','Sa'],'Mo':[],'Ma':['Me'],'Me':['Mo'],'Ju':['Me','Ve'],
           'Ve':['Su','Mo'],'Sa':['Su','Mo','Ma'],'Ra':['Su','Mo','Ma'],'Ke':['Su','Mo','Ma']}
MEAN_MOTION = {'Su':0.9856,'Mo':13.1764,'Me':1.3833,'Ve':1.2,'Ma':0.524,'Ju':0.0831,'Sa':0.0335,'Ra':-0.0529,'Ke':-0.0529}
COMB_ORB = {'Mo':12,'Me':14,'Ve':10,'Ma':17,'Ju':11,'Sa':15}
PLANETS  = ['Su','Mo','Me','Ve','Ma','Ju','Sa','Ra','Ke']
SPECIAL_ASPECTS = {'Ma':[4,8],'Ju':[5,9],'Sa':[3,10],'Ra':[5,9],'Ke':[5,9]}
CHOGHADIYA_DAY = {6:['U','C','L','A','K','S','R','U'],0:['A','K','S','R','U','C','L','A'],
                  1:['R','U','C','L','A','K','S','R'],2:['L','A','K','S','R','U','C','L'],
                  3:['S','R','U','C','L','A','K','S'],4:['C','L','A','K','S','R','U','C'],
                  5:['K','S','R','U','C','L','A','K']}
CHOGHADIYA_QUALITY = {'A':'best','L':'good','S':'good','C':'neutral','U':'avoid','K':'avoid','R':'avoid'}
HORA_SEQ = ['Su','Ve','Me','Mo','Sa','Ju','Ma']
HORA_IDX = {p:i for i,p in enumerate(HORA_SEQ)}
INCEPTION_MOON_NAK = 5
NATAL_MOON_SIGN    = 2
DASHA_PERIODS = {'Ke':7,'Ve':20,'Su':6,'Mo':10,'Ma':7,'Ra':18,'Ju':16,'Sa':19,'Me':17}
DASHA_ORDER   = ['Ke','Ve','Su','Mo','Ma','Ra','Ju','Sa','Me']
DASHA_TOTAL   = 120
DASHA_NATURE  = {'Ke':'malefic','Ve':'benefic','Su':'malefic','Mo':'benefic','Ma':'malefic',
                 'Ra':'malefic','Ju':'benefic','Sa':'malefic','Me':'neutral'}
INCEPTION_DATE = date(1996, 4, 22)

def sign_of(d): return int(d / 30) + 1
def deg_in_sign(d): return d % 30
def nak_of(d): return int(d / (360/27)) + 1

def planet_sign_lord(sg):
    m={1:'Ma',2:'Ve',3:'Me',4:'Mo',5:'Su',6:'Me',7:'Ve',8:'Ma',9:'Ju',10:'Sa',11:'Sa',12:'Ju'}
    return m[sg]

def dignity(planet, sid_deg):
    sg = sign_of(sid_deg); dg = deg_in_sign(sid_deg)
    if sg == DEBIL[planet][0]:
        return 'exact_debil' if abs(dg-DEBIL[planet][1])<=1 else 'debilitated'
    if sg == EXALT[planet][0]:
        return 'exact_exalt' if abs(dg-EXALT[planet][1])<=1 else 'exalted'
    if planet in MOOLA:
        ms,ml,mh = MOOLA[planet]
        if sg == ms and ml<=dg<=mh: return 'moolatrikona'
    if sg in OWN.get(planet,[]): return 'own'
    ruler = planet_sign_lord(sg)
    if ruler in FRIEND.get(planet,[]): return 'friendly'
    if ruler in ENEMY.get(planet,[]): return 'enemy'
    return 'neutral'

def speed_cat(planet, spd):
    mm = MEAN_MOTION[planet]
    if abs(spd) < 0.05: return 'stationary'
    if spd < -0.001: return 'retrograde'
    ratio = spd / abs(mm)
    if ratio >= 1.3: return 'very_fast'
    if ratio >= 1.1: return 'fast'
    if ratio >= 0.9: return 'mean'
    if ratio >= 0.7: return 'slow'
    return 'very_slow'

def compute_vimshottari(target_date):
    days_since = (target_date - INCEPTION_DATE).days
    years_since = days_since / 365.25
    idx = 0; cumulative = 0
    while cumulative <= years_since:
        p = DASHA_ORDER[idx % 9]
        dur = DASHA_PERIODS[p]
        if cumulative + dur > years_since:
            maha = p
            time_in_maha = years_since - cumulative
            ant_idx = idx % 9
            ac = 0
            for j in range(9):
                ap = DASHA_ORDER[(ant_idx+j) % 9]
                ad = DASHA_PERIODS[ap] * DASHA_PERIODS[p] / DASHA_TOTAL
                if ac + ad > time_in_maha:
                    return maha, ap
                ac += ad
            return maha, DASHA_ORDER[ant_idx]
        cumulative += dur; idx += 1
    return 'Me', 'Me'

def get_planets_swisseph(d):
    """Compute planetary positions using pyswisseph."""
    jd = swe.julday(d.year, d.month, d.day, 12.0)
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    pos = {}
    SWE_PLANETS = {
        'Su': swe.SUN, 'Mo': swe.MOON, 'Me': swe.MERCURY, 'Ve': swe.VENUS,
        'Ma': swe.MARS, 'Ju': swe.JUPITER, 'Sa': swe.SATURN,
        'Ra': swe.MEAN_NODE,
    }
    for name, body in SWE_PLANETS.items():
        result, _ = swe.calc_ut(jd, body, swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon = result[0] % 360
        spd = result[3]  # longitude speed
        pos[name] = (lon, spd)
    # Ketu = Rahu + 180
    ra_lon = pos['Ra'][0]
    pos['Ke'] = ((ra_lon + 180) % 360, -pos['Ra'][1])
    return pos

def compute_day_features(d, positions=None):
    """Compute all astrological features for a given date.
    positions: dict of {planet: (sidereal_lon, speed)} — if None, uses swisseph.
    """
    if positions is None:
        positions = get_planets_swisseph(d)

    feat = {'date': str(d)}
    feat['dow'] = d.weekday()
    vara_map = {0:'Mo',1:'Ma',2:'Me',3:'Ju',4:'Ve',5:'Sa',6:'Su'}
    feat['vara_lord'] = vara_map[d.weekday()]

    sid = {p: positions[p][0] for p in PLANETS}
    spd = {p: positions[p][1] for p in PLANETS}

    for p in PLANETS:
        feat[f'sid_{p}']   = round(sid[p], 4)
        feat[f'spd_{p}']   = round(spd[p], 4)
        feat[f'sign_{p}']  = sign_of(sid[p])
        feat[f'dig_{p}']   = dignity(p, sid[p])
        feat[f'elem_{p}']  = SIGN_ELEMENT[sign_of(sid[p])]
        feat[f'mod_{p}']   = SIGN_MODALITY[sign_of(sid[p])]
        feat[f'retro_{p}'] = int(spd[p] < -0.001)
        feat[f'stat_{p}']  = int(abs(spd[p]) < 0.05)

    # Combustion (using tropical ≈ sidereal since difference is ayanamsa which cancels out)
    for p in ['Mo','Me','Ve','Ma','Ju','Sa']:
        sep = abs(sid[p] - sid['Su'])
        sep = min(sep, 360 - sep)
        feat[f'comb_{p}'] = int(sep <= COMB_ORB[p])

    # Sun-Moon separation → Tithi, Paksha
    sep_sm = (sid['Mo'] - sid['Su'] + 360) % 360
    # Use tropical separation for tithi
    trop_su = (sid['Su'] + 24.2) % 360  # rough tropical approx
    trop_mo = (sid['Mo'] + 24.2) % 360
    sep_trop = (trop_mo - trop_su + 360) % 360
    tithi = int(sep_trop / 12) + 1
    tithi = max(1, min(30, tithi))
    feat['sun_moon_sep'] = round(sep_sm, 2)
    feat['tithi_num']    = tithi
    feat['paksha']       = 'SHUKLA' if tithi <= 15 else 'KRISHNA'
    tq_map = {1:'Nanda',2:'Bhadra',3:'Jaya',4:'Rikta',5:'Purna',6:'Nanda',7:'Bhadra',8:'Jaya',
              9:'Rikta',10:'Purna',11:'Nanda',12:'Bhadra',13:'Jaya',14:'Rikta',15:'Purna',
              16:'Nanda',17:'Bhadra',18:'Jaya',19:'Rikta',20:'Purna',21:'Nanda',22:'Bhadra',
              23:'Jaya',24:'Rikta',25:'Purna',26:'Nanda',27:'Bhadra',28:'Jaya',29:'Rikta',30:'Purna'}
    feat['tithi_quality'] = tq_map[tithi]

    # Nakshatra
    nak_num = nak_of(sid['Mo'])
    feat['nak_mo']       = nak_num
    feat['nak_mo_name']  = NAK_NAMES[nak_num-1]
    feat['nak_mo_lord']  = NAK_LORDS[nak_num-1]
    feat['nak_mo_qual']  = NAK_QUALITY[nak_num-1]

    # Panchanga Yoga
    yoga_pt  = (sid['Su'] + sid['Mo']) % 360
    yoga_num = int(yoga_pt / (360/27)) + 1
    feat['yoga_num']     = yoga_num
    feat['yoga_name']    = YOGA_NAMES[yoga_num-1]
    feat['yoga_quality'] = 'inauspicious' if yoga_num in YOGA_INAUSPICIOUS else 'auspicious'

    # Karana
    MOVABLE = ['Bava','Balava','Kaulava','Taitila','Garija','Vanija','Vishti']
    ki = int(sep_trop / 6)
    if ki == 0: kar = 'Kimstughna'
    elif ki <= 56: kar = MOVABLE[(ki-1)%7]
    elif ki == 57: kar = 'Shakuni'
    elif ki == 58: kar = 'Chatushpada'
    else: kar = 'Naga'
    feat['karana'] = kar
    KQ = {'Vishti':'inauspicious','Shakuni':'inauspicious','Chatushpada':'inauspicious','Naga':'inauspicious'}
    feat['karana_quality'] = KQ.get(kar, 'auspicious')

    # Yogas
    def _gajakesari(ms, js):
        diff = abs(ms - js); return int(min(diff, 12-diff) in [0,3,6,9])
    feat['gajakesari'] = _gajakesari(feat['sign_Mo'], feat['sign_Ju'])

    malefics = [feat[f'sign_{p}'] for p in ['Su','Ma','Sa','Ra','Ke']]
    before = ((feat['sign_Mo']-2)%12)+1; after = (feat['sign_Mo']%12)+1
    feat['papakartari'] = int(before in malefics and after in malefics)
    second = (feat['sign_Mo']%12)+1; fourth = (feat['sign_Mo']+2)%12+1
    eleventh = (feat['sign_Mo']+9)%12+1
    pos_signs = [second, fourth, eleventh]
    other_signs = [feat[f'sign_{p}'] for p in PLANETS if p != 'Mo']
    feat['argala_mo'] = int(any(s in pos_signs for s in other_signs))

    # Graha Yuddha
    feat['graha_yuddha'] = 0
    for p1,p2 in [('Me','Ve'),('Me','Ma'),('Ve','Ma'),('Ma','Ju'),('Ju','Sa')]:
        sep2 = abs(sid[p1]-sid[p2]); sep2 = min(sep2,360-sep2)
        if sep2 <= 1.0: feat['graha_yuddha'] = 1; break

    # Gandanta
    def _gand(x):
        dg = x%30; sg = int(x/30)+1
        return int((sg in [12,4,8] and dg>=26.67) or (sg in [1,5,9] and dg<=3.33))
    feat['gand_Mo'] = _gand(sid['Mo']); feat['gand_any'] = int(any(_gand(sid[p]) for p in PLANETS))
    feat['sandhi_mo'] = int(deg_in_sign(sid['Mo']) >= 29)

    # Tara Bala
    diff = (nak_num - INCEPTION_MOON_NAK) % 27
    tara = (diff % 9) + 1
    feat['tara_num'] = tara
    tara_quality_map = {1:'critical',2:'wealth',3:'danger',4:'prosperity',5:'obstacle',
                        6:'achievement',7:'worst',8:'good',9:'best'}
    feat['tara_quality'] = tara_quality_map[tara]

    # Nakshatra lord state
    nl = feat['nak_mo_lord']
    feat['nakl_dig'] = dignity(nl, sid[nl])
    feat['nakl_spd'] = speed_cat(nl, spd[nl])

    # Hora and Choghadiya
    dow = d.weekday()
    day_lord_map = {0:'Mo',1:'Ma',2:'Me',3:'Ju',4:'Ve',5:'Sa',6:'Su'}
    hora_start = HORA_IDX[day_lord_map[dow]]
    feat['hora_at_open'] = HORA_SEQ[(hora_start + 3) % 7]
    chog_seq = CHOGHADIYA_DAY[dow]
    feat['choghadiya'] = chog_seq[2]
    feat['choghadiya_quality'] = CHOGHADIYA_QUALITY[chog_seq[2]]

    # Dasha
    maha, ant = compute_vimshottari(d)
    feat['mahadasha']  = maha
    feat['antardasha'] = ant
    feat['dasha_quality'] = DASHA_NATURE.get(maha,'neutral')
    feat['dasha_lord_dig'] = dignity(maha, sid[maha])

    # Sade Sati / Ashtama Shani
    feat['sade_sati'] = int(feat['sign_Sa'] in [(NATAL_MOON_SIGN-2+12-1)%12+1, NATAL_MOON_SIGN, (NATAL_MOON_SIGN%12)+1])
    ashtama_sg = (NATAL_MOON_SIGN+7-1)%12+1
    feat['ashtama_shani'] = int(feat['sign_Sa'] == ashtama_sg)
    ss_phase_map = {(NATAL_MOON_SIGN-2+12-1)%12+1:'rising', NATAL_MOON_SIGN:'peak', (NATAL_MOON_SIGN%12)+1:'setting'}
    feat['sade_sati_phase'] = ss_phase_map.get(feat['sign_Sa'], 'none')

    # Panchaka
    feat['panchaka'] = int(nak_num >= 23)

    # Aspect: Jupiter, Saturn aspecting Moon
    def _asp(p_sign, p_name, target_sign):
        aspects = {(p_sign+5)%12+1}  # 7th
        for off in SPECIAL_ASPECTS.get(p_name,[]):
            aspects.add((p_sign+off-1)%12+1)
        return int(target_sign in aspects)
    feat['ju_asp_mo'] = _asp(feat['sign_Ju'], 'Ju', feat['sign_Mo'])
    feat['sa_asp_mo'] = _asp(feat['sign_Sa'], 'Sa', feat['sign_Mo'])
    feat['ma_asp_mo'] = _asp(feat['sign_Ma'], 'Ma', feat['sign_Mo'])

    # Interaction features
    feat['ix_paksha_ju_dig']    = feat['paksha'] + '_' + feat['dig_Ju']
    feat['ix_paksha_nak']       = feat['paksha'] + '_' + feat['nak_mo_name']
    feat['ix_paksha_moon_sign'] = feat['paksha'] + '_Mo' + str(feat['sign_Mo'])
    feat['ix_tithi_nak']        = feat['tithi_quality'] + '_' + feat['nak_mo_qual']
    feat['ix_ju_dig_moon_sign'] = feat['dig_Ju'] + '_Mo' + str(feat['sign_Mo'])
    feat['ix_vara_paksha']      = feat['vara_lord'] + '_' + feat['paksha']

    # Binary feature strings (for pattern matching)
    for col in ['gajakesari','papakartari','comb_Mo','comb_Me','comb_Ve',
                'retro_Me','retro_Ju','gand_Mo','sade_sati','ju_asp_mo','sa_asp_mo','graha_yuddha']:
        feat[col+'_str'] = col + '=' + str(feat.get(col, 0))

    return feat

# ═══════════════════════════════════════════════════════════════════════
# LOAD CONFIRMED PATTERNS
# ═══════════════════════════════════════════════════════════════════════
print("Loading confirmed patterns …")
conf_path = f"{REPO}/results/validation/confirmed_patterns.csv"
if not os.path.exists(conf_path):
    print(f"ERROR: {conf_path} not found. Run new_step3.py first.")
    exit(1)

confirmed = pd.read_csv(conf_path)
print(f"  {len(confirmed)} confirmed patterns")

# ═══════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE ON HISTORICAL DATA
# ═══════════════════════════════════════════════════════════════════════
print("\nComputing composite score on historical data …")

df = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df_clean = df.dropna(subset=['is_bull']).copy()
df_clean['is_bull'] = df_clean['is_bull'].astype(int)

BASE_BULL = df_clean['is_bull'].mean()

def eval_pattern_row(row_feat, features_str, condition_str):
    """Check if a pattern matches a single row (dict or Series)."""
    features = features_str.split('|')
    condition_parts = condition_str.split('||') if '||' in str(condition_str) else [str(condition_str)]
    if len(features) != len(condition_parts): return False
    for feat, val in zip(features, condition_parts):
        feat = feat.strip(); val = val.strip()
        v = row_feat.get(feat) if isinstance(row_feat, dict) else (row_feat[feat] if feat in row_feat.index else None)
        if v is None: return False
        if str(v) != val: return False
    return True

def compute_scores(df, confirmed):
    """Compute composite bull/bear score for each row."""
    scores = np.zeros(len(df))
    bull_patterns = confirmed[confirmed['signal_dir'] == 'BULL'] if 'signal_dir' in confirmed.columns else confirmed
    bear_patterns = confirmed[confirmed['signal_dir'] == 'BEAR'] if 'signal_dir' in confirmed.columns else pd.DataFrame()

    for _, pat in bull_patterns.iterrows():
        feat = pat['features']; cond = pat['condition']
        wlb = pat.get('wlb_train', pat.get('wilson_lower', 0.5))
        weight = wlb - BASE_BULL
        feats = feat.split('|')
        conds = cond.split('||') if '||' in str(cond) else [str(cond)]
        if len(feats) != len(conds): continue
        # Vectorized evaluation
        mask = pd.Series([True] * len(df), index=df.index)
        for f, v in zip(feats, conds):
            f = f.strip(); v = v.strip()
            if f in df.columns:
                mask = mask & (df[f].astype(str) == v)
            else:
                mask = pd.Series([False] * len(df), index=df.index)
                break
        scores += mask.values * weight

    bear_scores = np.zeros(len(df))
    for _, pat in bear_patterns.iterrows():
        feat = pat['features']; cond = pat['condition']
        wlb = pat.get('wlb_train', pat.get('wilson_lower', 0.5))
        weight = abs(wlb - BASE_BULL)
        feats = feat.split('|')
        conds = cond.split('||') if '||' in str(cond) else [str(cond)]
        if len(feats) != len(conds): continue
        mask = pd.Series([True] * len(df), index=df.index)
        for f, v in zip(feats, conds):
            f = f.strip(); v = v.strip()
            if f in df.columns:
                mask = mask & (df[f].astype(str) == v)
            else:
                mask = pd.Series([False] * len(df), index=df.index)
                break
        bear_scores += mask.values * weight

    # Normalize to 0-100 (50 = neutral)
    net = scores - bear_scores
    net_min, net_max = net.min(), net.max()
    if net_max - net_min > 0:
        normalized = (net - net_min) / (net_max - net_min) * 100
    else:
        normalized = np.full(len(df), 50.0)
    return normalized, scores, bear_scores

norm_scores, bull_scores, bear_scores = compute_scores(df_clean, confirmed)
df_clean['composite_score'] = norm_scores
df_clean['bull_score'] = bull_scores
df_clean['bear_score'] = bear_scores

comp_csv = f"{REPO}/results/synthesis/composite_scores.csv"
os.makedirs(f"{REPO}/results/synthesis", exist_ok=True)
df_clean[['date','composite_score','bull_score','bear_score','is_bull','fwd_ret_3d']].to_csv(comp_csv, index=False)
print(f"Composite scores saved: {len(df_clean)} rows")

# ═══════════════════════════════════════════════════════════════════════
# ACCURACY-SELECTIVITY SURFACE
# ═══════════════════════════════════════════════════════════════════════
print("\nComputing accuracy-selectivity surface …")
deciles = np.percentile(df_clean['composite_score'], np.arange(10, 100, 10))
surface = []
for thresh in [30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]:
    sub = df_clean[df_clean['composite_score'] >= thresh]
    n = len(sub); k = sub['is_bull'].sum()
    wr = k/n if n > 0 else 0
    wlb = wilson_lower(n, k)
    freq = n/len(df_clean)
    surface.append({'threshold': thresh, 'n': n, 'win_rate': round(wr,4),
                     'wilson_lower': round(wlb,4), 'frequency': round(freq,4)})
    if wlb > BASE_BULL + 0.05 and n > 50:
        print(f"  threshold={thresh}: n={n}, win_rate={wr:.3f}, wlb={wlb:.3f}, freq={freq:.3f}")

surface_df = pd.DataFrame(surface)
surface_df.to_csv(f"{REPO}/results/validation/accuracy_selectivity_surface.csv", index=False)

# Find empirical threshold
prime_thresh = 50
for _, row in surface_df.iterrows():
    if row['wilson_lower'] > BASE_BULL + 0.05 and row['n'] >= 50:
        prime_thresh = row['threshold']
        break
print(f"Empirical prime trade threshold: score >= {prime_thresh}")

# ═══════════════════════════════════════════════════════════════════════
# BACKTEST (signals only based on astrological composite score)
# ═══════════════════════════════════════════════════════════════════════
print("\nRunning backtest (astrological signals only) …")
df_bt = df_clean.dropna(subset=['fwd_ret_3d']).copy()
bull_signal = df_bt['composite_score'] >= prime_thresh
bear_signal = df_bt['composite_score'] <= (100 - prime_thresh)

trades = []
for idx, row in df_bt[bull_signal | bear_signal].iterrows():
    direction = 1 if row['composite_score'] >= prime_thresh else -1
    ret = row['fwd_ret_3d'] * direction
    trades.append({'date': row['date'], 'direction': direction,
                   'score': row['composite_score'], 'ret_3d': row['fwd_ret_3d'],
                   'trade_ret': ret, 'win': int(ret > 0)})

trades_df = pd.DataFrame(trades)
if len(trades_df) > 0:
    win_rate = trades_df['win'].mean()
    mean_ret = trades_df['trade_ret'].mean()
    std_ret  = trades_df['trade_ret'].std()
    sharpe   = mean_ret / std_ret * math.sqrt(252/3) if std_ret > 0 else 0
    # Equity curve
    cum_ret  = (1 + trades_df['trade_ret']).cumprod()
    max_dd_pct = ((cum_ret / cum_ret.cummax() - 1)).min()
    trades_df['cum_ret'] = cum_ret
    print(f"  Trades: {len(trades_df)}, win rate: {win_rate:.3f}, mean ret: {mean_ret:.4f}")
    print(f"  Sharpe: {sharpe:.2f}, max DD: {max_dd_pct:.3f}")
    # Stress tests
    stress_periods = {
        '2008': ('2008-01-01','2009-03-31'),
        '2013': ('2013-05-01','2013-08-31'),
        '2016': ('2016-10-01','2016-12-31'),
        '2020': ('2020-02-01','2020-04-30'),
        '2022': ('2022-01-01','2022-06-30'),
    }
    stress_records = []
    for period, (s, e) in stress_periods.items():
        sub = trades_df[(trades_df['date'] >= s) & (trades_df['date'] <= e)]
        if len(sub) == 0: continue
        stress_records.append({'period': period, 'n': len(sub), 'win_rate': round(sub['win'].mean(),3),
                                'mean_ret': round(sub['trade_ret'].mean(),4)})
    pd.DataFrame(stress_records).to_csv(f"{REPO}/results/synthesis/stress_tests.csv", index=False)

    # Backtest summary
    summary = pd.DataFrame([{
        'n_trades': len(trades_df), 'win_rate': round(win_rate,4),
        'mean_ret_per_trade': round(mean_ret,5), 'sharpe': round(sharpe,2),
        'max_drawdown': round(max_dd_pct,4), 'threshold': prime_thresh,
        'note': 'Pure astrological signals only — no market data in signal'
    }])
    summary.to_csv(f"{REPO}/results/synthesis/backtest_summary.csv", index=False)
    trades_df.to_csv(f"{REPO}/results/synthesis/backtest_trade_log.csv", index=False)
else:
    print("  No trades generated — check threshold")

# ═══════════════════════════════════════════════════════════════════════
# FORWARD CALENDAR — PYSWISSEPH ONLY
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding forward calendar using pyswisseph …")
import pandas_market_calendars as mcal
nse_cal = mcal.get_calendar('NSE')

start_date = date.today()
end_date   = start_date + timedelta(days=400)
schedule   = nse_cal.schedule(start_date=str(start_date), end_date=str(end_date))
trading_days = [d.date() for d in schedule.index][:252]
print(f"  Computing features for {len(trading_days)} trading days …")

# Load confirmed patterns for scoring
def score_day_forward(feat_dict, confirmed, base_bull):
    """Compute composite score for a day given its feature dict."""
    bull_score = 0.0; bear_score = 0.0
    active_bull = []; active_bear = []
    for _, pat in confirmed.iterrows():
        feats = pat['features'].split('|')
        conds = pat['condition'].split('||') if '||' in str(pat['condition']) else [str(pat['condition'])]
        if len(feats) != len(conds): continue
        match = all(str(feat_dict.get(f.strip(), '')) == v.strip() for f,v in zip(feats, conds))
        if not match: continue
        wlb = pat.get('wlb_train', 0.5)
        is_bull_pat = pat.get('signal_dir','BULL') == 'BULL' or wlb > base_bull
        w = abs(wlb - base_bull)
        if is_bull_pat: bull_score += w; active_bull.append({'pattern':pat['features']+'='+str(pat['condition']), 'wlb':wlb})
        else: bear_score += w; active_bear.append({'pattern':pat['features']+'='+str(pat['condition']), 'wlb':wlb})
    net = bull_score - bear_score
    # Classify
    n_bull = len(active_bull); n_bear = len(active_bear)
    if n_bull >= 3 and n_bear == 0: classification = 'PRIME_TRADE_BULL'
    elif n_bear >= 3 and n_bull == 0: classification = 'PRIME_TRADE_BEAR'
    elif n_bull >= 1 and bull_score > 0.05: classification = 'WATCH_BULL'
    elif n_bear >= 1 and bear_score > 0.05: classification = 'WATCH_BEAR'
    else: classification = 'NEUTRAL'
    return net, bull_score, bear_score, classification, active_bull, active_bear

calendar_rows = []
prev_positions = None
prev_day_feat = None
for d in trading_days:
    try:
        positions = get_planets_swisseph(d)
        feat = compute_day_features(d, positions)
        # Detect nakshatra transition (for sequential patterns)
        feat['nak_transition'] = 0
        if prev_day_feat and feat['nak_mo'] != prev_day_feat.get('nak_mo', feat['nak_mo']):
            feat['nak_transition'] = 1
        # Add binary _str versions
        for col in ['gajakesari','papakartari','comb_Mo','comb_Me','comb_Ve',
                    'retro_Me','retro_Ju','gand_Mo','sade_sati','ju_asp_mo','sa_asp_mo','graha_yuddha']:
            feat[col+'_str'] = col + '=' + str(feat.get(col, 0))
        net, bs, brs, cls, ab, abr = score_day_forward(feat, confirmed, BASE_BULL)
        row = {
            'date': str(d),
            'classification': cls,
            'net_score': round(net, 4),
            'bull_score': round(bs, 4),
            'bear_score': round(brs, 4),
            'n_bull_patterns': len(ab),
            'n_bear_patterns': len(abr),
            'paksha': feat.get('paksha',''),
            'tithi_num': feat.get('tithi_num',''),
            'tithi_quality': feat.get('tithi_quality',''),
            'nak_mo_name': feat.get('nak_mo_name',''),
            'yoga_name': feat.get('yoga_name',''),
            'yoga_quality': feat.get('yoga_quality',''),
            'karana': feat.get('karana',''),
            'vara_lord': feat.get('vara_lord',''),
            'dig_Ju': feat.get('dig_Ju',''),
            'dig_Sa': feat.get('dig_Sa',''),
            'dig_Mo': feat.get('dig_Mo',''),
            'retro_Me': feat.get('retro_Me',0),
            'comb_Mo': feat.get('comb_Mo',0),
            'gajakesari': feat.get('gajakesari',0),
            'sade_sati': feat.get('sade_sati',0),
            'mahadasha': feat.get('mahadasha',''),
            'active_bull_patterns': '|'.join(x['pattern'] for x in ab[:3]),
            'active_bear_patterns': '|'.join(x['pattern'] for x in abr[:3]),
        }
        calendar_rows.append(row)
        prev_day_feat = feat
    except Exception as e:
        print(f"  Error on {d}: {e}")
        calendar_rows.append({'date': str(d), 'classification': 'ERROR', 'net_score': 0})

cal_df = pd.DataFrame(calendar_rows)
cal_df.to_csv(f"{REPO}/results/forward_calendar/planetary_calendar_1yr.csv", index=False)
print(f"Forward calendar saved: {len(cal_df)} trading days")
print(cal_df['classification'].value_counts().to_string())

# Next PRIME_TRADE windows
prime_bull = cal_df[cal_df['classification']=='PRIME_TRADE_BULL']
prime_bear = cal_df[cal_df['classification']=='PRIME_TRADE_BEAR']
print(f"\nNext PRIME TRADE BULL: {prime_bull['date'].iloc[0] if len(prime_bull)>0 else 'none'}")
print(f"Next PRIME TRADE BEAR: {prime_bear['date'].iloc[0] if len(prime_bear)>0 else 'none'}")

print("\n=== STEP 4 COMPLETE ===")
