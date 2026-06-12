"""
Step 1 — Complete Feature Engineering
Builds nifty_enriched.csv and banknifty_enriched.csv
All forward-return columns use ONLY future data (shift verified).
Run: python step1_features.py
"""
import json, math, warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
AYAN = 'la'  # Lahiri — consistent throughout

# ── helpers ──────────────────────────────────────────────────────────────────
def sid(trop, ayan_val): return (trop - ayan_val + 360) % 360
def norm360(x):          return (x + 360) % 360

# ── load data.json ────────────────────────────────────────────────────────────
print("Loading data.json …")
with open(f"{REPO}/data.json") as f:
    raw = json.load(f)['records']

df = pd.DataFrame(raw)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# flatten nested dicts
for pk in ['Su','Mo','Me','Ve','Ma','Ju','Sa','Ra','Ke']:
    df[f'trop_{pk}'] = df['p'].apply(lambda x: x[pk])
for ak in ['la','kp','ra','yu','pu','va']:
    df[f'ayan_{ak}'] = df['ayan'].apply(lambda x: x[ak])
df.drop(columns=['p','ayan','chg'], inplace=True)

# sidereal longitudes using Lahiri
PLANETS = ['Su','Mo','Me','Ve','Ma','Ju','Sa','Ra','Ke']
for pk in PLANETS:
    df[f'sid_{pk}'] = df.apply(lambda r: sid(r[f'trop_{pk}'], r['ayan_la']), axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.1  PRICE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("1.1 Price features …")
df['log_ret']     = np.log(df['close'] / df['close'].shift(1))
df['range_pct']   = (df['high'] - df['low']) / df['close']

# forward returns — shift(-n) so row t gets return starting at t+1 close
for n in [1, 2, 3, 5, 10]:
    df[f'fwd_ret_{n}d'] = np.log(df['close'].shift(-n) / df['close'])
    # preserve NaN from fwd_ret so lookahead audit passes
    df[f'fwd_dir_{n}d'] = pd.array(
        [pd.NA if pd.isna(v) else int(v > 0) for v in df[f'fwd_ret_{n}d']], dtype='Int8'
    )

# rolling vol
for w in [5, 10, 20, 60]:
    df[f'rvol_{w}d'] = df['log_ret'].rolling(w).std() * math.sqrt(252)

# binary vol label: next-5d realised vol > rolling 60d median
fwd_rvol5 = df['log_ret'].shift(-1).rolling(5).std() * math.sqrt(252)
df['fwd_vol_hi'] = (fwd_rvol5 > df['rvol_60d'].rolling(60).median()).astype('Int8')

# ATR 14
tr = pd.concat([df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low']  - df['close'].shift()).abs()], axis=1).max(axis=1)
df['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()

# SMAs
df['sma50']  = df['close'].rolling(50).mean()
df['sma200'] = df['close'].rolling(200).mean()
df['dist_sma50']  = (df['close'] - df['sma50'])  / df['sma50']  * 100
df['dist_sma200'] = (df['close'] - df['sma200']) / df['sma200'] * 100

# market regime
def regime(r):
    if pd.isna(r['sma50']) or pd.isna(r['sma200']): return 'TRANSITIONAL'
    if r['close'] > r['sma50'] > r['sma200']:        return 'BULL'
    if r['close'] < r['sma50'] < r['sma200']:        return 'BEAR'
    return 'TRANSITIONAL'
df['market_regime'] = df.apply(regime, axis=1)

# 20-day momentum percentile rank in rolling 252-day window
df['mom20'] = df['close'] / df['close'].shift(20) - 1
df['mom_rank'] = df['mom20'].rolling(252).rank(pct=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.2  CORE VEDIC TIME FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("1.2 Vedic time features …")

# Sun-Moon separation → Tithi
df['sun_moon_sep'] = df.apply(lambda r: norm360(r['trop_Mo'] - r['trop_Su']), axis=1)
df['tithi_num']    = (df['sun_moon_sep'] / 12).apply(math.floor) + 1  # 1-30
df['paksha']       = df['tithi_num'].apply(lambda t: 'SHUKLA' if t <= 15 else 'KRISHNA')

# Moon phase quadrant
def moon_phase_quad(sep):
    if sep < 90:   return 'NEW'
    if sep < 180:  return 'WAXING'
    if sep < 270:  return 'FULL'
    return 'WANING'
df['moon_phase_quad'] = df['sun_moon_sep'].apply(moon_phase_quad)

# Tithi quality
TITHI_QUALITY = {}
for t in [1,6,11,16,21,26]:  TITHI_QUALITY[t] = 'NANDA'
for t in [2,7,12,17,22,27]:  TITHI_QUALITY[t] = 'BHADRA'
for t in [3,8,13,18,23,28]:  TITHI_QUALITY[t] = 'JAYA'
for t in [4,9,14,19,24,29]:  TITHI_QUALITY[t] = 'RIKTA'
for t in [5,10,15,20,25,30]: TITHI_QUALITY[t] = 'PURNA'
df['tithi_quality'] = df['tithi_num'].apply(lambda t: TITHI_QUALITY.get(t % 30 if t != 30 else 30, 'PURNA'))

# Moon Nakshatra (sidereal) — 1-27
df['moon_nak_num'] = df['sid_Mo'].apply(lambda l: int(l / (360/27)) + 1).clip(1, 27)

SIGNS = ['Ari','Tau','Gem','Can','Leo','Vir','Lib','Sco','Sag','Cap','Aqu','Pis']
NAKS  = ['Ashwini','Bharani','Krittika','Rohini','Mrigashira','Ardra',
         'Punarvasu','Pushya','Ashlesha','Magha','Purva_Phalguni','Uttara_Phalguni',
         'Hasta','Chitra','Swati','Vishakha','Anuradha','Jyeshtha',
         'Mula','Purva_Ashadha','Uttara_Ashadha','Shravana','Dhanishtha',
         'Shatabhisha','Purva_Bhadrapada','Uttara_Bhadrapada','Revati']
df['moon_nak_name'] = df['moon_nak_num'].apply(lambda n: NAKS[n-1])

# Nakshatra lords (Vimshottari — 9-lord sequence repeating 3x)
NK_LORDS = ['Ketu','Venus','Sun','Moon','Mars','Rahu','Jupiter','Saturn','Mercury'] * 3
df['moon_nak_lord'] = df['moon_nak_num'].apply(lambda n: NK_LORDS[n-1])

# Nakshatra quality type
NAK_QUALITY = {
    'DHRUVA':  ['Rohini','Uttara_Phalguni','Uttara_Ashadha','Uttara_Bhadrapada'],
    'CHARA':   ['Punarvasu','Swati','Shravana','Dhanishtha','Shatabhisha'],
    'UGRA':    ['Bharani','Magha','Purva_Phalguni','Purva_Ashadha','Purva_Bhadrapada'],
    'MRIDU':   ['Mrigashira','Chitra','Anuradha','Revati'],
    'TIKSHNA': ['Ardra','Ashlesha','Jyeshtha','Mula'],
    'MISHRA':  ['Krittika','Vishakha'],
    'LAGHU':   ['Ashwini','Pushya','Hasta'],
}
NAK_TO_QUAL = {n: q for q, naks in NAK_QUALITY.items() for n in naks}
df['moon_nak_quality'] = df['moon_nak_name'].apply(lambda n: NAK_TO_QUAL.get(n, 'LAGHU'))

# Karana (half-tithi)
KARANA_MOVABLE = ['Bava','Balava','Kaulava','Taitila','Garija','Vanija','Vishti']
KARANA_FIXED_SEQ = {0:'Kimstughna', 57:'Shakuni', 58:'Chatushpada', 59:'Naga'}
KARANA_LORDS = {
    'Bava':'Sun','Balava':'Moon','Kaulava':'Mars','Taitila':'Mercury',
    'Garija':'Jupiter','Vanija':'Venus','Vishti':'Saturn',
    'Shakuni':'Saturn','Chatushpada':'Jupiter','Naga':'Mercury','Kimstughna':'Moon'
}

def get_karana(sun_moon_sep):
    idx = int(sun_moon_sep / 6)
    if idx == 0 or idx == 60: return 'Kimstughna'
    if idx in KARANA_FIXED_SEQ: return KARANA_FIXED_SEQ[idx]
    if idx > 56: return KARANA_FIXED_SEQ.get(idx, 'Naga')
    return KARANA_MOVABLE[(idx - 1) % 7]

df['karana']      = df['sun_moon_sep'].apply(get_karana)
df['karana_lord'] = df['karana'].apply(lambda k: KARANA_LORDS.get(k, ''))

# Yogi point: (Su_sid + Mo_sid + 93.333) % 360
df['yogi_pt']     = df.apply(lambda r: (r['sid_Su'] + r['sid_Mo'] + 93.333) % 360, axis=1)
df['yogi_nak_num']= df['yogi_pt'].apply(lambda l: int(l / (360/27)) + 1).clip(1, 27)
df['yogi_nak']    = df['yogi_nak_num'].apply(lambda n: NAKS[n-1])
df['yogi_lord']   = df['yogi_nak_num'].apply(lambda n: NK_LORDS[n-1])

# Avayogi: 6 nakshatras ahead of Yogi
df['avayogi_nak_num'] = ((df['yogi_nak_num'] - 1 + 6) % 27) + 1
df['avayogi_nak']     = df['avayogi_nak_num'].apply(lambda n: NAKS[n-1])
df['avayogi_lord']    = df['avayogi_nak_num'].apply(lambda n: NK_LORDS[n-1])

# Calendar features
df['dow']     = df['date'].dt.dayofweek       # 0=Mon
df['month']   = df['date'].dt.month
df['week']    = df['date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['date'].dt.quarter

# Planet signs (sidereal)
for pk in PLANETS:
    df[f'sign_{pk}'] = df[f'sid_{pk}'].apply(lambda l: SIGNS[int(l/30)])

# ═══════════════════════════════════════════════════════════════════════════════
# 1.3  PLANETARY MOTION & STATE
# ═══════════════════════════════════════════════════════════════════════════════
print("1.3 Planetary motion & state …")

# Daily speed (tropical change) — Ra/Ke always retrograde, Mo never
for pk in PLANETS:
    spd = df[f'trop_{pk}'].diff()
    # handle wrap-around
    spd = spd.apply(lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x))
    df[f'spd_{pk}'] = spd

# Retrograde flags
for pk in ['Me','Ve','Ma','Ju','Sa']:
    df[f'retro_{pk}'] = (df[f'spd_{pk}'] < 0).astype('Int8')
df['retro_Ra'] = 1
df['retro_Ke'] = 1
df['retro_Mo'] = 0
df['retro_Su'] = 0

# Mercury retrograde: negative speed for 2+ consecutive days
me_retro = df['spd_Me'] < 0
df['me_retro_confirmed'] = (me_retro & me_retro.shift(1)).astype('Int8')

# Stationary flags: abs speed < 0.05 deg/day
for pk in ['Me','Ve','Ma','Ju','Sa','Ra']:
    df[f'stationary_{pk}'] = (df[f'spd_{pk}'].abs() < 0.05).astype('Int8')

# Combustion flags (tropical angular distance from Sun)
# Mercury: 14° direct, 12° retrograde; Venus: 10° direct, 16° retrograde
COMB_ORB = {'Mo':12,'Me':14,'Ve':10,'Ma':17,'Ju':11,'Sa':15}
for pk, orb in COMB_ORB.items():
    d_sun = df.apply(lambda r: min(abs(r[f'trop_{pk}'] - r['trop_Su']),
                                   360 - abs(r[f'trop_{pk}'] - r['trop_Su'])), axis=1)
    if pk == 'Me':
        effective_orb = df['retro_Me'].apply(lambda x: 12 if x == 1 else 14)
    elif pk == 'Ve':
        effective_orb = df['retro_Ve'].apply(lambda x: 16 if x == 1 else 10)
    else:
        effective_orb = orb
    df[f'combust_{pk}'] = ((d_sun > 1) & (d_sun < effective_orb)).astype('Int8')
df['combust_any_benefic'] = ((df['combust_Ju'] == 1) | (df['combust_Ve'] == 1)).astype('Int8')

# Planetary war: Me/Ve/Ma/Ju/Sa within 1° AND same sidereal sign
WAR_PLANETS = ['Me','Ve','Ma','Ju','Sa']
df['graha_yuddha'] = 0
df['graha_yuddha_pair'] = ''
for i, p1 in enumerate(WAR_PLANETS):
    for p2 in WAR_PLANETS[i+1:]:
        same_sign = df[f'sign_{p1}'] == df[f'sign_{p2}']
        ang_dist  = (df[f'trop_{p1}'] - df[f'trop_{p2}']).abs()
        ang_dist  = ang_dist.apply(lambda x: min(x, 360-x))
        war = (same_sign & (ang_dist <= 1)).astype(int)
        df['graha_yuddha'] = (df['graha_yuddha'] | war).astype(int)
        mask = war == 1
        df.loc[mask, 'graha_yuddha_pair'] = f'{p1}-{p2}'

# Sign ingress flag: planet crossed sign boundary since yesterday
for pk in PLANETS:
    df[f'ingress_{pk}'] = (df[f'sign_{pk}'] != df[f'sign_{pk}'].shift(1)).astype('Int8')
    df.loc[df.index[0], f'ingress_{pk}'] = 0

# Gandanta: within 3° of water-fire junctions (sidereal)
# Junctions: Pis/Ari (357-3), Can/Leo (117-123), Sco/Sag (237-243)
def in_gandanta(sid_lon):
    l = sid_lon % 360
    return (l >= 357 or l <= 3) or (117 <= l <= 123) or (237 <= l <= 243)
for pk in PLANETS:
    df[f'gandanta_{pk}'] = df[f'sid_{pk}'].apply(in_gandanta).astype('Int8')
df['gandanta_moon'] = df['gandanta_Mo']
df['gandanta_any']  = df[[f'gandanta_{pk}' for pk in PLANETS]].max(axis=1).astype('Int8')

# Eclipse corridor
# Solar eclipse: New Moon (tithi 30 or 1) AND Rahu within 18° of Sun
# Lunar eclipse: Full Moon (tithi 15) AND Rahu within 12° of Moon
def rahu_sun_dist(r):
    d = abs(r['trop_Ra'] - r['trop_Su'])
    return min(d, 360-d)
def rahu_moon_dist(r):
    d = abs(r['trop_Ra'] - r['trop_Mo'])
    return min(d, 360-d)
df['rahu_sun_dist']  = df.apply(rahu_sun_dist, axis=1)
df['rahu_moon_dist'] = df.apply(rahu_moon_dist, axis=1)

solar_eclipse_day = ((df['tithi_num'].isin([1,30])) & (df['rahu_sun_dist'] < 18))
lunar_eclipse_day = ((df['tithi_num'] == 15) & (df['rahu_moon_dist'] < 12))
eclipse_day = solar_eclipse_day | lunar_eclipse_day

# Corridor: 15 days either side of eclipse
df['eclipse_corridor'] = 0
eclipse_dates = df.loc[eclipse_day, 'date'].tolist()
for ed in eclipse_dates:
    mask = (df['date'] >= ed - timedelta(days=15)) & (df['date'] <= ed + timedelta(days=15))
    df.loc[mask, 'eclipse_corridor'] = 1
df['eclipse_corridor'] = df['eclipse_corridor'].astype('Int8')
df['eclipse_day'] = eclipse_day.astype('Int8')

# Rahu sign
df['rahu_sign'] = df['sid_Ra'].apply(lambda l: SIGNS[int(l/30)])
df['ketu_sign'] = df['sid_Ke'].apply(lambda l: SIGNS[int(l/30)])
df['rahu_ketu_axis'] = df['rahu_sign'] + '-' + df['ketu_sign']

# ═══════════════════════════════════════════════════════════════════════════════
# 1.4  PLANETARY STRENGTH FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("1.4 Planetary strength …")

TRIKONA_HOUSES = {3, 5, 7, 9, 10, 11}

def sign_idx(lon): return int(lon / 30) % 12

def ashtakavarga_score(row, focal_pk):
    focal_sign = sign_idx(row[f'sid_{focal_pk}'])
    score = 0
    for pk in PLANETS:
        if pk == focal_pk: continue
        other_sign = sign_idx(row[f'sid_{pk}'])
        house = (other_sign - focal_sign) % 12 + 1
        if house in TRIKONA_HOUSES:
            score += 1
    return score

print("  Computing Ashtakavarga (takes ~30s) …")
for pk in PLANETS:
    df[f'ashta_{pk}'] = df.apply(lambda r: ashtakavarga_score(r, pk), axis=1)
df['sarvashtakavarga'] = df[[f'ashta_{pk}' for pk in PLANETS]].sum(axis=1)
df['sarva_bucket'] = pd.cut(df['sarvashtakavarga'], bins=[-1,24,48,72],
                             labels=['LOW','MEDIUM','HIGH'])

# Jupiter dignity
JU_SIGN_DIGNITY = {
    8:3, 11:3,  # own: Sag(8), Pis(11)
    3:4,         # exalt: Can(3)
    9:-1,        # debil: Cap(9)
    # friendly: Ta(1), Ge(2), Li(6), Sc(7) → 2; neutral/enemy rest → 1/0
}
def jupiter_dignity(sid_ju):
    si = sign_idx(sid_ju)
    if si in JU_SIGN_DIGNITY: return JU_SIGN_DIGNITY[si]
    if si in {1,2,6,7}: return 2    # friendly
    if si in {0,4,5}:   return 1    # neutral
    return 0                          # enemy
df['ju_dignity'] = df['sid_Ju'].apply(jupiter_dignity)

# Saturn dignity
SA_SIGN_DIGNITY = {9:3, 10:3, 6:4, 0:-1}
def saturn_dignity(sid_sa):
    si = sign_idx(sid_sa)
    if si in SA_SIGN_DIGNITY: return SA_SIGN_DIGNITY[si]
    if si in {2,5,11}: return 2
    if si in {3,7,8}:  return 1
    return 0
df['sa_dignity'] = df['sid_Sa'].apply(saturn_dignity)

# Moon strength composite
nak_qual_score = {'LAGHU':2,'MRIDU':2,'DHRUVA':1,'CHARA':1,'MISHRA':0,'UGRA':-1,'TIKSHNA':-2}
df['moon_strength'] = (
    df['paksha'].apply(lambda p: 1 if p=='SHUKLA' else -1) +
    df['moon_nak_quality'].apply(lambda q: nak_qual_score.get(q, 0)) +
    df['combust_Mo'].apply(lambda c: -2 if c == 1 else 0)
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.5  MACRO CYCLE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("1.5 Macro cycle features …")

# Jupiter-Saturn synodic angle
df['ju_sa_angle'] = df.apply(lambda r: norm360(r['sid_Sa'] - r['sid_Ju']), axis=1)
def ju_sa_quadrant(ang):
    if ang < 90:   return 'Q1_CONJUNCTION'
    if ang < 180:  return 'Q2_WAXING'
    if ang < 270:  return 'Q3_OPPOSITION'
    return 'Q4_WANING'
df['ju_sa_quad'] = df['ju_sa_angle'].apply(ju_sa_quadrant)

# Vimshottari Dasha from Moon Nakshatra on Nifty 50 inception: 1996-04-22
# Moon nak on that day:
inception_rec = next(r for r in json.load(open(f"{REPO}/data.json"))['records'] if r['date']=='1996-04-22')
ayan_la_inception = inception_rec['ayan']['la']
mo_sid_inception  = (inception_rec['p']['Mo'] - ayan_la_inception + 360) % 360
inception_nak_idx = int(mo_sid_inception / (360/27))  # 0-based
inception_nak_num = inception_nak_idx + 1              # 1-based

DASHA_ORDER  = ['Ketu','Venus','Sun','Moon','Mars','Rahu','Jupiter','Saturn','Mercury']
DASHA_YEARS  = {'Ketu':7,'Venus':20,'Sun':6,'Moon':10,'Mars':7,'Rahu':18,'Jupiter':16,'Saturn':19,'Mercury':17}
DASHA_NATURE = {
    'Jupiter':'BENEFIC','Venus':'BENEFIC','Moon':'MIXED',
    'Mercury':'MIXED','Sun':'MIXED','Mars':'MALEFIC',
    'Saturn':'MALEFIC','Rahu':'MALEFIC','Ketu':'MALEFIC'
}

# compute dasha balance at inception
# Moon was at nak_num=inception_nak_num, position within nak = fraction
nak_deg_size = 360 / 27
mo_deg_in_nak = mo_sid_inception % nak_deg_size
nak_fraction_elapsed = mo_deg_in_nak / nak_deg_size
start_lord_idx = (inception_nak_idx) % 9  # NK lord order aligns with dasha order
start_lord = DASHA_ORDER[start_lord_idx]
balance_years = DASHA_YEARS[start_lord] * (1 - nak_fraction_elapsed)

# Build Dasha timeline from inception
INCEPTION_DATE = pd.Timestamp('1996-04-22')
dasha_timeline = []  # list of (start_date, end_date, mahadasha)
cur_date = INCEPTION_DATE
lord_idx = start_lord_idx
remaining = balance_years
dasha_timeline.append((cur_date, cur_date + pd.DateOffset(days=int(remaining*365.25)), DASHA_ORDER[lord_idx]))
cur_date = dasha_timeline[-1][1]
lord_idx = (lord_idx + 1) % 9
while cur_date < pd.Timestamp('2030-01-01'):
    lord = DASHA_ORDER[lord_idx]
    end = cur_date + pd.DateOffset(days=int(DASHA_YEARS[lord]*365.25))
    dasha_timeline.append((cur_date, end, lord))
    cur_date = end
    lord_idx = (lord_idx + 1) % 9

def get_mahadasha(dt):
    for start, end, lord in dasha_timeline:
        if start <= dt < end: return lord
    return dasha_timeline[-1][2]

# Antardasha: sub-period within mahadasha proportional to dasha years
def get_antardasha(dt):
    for mstart, mend, mlord in dasha_timeline:
        if not (mstart <= dt < mend): continue
        m_dur = (mend - mstart).days
        # antardasha starts from same lord as mahadasha
        start_sub_idx = DASHA_ORDER.index(mlord)
        total_cycle = sum(DASHA_YEARS.values())
        ad_cur = mstart
        for i in range(9):
            sub_lord = DASHA_ORDER[(start_sub_idx + i) % 9]
            ad_dur_days = int(m_dur * DASHA_YEARS[sub_lord] / total_cycle)
            ad_end = ad_cur + pd.DateOffset(days=ad_dur_days)
            if ad_cur <= dt < ad_end: return sub_lord
            ad_cur = ad_end
        return mlord
    return 'Unknown'

print("  Computing Dasha for all dates …")
df['mahadasha']        = df['date'].apply(get_mahadasha)
df['antardasha']       = df['date'].apply(get_antardasha)
df['mahadasha_nature'] = df['mahadasha'].apply(lambda l: DASHA_NATURE.get(l,'MIXED'))
df['antardasha_nature']= df['antardasha'].apply(lambda l: DASHA_NATURE.get(l,'MIXED'))

def dasha_quality(maha_nat, antar_nat):
    score_map = {'BENEFIC':2,'MIXED':1,'MALEFIC':0}
    s = score_map[maha_nat] + score_map[antar_nat]
    if s >= 4: return 'BENEVOLENT'
    if s >= 2: return 'MIXED'
    return 'MALEFIC'
df['dasha_quality'] = df.apply(lambda r: dasha_quality(r['mahadasha_nature'], r['antardasha_nature']), axis=1)

# Yogi/Avayogi activation: any planet within 3° of Yogi or Avayogi point
df['yogi_activation']   = 0
df['avayogi_activation']= 0
for pk in PLANETS:
    yogi_dist   = (df[f'sid_{pk}'] - df['yogi_pt']).abs()
    yogi_dist   = yogi_dist.apply(lambda x: min(x, 360-x))
    avayogi_pt  = (df['yogi_pt'] + 6*(360/27)) % 360
    avayogi_dist= (df[f'sid_{pk}'] - avayogi_pt).abs()
    avayogi_dist= avayogi_dist.apply(lambda x: min(x, 360-x))
    df['yogi_activation']    = (df['yogi_activation']    | (yogi_dist   < 3).astype(int)).astype(int)
    df['avayogi_activation'] = (df['avayogi_activation'] | (avayogi_dist < 3).astype(int)).astype(int)

# Moon nak lord chain: lord's current sign and speed
df['moon_nakl_sign'] = df.apply(lambda r: {
    'Ketu':'sid_Ke','Venus':'sid_Ve','Sun':'sid_Su','Moon':'sid_Mo',
    'Mars':'sid_Ma','Rahu':'sid_Ra','Jupiter':'sid_Ju','Saturn':'sid_Sa','Mercury':'sid_Me'
}.get(r['moon_nak_lord'], 'sid_Su'), axis=1).apply(
    lambda col: 'unknown')  # placeholder — computed below

# Properly compute moon nak lord chain
def lord_col(lord, typ):
    lmap = {'Ketu':'Ke','Venus':'Ve','Sun':'Su','Moon':'Mo','Mars':'Ma',
            'Rahu':'Ra','Jupiter':'Ju','Saturn':'Sa','Mercury':'Me'}
    pk = lmap.get(lord,'Su')
    return f'{typ}_{pk}'

df['moon_nakl_cur_sign'] = df.apply(lambda r: SIGNS[int(r[lord_col(r['moon_nak_lord'],'sid')]/30)], axis=1)
df['moon_nakl_speed_flag']= df.apply(lambda r:
    'RETRO'     if r.get(f"retro_{lord_col(r['moon_nak_lord'],'spd').split('_')[1]}", 0) == 1
    else 'DIRECT', axis=1)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.6  INTERACTION FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("1.6 Interaction features …")

INTERACTIONS = {
    'ix_nak_tithi_qual':    lambda r: f"{r['moon_nak_name']}|{r['tithi_quality']}",
    'ix_nak_paksha':        lambda r: f"{r['moon_nak_name']}|{r['paksha']}",
    'ix_nak_regime':        lambda r: f"{r['moon_nak_name']}|{r['market_regime']}",
    'ix_nak_me_retro':      lambda r: f"{r['moon_nak_name']}|MeRetro={r['me_retro_confirmed']}",
    'ix_karana_moon_phase': lambda r: f"{r['karana']}|{r['moon_phase_quad']}",
    'ix_tithiq_nakq':       lambda r: f"{r['tithi_quality']}|{r['moon_nak_quality']}",
    'ix_dasha_regime':      lambda r: f"{r['dasha_quality']}|{r['market_regime']}",
    'ix_gandanta_war':      lambda r: f"Gand={r['gandanta_any']}|War={r['graha_yuddha']}",
    'ix_eclipse_moonphase': lambda r: f"Ecl={r['eclipse_corridor']}|{r['moon_phase_quad']}",
    'ix_sarva_regime':      lambda r: f"{r['sarva_bucket']}|{r['market_regime']}",
    'ix_sarva_dasha':       lambda r: f"{r['sarva_bucket']}|{r['dasha_quality']}",
    'ix_jusa_paksha':       lambda r: f"{r['ju_sa_quad']}|{r['paksha']}",
    'ix_rahu_sign_nak':     lambda r: f"{r['rahu_sign']}|{r['moon_nak_name']}",
    'ix_comb_benefic_regime': lambda r: f"CombBenefic={r['combust_any_benefic']}|{r['market_regime']}",
    'ix_yogi_paksha_tithiq':  lambda r: f"YogiAct={r['yogi_activation']}|{r['paksha']}|{r['tithi_quality']}",
}

for col, fn in INTERACTIONS.items():
    df[col] = df.apply(fn, axis=1)

# Prune levels with < 30 occurrences
for col in INTERACTIONS:
    counts = df[col].value_counts()
    rare   = counts[counts < 30].index
    df[col] = df[col].apply(lambda x: 'RARE' if x in rare else x)

# ═══════════════════════════════════════════════════════════════════════════════
# LOOKAHEAD AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print("Lookahead audit …")
import re
fwd_cols = [c for c in df.columns if re.match(r'fwd_(ret|dir)_\d+d', c)]
for col in fwd_cols:
    n = int(re.search(r'_(\d+)d', col).group(1))
    tail_nulls = df[col].tail(n).isna().all()
    assert tail_nulls, f"LOOKAHEAD FAIL on {col} — tail({n}) has non-NaN values"
print("  PASS: All forward-return columns have NaN in final rows (no lookahead)")

# ═══════════════════════════════════════════════════════════════════════════════
# BANK NIFTY MERGE
# ═══════════════════════════════════════════════════════════════════════════════
print("Building Bank Nifty enriched dataset …")
with open(f"{REPO}/ohlc_banknifty.json") as f:
    bn_raw = json.load(f)['records']

bn = pd.DataFrame([
    {'date': pd.Timestamp(dt), 'bn_open':v['o'],'bn_high':v['h'],
     'bn_low':v['l'],'bn_close':v['c'],'bn_chg':v['g']}
    for dt, v in bn_raw.items()
]).sort_values('date').reset_index(drop=True)

# merge planets onto BN dates
dfbn = bn.merge(df.drop(columns=['open','high','low','close']), on='date', how='left')
dfbn['log_ret']   = np.log(dfbn['bn_close'] / dfbn['bn_close'].shift(1))
dfbn['range_pct'] = (dfbn['bn_high'] - dfbn['bn_low']) / dfbn['bn_close']
for n in [1,2,3,5,10]:
    dfbn[f'fwd_ret_{n}d'] = np.log(dfbn['bn_close'].shift(-n) / dfbn['bn_close'])
    dfbn[f'fwd_dir_{n}d'] = pd.array(
        [pd.NA if pd.isna(v) else int(v > 0) for v in dfbn[f'fwd_ret_{n}d']], dtype='Int8'
    )
dfbn.drop(columns=['bn_chg'], inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
print("Saving enriched datasets …")
df.to_csv(f"{REPO}/data/nifty_enriched.csv", index=False)
dfbn.to_csv(f"{REPO}/data/banknifty_enriched.csv", index=False)

# Column + null report
print(f"\n{'='*60}")
print(f"nifty_enriched.csv   : {len(df)} rows × {len(df.columns)} columns")
print(f"banknifty_enriched.csv: {len(dfbn)} rows × {len(dfbn.columns)} columns")
null_nifty = df.isnull().sum()
null_nifty = null_nifty[null_nifty > 0]
print(f"\nNifty nulls (expected for warm-up only):")
print(null_nifty.to_string())
print(f"\nAll forward return nulls due to insufficient future rows — EXPECTED, NOT DATA ERROR")
print("\nStep 1 COMPLETE")
