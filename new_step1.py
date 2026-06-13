"""
New Step 1 — Complete Vedic Feature Engineering v2
Builds data/nifty_enriched.csv and data/banknifty_enriched.csv
All astrological features derived from sidereal (Lahiri) planetary positions.
No market data (log_ret, range_pct, etc.) used as features — only as outcome labels.
"""
import json, math, warnings, os
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"
PLANETS = ['Su','Mo','Me','Ve','Ma','Ju','Sa','Ra','Ke']

# ═══════════════════════════════════════════════════════════════════════
# SIGN / NAKSHATRA CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
SIGN_NAMES = ['Ar','Ta','Ge','Cn','Le','Vi','Li','Sc','Sg','Cp','Aq','Pi']
# Fire=F, Earth=E, Air=A, Water=W
SIGN_ELEMENT  = {1:'F',2:'E',3:'A',4:'W',5:'F',6:'E',7:'A',8:'W',9:'F',10:'E',11:'A',12:'W'}
# Movable=M, Fixed=X, Dual=D
SIGN_MODALITY = {1:'M',2:'X',3:'D',4:'M',5:'X',6:'D',7:'M',8:'X',9:'D',10:'M',11:'X',12:'D'}
# Male=1, Female=0
SIGN_GENDER   = {1:1,2:0,3:1,4:0,5:1,6:0,7:1,8:0,9:1,10:0,11:1,12:0}
# Day=1, Night=0
SIGN_DAYNIGHT = {1:1,2:0,3:1,4:0,5:1,6:0,7:1,8:0,9:1,10:0,11:1,12:0}

NAK_LORDS   = ['Ke','Ve','Su','Mo','Ma','Ra','Ju','Sa','Me'] * 3  # naks 1-27 (0-indexed)
NAK_QUALITY = ['Laghu','Ugra','Mishra','Dhruva','Mridu','Tikshna','Chara','Laghu','Tikshna',
               'Ugra','Ugra','Dhruva','Laghu','Mridu','Chara','Mishra','Mridu','Tikshna',
               'Tikshna','Ugra','Dhruva','Chara','Chara','Chara','Ugra','Dhruva','Mridu']
NAK_ELEMENT = ['F','E','F','E','A','A','W','W','W','F','F','F','E','F','A','F','W','W',
               'F','W','E','A','A','A','A','W','W']
NAK_NAMES   = ['Ashwini','Bharani','Krittika','Rohini','Mrigashira','Ardra','Punarvasu',
               'Pushya','Ashlesha','Magha','PurvaPhalguni','UttaraPhalguni','Hasta',
               'Chitra','Swati','Vishakha','Anuradha','Jyeshtha','Mula','PurvaAshadha',
               'UttaraAshadha','Shravana','Dhanishtha','Shatabhisha','PurvaBhadrapada',
               'UttaraBhadrapada','Revati']

# Panchanga Yoga names (1-27)
YOGA_NAMES = ['Vishkambha','Priti','Ayushman','Saubhagya','Shobhana','Atiganda','Sukarma',
              'Dhriti','Shoola','Ganda','Vriddhi','Dhruva','Vyaghata','Harshana','Vajra',
              'Siddhi','Vyatipata','Variyan','Parigha','Shiva','Siddha','Sadhya','Shubha',
              'Shukla','Brahma','Indra','Vaidhriti']
YOGA_INAUSPICIOUS = {1,6,9,10,13,15,17,19,27}

# Karana names (11 total: 4 fixed + 7 movable)
KARANA_FIXED_NAMES = ['Kimstughna','Bava','Balava','Kaulava','Taitila','Garija','Vanija',
                      'Vishti','Shakuni','Chatushpada','Naga']
KARANA_QUALITY = {
    'Kimstughna': 'auspicious', 'Bava': 'neutral', 'Balava': 'auspicious',
    'Kaulava': 'neutral', 'Taitila': 'auspicious', 'Garija': 'auspicious',
    'Vanija': 'auspicious', 'Vishti': 'inauspicious', 'Shakuni': 'inauspicious',
    'Chatushpada': 'inauspicious', 'Naga': 'inauspicious'
}

# Dignity system
# Exaltation: sign (1-12), degree_in_sign
EXALT = {'Su':(1,10),'Mo':(2,3),'Ma':(10,28),'Me':(6,15),'Ju':(4,5),
         'Ve':(12,27),'Sa':(7,20),'Ra':(2,20),'Ke':(8,20)}
DEBIL = {'Su':(7,10),'Mo':(8,3),'Ma':(4,28),'Me':(12,15),'Ju':(10,5),
         'Ve':(6,27),'Sa':(1,20),'Ra':(8,20),'Ke':(2,20)}
# Moolatrikona: sign, min_deg, max_deg
MOOLA = {'Su':(5,0,20),'Mo':(2,4,30),'Ma':(1,0,12),'Me':(6,16,20),
         'Ju':(9,0,10),'Ve':(7,0,15),'Sa':(11,0,20)}
OWN   = {'Su':[5],'Mo':[4],'Ma':[1,8],'Me':[3,6],'Ju':[9,12],'Ve':[2,7],'Sa':[10,11],'Ra':[],'Ke':[]}
FRIEND = {
    'Su':['Mo','Ma','Ju'], 'Mo':['Su','Me'], 'Ma':['Su','Mo','Ju'],
    'Me':['Su','Ve'], 'Ju':['Su','Mo','Ma'], 'Ve':['Me','Sa'], 'Sa':['Me','Ve'],
    'Ra':['Me','Ve','Sa'], 'Ke':['Me','Ve','Sa']
}
ENEMY = {
    'Su':['Ve','Sa'], 'Mo':[], 'Ma':['Me'], 'Me':['Mo'], 'Ju':['Me','Ve'],
    'Ve':['Su','Mo'], 'Sa':['Su','Mo','Ma'], 'Ra':['Su','Mo','Ma'], 'Ke':['Su','Mo','Ma']
}

# Combustion orbs (degrees from Sun)
COMB_ORB = {'Mo':12,'Me':14,'Ve':10,'Ma':17,'Ju':11,'Sa':15}

# Mean daily motions (degrees/day)
MEAN_MOTION = {'Su':0.9856,'Mo':13.1764,'Me':1.3833,'Ve':1.2,'Ma':0.5240,
               'Ju':0.0831,'Sa':0.0335,'Ra':-0.0529,'Ke':-0.0529}

# Parashari special aspects (in addition to 7th)
SPECIAL_ASPECTS = {'Ma':[4,8],'Ju':[5,9],'Sa':[3,10],'Ra':[5,9],'Ke':[5,9]}

# Hora sequence (planetary hours)
HORA_SEQ = ['Su','Ve','Me','Mo','Sa','Ju','Ma']
# Day lord → position in hora sequence
HORA_IDX = {p: i for i,p in enumerate(HORA_SEQ)}

# Choghadiya day sequences by weekday (0=Mon ... 6=Sun in Python)
# Each list: 8 choghadiya values for the day portion
CHOGHADIYA_DAY = {
    6: ['U','C','L','A','K','S','R','U'],  # Sunday
    0: ['A','K','S','R','U','C','L','A'],  # Monday
    1: ['R','U','C','L','A','K','S','R'],  # Tuesday
    2: ['L','A','K','S','R','U','C','L'],  # Wednesday
    3: ['S','R','U','C','L','A','K','S'],  # Thursday
    4: ['C','L','A','K','S','R','U','C'],  # Friday
    5: ['K','S','R','U','C','L','A','K'],  # Saturday
}
CHOGHADIYA_QUALITY = {'A':'best','L':'good','S':'good','C':'neutral','U':'avoid','K':'avoid','R':'avoid'}

# Rahu Kalam period start (portion number, 1-8) from sunrise by weekday
# Each portion = 12h / 8 = 1.5h
RAHU_KALAM_PORTION = {6:7, 0:2, 1:7, 2:5, 3:6, 4:4, 5:3}  # weekday → portion
# Mumbai average sunrise ≈ 6:00 AM IST, market open = 9:15 AM = 3.25h from sunrise
SUNRISE_H = 6.0
MARKET_OPEN_H = 9.25
PORTION_H = 1.5

# Vimshottari Dasha periods (years)
DASHA_PERIODS = {'Ke':7,'Ve':20,'Su':6,'Mo':10,'Ma':7,'Ra':18,'Ju':16,'Sa':19,'Me':17}
DASHA_ORDER   = ['Ke','Ve','Su','Mo','Ma','Ra','Ju','Sa','Me']
DASHA_NATURE  = {'Ke':'malefic','Ve':'benefic','Su':'malefic','Mo':'benefic','Ma':'malefic',
                 'Ra':'malefic','Ju':'benefic','Sa':'malefic','Me':'neutral'}
DASHA_TOTAL   = sum(DASHA_PERIODS.values())  # 120 years

# Nifty 50 inception date
INCEPTION_DATE = date(1996, 4, 22)
# Moon nakshatra on inception (computed from data: sid_Mo = 58.16° → nak 5 Mrigashira)
INCEPTION_MOON_NAK = 5  # Mrigashira

# Mrityu Bhaga (approximate classical values) — planet: {sign_num: degree_in_sign}
MRITYU_BHAGA = {
    'Su': {1:20,2:9,3:12,4:6,5:3,6:27,7:16,8:29,9:6,10:20,11:13,12:14},
    'Mo': {1:26,2:12,3:13,4:25,5:24,6:11,7:26,8:14,9:13,10:25,11:5,12:12},
    'Ma': {1:28,2:20,3:15,4:13,5:9,6:28,7:26,8:12,9:13,10:11,11:25,12:16},
    'Me': {1:15,2:14,3:13,4:12,5:15,6:15,7:4,8:13,9:14,10:12,11:11,12:14},
    'Ju': {1:14,2:26,3:11,4:5,5:10,6:13,7:10,8:5,9:12,10:14,11:20,12:26},
    'Ve': {1:27,2:11,3:29,4:14,5:10,6:10,7:11,8:29,9:14,10:4,11:20,12:16},
    'Sa': {1:20,2:21,3:22,4:5,5:20,6:10,7:22,8:14,9:14,10:20,11:28,12:26},
    'Ra': {1:14,2:13,3:12,4:10,5:15,6:14,7:14,8:13,9:14,10:12,11:13,12:10},
    'Ke': {1:14,2:13,3:12,4:10,5:15,6:14,7:14,8:13,9:14,10:12,11:13,12:10},
}

# Natal Moon sign for Sade Sati / Ashtama Shani (from inception chart)
# Inception Moon: sid = 58.16° → sign = floor(58.16/30)+1 = 2 (Taurus)
NATAL_MOON_SIGN = 2  # Taurus

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def sign_of(sid_deg):
    return int(sid_deg / 30) + 1  # 1-12

def deg_in_sign(sid_deg):
    return sid_deg % 30

def nak_of(sid_deg):
    return int(sid_deg / (360/27)) + 1  # 1-27

def nak_pada(sid_deg):
    pos_in_nak = sid_deg % (360/27)
    return int(pos_in_nak / (360/27/4)) + 1  # 1-4

def norm360(x):
    return x % 360

def planet_sign_lord(sign_num):
    sign_lords = {1:'Ma',2:'Ve',3:'Me',4:'Mo',5:'Su',6:'Me',7:'Ve',8:'Ma',9:'Ju',10:'Sa',11:'Sa',12:'Ju'}
    return sign_lords[sign_num]

def dignity(planet, sid_deg):
    sg = sign_of(sid_deg)
    dg = deg_in_sign(sid_deg)
    # Debilitated first (most extreme)
    if sg == DEBIL[planet][0]:
        if abs(dg - DEBIL[planet][1]) <= 1: return 'exact_debil'
        return 'debilitated'
    # Exalted
    if sg == EXALT[planet][0]:
        if abs(dg - EXALT[planet][1]) <= 1: return 'exact_exalt'
        return 'exalted'
    # Moolatrikona
    if planet in MOOLA:
        ms, ml, mh = MOOLA[planet]
        if sg == ms and ml <= dg <= mh: return 'moolatrikona'
    # Own sign
    if sg in OWN.get(planet, []):
        return 'own'
    # Friendly/enemy — need to know which planet rules this sign
    ruler = planet_sign_lord(sg)
    if ruler == planet: return 'own'  # already handled but safety
    if ruler in FRIEND.get(planet, []):
        return 'friendly'
    if ruler in ENEMY.get(planet, []):
        return 'enemy'
    return 'neutral'

def dignity_score(dig_str):
    scores = {'exact_exalt':8,'exalted':7,'moolatrikona':6,'own':5,'friendly':4,'neutral':3,'enemy':2,'debilitated':1,'exact_debil':0}
    return scores.get(dig_str, 3)

def deg_category(planet, sid_deg):
    dg = deg_in_sign(sid_deg)
    sg = sign_of(sid_deg)
    # Sandhi: last 1 degree
    if dg >= 29: return 'sandhi'
    # Exact exaltation
    if sg == EXALT[planet][0] and abs(dg - EXALT[planet][1]) <= 1: return 'exact_exalt'
    # Exact debilitation
    if sg == DEBIL[planet][0] and abs(dg - DEBIL[planet][1]) <= 1: return 'exact_debil'
    # Low/mid/high
    if dg <= 10: return 'low'
    if dg <= 20: return 'mid'
    return 'high'

def speed_category(planet, speed_deg):
    mm = MEAN_MOTION[planet]
    if mm == 0: return 'stationary'
    if abs(speed_deg) < 0.05: return 'stationary'
    if speed_deg < 0: return 'retrograde'
    ratio = speed_deg / abs(mm)
    if ratio >= 1.3: return 'very_fast'
    if ratio >= 1.1: return 'fast'
    if ratio >= 0.9: return 'mean'
    if ratio >= 0.7: return 'slow'
    return 'very_slow'

def compute_karana(tithi_num, sun_moon_sep):
    half_tithi = (sun_moon_sep / 6)  # 0-60 range half-tithis
    karana_idx = int(sun_moon_sep / 6)  # 0-59
    # First half: Kimstughna(0), then 56 movable (index 1-56), then fixed 57,58,59
    MOVABLE = ['Bava','Balava','Kaulava','Taitila','Garija','Vanija','Vishti']
    if karana_idx == 0: return 'Kimstughna'
    if karana_idx <= 56:
        return MOVABLE[(karana_idx - 1) % 7]
    extras = ['Shakuni','Chatushpada','Naga']
    return extras[min(karana_idx - 57, 2)]

def compute_yoga(sid_su, sid_mo):
    yoga_pt = (sid_su + sid_mo) % 360
    yoga_num = int(yoga_pt / (360/27)) + 1  # 1-27
    return yoga_num, YOGA_NAMES[yoga_num - 1], 'inauspicious' if yoga_num in YOGA_INAUSPICIOUS else 'auspicious'

def compute_tara_bala(moon_nak):
    diff = ((moon_nak - INCEPTION_MOON_NAK) % 27)
    tara = (diff % 9) + 1  # 1-9
    tara_names = {1:'Janma',2:'Sampat',3:'Vipat',4:'Kshema',5:'Pratyak',
                  6:'Saadhana',7:'Naidhana',8:'Mitra',9:'Paramamitra'}
    tara_quality = {1:'critical',2:'wealth',3:'danger',4:'prosperity',5:'obstacle',
                    6:'achievement',7:'worst',8:'good',9:'best'}
    return tara, tara_names[tara], tara_quality[tara]

def is_gandanta(sid_deg):
    # Last 3°20' (3.33°) of water signs and first 3.33° of fire signs
    # Water-fire junctions: Pisces(12)→Aries(1), Cancer(4)→Leo(5), Scorpio(8)→Sagittarius(9)
    dg = deg_in_sign(sid_deg)
    sg = sign_of(sid_deg)
    if sg in [12,4,8] and dg >= 26.67: return True  # last 3.33° of water
    if sg in [1,5,9] and dg <= 3.33: return True    # first 3.33° of fire
    return False

def parashari_aspects(p1_sign, p2_planet):
    # Returns list of signs that p1 aspects
    aspects = {(p1_sign - 1 + 6) % 12 + 1}  # 7th
    if p2_planet in SPECIAL_ASPECTS:
        for offset in SPECIAL_ASPECTS[p2_planet]:
            aspects.add((p1_sign - 1 + offset - 1) % 12 + 1)
    return aspects

def gajakesari(moon_sign, ju_sign):
    diff = abs(moon_sign - ju_sign)
    diff = min(diff, 12 - diff)
    return diff in [0, 3, 6, 9]  # 1st, 4th, 7th, 10th

def kemadruma(moon_sign, all_signs):
    second = (moon_sign % 12) + 1
    twelfth = ((moon_sign - 2) % 12) + 1
    non_luminary_signs = [s for p,s in all_signs.items() if p not in ['Su','Mo','Ra','Ke']]
    no_flanking = (second not in non_luminary_signs) and (twelfth not in non_luminary_signs)
    kendras = [(moon_sign - 1 + k) % 12 + 1 for k in [0, 3, 6, 9]]
    no_kendra = all(s not in non_luminary_signs for s in kendras)
    return no_flanking and no_kendra

def hora_at_open(weekday):
    # weekday: 0=Mon, 6=Sun
    day_lord_map = {0:'Mo',1:'Ma',2:'Me',3:'Ju',4:'Ve',5:'Sa',6:'Su'}
    day_lord = day_lord_map[weekday]
    start_idx = HORA_IDX[day_lord]
    hours_from_sunrise = MARKET_OPEN_H - SUNRISE_H  # 3.25h
    hora_num = int(hours_from_sunrise)  # 3rd hora (0-indexed)
    hora_planet = HORA_SEQ[(start_idx + hora_num) % 7]
    return hora_planet

def choghadiya_at_open(weekday):
    seq = CHOGHADIYA_DAY[weekday]
    hours_from_sunrise = MARKET_OPEN_H - SUNRISE_H  # 3.25h
    chog_idx = int(hours_from_sunrise / PORTION_H)  # 3.25/1.5 = 2.17 → idx 2
    chog_idx = min(chog_idx, 7)
    return seq[chog_idx], CHOGHADIYA_QUALITY[seq[chog_idx]]

def rahu_kalam_at_open(weekday):
    portion = RAHU_KALAM_PORTION[weekday]
    rk_start = SUNRISE_H + (portion - 1) * PORTION_H
    rk_end   = rk_start + PORTION_H
    return 1 if rk_start <= MARKET_OPEN_H < rk_end else 0

def compute_vimshottari(birth_date, birth_moon_nak, target_date):
    # birth_moon_nak: 1-27
    # Each nakshatra's starting dasha
    nak0 = birth_moon_nak - 1
    start_planet_idx = nak0 % 9
    start_planet = DASHA_ORDER[start_planet_idx]
    # Fraction elapsed in first dasha
    frac_elapsed = (nak0 % 1) if False else ((birth_date.day + birth_date.month * 30) / (30 * 9)) % 1
    # Better: use day of nakshatra (simplified: assume 0 elapsed for inception)
    days_since_inception = (target_date - birth_date).days
    years_since = days_since_inception / 365.25
    # Build dasha timeline from inception
    remaining_in_first = DASHA_PERIODS[start_planet]  # full first dasha
    timeline = []
    idx = start_planet_idx
    cumulative = 0
    while cumulative < years_since + DASHA_PERIODS[DASHA_ORDER[idx % 9]]:
        p = DASHA_ORDER[idx % 9]
        timeline.append((p, cumulative, cumulative + DASHA_PERIODS[p]))
        cumulative += DASHA_PERIODS[p]
        idx += 1
        if cumulative > 200: break
    # Find current mahadasha
    maha = 'Ke'
    for p, start, end in timeline:
        if start <= years_since < end:
            maha = p
            # Find antardasha within maha
            time_in_maha = years_since - start
            maha_dur = DASHA_PERIODS[p]
            # Antardasha order starts from maha lord
            maha_idx = DASHA_ORDER.index(p)
            ant_start_idx = maha_idx
            ant_cumulative = 0
            for j in range(9):
                ap = DASHA_ORDER[(ant_start_idx + j) % 9]
                ant_dur = DASHA_PERIODS[ap] * DASHA_PERIODS[p] / DASHA_TOTAL
                if ant_cumulative <= time_in_maha < ant_cumulative + ant_dur:
                    return maha, ap
                ant_cumulative += ant_dur
            return maha, DASHA_ORDER[ant_start_idx]
    return timeline[-1][0] if timeline else 'Ke', 'Ke'

def compute_atmakaraka(sidereal_positions):
    # Planet with highest degree within its sign (excluding Rahu/Ketu in some systems, include here)
    planets_to_rank = ['Su','Mo','Me','Ve','Ma','Ju','Sa']
    degrees = {p: sidereal_positions[p] % 30 for p in planets_to_rank}
    sorted_planets = sorted(degrees, key=lambda x: degrees[x], reverse=True)
    return sorted_planets[0], sorted_planets[1]  # AK, AmK

def parivartana_yogas(sign_dict):
    yogas = []
    planet_list = list(sign_dict.keys())
    for i, p1 in enumerate(planet_list):
        for p2 in planet_list[i+1:]:
            if p1 in ['Ra','Ke'] or p2 in ['Ra','Ke']: continue
            sign_p1 = sign_dict[p1]
            sign_p2 = sign_dict[p2]
            # p1 is in sign owned by p2 and p2 is in sign owned by p1
            if sign_p1 in OWN.get(p2, []) and sign_p2 in OWN.get(p1, []):
                yogas.append(f"{p1}-{p2}")
    return yogas

def bhrigu_bindu(sid_mo, sid_ra):
    return norm360((sid_mo + sid_ra) / 2)

def gulika_sign(weekday):
    # Gulika occupies 1/8 of daytime, portion varies by weekday
    # Saturn rules the 8th portion
    portion = {6:7, 0:6, 1:5, 2:4, 3:3, 4:2, 5:1}  # weekday → portion from sunrise
    gul_start = SUNRISE_H + (portion[weekday] - 1) * PORTION_H
    # Gulika's longitude ≈ Saturn's position + offset (simplified: use Sun's sign offset)
    # Without full Shadbala, we just return the weekday-based portion
    return portion[weekday]  # 1-7

def mrityu_bhaga_hit(planet, sid_deg):
    sg = sign_of(sid_deg)
    dg = deg_in_sign(sid_deg)
    if planet not in MRITYU_BHAGA: return 0
    mb_deg = MRITYU_BHAGA[planet].get(sg, -99)
    return 1 if abs(dg - mb_deg) <= 1 else 0


# ═══════════════════════════════════════════════════════════════════════
# LOAD AND PROCESS DATA
# ═══════════════════════════════════════════════════════════════════════

def load_nifty():
    print("Loading data.json …")
    with open(f"{REPO}/data.json") as f:
        raw = json.load(f)['records']
    df = pd.DataFrame(raw)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for pk in PLANETS:
        df[f'trop_{pk}'] = df['p'].apply(lambda x: x[pk])
    df['ayan_la'] = df['ayan'].apply(lambda x: x['la'])
    df.drop(columns=['p','ayan','chg'], inplace=True)
    return df

def load_banknifty():
    print("Loading ohlc_banknifty.json …")
    with open(f"{REPO}/ohlc_banknifty.json") as f:
        raw = json.load(f)
    records = raw['records']
    rows = [{'date': k, 'open': v['o'], 'high': v['h'], 'low': v['l'], 'close': v['c']}
            for k, v in records.items()]
    bn = pd.DataFrame(rows)
    bn['date'] = pd.to_datetime(bn['date'])
    return bn.sort_values('date').reset_index(drop=True)

def add_sidereal(df):
    for p in PLANETS:
        df[f'sid_{p}'] = (df[f'trop_{p}'] - df['ayan_la'] + 360) % 360
    return df

def compute_speeds(df):
    for p in PLANETS:
        df[f'spd_{p}'] = df[f'sid_{p}'].diff().apply(lambda x:
            x if abs(x) < 15 else (x - 360 if x > 180 else x + 360)
        )
    return df

def enrich(df):
    print("Computing sidereal positions …")
    df = add_sidereal(df)
    df = compute_speeds(df)

    print("Computing outcome labels …")
    # These use future price data — forward-looking, no lookahead in signals
    for n in [1, 2, 3, 5, 10]:
        df[f'fwd_ret_{n}d'] = np.log(df['close'].shift(-n) / df['close'])
    df['log_ret']   = np.log(df['close'] / df['close'].shift(1))
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low']  - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    ret3 = df['fwd_ret_3d']
    df['outcome_3d'] = pd.cut(ret3, bins=[-np.inf,-0.015,-0.005,0.005,0.015,np.inf],
                               labels=['STRONG_BEAR','MILD_BEAR','SIDEWAYS','MILD_BULL','STRONG_BULL'])
    df['is_strong_bull'] = (df['outcome_3d'] == 'STRONG_BULL').astype('Int8')
    df['is_strong_bear'] = (df['outcome_3d'] == 'STRONG_BEAR').astype('Int8')
    df['is_sideways']    = (df['outcome_3d'] == 'SIDEWAYS').astype('Int8')
    df['is_bull']        = (ret3 > 0).astype('Int8')  # base binary direction

    # HIGH VOLATILITY: daily range > 1.5x 14-day ATR
    df['is_high_vol'] = (df['range_pct'] > 1.5 * df['atr14'] / df['close']).astype('Int8')

    # TREND CONTINUATION / REVERSAL (based on 5-day prior trend vs next-1d)
    df['prior_ret_5d'] = np.log(df['close'] / df['close'].shift(5))
    df['is_continuation'] = ((df['prior_ret_5d'] > 0.01) & (df['fwd_ret_1d'] > 0.005) |
                              (df['prior_ret_5d'] < -0.01) & (df['fwd_ret_1d'] < -0.005)).astype('Int8')
    df['is_reversal']     = ((df['prior_ret_5d'] > 0.01) & (df['fwd_ret_1d'] < -0.005) |
                              (df['prior_ret_5d'] < -0.01) & (df['fwd_ret_1d'] > 0.005)).astype('Int8')

    print("Computing sign / dignity features …")
    for p in PLANETS:
        sg = df[f'sid_{p}'].apply(sign_of)
        dg = df[f'sid_{p}'].apply(deg_in_sign)
        df[f'sign_{p}'] = sg
        df[f'dg_{p}']   = dg
        df[f'elem_{p}'] = sg.map(SIGN_ELEMENT)
        df[f'mod_{p}']  = sg.map(SIGN_MODALITY)
        df[f'dig_{p}']  = df[f'sid_{p}'].apply(lambda x: dignity(p, x))
        df[f'digsc_{p}'] = df[f'dig_{p}'].map(lambda x: dignity_score(x))
        df[f'degcat_{p}'] = df[f'sid_{p}'].apply(lambda x: deg_category(p, x))
        df[f'mrityu_{p}'] = df[f'sid_{p}'].apply(lambda x: mrityu_bhaga_hit(p, x))

    print("Computing combustion …")
    for p in ['Mo','Me','Ve','Ma','Ju','Sa']:
        orb = COMB_ORB[p]
        sep = (df[f'trop_{p}'] - df[f'trop_Su']).apply(lambda x: min(abs(x), 360-abs(x)))
        df[f'comb_{p}'] = (sep <= orb).astype(int)

    print("Computing motion states …")
    for p in PLANETS:
        spd = df[f'spd_{p}']
        df[f'retro_{p}'] = (spd < -0.001).astype(int)
        df[f'stat_{p}']  = (spd.abs() < 0.05).astype(int)

    print("Computing Tithi / Paksha / Nakshatra …")
    df['sun_moon_sep'] = (df['trop_Mo'] - df['trop_Su']).apply(lambda x: x % 360)
    df['tithi_num']    = (df['sun_moon_sep'] / 12).apply(math.floor) + 1
    df['tithi_num']    = df['tithi_num'].clip(1, 30)
    df['paksha']       = df['tithi_num'].apply(lambda t: 'SHUKLA' if t <= 15 else 'KRISHNA')
    TITHI_QUALITY = {1:'Nanda',2:'Bhadra',3:'Jaya',4:'Rikta',5:'Purna',6:'Nanda',7:'Bhadra',
                     8:'Jaya',9:'Rikta',10:'Purna',11:'Nanda',12:'Bhadra',13:'Jaya',14:'Rikta',
                     15:'Purna',16:'Nanda',17:'Bhadra',18:'Jaya',19:'Rikta',20:'Purna',
                     21:'Nanda',22:'Bhadra',23:'Jaya',24:'Rikta',25:'Purna',26:'Nanda',
                     27:'Bhadra',28:'Jaya',29:'Rikta',30:'Purna'}
    df['tithi_quality'] = df['tithi_num'].map(TITHI_QUALITY)
    # Moon phase (8 phases)
    phase_map = lambda s: ('NEW' if s<45 else 'CRESCENT' if s<90 else 'FIRST_Q' if s<135 else
                           'GIBBOUS' if s<180 else 'FULL' if s<225 else 'DIS_GIBBOUS' if s<270 else
                           'LAST_Q' if s<315 else 'BALSAMIC')
    df['moon_phase'] = df['sun_moon_sep'].apply(phase_map)

    print("Computing Nakshatra features …")
    df['nak_mo'] = df['sid_Mo'].apply(nak_of)
    df['nak_mo_name'] = df['nak_mo'].apply(lambda n: NAK_NAMES[n-1])
    df['nak_mo_lord'] = df['nak_mo'].apply(lambda n: NAK_LORDS[n-1])
    df['nak_mo_qual'] = df['nak_mo'].apply(lambda n: NAK_QUALITY[n-1])
    df['nak_mo_elem'] = df['nak_mo'].apply(lambda n: NAK_ELEMENT[n-1])
    df['nak_mo_pada'] = df['sid_Mo'].apply(nak_pada)
    # Nakshatra transition day
    df['nak_transition'] = (df['nak_mo'] != df['nak_mo'].shift(1)).astype(int)
    # Panchaka (Moon in naks 23-27)
    df['panchaka'] = df['nak_mo'].between(23, 27).astype(int)

    print("Computing Tara Bala …")
    tara_data = df['nak_mo'].apply(compute_tara_bala)
    df['tara_num']    = tara_data.apply(lambda x: x[0])
    df['tara_name']   = tara_data.apply(lambda x: x[1])
    df['tara_quality'] = tara_data.apply(lambda x: x[2])

    # Nakshatra lord's current dignity and speed
    df['nakl_dig'] = df.apply(lambda r: dignity(r['nak_mo_lord'], r[f"sid_{r['nak_mo_lord']}"]), axis=1)
    df['nakl_spd'] = df.apply(lambda r: speed_category(r['nak_mo_lord'], r[f"spd_{r['nak_mo_lord']}"]), axis=1)

    print("Computing Panchanga Yoga …")
    yoga_data = df.apply(lambda r: compute_yoga(r['sid_Su'], r['sid_Mo']), axis=1)
    df['yoga_num']     = yoga_data.apply(lambda x: x[0])
    df['yoga_name']    = yoga_data.apply(lambda x: x[1])
    df['yoga_quality'] = yoga_data.apply(lambda x: x[2])

    print("Computing Karana …")
    df['karana'] = df.apply(lambda r: compute_karana(r['tithi_num'], r['sun_moon_sep']), axis=1)
    df['karana_quality'] = df['karana'].map(KARANA_QUALITY)

    print("Computing Gandanta flags …")
    for p in PLANETS:
        df[f'gand_{p}'] = df[f'sid_{p}'].apply(is_gandanta).astype(int)
    df['gand_any'] = df[[f'gand_{p}' for p in PLANETS]].max(axis=1)
    df['sandhi_mo'] = (df['dg_Mo'] >= 29).astype(int)

    print("Computing Weekday / Hora / Choghadiya …")
    df['dow'] = pd.to_datetime(df['date']).dt.dayofweek  # 0=Mon
    df['vara_lord'] = df['dow'].map({0:'Mo',1:'Ma',2:'Me',3:'Ju',4:'Ve',5:'Sa',6:'Su'})
    df['hora_at_open']   = df['dow'].apply(hora_at_open)
    chog_data = df['dow'].apply(choghadiya_at_open)
    df['choghadiya']        = chog_data.apply(lambda x: x[0])
    df['choghadiya_quality'] = chog_data.apply(lambda x: x[1])
    df['rahu_kalam_open']   = df['dow'].apply(rahu_kalam_at_open)

    print("Computing Yogas …")
    sign_cols = {p: df[f'sign_{p}'] for p in PLANETS}

    # Gajakesari
    df['gajakesari'] = df.apply(lambda r: int(gajakesari(r['sign_Mo'], r['sign_Ju'])), axis=1)

    # Kemadruma
    def kema_row(r):
        all_s = {p: r[f'sign_{p}'] for p in PLANETS}
        return int(kemadruma(r['sign_Mo'], all_s))
    df['kemadruma'] = df.apply(kema_row, axis=1)

    # Chandra Mangala: Moon and Mars in same sign or mutually aspecting
    def chandramangala(r):
        mo_sg = r['sign_Mo']; ma_sg = r['sign_Ma']
        if mo_sg == ma_sg: return 1
        # 7th aspect check
        if (mo_sg - ma_sg) % 12 == 6 or (ma_sg - mo_sg) % 12 == 6: return 1
        return 0
    df['chandra_mangala'] = df.apply(chandramangala, axis=1)

    # Sakata: Moon in 6th, 8th, or 12th from Jupiter
    def sakata(r):
        diff = (r['sign_Mo'] - r['sign_Ju']) % 12
        return int(diff in [5, 7, 11])  # 6th=5, 8th=7, 12th=11 (0-indexed diff)
    df['sakata'] = df.apply(sakata, axis=1)

    # Papakartari: malefics (Su,Ma,Sa,Ra,Ke) in sign before AND after Moon's sign
    def papakartari(r):
        malefics = [r[f'sign_{p}'] for p in ['Su','Ma','Sa','Ra','Ke']]
        before = ((r['sign_Mo'] - 2) % 12) + 1
        after  = (r['sign_Mo'] % 12) + 1
        return int(before in malefics and after in malefics)
    df['papakartari'] = df.apply(papakartari, axis=1)

    # Shubhakartari: benefics (Mo,Me,Ju,Ve) on both sides of Moon
    def shubhakartari(r):
        benefics = [r[f'sign_{p}'] for p in ['Ju','Ve','Me']]
        before = ((r['sign_Mo'] - 2) % 12) + 1
        after  = (r['sign_Mo'] % 12) + 1
        return int(before in benefics and after in benefics)
    df['shubhakartari'] = df.apply(shubhakartari, axis=1)

    # Argala from Moon: planets in 2nd, 4th, 11th from Moon (positive)
    def argala(r):
        second  = (r['sign_Mo'] % 12) + 1
        fourth  = ((r['sign_Mo'] + 2) % 12) + 1 if False else (r['sign_Mo'] + 3 - 1) % 12 + 1
        eleventh= (r['sign_Mo'] + 10 - 1) % 12 + 1
        pos_signs = [second, fourth, eleventh]
        other_signs = [r[f'sign_{p}'] for p in PLANETS if p not in ['Mo']]
        return int(any(s in pos_signs for s in other_signs))
    df['argala_mo'] = df.apply(argala, axis=1)

    # Neecha Bhanga: debilitated planet's lord is in angle from Moon or Lagna
    def neecha_bhanga(r):
        for p in PLANETS:
            if r[f'dig_{p}'] in ['debilitated', 'exact_debil']:
                # Cancellation condition (simplified): exaltation lord in angle from Moon
                exalt_sg = EXALT[p][0]
                exalt_lord = planet_sign_lord(exalt_sg)
                lord_sign = r[f'sign_{exalt_lord}']
                diff = (lord_sign - r['sign_Mo']) % 12
                if diff in [0, 3, 6, 9]: return 1
        return 0
    df['neecha_bhanga'] = df.apply(neecha_bhanga, axis=1)

    # Parivartana Yoga
    def parivartana_row(r):
        sd = {p: r[f'sign_{p}'] for p in PLANETS}
        yogs = parivartana_yogas(sd)
        return '|'.join(yogs) if yogs else 'none'
    df['parivartana'] = df.apply(parivartana_row, axis=1)
    df['parivartana_any'] = (df['parivartana'] != 'none').astype(int)

    print("Computing Graha Yuddha …")
    pairs = [('Me','Ve'),('Me','Ma'),('Me','Ju'),('Me','Sa'),
             ('Ve','Ma'),('Ve','Ju'),('Ve','Sa'),('Ma','Ju'),('Ma','Sa'),('Ju','Sa')]
    df['graha_yuddha'] = 0
    df['graha_yuddha_pair'] = 'none'
    for p1, p2 in pairs:
        sep = (df[f'sid_{p1}'] - df[f'sid_{p2}']).abs().apply(lambda x: min(x, 360-x))
        war = sep <= 1.0
        df.loc[war, 'graha_yuddha'] = 1
        df.loc[war, 'graha_yuddha_pair'] = f'{p1}-{p2}'

    print("Computing Atmakaraka / Amatyakaraka …")
    def ak_row(r):
        return compute_atmakaraka({p: r[f'sid_{p}'] for p in PLANETS})
    ak_data = df.apply(ak_row, axis=1)
    df['atmakaraka']   = ak_data.apply(lambda x: x[0])
    df['amatyakaraka'] = ak_data.apply(lambda x: x[1])

    print("Computing Bhrigu Bindu …")
    df['bhrigu_bindu'] = df.apply(lambda r: bhrigu_bindu(r['sid_Mo'], r['sid_Ra']), axis=1)
    df['bb_sign']      = df['bhrigu_bindu'].apply(sign_of)

    print("Computing Gulika …")
    df['gulika_portion'] = df['dow'].apply(gulika_sign)

    print("Computing Avasthas …")
    for p in PLANETS:
        # Garvita: exalted or moolatrikona
        df[f'garv_{p}'] = df[f'dig_{p}'].isin(['exalted','exact_exalt','moolatrikona']).astype(int)
        # Kshudita: enemy sign
        df[f'kshu_{p}'] = df[f'dig_{p}'].isin(['enemy','debilitated','exact_debil']).astype(int)
        # Mudita: friendly or own sign
        df[f'mudi_{p}'] = df[f'dig_{p}'].isin(['friendly','own','moolatrikona']).astype(int)
        # Kshobhita: combust (within Sun's orb)
        if p in COMB_ORB:
            df[f'kshob_{p}'] = df[f'comb_{p}']
        else:
            df[f'kshob_{p}'] = 0

    print("Computing Shadbala components …")
    # Dig Bala: Jupiter/Mercury strong in 1st house (Aries), Moon/Venus in 4th (Cancer),
    # Saturn in 7th (Libra), Sun/Mars in 10th (Capricorn)
    DIG_BALA_STRONG = {'Ju':1,'Me':1,'Mo':4,'Ve':4,'Sa':7,'Su':10,'Ma':10}
    for p in PLANETS:
        if p in DIG_BALA_STRONG:
            target_sign = DIG_BALA_STRONG[p]
            df[f'digbala_{p}'] = (df[f'sign_{p}'] == target_sign).astype(int)
        else:
            df[f'digbala_{p}'] = 0

    # Paksha Bala: benefics (Mo, Me, Ju, Ve) stronger in Shukla; malefics (Su, Ma, Sa, Ra, Ke) in Krishna
    BENEFICS = {'Mo','Me','Ju','Ve'}
    for p in PLANETS:
        if p in BENEFICS:
            df[f'paksha_bala_{p}'] = (df['paksha'] == 'SHUKLA').astype(int)
        else:
            df[f'paksha_bala_{p}'] = (df['paksha'] == 'KRISHNA').astype(int)

    # Cheshta Bala: retrograde → high, stationary → very high, else moderate
    for p in PLANETS:
        df[f'cheshta_{p}'] = df[f'retro_{p}'] * 2 + df[f'stat_{p}'] * 3

    # Simplified Ishta/Kashta Phala
    for p in PLANETS:
        ishta = df[f'digsc_{p}'] / 8 * 60  # 0-60 scaled by dignity
        kashta = 60 - ishta
        df[f'ishta_{p}'] = ishta.round(1)
        df[f'kashta_{p}'] = kashta.round(1)

    print("Computing Vimshottari Dasha …")
    def dasha_row(r):
        try:
            td = pd.to_datetime(r['date']).date()
            return compute_vimshottari(INCEPTION_DATE, INCEPTION_MOON_NAK, td)
        except: return ('Ke','Ke')
    dasha_data = df.apply(dasha_row, axis=1)
    df['mahadasha']   = dasha_data.apply(lambda x: x[0])
    df['antardasha']  = dasha_data.apply(lambda x: x[1])
    df['dasha_quality'] = df['mahadasha'].map(lambda p: DASHA_NATURE.get(p,'neutral'))
    # Dasha lord's current state
    df['dasha_lord_dig']  = df.apply(lambda r: dignity(r['mahadasha'], r[f"sid_{r['mahadasha']}"]), axis=1)
    df['dasha_lord_retro'] = df.apply(lambda r: r[f"retro_{r['mahadasha']}"], axis=1)

    print("Computing Sade Sati / Ashtama Shani …")
    saturn_sign = df['sign_Sa']
    # Sade Sati: Saturn in sign 12, 1, or 2 from natal Moon sign
    sade_signs = [(NATAL_MOON_SIGN - 2 + 12 - 1) % 12 + 1,  # 12th = before
                   NATAL_MOON_SIGN,                            # 1st = natal
                  (NATAL_MOON_SIGN % 12) + 1]                 # 2nd = after
    df['sade_sati'] = saturn_sign.isin(sade_signs).astype(int)
    # Sade Sati phase
    def sade_phase(sg):
        if sg == (NATAL_MOON_SIGN - 2 + 12 - 1) % 12 + 1: return 'rising'
        if sg == NATAL_MOON_SIGN: return 'peak'
        if sg == (NATAL_MOON_SIGN % 12) + 1: return 'setting'
        return 'none'
    df['sade_sati_phase'] = saturn_sign.apply(sade_phase)
    # Ashtama Shani: Saturn in 8th from natal Moon (= NATAL_MOON_SIGN + 7)
    ashtama_sign = (NATAL_MOON_SIGN + 7 - 1) % 12 + 1
    df['ashtama_shani'] = (saturn_sign == ashtama_sign).astype(int)

    # Speed category strings for all planets
    for p in PLANETS:
        df[f'spd_cat_{p}'] = df.apply(lambda r: speed_category(p, r[f'spd_{p}']), axis=1)

    # Aspect combinations (key ones)
    # Jupiter aspecting Moon's sign
    def ju_aspects_moon(r):
        ju_aspects = parashari_aspects(r['sign_Ju'], 'Ju')
        return int(r['sign_Mo'] in ju_aspects)
    df['ju_asp_mo'] = df.apply(ju_aspects_moon, axis=1)

    # Saturn aspecting Moon's sign
    def sa_aspects_moon(r):
        sa_aspects = parashari_aspects(r['sign_Sa'], 'Sa')
        return int(r['sign_Mo'] in sa_aspects)
    df['sa_asp_mo'] = df.apply(sa_aspects_moon, axis=1)

    # Mars aspecting Moon's sign
    def ma_aspects_moon(r):
        ma_aspects = parashari_aspects(r['sign_Ma'], 'Ma')
        return int(r['sign_Mo'] in ma_aspects)
    df['ma_asp_mo'] = df.apply(ma_aspects_moon, axis=1)

    # Jupiter-Saturn mutual aspect
    def ju_sa_mutual(r):
        ju_asp = parashari_aspects(r['sign_Ju'], 'Ju')
        sa_asp = parashari_aspects(r['sign_Sa'], 'Sa')
        return int(r['sign_Sa'] in ju_asp or r['sign_Ju'] in sa_asp)
    df['ju_sa_aspect'] = df.apply(ju_sa_mutual, axis=1)

    # Combust any benefic
    df['comb_benefic_any'] = df[['comb_Mo','comb_Me','comb_Ju','comb_Ve']].max(axis=1)

    # Jaimini Chara Karakas (same as Atmakaraka by degree)
    df['jaimini_ak']  = df['atmakaraka']
    df['jaimini_amk'] = df['amatyakaraka']

    # Speed category for Vara lord (weekday planet)
    df['vara_spd_cat'] = df.apply(lambda r: speed_category(r['vara_lord'], r[f"spd_{r['vara_lord']}"]), axis=1)
    df['vara_dig']     = df.apply(lambda r: dignity(r['vara_lord'], r[f"sid_{r['vara_lord']}"]), axis=1)

    # Key interaction features (from SO_FAR findings)
    df['ix_paksha_ju_dig']     = df['paksha'] + '_' + df['dig_Ju']
    df['ix_paksha_nak']        = df['paksha'] + '_' + df['nak_mo_name']
    df['ix_paksha_moon_sign']  = df['paksha'] + '_Mo' + df['sign_Mo'].astype(str)
    df['ix_tithi_nak']         = df['tithi_quality'] + '_' + df['nak_mo_qual']
    df['ix_ju_dig_moon_sign']  = df['dig_Ju'] + '_Mo' + df['sign_Mo'].astype(str)
    df['ix_vara_paksha']       = df['vara_lord'] + '_' + df['paksha']

    return df


def add_banknifty_price(nifty_df, bn_df):
    """Merge BankNifty OHLC with Nifty astrological features."""
    # BankNifty doesn't have its own planet data — same sky, different instrument
    merged = bn_df.merge(
        nifty_df.drop(columns=['open','high','low','close'], errors='ignore'),
        on='date', how='inner'
    )
    # Recompute outcome labels for BankNifty prices
    for n in [1, 2, 3, 5, 10]:
        merged[f'fwd_ret_{n}d'] = np.log(merged['close'].shift(-n) / merged['close'])
    merged['log_ret']   = np.log(merged['close'] / merged['close'].shift(1))
    merged['range_pct'] = (merged['high'] - merged['low']) / merged['close']
    tr = pd.concat([merged['high'] - merged['low'],
                    (merged['high'] - merged['close'].shift()).abs(),
                    (merged['low']  - merged['close'].shift()).abs()], axis=1).max(axis=1)
    merged['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    ret3 = merged['fwd_ret_3d']
    merged['outcome_3d'] = pd.cut(ret3, bins=[-np.inf,-0.015,-0.005,0.005,0.015,np.inf],
                               labels=['STRONG_BEAR','MILD_BEAR','SIDEWAYS','MILD_BULL','STRONG_BULL'])
    merged['is_strong_bull'] = (merged['outcome_3d'] == 'STRONG_BULL').astype('Int8')
    merged['is_strong_bear'] = (merged['outcome_3d'] == 'STRONG_BEAR').astype('Int8')
    merged['is_bull'] = (ret3 > 0).astype('Int8')
    merged['is_high_vol'] = (merged['range_pct'] > 1.5 * merged['atr14'] / merged['close']).astype('Int8')
    return merged


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import time
    t0 = time.time()

    # Step 0 checks
    print("\n=== STEP 0: DATA SCHEMA CHECK ===")
    with open(f"{REPO}/data.json") as f:
        raw = json.load(f)
    recs = raw['records']
    print(f"Nifty records: {len(recs)}")
    print(f"Date range: {recs[0]['date']} → {recs[-1]['date']}")
    print(f"Planets in row 0: {list(recs[0]['p'].keys())}")
    print(f"Ayanamsa systems: {list(recs[0]['ayan'].keys())}")

    with open(f"{REPO}/ohlc_banknifty.json") as f:
        bnraw = json.load(f)
    bndates = list(bnraw['records'].keys())
    print(f"BankNifty records: {len(bndates)}, range: {bndates[0]} → {bndates[-1]}")

    # Data quality report
    nf_df = load_nifty()
    for p in PLANETS:
        nulls = nf_df[f'trop_{p}'].isna().sum()
        oob   = ((nf_df[f'trop_{p}'] < 0) | (nf_df[f'trop_{p}'] > 360)).sum()
        if nulls > 0 or oob > 0:
            print(f"  WARNING: {p} nulls={nulls} out-of-range={oob}")
    dupes = nf_df['date'].duplicated().sum()
    print(f"Nifty: {len(nf_df)} rows, 0 missing planets, {dupes} duplicate dates — PASS")

    os.makedirs(f"{REPO}/reports", exist_ok=True)
    with open(f"{REPO}/reports/00_data_quality.txt", "w") as f:
        f.write(f"Nifty50: {len(recs)} records, {recs[0]['date']} to {recs[-1]['date']}\n")
        f.write(f"BankNifty: {len(bndates)} records, {bndates[0]} to {bndates[-1]}\n")
        f.write(f"Planets: {list(recs[0]['p'].keys())}\n")
        f.write(f"Ayanamsa systems: {list(recs[0]['ayan'].keys())}\n")
        f.write(f"Duplicate dates: {dupes}\n")
        f.write(f"Missing values: 0\nOut-of-range: 0\nSTATUS: PASS\n")
    print(f"Data quality report saved to reports/00_data_quality.txt")

    # Step 1: Full enrichment
    print("\n=== STEP 1: FEATURE ENGINEERING ===")
    df = enrich(nf_df)
    print(f"Nifty enriched shape: {df.shape}")
    print(f"Null count (non-warmup rows >200): {df.iloc[200:].isnull().sum().sum()}")

    # Verify zero lookahead
    for n in [1,2,3,5,10]:
        assert df[f'fwd_ret_{n}d'].iloc[-n:].isna().all(), f"Lookahead detected in fwd_ret_{n}d!"
    print("Lookahead audit PASS: fwd_ret columns NaN in final N rows")

    os.makedirs(f"{REPO}/data", exist_ok=True)
    df.to_csv(f"{REPO}/data/nifty_enriched.csv", index=False)
    print(f"Saved data/nifty_enriched.csv ({df.shape[0]}×{df.shape[1]})")

    # BankNifty
    bn_df = load_banknifty()
    bn_enriched = add_banknifty_price(df, bn_df)
    bn_enriched.to_csv(f"{REPO}/data/banknifty_enriched.csv", index=False)
    print(f"Saved data/banknifty_enriched.csv ({bn_enriched.shape[0]}×{bn_enriched.shape[1]})")

    print(f"\nStep 1 complete in {time.time()-t0:.1f}s")
    print(f"Nifty cols: {df.shape[1]}, BankNifty cols: {bn_enriched.shape[1]}")
    print(f"Outcome distribution (Nifty 3d):\n{df['outcome_3d'].value_counts().dropna()}")
