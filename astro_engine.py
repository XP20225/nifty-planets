"""
Shared astrological computation engine.
Pure functions only — safe to import without side effects.
Used by generate_signal.py and new_step4.py.
"""
import math, os
from datetime import date
import swisseph as swe

REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"

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
RAHU_KALAM_PORTION  = {6:7, 0:2, 1:7, 2:5, 3:6, 4:4, 5:3}
GULIKA_KALAM_PORTION = {6:6, 0:5, 1:4, 2:3, 3:2, 4:1, 5:7}
MRITYU_BHAGA = {
    'Su': {1:20,2:9,3:12,4:6,5:3,6:27,7:16,8:29,9:6,10:20,11:13,12:14},
    'Mo': {1:26,2:12,3:13,4:25,5:24,6:11,7:26,8:14,9:13,10:25,11:5,12:12},
    'Ma': {1:28,2:20,3:15,4:13,5:9,6:28,7:26,8:12,9:13,10:11,11:25,12:16},
    'Me': {1:15,2:14,3:13,4:12,5:15,6:15,7:4,8:13,9:14,10:12,11:11,12:14},
    'Ju': {1:14,2:26,3:11,4:5,5:10,6:13,7:10,8:5,9:12,10:14,11:20,12:26},
    'Ve': {1:27,2:11,3:29,4:14,5:10,6:10,7:11,8:29,9:14,10:4,11:20,12:16},
    'Sa': {1:20,2:21,3:22,4:5,5:20,6:10,7:22,8:14,9:14,10:20,11:28,12:26},
}
_MUMBAI_LAT = 18.9750
_MUMBAI_LON = 72.8258
_MUMBAI_ALT = 14.0
_MARKET_OPEN_H = 9.25  # 9:15 AM IST as fractional hours

def _get_sunrise_sunset_ist(d):
    jd = swe.julday(d.year, d.month, d.day, 0.0)
    geopos = (_MUMBAI_LON, _MUMBAI_LAT, _MUMBAI_ALT)
    _, tret_r = swe.rise_trans(jd, swe.SUN, 1, geopos, 0.0, 0.0)
    _, tret_s = swe.rise_trans(jd, swe.SUN, 2, geopos, 0.0, 0.0)
    rise_ist = (tret_r[0] - jd) * 24.0 + 5.5
    set_ist  = (tret_s[0] - jd) * 24.0 + 5.5
    return rise_ist, set_ist

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
    """Compute sidereal (Lahiri) planetary positions using pyswisseph."""
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
        spd = result[3]
        pos[name] = (lon, spd)
    ra_lon = pos['Ra'][0]
    pos['Ke'] = ((ra_lon + 180) % 360, -pos['Ra'][1])
    return pos

def compute_day_features(d, positions=None):
    """Compute all astrological features for a given date."""
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

    for p in ['Mo','Me','Ve','Ma','Ju','Sa']:
        sep = abs(sid[p] - sid['Su'])
        sep = min(sep, 360 - sep)
        feat[f'comb_{p}'] = int(sep <= COMB_ORB[p])

    sep_sm = (sid['Mo'] - sid['Su'] + 360) % 360
    trop_su = (sid['Su'] + 24.2) % 360
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

    nak_num = nak_of(sid['Mo'])
    feat['nak_mo']       = nak_num
    feat['nak_mo_name']  = NAK_NAMES[nak_num-1]
    feat['nak_mo_lord']  = NAK_LORDS[nak_num-1]
    feat['nak_mo_qual']  = NAK_QUALITY[nak_num-1]

    yoga_pt  = (sid['Su'] + sid['Mo']) % 360
    yoga_num = int(yoga_pt / (360/27)) + 1
    feat['yoga_num']     = yoga_num
    feat['yoga_name']    = YOGA_NAMES[yoga_num-1]
    feat['yoga_quality'] = 'inauspicious' if yoga_num in YOGA_INAUSPICIOUS else 'auspicious'

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

    feat['graha_yuddha'] = 0
    for p1,p2 in [('Me','Ve'),('Me','Ma'),('Ve','Ma'),('Ma','Ju'),('Ju','Sa')]:
        sep2 = abs(sid[p1]-sid[p2]); sep2 = min(sep2,360-sep2)
        if sep2 <= 1.0: feat['graha_yuddha'] = 1; break

    def _gand(x):
        dg = x%30; sg = int(x/30)+1
        return int((sg in [12,4,8] and dg>=26.67) or (sg in [1,5,9] and dg<=3.33))
    feat['gand_Mo'] = _gand(sid['Mo']); feat['gand_any'] = int(any(_gand(sid[p]) for p in PLANETS))
    feat['sandhi_mo'] = int(deg_in_sign(sid['Mo']) >= 29)

    diff = (nak_num - INCEPTION_MOON_NAK) % 27
    tara = (diff % 9) + 1
    feat['tara_num'] = tara
    tara_quality_map = {1:'critical',2:'wealth',3:'danger',4:'prosperity',5:'obstacle',
                        6:'achievement',7:'worst',8:'good',9:'best'}
    feat['tara_quality'] = tara_quality_map[tara]

    nl = feat['nak_mo_lord']
    feat['nakl_dig'] = dignity(nl, sid[nl])
    feat['nakl_spd'] = speed_cat(nl, spd[nl])

    dow = d.weekday()
    day_lord_map = {0:'Mo',1:'Ma',2:'Me',3:'Ju',4:'Ve',5:'Sa',6:'Su'}
    try:
        rise_ist, set_ist = _get_sunrise_sunset_ist(d)
    except Exception:
        rise_ist, set_ist = 6.0, 18.0
    day_dur   = set_ist - rise_ist
    portion_h = day_dur / 8.0

    hora_start = HORA_IDX[day_lord_map[dow]]
    hora_num   = max(0, int(_MARKET_OPEN_H - rise_ist))
    feat['hora_at_open'] = HORA_SEQ[(hora_start + hora_num) % 7]

    chog_seq  = CHOGHADIYA_DAY[dow]
    chog_idx  = min(int((_MARKET_OPEN_H - rise_ist) / portion_h), 7) if portion_h > 0 else 2
    feat['choghadiya']         = chog_seq[chog_idx]
    feat['choghadiya_quality'] = CHOGHADIYA_QUALITY[chog_seq[chog_idx]]

    rk_portion = RAHU_KALAM_PORTION[dow]
    rk_start   = rise_ist + (rk_portion - 1) * portion_h
    feat['rahu_kalam_open'] = int(rk_start <= _MARKET_OPEN_H < rk_start + portion_h)

    gk_portion = GULIKA_KALAM_PORTION[dow]
    gk_start   = rise_ist + (gk_portion - 1) * portion_h
    feat['gulika_kalam_open'] = int(gk_start <= _MARKET_OPEN_H < gk_start + portion_h)

    mo_sg = feat.get('sign_Mo', sign_of(sid.get('Mo', 0)))
    mo_dg = sid.get('Mo', 0) % 30
    mb_deg = MRITYU_BHAGA.get('Mo', {}).get(mo_sg, -99)
    feat['mrityu_Mo'] = int(abs(mo_dg - mb_deg) <= 1)

    maha, ant = compute_vimshottari(d)
    feat['mahadasha']  = maha
    feat['antardasha'] = ant
    feat['dasha_quality'] = DASHA_NATURE.get(maha,'neutral')
    feat['dasha_lord_dig'] = dignity(maha, sid[maha])

    feat['sade_sati'] = int(feat['sign_Sa'] in [(NATAL_MOON_SIGN-2+12-1)%12+1, NATAL_MOON_SIGN, (NATAL_MOON_SIGN%12)+1])
    ashtama_sg = (NATAL_MOON_SIGN+7-1)%12+1
    feat['ashtama_shani'] = int(feat['sign_Sa'] == ashtama_sg)
    ss_phase_map = {(NATAL_MOON_SIGN-2+12-1)%12+1:'rising', NATAL_MOON_SIGN:'peak', (NATAL_MOON_SIGN%12)+1:'setting'}
    feat['sade_sati_phase'] = ss_phase_map.get(feat['sign_Sa'], 'none')

    feat['panchaka'] = int(nak_num >= 23)

    def _asp(p_sign, p_name, target_sign):
        aspects = {(p_sign+5)%12+1}
        for off in SPECIAL_ASPECTS.get(p_name,[]):
            aspects.add((p_sign+off-1)%12+1)
        return int(target_sign in aspects)
    feat['ju_asp_mo'] = _asp(feat['sign_Ju'], 'Ju', feat['sign_Mo'])
    feat['sa_asp_mo'] = _asp(feat['sign_Sa'], 'Sa', feat['sign_Mo'])
    feat['ma_asp_mo'] = _asp(feat['sign_Ma'], 'Ma', feat['sign_Mo'])

    # Fix 6: Comprehensive inter-planetary aspects (398 features)
    _P_ASP = {'Su':[7],'Mo':[7],'Me':[7],'Ve':[7],
              'Ma':[4,7,8],'Ju':[5,7,9],'Sa':[3,7,10],
              'Ra':[5,7,9],'Ke':[5,7,9]}
    _S_LORD = {1:'Ma',2:'Ve',3:'Me',4:'Mo',5:'Su',6:'Me',
               7:'Ve',8:'Ma',9:'Ju',10:'Sa',11:'Sa',12:'Ju'}
    _OWN_S  = {'Ma':{1,8},'Ve':{2,7},'Me':{3,6},'Mo':{4},
               'Su':{5},'Ju':{9,12},'Sa':{10,11}}
    _NATAL  = 2
    _ALL_P  = ['Su','Mo','Ma','Me','Ju','Ve','Sa','Ra','Ke']
    _KEY_PR = [('Ju','Sa'),('Ju','Ma'),('Sa','Ma'),('Ju','Su'),('Sa','Su'),
               ('Ma','Su'),('Ju','Mo'),('Sa','Mo'),('Ma','Mo'),('Ve','Ju'),
               ('Ve','Sa'),('Ve','Ma'),('Me','Ju'),('Me','Sa'),('Me','Ma'),
               ('Ra','Ju'),('Ra','Sa'),('Ra','Mo'),('Su','Mo'),('Me','Ve')]
    _DASP   = {'conj':0,'opp':180,'trine':120,'sq':90,'sext':60}
    _DORB   = {'conj':8,'opp':8,'trine':7,'sq':7,'sext':6}

    def _asign(sg, H): return ((sg - 1 + H - 1) % 12) + 1

    for _p1 in _ALL_P:
        _sg1 = feat.get(f'sign_{_p1}', 0)
        if not _sg1: continue
        _all_asp = {}
        for _H in _P_ASP.get(_p1, []):
            _asg = _asign(_sg1, _H)
            _all_asp[_H] = _asg
            feat[f'asp{_H}_{_p1}_lord']       = _S_LORD[_asg]
            feat[f'aspected_sign_{_p1}_{_H}'] = _asg
            feat[f'asp{_H}_{_p1}_natal_mo']   = int(_asg == _NATAL)
            for _p2 in _ALL_P:
                if _p2 == _p1: continue
                feat[f'asp{_H}_{_p1}_{_p2}'] = int(feat.get(f'sign_{_p2}', 0) == _asg)
        _asp_set = set(_all_asp.values())
        for _p2 in list(_OWN_S.keys()):
            feat[f'asp_{_p1}_dom_{_p2}'] = int(bool(_asp_set & _OWN_S[_p2]))

    for _p1, _p2 in _KEY_PR:
        _s1 = sid.get(_p1); _s2 = sid.get(_p2)
        if _s1 is None or _s2 is None: continue
        _raw = abs(_s1 - _s2) % 360
        _sep = _raw if _raw <= 180 else 360 - _raw
        for _an, _ad in _DASP.items():
            _orb = _DORB[_an]
            _in = int(abs(_sep - _ad) <= _orb)
            feat[f'deg_{_an}_{_p1}_{_p2}'] = _in
            feat[f'ex_{_an}_{_p1}_{_p2}']  = int(_in and abs(_sep - _ad) <= 3)

    for _slow in ['Sa','Ju','Ma']:
        _sg_s = feat.get(f'sign_{_slow}', 0)
        if not _sg_s: continue
        _asgs = {_asign(_sg_s, _H) for _H in _P_ASP[_slow]}
        feat[f'n_asp_under_{_slow}'] = sum(1 for _p in _ALL_P if _p != _slow
                                            and feat.get(f'sign_{_p}', 0) in _asgs)

    feat['ix_paksha_ju_dig']    = feat['paksha'] + '_' + feat['dig_Ju']
    feat['ix_paksha_nak']       = feat['paksha'] + '_' + feat['nak_mo_name']
    feat['ix_paksha_moon_sign'] = feat['paksha'] + '_Mo' + str(feat['sign_Mo'])
    feat['ix_tithi_nak']        = feat['tithi_quality'] + '_' + feat['nak_mo_qual']
    feat['ix_ju_dig_moon_sign'] = feat['dig_Ju'] + '_Mo' + str(feat['sign_Mo'])
    feat['ix_vara_paksha']      = feat['vara_lord'] + '_' + feat['paksha']

    for col in ['gajakesari','papakartari','comb_Mo','comb_Me','comb_Ve',
                'retro_Me','retro_Ju','gand_Mo','sade_sati','ju_asp_mo','sa_asp_mo','graha_yuddha']:
        feat[col+'_str'] = col + '=' + str(feat.get(col, 0))

    return feat
