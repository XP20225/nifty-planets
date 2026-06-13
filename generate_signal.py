#!/usr/bin/env python3
"""
AstroQuant Signal Generator v2
Usage: python generate_signal.py [YYYY-MM-DD]
If no date given, uses today.

Outputs:
  - Complete Panchanga for the date
  - All planetary dignities, speeds, states
  - All active confirmed patterns (Wilson LB, n, p-value)
  - Active veto conditions
  - Composite score
  - TRADE BULL / TRADE BEAR / WATCH / NO TRADE decision
  - Next 30 days upcoming pattern windows
"""
import sys, os, json, math
from datetime import date, timedelta
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))

# ── Import helpers from astro_engine (no side effects) ───────────────
sys.path.insert(0, REPO)
from astro_engine import (
    compute_day_features, get_planets_swisseph, wilson_lower,
    PLANETS, NAK_NAMES, YOGA_NAMES, DASHA_NATURE
)

def load_confirmed():
    path = f"{REPO}/results/validation/confirmed_patterns.csv"
    if not os.path.exists(path):
        print("ERROR: confirmed_patterns.csv not found. Run new_step3.py first.")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)

def eval_pattern(feat_dict, features_str, condition_str):
    """Check if pattern matches given feature dict."""
    features = features_str.split('|')
    conditions = condition_str.split('||') if '||' in str(condition_str) else [str(condition_str)]
    if len(features) != len(conditions): return False
    for f, v in zip(features, conditions):
        f = f.strip(); v = v.strip()
        fval = str(feat_dict.get(f, ''))
        if fval != v: return False
    return True

def generate_signal(target_date=None):
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    print(f"\n{'='*60}")
    print(f"  AstroQuant Signal — {target_date.strftime('%A, %B %d %Y')}")
    print(f"{'='*60}")

    # Get planetary positions
    try:
        positions = get_planets_swisseph(target_date)
    except Exception as e:
        print(f"ERROR computing positions: {e}")
        return

    # Compute features
    feat = compute_day_features(target_date, positions)

    # Add _str versions for binary features
    for col in ['gajakesari','papakartari','comb_Mo','comb_Me','comb_Ve',
                'retro_Me','retro_Ju','gand_Mo','sade_sati','ju_asp_mo','sa_asp_mo','graha_yuddha']:
        feat[col+'_str'] = col + '=' + str(feat.get(col, 0))

    # ── PANCHANGA ──────────────────────────────────────────────────────
    vara_full = {'Mo':'Monday (Moon)','Ma':'Tuesday (Mars)','Me':'Wednesday (Mercury)',
                 'Ju':'Thursday (Jupiter)','Ve':'Friday (Venus)','Sa':'Saturday (Saturn)','Su':'Sunday (Sun)'}
    print(f"\n  PANCHANGA")
    print(f"  ─────────────────────────────────────────")
    print(f"  Vara:      {vara_full.get(feat['vara_lord'], feat['vara_lord'])}")
    print(f"  Tithi:     {feat['tithi_num']} ({feat['paksha']}) — {feat['tithi_quality']}")
    print(f"  Nakshatra: {feat['nak_mo_name']}")
    print(f"  Yoga:      {feat['yoga_name']} — {feat['yoga_quality'].upper()}")
    print(f"  Karana:    {feat['karana']} ({feat['karana_quality']})")
    print(f"  Hora@9:15: {feat['hora_at_open']}")
    print(f"  Choghadiya:{feat['choghadiya']} ({feat['choghadiya_quality'].upper()})")

    # ── PLANETARY STATES ───────────────────────────────────────────────
    print(f"\n  PLANETARY STATES (Sidereal Lahiri)")
    print(f"  ─────────────────────────────────────────")
    print(f"  {'Planet':<10} {'Sign':>12} {'Degree':>8} {'Dignity':<14} {'Retro':>5} {'Comb':>5}")
    planet_names = {'Su':'Sun','Mo':'Moon','Me':'Mercury','Ve':'Venus','Ma':'Mars',
                    'Ju':'Jupiter','Sa':'Saturn','Ra':'Rahu','Ke':'Ketu'}
    sign_names_full = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']
    for p in PLANETS:
        lon = feat[f'sid_{p}']
        sg  = feat[f'sign_{p}']
        dg  = lon % 30
        dig = feat[f'dig_{p}']
        retro = '♺' if feat.get(f'retro_{p}',0) else ''
        comb  = '🔥' if feat.get(f'comb_{p}',0) else ''
        sgn   = sign_names_full[sg-1] if 1<=sg<=12 else str(sg)
        print(f"  {planet_names[p]:<10} {sgn:>12}  {dg:>5.1f}°  {dig:<14} {retro:>5} {comb:>5}")

    # ── DASHA ──────────────────────────────────────────────────────────
    print(f"\n  DASHA STATE")
    print(f"  ─────────────────────────────────────────")
    print(f"  Mahadasha:  {feat['mahadasha']} ({DASHA_NATURE.get(feat['mahadasha'],'?')})")
    print(f"  Antardasha: {feat['antardasha']}")
    print(f"  Dasha Lord Dignity: {feat['dasha_lord_dig']}")
    print(f"  Sade Sati: {'ACTIVE — ' + feat['sade_sati_phase'].upper() if feat.get('sade_sati') else 'No'}")
    print(f"  Ashtama Shani: {'ACTIVE' if feat.get('ashtama_shani') else 'No'}")

    # ── SPECIAL CONDITIONS ─────────────────────────────────────────────
    specials = []
    if feat.get('gajakesari'):    specials.append("GAJAKESARI YOGA (Jupiter-Moon angular)")
    if feat.get('kemadruma'):     specials.append("KEMADRUMA YOGA (Moon isolated)")
    if feat.get('papakartari'):   specials.append("PAPAKARTARI (malefics hem Moon)")
    if feat.get('graha_yuddha'):  specials.append("GRAHA YUDDHA (planetary war)")
    if feat.get('gand_Mo'):       specials.append("GANDANTA MOON (water-fire junction)")
    if feat.get('sandhi_mo'):     specials.append("SANDHI MOON (last degree of sign)")
    if feat.get('panchaka'):      specials.append("PANCHAKA PERIOD (naks 23-27)")
    if feat.get('retro_Me'):      specials.append("MERCURY RETROGRADE")
    if feat.get('retro_Ju'):      specials.append("JUPITER RETROGRADE")
    if feat.get('retro_Sa'):      specials.append("SATURN RETROGRADE")
    if feat.get('sade_sati'):     specials.append(f"SADE SATI ({feat.get('sade_sati_phase','').upper()})")
    if feat.get('ashtama_shani'): specials.append("ASHTAMA SHANI")
    print(f"\n  SPECIAL CONDITIONS")
    print(f"  ─────────────────────────────────────────")
    for s in (specials or ['None active']):
        print(f"  • {s}")

    # ── PATTERN MATCHING ───────────────────────────────────────────────
    confirmed = load_confirmed()
    if len(confirmed) == 0:
        print("\n  No confirmed patterns available.")
        return

    BASE_BULL = 0.551
    active_bull = []; active_bear = []
    for _, pat in confirmed.iterrows():
        if eval_pattern(feat, pat['features'], pat['condition']):
            wlb = pat.get('wlb_train', 0.5)
            is_bull = pat.get('signal_dir','BULL') == 'BULL' or wlb > BASE_BULL
            entry = {
                'features': pat['features'], 'condition': pat['condition'],
                'outcome': pat.get('outcome','?'), 'wlb': wlb,
                'n_train': pat.get('n_train',0), 'p': pat.get('p_value',1),
                'wr_oos': pat.get('wr_oos',0), 'n_oos': pat.get('n_oos',0),
            }
            if is_bull: active_bull.append(entry)
            else: active_bear.append(entry)

    print(f"\n  ACTIVE CONFIRMED PATTERNS")
    print(f"  ─────────────────────────────────────────")
    if active_bull:
        print(f"  BULL patterns ({len(active_bull)}):")
        for p in active_bull:
            print(f"    ✓ {p['features']} = {p['condition']}")
            print(f"      WilsonLB={p['wlb']:.3f}  n={p['n_train']}  p={p['p']:.4f}  OOS_wr={p['wr_oos']:.3f}(n={p['n_oos']})")
    else:
        print("  BULL patterns: none active")
    if active_bear:
        print(f"  BEAR patterns ({len(active_bear)}):")
        for p in active_bear:
            print(f"    ✗ {p['features']} = {p['condition']}")
            print(f"      WilsonLB={p['wlb']:.3f}  n={p['n_train']}  p={p['p']:.4f}  OOS_wr={p['wr_oos']:.3f}(n={p['n_oos']})")
    else:
        print("  BEAR patterns: none active")

    # ── COMPOSITE SCORE ─────────────────────────────────────────────────
    bull_score = sum(max(0, p['wlb'] - BASE_BULL) for p in active_bull)
    bear_score = sum(max(0, BASE_BULL - p['wlb']) for p in active_bear)
    net = bull_score - bear_score
    norm_score = 50 + net * 100

    print(f"\n  COMPOSITE SCORE")
    print(f"  ─────────────────────────────────────────")
    print(f"  Bull component:  +{bull_score:.4f}")
    print(f"  Bear component:  -{bear_score:.4f}")
    print(f"  Net:              {net:+.4f}")
    print(f"  Score (0-100):   {norm_score:.1f}  (50 = neutral)")

    # ── DECISION ──────────────────────────────────────────────────────
    n_bull = len(active_bull); n_bear = len(active_bear)
    print(f"\n  TRADE DECISION")
    print(f"  ─────────────────────────────────────────")
    if n_bull >= 3 and n_bear == 0:
        decision = "TRADE BULL"
        print(f"  ⬆️  {decision}")
        print(f"  Instrument: Nifty 50 (long / call)")
        print(f"  Entry: Next open (9:15 AM IST)")
        print(f"  Stop:  -1.5 ATR from entry")
        print(f"  Target: +2.5 ATR from entry")
        print(f"  Hold:  max 3 trading days")
    elif n_bear >= 3 and n_bull == 0:
        decision = "TRADE BEAR"
        print(f"  ⬇️  {decision}")
        print(f"  Instrument: Nifty 50 (short / put)")
        print(f"  Entry: Next open (9:15 AM IST)")
        print(f"  Stop:  +1.5 ATR from entry")
        print(f"  Target: -2.5 ATR from entry")
        print(f"  Hold:  max 3 trading days")
    elif n_bull >= 1 or n_bear >= 1:
        decision = "WATCH"
        print(f"  👁  {decision} — insufficient confluence ({n_bull} bull, {n_bear} bear active)")
    else:
        decision = "NO TRADE"
        print(f"  ○  {decision} — no confirmed patterns active today")

    # ── NEXT 30 DAYS ──────────────────────────────────────────────────
    cal_path = f"{REPO}/results/forward_calendar/planetary_calendar_1yr.csv"
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path)
        cal['date'] = pd.to_datetime(cal['date'])
        future = cal[cal['date'] > pd.Timestamp(target_date)]
        prime = future[future['classification'].isin(['PRIME_TRADE_BULL','PRIME_TRADE_BEAR'])].head(10)
        if len(prime) > 0:
            print(f"\n  UPCOMING PRIME TRADE WINDOWS")
            print(f"  ─────────────────────────────────────────")
            for _, r in prime.iterrows():
                icon = "⬆️" if r['classification']=='PRIME_TRADE_BULL' else "⬇️"
                nak  = str(r.get('nak_mo_name','?'))[:12]
                ju   = str(r.get('dig_Ju','?'))
                pak  = str(r.get('paksha','?'))[:1]
                print(f"  {icon} {str(r['date'])[:10]}  {r['classification']:<20} nak={nak:<12} ju={ju} pak={pak}")
        else:
            watch = future[future['classification'].str.startswith('WATCH')].head(5)
            if len(watch) > 0:
                print(f"\n  Next WATCH days (no PRIME_TRADE in upcoming window):")
                for _, r in watch.iterrows():
                    print(f"  👁  {str(r['date'])[:10]}  {r['classification']}")

    print(f"\n{'='*60}\n")
    return decision, norm_score, active_bull, active_bear

if __name__ == '__main__':
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = str(date.today())
    generate_signal(target)
