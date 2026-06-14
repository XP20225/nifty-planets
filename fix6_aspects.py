"""
Fix 6: Comprehensive inter-planetary aspects.

What was missing:
  - Only 4 aspect features existed: ju_asp_mo, sa_asp_mo, ma_asp_mo, ju_sa_aspect
  - All other planet pairs were unrepresented
  - No degree-based aspects (conjunction, opposition, trine, square, sextile)
  - No "lord chain" (Saturn aspecting Sagittarius = Saturn influences Jupiter's domain)
  - No mutual special aspects (Jupiter 9th aspecting Saturn's sign, Saturn 10th returning)
  - No natal-Moon-sign aspect tracking across the history

What this adds:
  A. Sign-based Vedic aspects — every planet, all its special houses, every target planet
  B. Aspected sign & lord — for each aspect, what sign is hit and who rules it
  C. Lord-domain aspects — P1 aspects any sign owned by P2 (the chain of influence)
  D. Natal Moon sign (Taurus) aspects — dynamic tracking across all history
  E. Degree-based aspects — conjunction/opp/trine/square/sextile for 20 key pairs
  F. Exact tight aspects (≤3°) for Saturn-Jupiter, Saturn-Mars, Jupiter-Mars
  G. Aggregate counts — n planets under Jupiter's / Saturn's / Mars's full field
  H. Key interaction features — aspect + dignity, aspect + paksha

Vedic aspect house rules (inclusive counting, planet's own sign = 1):
  All planets : 7th (opposition)
  Mars        : 4th, 7th, 8th
  Jupiter     : 5th, 7th, 9th
  Saturn      : 3rd, 7th, 10th
  Rahu        : 5th, 7th, 9th  (Jupiter-like, most common Jyotish usage)
  Ketu        : 5th, 7th, 9th  (Jupiter-like)

Sign lord table (classical Parashari):
  Aries(1):Ma  Taurus(2):Ve  Gemini(3):Me  Cancer(4):Mo  Leo(5):Su  Virgo(6):Me
  Libra(7):Ve  Scorpio(8):Ma  Sagittarius(9):Ju  Capricorn(10):Sa  Aquarius(11):Sa  Pisces(12):Ju
"""

import os, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
REPO = "/Users/vasanthakumaranpalanisamy/Nifty Planets"

# ── Constants ──────────────────────────────────────────────────────────────────

PLANETS = ['Su', 'Mo', 'Ma', 'Me', 'Ju', 'Ve', 'Sa', 'Ra', 'Ke']

SIGN_LORD = {
    1:'Ma', 2:'Ve', 3:'Me', 4:'Mo', 5:'Su', 6:'Me',
    7:'Ve', 8:'Ma', 9:'Ju', 10:'Sa', 11:'Sa', 12:'Ju'
}

PLANET_OWN_SIGNS = {
    'Ma': {1, 8}, 'Ve': {2, 7}, 'Me': {3, 6}, 'Mo': {4},
    'Su': {5}, 'Ju': {9, 12}, 'Sa': {10, 11}
}

# Aspect houses per planet (inclusive Vedic counting, own sign = 1)
PLANET_ASPECTS = {
    'Su': [7],
    'Mo': [7],
    'Me': [7],
    'Ve': [7],
    'Ma': [4, 7, 8],
    'Ju': [5, 7, 9],
    'Sa': [3, 7, 10],
    'Ra': [5, 7, 9],
    'Ke': [5, 7, 9],
}

# Nifty/BankNifty inception — Moon in Taurus
NATAL_MO_SIGN = 2  # Taurus

# 20 key pairs for degree-based aspects (chosen for astrological significance)
KEY_PAIRS_DEG = [
    ('Ju', 'Sa'), ('Ju', 'Ma'), ('Sa', 'Ma'),
    ('Ju', 'Su'), ('Sa', 'Su'), ('Ma', 'Su'),
    ('Ju', 'Mo'), ('Sa', 'Mo'), ('Ma', 'Mo'),
    ('Ve', 'Ju'), ('Ve', 'Sa'), ('Ve', 'Ma'),
    ('Me', 'Ju'), ('Me', 'Sa'), ('Me', 'Ma'),
    ('Ra', 'Ju'), ('Ra', 'Sa'), ('Ra', 'Mo'),
    ('Su', 'Mo'), ('Me', 'Ve'),
]

# Degree-based aspect targets and orbs
ASP_DEG  = {'conj': 0,   'opp': 180, 'trine': 120, 'sq': 90, 'sext': 60}
ASP_ORB  = {'conj': 8,   'opp': 8,   'trine': 7,   'sq': 7,  'sext': 6}
ASP_ORB_TIGHT = 3  # tight orb for "exact" aspect flags

# ── Helper ──────────────────────────────────────────────────────────────────────

def asp_sign_series(sign_series, H):
    """Vectorised: given planet's sign (1-12), return the sign it aspects at house H."""
    return ((sign_series - 1 + H - 1) % 12) + 1

def angular_sep(sid1, sid2):
    """Shortest arc between two sidereal longitudes (0–180°)."""
    raw = (sid1 - sid2).abs() % 360
    return raw.where(raw <= 180, 360 - raw)

# ── Main feature builder ────────────────────────────────────────────────────────

def add_aspect_features(df):
    t0 = time.time()
    new_cols = {}

    # ── A. Sign-based Vedic aspects: asp{H}_{P1}_{P2} ─────────────────────────
    # P1 aspects P2 through house H → binary 0/1
    print("  A. Sign-based Vedic aspects …", flush=True)
    for p1, asp_houses in PLANET_ASPECTS.items():
        s1 = df[f'sign_{p1}']
        for H in asp_houses:
            a_sign = asp_sign_series(s1, H)          # series of aspected signs
            for p2 in PLANETS:
                if p2 == p1: continue
                new_cols[f'asp{H}_{p1}_{p2}'] = (df[f'sign_{p2}'] == a_sign).astype(np.int8)

    # ── B. Aspected sign category & its lord ──────────────────────────────────
    # asp{H}_{P1}_sign  : integer 1-12 — which sign P1 hits through house H
    # asp{H}_{P1}_lord  : string   — who rules that sign
    print("  B. Aspected sign & lord …", flush=True)
    for p1, asp_houses in PLANET_ASPECTS.items():
        s1 = df[f'sign_{p1}']
        for H in asp_houses:
            a_sign = asp_sign_series(s1, H)
            new_cols[f'asp{H}_{p1}_sign'] = a_sign.astype(np.int8)
            new_cols[f'asp{H}_{p1}_lord'] = a_sign.map(SIGN_LORD)

    # ── C. Lord-domain aspect: does P1's field touch P2's own signs? ──────────
    # asp_{P1}_dom_{P2} = 1 if P1 aspects ANY sign ruled by P2 (via any aspect house)
    # Example: Saturn in Pisces → 10th aspect hits Sagittarius → asp_Sa_dom_Ju = 1
    print("  C. Lord-domain aspects …", flush=True)
    LORD_PLANETS = ['Su', 'Mo', 'Ma', 'Me', 'Ju', 'Ve', 'Sa']
    for p1, asp_houses in PLANET_ASPECTS.items():
        s1 = df[f'sign_{p1}']
        aspected_signs_list = [asp_sign_series(s1, H) for H in asp_houses]

        for p2 in LORD_PLANETS:
            if p2 == p1: continue
            if p2 not in PLANET_OWN_SIGNS: continue
            p2_signs = PLANET_OWN_SIGNS[p2]
            # 1 if any of P1's aspected signs falls inside P2's own signs
            combined = pd.Series(False, index=df.index)
            for a_sign in aspected_signs_list:
                combined |= a_sign.isin(p2_signs)
            new_cols[f'asp_{p1}_dom_{p2}'] = combined.astype(np.int8)

    # ── D. Natal Moon sign aspects ─────────────────────────────────────────────
    # Does P1's field (any aspect house) touch Taurus — the market's natal Moon sign?
    print("  D. Natal Moon sign (Taurus) aspects …", flush=True)
    for p1, asp_houses in PLANET_ASPECTS.items():
        s1 = df[f'sign_{p1}']
        hits_natal = pd.Series(False, index=df.index)
        for H in asp_houses:
            hits_natal |= (asp_sign_series(s1, H) == NATAL_MO_SIGN)
        new_cols[f'asp_{p1}_natal_mo'] = hits_natal.astype(np.int8)

    # ── E. Degree-based aspects for 20 key pairs ──────────────────────────────
    # deg_{type}_{P1}_{P2}: conjunction / opposition / trine / square / sextile
    print("  E. Degree-based aspects …", flush=True)
    for p1, p2 in KEY_PAIRS_DEG:
        sep = angular_sep(df[f'sid_{p1}'], df[f'sid_{p2}'])
        for asp_name, target in ASP_DEG.items():
            orb = ASP_ORB[asp_name]
            new_cols[f'deg_{asp_name}_{p1}_{p2}'] = (
                (sep - target).abs() <= orb
            ).astype(np.int8)

    # ── F. Tight (≤3°) exact aspects for the three slow-planet pairs ──────────
    # These fire rarely but are extremely potent when they do
    print("  F. Tight exact aspects (≤3°) …", flush=True)
    TIGHT_PAIRS = [('Ju','Sa'), ('Sa','Ma'), ('Ju','Ma'), ('Ju','Su'), ('Sa','Su')]
    for p1, p2 in TIGHT_PAIRS:
        sep = angular_sep(df[f'sid_{p1}'], df[f'sid_{p2}'])
        for asp_name, target in ASP_DEG.items():
            new_cols[f'ex_{asp_name}_{p1}_{p2}'] = (
                (sep - target).abs() <= ASP_ORB_TIGHT
            ).astype(np.int8)

    # ── G. Aggregate counts ───────────────────────────────────────────────────
    # n_asp_{P1}: how many planets fall inside P1's full aspect field (sign-based)
    print("  G. Aggregate planet-count under each major planet's field …", flush=True)
    for p1, asp_houses in [('Ju',[5,7,9]), ('Sa',[3,7,10]), ('Ma',[4,7,8])]:
        s1 = df[f'sign_{p1}']
        count = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
        for H in asp_houses:
            a_sign = asp_sign_series(s1, H)
            for p2 in PLANETS:
                if p2 == p1: continue
                count += (df[f'sign_{p2}'] == a_sign).astype(np.int8)
        new_cols[f'n_asp_{p1}'] = count

    # ── H. Key interaction features ───────────────────────────────────────────
    # Combine the most astrologically significant aspect signals with dignity / paksha
    print("  H. Interaction features …", flush=True)

    # Saturn aspecting Jupiter's domain × Jupiter's own dignity
    if 'asp_Sa_dom_Ju' in new_cols and 'dig_Ju' in df.columns:
        new_cols['ix_sadom_ju_digju'] = (
            new_cols['asp_Sa_dom_Ju'].astype(str) + '_' + df['dig_Ju'].astype(str)
        )

    # Jupiter aspecting Saturn's domain × Saturn's dignity
    if 'asp_Ju_dom_Sa' in new_cols and 'dig_Sa' in df.columns:
        new_cols['ix_judom_sa_digsa'] = (
            new_cols['asp_Ju_dom_Sa'].astype(str) + '_' + df['dig_Sa'].astype(str)
        )

    # Mars aspecting Moon (any of Mars's 3 aspects) × paksha
    ma_asp_mo_any = (
        new_cols.get('asp4_Ma_Mo', pd.Series(0, index=df.index)) |
        new_cols.get('asp7_Ma_Mo', pd.Series(0, index=df.index)) |
        new_cols.get('asp8_Ma_Mo', pd.Series(0, index=df.index))
    ).astype(np.int8)
    new_cols['ma_asp_mo_any'] = ma_asp_mo_any
    if 'paksha' in df.columns:
        new_cols['ix_ma_asp_mo_paksha'] = ma_asp_mo_any.astype(str) + '_' + df['paksha'].astype(str)

    # Jupiter aspecting Moon (any of Ju's 3 aspects) × dig_Ju
    ju_asp_mo_any = (
        new_cols.get('asp5_Ju_Mo', pd.Series(0, index=df.index)) |
        new_cols.get('asp7_Ju_Mo', pd.Series(0, index=df.index)) |
        new_cols.get('asp9_Ju_Mo', pd.Series(0, index=df.index))
    ).astype(np.int8)
    new_cols['ju_asp_mo_any'] = ju_asp_mo_any
    if 'dig_Ju' in df.columns:
        new_cols['ix_ju_asp_mo_digju'] = ju_asp_mo_any.astype(str) + '_' + df['dig_Ju'].astype(str)

    # Saturn aspecting Moon (any of Sa's 3 aspects) × dig_Sa
    sa_asp_mo_any = (
        new_cols.get('asp3_Sa_Mo', pd.Series(0, index=df.index)) |
        new_cols.get('asp7_Sa_Mo', pd.Series(0, index=df.index)) |
        new_cols.get('asp10_Sa_Mo', pd.Series(0, index=df.index))
    ).astype(np.int8)
    new_cols['sa_asp_mo_any'] = sa_asp_mo_any
    if 'dig_Sa' in df.columns:
        new_cols['ix_sa_asp_mo_digsa'] = sa_asp_mo_any.astype(str) + '_' + df['dig_Sa'].astype(str)

    # Jupiter-Saturn mutual field (sign-based): Ju aspects Sa's sign AND Sa aspects Ju's sign
    ju_on_sa = new_cols.get('asp7_Ju_Sa', pd.Series(0, index=df.index))  # 7th
    sa_on_ju = new_cols.get('asp7_Sa_Ju', pd.Series(0, index=df.index))
    new_cols['mut_asp_Ju_Sa'] = (ju_on_sa & sa_on_ju).astype(np.int8)

    ju_5_sa = new_cols.get('asp5_Ju_Sa', pd.Series(0, index=df.index))
    ju_9_sa = new_cols.get('asp9_Ju_Sa', pd.Series(0, index=df.index))
    sa_3_ju = new_cols.get('asp3_Sa_Ju', pd.Series(0, index=df.index))
    sa_10_ju = new_cols.get('asp10_Sa_Ju', pd.Series(0, index=df.index))
    new_cols['ju_special_asp_Sa'] = (ju_5_sa | ju_9_sa).astype(np.int8)   # Ju's 5th or 9th on Sa
    new_cols['sa_special_asp_Ju'] = (sa_3_ju | sa_10_ju).astype(np.int8)  # Sa's 3rd or 10th on Ju
    new_cols['any_Ju_Sa_asp'] = (
        new_cols['mut_asp_Ju_Sa'] | new_cols['ju_special_asp_Sa'] | new_cols['sa_special_asp_Ju']
    ).astype(np.int8)

    # Jupiter-Mars
    ju_on_ma = new_cols.get('asp5_Ju_Ma', pd.Series(0, index=df.index)) | \
               new_cols.get('asp7_Ju_Ma', pd.Series(0, index=df.index)) | \
               new_cols.get('asp9_Ju_Ma', pd.Series(0, index=df.index))
    ma_on_ju = new_cols.get('asp4_Ma_Ju', pd.Series(0, index=df.index)) | \
               new_cols.get('asp7_Ma_Ju', pd.Series(0, index=df.index)) | \
               new_cols.get('asp8_Ma_Ju', pd.Series(0, index=df.index))
    new_cols['any_Ju_Ma_asp'] = (ju_on_ma | ma_on_ju).astype(np.int8)

    # Saturn-Mars
    sa_on_ma = new_cols.get('asp3_Sa_Ma', pd.Series(0, index=df.index)) | \
               new_cols.get('asp7_Sa_Ma', pd.Series(0, index=df.index)) | \
               new_cols.get('asp10_Sa_Ma', pd.Series(0, index=df.index))
    ma_on_sa = new_cols.get('asp4_Ma_Sa', pd.Series(0, index=df.index)) | \
               new_cols.get('asp7_Ma_Sa', pd.Series(0, index=df.index)) | \
               new_cols.get('asp8_Ma_Sa', pd.Series(0, index=df.index))
    new_cols['any_Sa_Ma_asp'] = (sa_on_ma | ma_on_sa).astype(np.int8)

    # Combined dignities of aspecting planets on Moon
    if 'dig_Ju' in df.columns and 'dig_Sa' in df.columns and 'dig_Ma' in df.columns:
        new_cols['ix_asp_mo_ju_sa_dig'] = (
            ju_asp_mo_any.astype(str) + '_' + df['dig_Ju'].astype(str) + '|' +
            sa_asp_mo_any.astype(str) + '_' + df['dig_Sa'].astype(str)
        )

    elapsed = time.time() - t0
    print(f"  Done: {len(new_cols)} new features in {elapsed:.1f}s", flush=True)
    return pd.DataFrame(new_cols, index=df.index)


# ── Process both CSVs ──────────────────────────────────────────────────────────

for fname, label in [
    ('data/nifty_enriched.csv',     'Nifty'),
    ('data/banknifty_enriched.csv', 'BankNifty'),
]:
    path = f"{REPO}/{fname}"
    print(f"\n{'='*60}")
    print(f"Processing {label}: {path}")
    t0 = time.time()

    df = pd.read_csv(path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Drop any previously added aspect cols to avoid duplicates on re-run
    existing_asp = [c for c in df.columns if c.startswith(('asp', 'deg_', 'ex_', 'mut_', 'ix_asp',
                                                             'ma_asp_mo_any','ju_asp_mo_any',
                                                             'sa_asp_mo_any','n_asp_Ju','n_asp_Sa',
                                                             'n_asp_Ma','any_Ju','any_Sa'))]
    if existing_asp:
        df = df.drop(columns=existing_asp)
        print(f"  Dropped {len(existing_asp)} stale aspect columns")

    new_feats = add_aspect_features(df)
    df = pd.concat([df, new_feats], axis=1)

    print(f"  Final shape: {df.shape[0]} rows × {df.shape[1]} columns  (+{len(new_feats.columns)} aspect features)")
    df.to_csv(path, index=False)
    print(f"  Saved {path}  [{time.time()-t0:.1f}s]")

print("\n=== Fix 6 complete ===")
print("New feature categories:")
sample = pd.read_csv(f"{REPO}/data/nifty_enriched.csv", nrows=1)
asp_cols = [c for c in sample.columns if c.startswith('asp') and not c.startswith('ix')]
deg_cols = [c for c in sample.columns if c.startswith('deg_')]
ex_cols  = [c for c in sample.columns if c.startswith('ex_')]
dom_cols = [c for c in sample.columns if '_dom_' in c]
lord_cols = [c for c in sample.columns if c.endswith('_lord') or c.endswith('_sign') and 'asp' in c]
ix_new   = [c for c in sample.columns if c.startswith('ix_') and 'asp' in c]
misc     = [c for c in sample.columns if any(c.startswith(p) for p in ['mut_','any_','n_asp','ju_asp_mo_any','sa_asp_mo_any','ma_asp_mo_any'])]
print(f"  Sign-based asp{{H}}_P1_P2:  {len(asp_cols)}")
print(f"  Lord-domain asp_P1_dom_P2: {len(dom_cols)}")
print(f"  Aspected sign/lord:        {len(lord_cols)}")
print(f"  Degree-based deg_*:        {len(deg_cols)}")
print(f"  Tight exact ex_*:          {len(ex_cols)}")
print(f"  Interaction ix_*:          {len(ix_new)}")
print(f"  Misc (mut,any,n_asp):      {len(misc)}")
print(f"  Total new columns:         {df.shape[1] - 353}")
print(f"\nTotal columns: {sample.shape[1]}")
