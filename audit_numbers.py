#!/usr/bin/env python
"""
Re-derive every headline number the paper states, from the result files.

    python3 audit_numbers.py            # check
    python3 audit_numbers.py --list     # also dump every prose number, with context

WHY THIS EXISTS. In two days of checking, four numeric claims in this paper
turned out to be wrong: the corpora were described as 200 questions where every
file holds 100; the observation count summed duplicate rows; a sentence claimed
alpha was absent from the certificate when it is not; and a "net harmful"
finding had no test behind it. Each was typed rather than derived, and each
survived several readings. The SaTML CfP desk-rejects for falsified results and
says reviewers may check artifacts against claims, so the standard here is that
every load-bearing number is recomputed from the CSVs, not remembered.

A claim that fails here is either a wrong number in the paper or a wrong
expectation in this file. Both need a human. Neither is allowed to pass silently.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--results', default='results')
ap.add_argument('--sections', default='../../../paper-satml2027/sections')
ap.add_argument('--list', action='store_true')
A = ap.parse_args()
R, S = A.results, A.sections

FAIL, OK, SKIP = [], [], []


def rows(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def pct(sel, tot):
    return 100.0 * sel / tot if tot else float('nan')


def check(label, got, want, tol=0.6, unit='%'):
    """Compare a recomputed value against what the paper prints."""
    if got is None:
        SKIP.append(f'{label}: data not present')
        return
    if isinstance(want, str):
        ok = str(got) == want
    else:
        ok = abs(got - want) <= tol
    line = f'{label}: paper says {want}{unit}, data gives {got:.4g}{unit}' \
        if not isinstance(want, str) else f'{label}: paper {want}, data {got}'
    (OK if ok else FAIL).append(line)


# ---------------------------------------------------------------------------
om = rows('omega_realtimeqa_k10_all.csv')
van = rows('vanilla_realtimeqa_k10.csv')
sweep = rows('sweep.csv')

# --- low-n survival: "100% of queries with n<=3 against 2% otherwise"
sub = [r for r in om if r['group_size'] == '1' and r['k_prime'] == '1'
       and str(r.get('attacker_kw_survived', '')).strip() != '']
lo = [r for r in sub if int(r['n_retained']) <= 3]
hi = [r for r in sub if int(r['n_retained']) > 3]
check('low-n survival (n<=3)',
      pct(sum(int(r['attacker_kw_survived']) for r in lo), len(lo)), 100.0)
check('low-n survival N (n<=3)', float(len(lo)), 88.0, tol=0, unit='')
check('high-n survival (n>3)',
      pct(sum(int(r['attacker_kw_survived']) for r in hi), len(hi)), 1.8)
check('high-n survival N', float(len(hi)), 512.0, tol=0, unit='')

# --- mu surface: "99.4% where mu<=1 vs 4.0% where mu>1" over the sweep
# k' MATTERS HERE. sweep.csv holds three corruption sizes (1,800 rows each) and
# the vacuity boundary generalises to mu <= k', so pooling them against a fixed
# mu <= 1 boundary mixes three different contrasts and gives 52.7% for mu>1.
# The paper's 99.4 / 4.0 figures are the k'=1 subset; this filter must match.
sw = [r for r in sweep if str(r.get('attacker_kw_survived', '')).strip() != ''
      and str(r.get('mu', '')).strip() != '' and r.get('k_prime') == '1']
vac = [r for r in sw if float(r['mu']) <= 1.0]
non = [r for r in sw if float(r['mu']) > 1.0]
check('mu<=1 survival',
      pct(sum(int(r['attacker_kw_survived']) for r in vac), len(vac)), 99.4)
check('mu<=1 N', float(len(vac)), 519.0, tol=0, unit='')
check('mu>1 survival',
      pct(sum(int(r['attacker_kw_survived']) for r in non), len(non)), 4.0)
check('mu>1 N', float(len(non)), 1281.0, tol=0, unit='')

# --- the non-strict boundary: every observation with mu == 1.00 exactly
ex = [r for r in sw if abs(float(r['mu']) - 1.0) < 1e-9]
check('mu==1.00 exactly, N', float(len(ex)), 100.0, tol=0, unit='')
check('mu==1.00 exactly, survived',
      float(sum(int(r['attacker_kw_survived']) for r in ex)), 100.0, tol=0, unit='')
check('mu==1.00 reached by how many (alpha,n) routes',
      float(len({(r['alpha'], r['n_retained']) for r in ex})), 3.0, tol=0, unit='')

# --- omega>=4 vacuity: every model, every query
bad = []
for m in {r['model'] for r in om}:
    for w in ('4', '5'):
        cell = [r for r in om if r['model'] == m and r['group_size'] == w
                and str(r.get('mu', '')).strip() != '']
        if cell and not all(float(r['mu']) <= 1.0 for r in cell):
            bad.append(f'{m}@w={w}')
(OK if not bad else FAIL).append(
    'omega>=4 vacuous on every query for every model'
    + ('' if not bad else f' -- VIOLATED for {bad}'))

# --- undefended vs defended on llama70b: "8% undefended vs 16% defended"
u = [int(r['undefended_asr']) for r in van
     if r['model'] == 'llama70b-bedrock' and r['attack'] != 'none'
     and str(r.get('undefended_asr', '')).strip() != '']
d = [int(r['defended_asr']) for r in om
     if r['model'] == 'llama70b-bedrock' and r['group_size'] == '1'
     and r['k_prime'] == '1' and str(r.get('defended_asr', '')).strip() != '']
check('llama70b undefended ASR', pct(sum(u), len(u)), 8.0)
check('llama70b defended ASR (w=1)', pct(sum(d), len(d)), 16.0)

# --- and the McNemar behind it: b=4, c=12, p=0.077
vi = {r['q_idx']: int(r['undefended_asr']) for r in van
      if r['model'] == 'llama70b-bedrock' and r['attack'] != 'none'
      and str(r.get('undefended_asr', '')).strip() != ''}
di = {r['q_idx']: int(r['defended_asr']) for r in om
      if r['model'] == 'llama70b-bedrock' and r['group_size'] == '1'
      and r['k_prime'] == '1' and str(r.get('defended_asr', '')).strip() != ''}
keys = sorted(set(vi) & set(di))
b = sum(1 for k in keys if vi[k] == 1 and di[k] == 0)
c = sum(1 for k in keys if vi[k] == 0 and di[k] == 1)
n = b + c
p = min(1.0, sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2) \
    if n else 1.0
check('llama70b McNemar b', float(b), 4.0, tol=0, unit='')
check('llama70b McNemar c', float(c), 12.0, tol=0, unit='')
check('llama70b McNemar p', p, 0.077, tol=0.001, unit='')
check('llama70b paired N', float(len(keys)), 100.0, tol=0, unit='')

# --- corpus coverage: 5.0 / 11.0 / 43.0 at omega=1, k'=0, mistral
for fn, lab, want in (('omega_open_nq_k10.csv', 'NQ', 5.0),
                      ('omega_realtimeqa_k10_all.csv', 'RealtimeQA', 11.0),
                      ('omega_hotpotqa_k10.csv', 'HotpotQA', 43.0)):
    rr = [r for r in rows(fn) if r['group_size'] == '1' and r['k_prime'] == '0'
          and r.get('model') == 'mistral7b-bedrock']
    check(f'coverage {lab} (n<=3)',
          pct(sum(1 for r in rr if int(r['n_retained']) <= 3), len(rr)) if rr else None,
          want)
    check(f'coverage {lab} N', float(len(rr)) if rr else None, 100.0, tol=0, unit='')

# --- the floor: 16%->4% at omega=1, 61%->2% at omega=4
fix = rows('fix_realtimeqa_k10_all.csv')
for w, wn, wf in (('1', 16.0, 4.0), ('4', 61.0, 2.0)):
    for mode, want in (('n', wn), ('floor2', wf)):
        rr = [r for r in fix if r['group_size'] == w and r['threshold_mode'] == mode
              and r['k_prime'] == '1'
              and str(r.get('attacker_kw_survived', '')).strip() != '']
        check(f'floor w={w} mode={mode}',
              pct(sum(int(r['attacker_kw_survived']) for r in rr), len(rr)) if rr else None,
              want)

# --- certification (only once the files exist)
cn = {}
cf = {}
for p_ in glob.glob(os.path.join(R, 'certify_n_*.csv')):
    for r in rows(os.path.basename(p_)):
        cn[(r['model'], r['q_idx'])] = r
for p_ in glob.glob(os.path.join(R, 'certify_floor2_*.csv')):
    for r in rows(os.path.basename(p_)):
        cf[(r['model'], r['q_idx'])] = r
if cn and cf:
    both = sorted(set(cn) & set(cf))
    cb = sum(1 for k in both
             if int(cn[k]['certified']) == 1 and int(cf[k]['certified']) == 0)
    cc = sum(1 for k in both
             if int(cn[k]['certified']) == 0 and int(cf[k]['certified']) == 1)
    # Look in the paper tree first, then beside this script. A standalone copy of
    # the code has no paper tree, and make_paper_assets.py then writes its assets
    # to ./generated, so these three checks are still runnable there.
    #
    # The missing branch below used to be absent entirely: when the file was not
    # found, three checks simply did not happen and the total silently dropped
    # from 34 to 31 with nothing reported. A check that disappears is worse than
    # one that fails, because the summary line still says 0 FAILED.
    facts = os.path.join(os.path.dirname(S), 'generated', 'certify_facts.tex')
    if not os.path.exists(facts):
        facts = os.path.join('generated', 'certify_facts.tex')
    if os.path.exists(facts):
        txt = open(facts).read()

        def mac(nm):
            m = re.search(r'\\newcommand\{\\' + nm + r'\}\{([^}]*)\}', txt)
            return m.group(1) if m else None
        check('certify b vs macro', float(cb), float(mac('certb')), tol=0, unit='')
        check('certify c vs macro', float(cc), float(mac('certc')), tol=0, unit='')
        check('certify N vs macro', float(len(both)), float(mac('certtotal')),
              tol=0, unit='')
    else:
        SKIP.append('certify b/c/N vs macro: certify_facts.tex not found '
                    '(run make_paper_assets.py first)')
else:
    SKIP.append('certification: certify_*.csv not present yet')

# --- the enumeration cap Section VIII discloses
# The released certifier abandons a query whose addable keyword set exceeds 14,
# returning uncertified. Section VIII states it binds on exactly two of
# Mistral-7B's queries and on no other model's. That claim comes from the
# profiler's JSON, so re-derive it rather than trusting the sentence.
import json as _json
_cap = {}
for _f in glob.glob('logs/certify_cost_*.json'):
    _m = os.path.basename(_f).replace('certify_cost_', '').rsplit('_', 1)[0]
    try:
        _d = _json.load(open(_f))
    except Exception:
        continue
    _cap[_m] = max(_cap.get(_m, 0), sum(1 for _x in _d if _x.get('skipped')))
if _cap:
    _hit = {m: v for m, v in _cap.items() if v}
    check('cap: models affected', float(len(_hit)), 1.0, tol=0, unit='')
    check('cap: Mistral queries abandoned',
          float(_cap.get('mistral7b-bedrock', 0)), 2.0, tol=0, unit='')
    _others = sum(v for m, v in _cap.items() if m != 'mistral7b-bedrock')
    check('cap: any other model affected', float(_others), 0.0, tol=0, unit='')
else:
    SKIP.append('enumeration cap: no profiler JSON to check against')

# ---------------------------------------------------------------------------
print('=' * 72)
for l in OK:
    print(f'  [ ok ] {l}')
for l in SKIP:
    print(f'  [skip] {l}')
for l in FAIL:
    print(f'  [FAIL] {l}')
print('=' * 72)
print(f'{len(OK)} ok, {len(SKIP)} skipped, {len(FAIL)} FAILED')

if A.list:
    print('\nevery numeric literal in the prose, for eyeball review:')
    for f in sorted(glob.glob(os.path.join(S, '*.tex'))):
        body = [l for l in open(f).read().splitlines()
                if not l.lstrip().startswith('%')]
        for i, l in enumerate(body, 1):
            for m in re.finditer(r'\$?\\?[\d]+(?:[.,]\d+)?\\?%?\$?', l):
                t = m.group(0)
                if re.search(r'\d', t):
                    print(f'  {os.path.basename(f):22s} {t:12s} '
                          f'{l.strip()[:80]}')
                    break

sys.exit(1 if FAIL else 0)
