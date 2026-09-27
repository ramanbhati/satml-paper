#!/usr/bin/env python
"""
The benign performance drop relative to VANILLA RAG, per model and per omega.

    python3 analyze_vanilla.py results/vanilla_realtimeqa_k10.csv \
                                        results/omega_realtimeqa_k10_all.csv

WHAT THIS ANSWERS. RobustRAG's utility claim is stated against undefended RAG:
"with omega=3, we reduce the benign performance drop from 7% to 0%". That is a
claim about (vanilla - defended), which no omega CSV can evaluate on its own,
because every omega run used --no_vanilla. This joins the two files on
(model, q_idx) and reports the drop directly.

WHY THE JOIN IS PAIRED. Both files record the same 100 queries under the same
q_idx, so vanilla and defended outcomes can be compared query by query. That
makes McNemar the right test again -- the same argument as analyze_significance.py.
Comparing the two accuracy RATES with an independent-samples test would ignore
that the same questions appear on both sides.

A caveat worth carrying into the paper: vanilla is measured ONCE per model,
because it does not depend on omega. The same vanilla column is therefore reused
across the omega rows. That is correct, not a shortcut -- but it does mean the
drop estimates at different omega share a common reference and are not
independent of one another.
"""
import csv
import math
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    sys.exit(__doc__.strip().splitlines()[2].strip())
van_path, omega_path = sys.argv[1], sys.argv[2]


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mcnemar(pairs):
    b = sum(1 for a, x in pairs if a == 1 and x == 0)
    c = sum(1 for a, x in pairs if a == 0 and x == 1)
    n = b + c
    if n == 0:
        return dict(b=b, c=c, z=0.0, p=1.0)
    chi2 = (abs(b - c) - 1) ** 2 / n
    z = math.copysign(math.sqrt(chi2), c - b)
    if n < 25:
        lo = min(b, c)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / 2.0 ** n)
    else:
        p = 2.0 * (1.0 - _phi(abs(z)))
    return dict(b=b, c=c, z=z, p=p)


# vanilla: clean arm only (k_prime == 0) for the utility baseline
van = {}
van_asr = {}
for r in csv.DictReader(open(van_path)):
    key = (r['model'], int(r['q_idx']))
    if str(r.get('k_prime', '0')).strip() in ('0', ''):
        van[key] = int(r['undefended_correct'])
    else:
        a = str(r.get('undefended_asr', '')).strip()
        if a != '':
            van_asr[key] = int(a)

dfn = defaultdict(dict)
for r in csv.DictReader(open(omega_path)):
    if r['k_prime'] != '0':
        continue
    v = str(r.get('defended_correct', '')).strip()
    if v == '':
        continue
    dfn[(r['model'], int(r['group_size']))][int(r['q_idx'])] = int(v)

models = sorted({m for m, _ in van}) or sorted({m for m, _ in dfn})
omegas = sorted({w for _, w in dfn})
if not van:
    sys.exit(f'no clean-arm vanilla rows in {van_path}')

print(f'\nBENIGN PERFORMANCE DROP vs VANILLA RAG   (clean arm, paired by q_idx)')
print(f'  vanilla: {van_path}')
print(f'  defended: {omega_path}\n')
print(f"  {'model':<20}{'vanilla':>8}" + ''.join(f'{f"w={w}":>18}' for w in omegas))
print('  ' + '-' * (28 + 18 * len(omegas)))

for m in models:
    vq = {q: y for (mm, q), y in van.items() if mm == m}
    if not vq:
        continue
    vacc = sum(vq.values()) / len(vq)
    cells = []
    for w in omegas:
        dq = dfn.get((m, w), {})
        common = sorted(set(vq) & set(dq))
        if not common:
            cells.append(f'{"--":>18}')
            continue
        dacc = sum(dq[q] for q in common) / len(common)
        st = mcnemar([(vq[q], dq[q]) for q in common])
        star = '***' if st['p'] < .001 else '**' if st['p'] < .01 else '*' if st['p'] < .05 else ''
        cells.append(f'{(dacc - vacc) * 100:>+8.0f}pt {star:<3}{"":>3}')
    print(f'  {m:<20}{vacc:>7.0%} ' + ''.join(cells))

print('\n  Negative = the defense COSTS accuracy relative to undefended RAG.')
print('  Stars are uncorrected McNemar on the paired queries; the drops at')
print('  different omega share one vanilla reference and are not independent.')

if van_asr:
    print('\n\nTHE GATE -- does the attack work WITHOUT a defense?')
    print('  If undefended ASR is near zero, the attack is not a threat and a')
    print('  defense "stopping" it demonstrates nothing. This is the control that')
    print('  makes the omega result meaningful.\n')
    print(f"  {'model':<20}{'undefended ASR':>16}")
    for m in models:
        aq = [y for (mm, _), y in van_asr.items() if mm == m]
        if aq:
            print(f'  {m:<20}{sum(aq) / len(aq):>15.0%}')
print()
