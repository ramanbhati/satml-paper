#!/usr/bin/env python
"""
Does in-group resistance track model CAPABILITY, or is llama70b just unusual?

    python3 analyze_by_model.py results/omega_realtimeqa_k10_all.csv

THE QUESTION. At omega >= 4 the keyword filter is vacuous (mu <= 1) on 100% of
rows for every model -- that part is arithmetic. Whether the attacker then WINS
is decided by the in-group contest: the poisoned passage shares a prompt with an
honest one and has to out-argue it. Measured so far:

    llama8b   (8B)   95%
    mistral7b (7B)   74%
    llama70b (70B)   15%

Three small models lose; one large model wins. That is a capability hypothesis
with n=1 on the deciding side. Adding llama3b gives a WITHIN-FAMILY series --
3B, 8B, 70B, same training recipe, same template, same system_prompt setting --
so size is the only thing varying. If survival falls monotonically with size,
the weakest part of the paper becomes an interpretable finding:

    at omega >= 4 the defense contributes nothing, and what remains is model
    capability: small models lose the in-group contest, large models resist.

If it does NOT fall monotonically, llama70b is an outlier and the honest report
is a spread rather than a trend. Either answer is publishable; guessing is not.
"""
import csv
import math
import sys

def _load_registry():
    """Read BEDROCK_MODELS out of src/models.py WITHOUT importing it.

    Importing src.models pulls in torch, transformers, litellm and spacy, which
    makes this analyser unusable anywhere those are absent. The registry is a
    plain dict literal, so exec just that slice instead."""
    try:
        src = open('src/models.py').read()
        i = src.index('BEDROCK_MODELS = {')
        j = src.index('\n}\n', i) + 3
        ns = {}
        exec(compile(src[i:j], 'registry', 'exec'), ns)
        reg = ns['BEDROCK_MODELS']
        return ({k: v['family'] for k, v in reg.items()},
                {k: v.get('params_b') for k, v in reg.items()},
                {k: v.get('rank') for k, v in reg.items()})
    except Exception as e:
        print(f'note: could not read BEDROCK_MODELS ({e}); '
              f'family/params columns will be blank')
        return {}, {}, {}


FAM, PAR, RANK = _load_registry()


def ladder_pos(m):
    """Position of a model on its family's capability ladder, or None.

    Two kinds of ladder exist and they are NOT interchangeable:
      params_b  llama 8B/70B     -- a real parameter axis, ratios mean something
      rank      nova micro/lite/pro -- ORDINAL only; Amazon publishes no sizes
    Both order a family correctly, which is all a monotonicity test needs, so
    this returns whichever exists. Anything that wants to plot against SIZE must
    use PAR directly and skip models where it is None."""
    return PAR.get(m) if PAR.get(m) is not None else RANK.get(m)

path = sys.argv[1] if len(sys.argv) > 1 else 'results/omega_realtimeqa_k10_all.csv'
rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit(f'no rows in {path}')

models = sorted({r['model'] for r in rows},
                key=lambda m: (PAR.get(m) is None, PAR.get(m) or 0))
omegas = sorted({int(r['group_size']) for r in rows})
alpha = float(rows[0]['alpha']); beta = float(rows[0]['beta'])
K = int(rows[0].get('top_k') or 10)


def pc(g, c):
    v = [int(x[c] or 0) for x in g if str(x.get(c, '')).strip() != '']
    return 100 * sum(v) / len(v) if v else float('nan')


print(f'\n{len(rows)} rows | k={K} alpha={alpha} beta={beta}\n')
print('ATTACKER KEYWORD SURVIVAL, by model and omega')
hdr = f"  {'model':<20} {'family':<10} {'params':>7} | " + ' '.join(f'w={w:<4}' for w in omegas)
print(hdr); print('  ' + '-' * (len(hdr) - 2))
for m in models:
    cells = []
    for w in omegas:
        g = [x for x in rows if x['model'] == m and int(x['group_size']) == w
             and x['k_prime'] == '1']
        cells.append(f'{pc(g,"attacker_kw_survived"):>4.0f}%' if g else '   - ')
    p = PAR.get(m)
    print(f"  {m:<20} {FAM.get(m,'?'):<10} {(str(p)+'B') if p else '?':>7} | "
          + ' '.join(f'{c:<6}' for c in cells))

# the vacuous-omega columns are where the in-group contest is the only obstacle
vac = [w for w in omegas if min(alpha * math.ceil(K / w), beta) <= 1]
print(f'\n  omega with mu<=1 (filter vacuous, in-group contest is the ONLY obstacle): {vac}')

print('\nCAPABILITY TEST -- within a family, capability is the only variable')
_fams = {}
for m in models:
    if ladder_pos(m) is not None:
        _fams.setdefault(FAM.get(m, '?'), []).append(m)
_ladders = {f: sorted(ms, key=ladder_pos) for f, ms in _fams.items() if len(ms) >= 2}
if not _ladders:
    print('  no family in this CSV has two or more models on a capability ladder.')
    print('  llama is 8B/70B only (3B was end-of-lifed by AWS); the nova ladder')
    print('  is micro/lite/pro and needs calibrate_model.py before it can run.')
for fam, ms in sorted(_ladders.items()):
    # label the axis honestly: a parameter count is not an ordinal tier
    ordinal = all(PAR.get(m) is None for m in ms)
    axis = 'tier (ordinal)' if ordinal else 'params'
    print(f'\n  family={fam}   axis={axis}' +
          ('   [Amazon publishes no parameter counts -- rank order only]'
           if ordinal else ''))
    print(f"  {axis:>14}  {'model':<20} " + ' '.join(f'w={w:<5}' for w in vac))
    vals = []
    for m in ms:
        cells = []
        for w in vac:
            g = [x for x in rows if x['model'] == m and int(x['group_size']) == w
                 and x['k_prime'] == '1']
            cells.append(pc(g, 'attacker_kw_survived') if g else float('nan'))
        vals.append((m, cells))
        lab = f'{PAR[m]}B' if PAR.get(m) is not None else f'#{RANK.get(m)}'
        print(f"  {lab:>14}  {m:<20} " + ' '.join(f'{c:>5.0f}%' for c in cells))
    if len(vals) >= 3 and vac:
        col = [v[1][0] for v in vals]
        if any(c != c for c in col):          # NaN: a cell never ran
            print('  => INCOMPLETE: a model is missing the vacuous-omega cell; '
                  'no trend claim.')
        elif all(col[i] >= col[i + 1] for i in range(len(col) - 1)):
            print('  => MONOTONE: survival falls as capability rises.')
            print('     Report as: at omega>=4 the defense contributes nothing and')
            print('     residual protection is model capability.')
        else:
            print('  => NOT monotone. Report the SPREAD honestly; do not claim a')
            print('     capability law from a non-monotone ladder.')
    elif vac:
        print(f'  => {len(vals)} points is not a trend. Descriptive only.')
print()
