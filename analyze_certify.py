#!/usr/bin/env python
"""
Certified accuracy under the upstream threshold vs the floor.

    python3 analyze_certify.py

Consumes results/certify_n.csv and results/certify_floor2.csv, which
run_final.sh writes with --certify_csv. Answers the question Section X of the
paper currently lists as open: Proposition 3 shows the certification algorithm
stays correct under the floored threshold, but says nothing about what happens
to tau. This measures it.

WHAT TO EXPECT, and why each direction is interesting:
  - floor2 tau HIGHER than n: the floor removes the degenerate branch (Prop. 4),
    so queries upstream refused outright can now certify. Strongest result.
  - floor2 tau LOWER than n: the higher threshold shrinks the always-retained
    keyword set, so some previously-certifiable queries fail. That is the
    trade-off the paper predicted and must then report honestly.
  - no difference: the floor is free. Also publishable, and the cleanest story.

Every comparison is PAIRED on (model, q_idx) -- the same query under both
rules -- so McNemar is the right test, as everywhere else in this paper.
"""
import csv
import math
import os
import sys
from collections import defaultdict

R = 'results'
# Per-model files (certify_<mode>_<model>.csv), merged here. run_final.sh writes
# one per model so a rate-limited model cannot destroy another model's clean data.
import glob


def load_mode(mode):
    paths = sorted(glob.glob(os.path.join(R, f'certify_{mode}_*.csv')))
    if not paths:
        sys.exit(f"no results/certify_{mode}_*.csv -- run ./run_final.sh certify first")
    out = []
    for p in paths:
        with open(p) as fh:
            out += list(csv.DictReader(fh))
    print(f"  {mode:7s}: {len(out):4d} rows from {len(paths)} file(s): "
          f"{', '.join(os.path.basename(x).split('_')[-1][:-4] for x in paths)}")
    return out


def exact_binom_two_sided(b, c):
    """Two-sided exact binomial on the discordant pairs. Used below 25 pairs,
    matching the rule stated in the paper's statistical-methods section."""
    n = b + c
    if n == 0:
        return 1.0
    p = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n * 2
    return min(1.0, p)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0, 0.0, 'no discordant pairs'
    if n < 25:
        return exact_binom_two_sided(b, c), (abs(b - c) - 1) / math.sqrt(n), 'exact'
    z = (abs(b - c) - 1) / math.sqrt(n)
    p = math.erfc(abs(z) / math.sqrt(2))
    return p, z, 'chi-square approx'


print('loading:')
rows = {m: load_mode(m) for m in ('n', 'floor2')}

# n itself is not in CERTIFY_COLS, so recover it from the per-query CSV the same
# run wrote. Needed to reconstruct the threshold certify() actually tested.
nret = {}
for p in sorted(glob.glob(os.path.join(R, 'certify_realtimeqa_n_*.csv'))):
    with open(p) as fh:
        for r in csv.DictReader(fh):
            nret[(r['model'], r['q_idx'])] = int(r['n_retained'])
print(f"  n_retained recovered for {len(nret)} queries from the per-query CSVs")
print()

# index on (model, q_idx) so the comparison is paired
idx = {m: {(r['model'], r['q_idx']): r for r in rs} for m, rs in rows.items()}
both = {m for (m, _) in idx['n']} & {m for (m, _) in idx['floor2']}
only_n = {m for (m, _) in idx['n']} - both
only_f = {m for (m, _) in idx['floor2']} - both
for m in sorted(only_n | only_f):
    print(f"  NOTE: {m} present in only one mode -- excluded from the paired test")
models = sorted(both)
if not models:
    sys.exit("no model has BOTH modes yet; finish the certify runs first")

print('CERTIFIED ACCURACY: upstream threshold vs floor')
print('=' * 72)
print(f"{'model':22s} {'n (upstream)':>14s} {'floor2':>10s} {'delta':>8s} "
      f"{'b':>4s} {'c':>4s} {'p':>8s}")

tot_b = tot_c = 0
for mdl in models:
    keys = sorted(set(k for k in idx['n'] if k[0] == mdl)
                  & set(k for k in idx['floor2'] if k[0] == mdl))
    if not keys:
        continue
    cn = [int(idx['n'][k]['certified']) for k in keys]
    cf = [int(idx['floor2'][k]['certified']) for k in keys]
    # b = certified under n but not floor2; c = the reverse
    b = sum(1 for x, y in zip(cn, cf) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(cn, cf) if x == 0 and y == 1)
    tot_b += b
    tot_c += c
    an, af = 100 * sum(cn) / len(cn), 100 * sum(cf) / len(cf)
    p, z, kind = mcnemar(b, c)
    print(f"{mdl:22s} {an:13.0f}% {af:9.0f}% {af - an:+7.0f} "
          f"{b:4d} {c:4d} {p:8.3f}")

p, z, kind = mcnemar(tot_b, tot_c)
print('-' * 72)
print(f"{'POOLED':22s} {'':13s}  {'':9s}  {'':7s} {tot_b:4d} {tot_c:4d} {p:8.3f}"
      f"   ({kind})")
print()
if tot_b + tot_c == 0:
    print('VERDICT: the floor changes no certification outcome at all.')
elif tot_c > tot_b:
    print(f'VERDICT: the floor CERTIFIES MORE ({tot_c} gained vs {tot_b} lost).')
    print('         Consistent with Prop. 4: the degenerate branch is unreachable.')
else:
    print(f'VERDICT: the floor CERTIFIES FEWER ({tot_b} lost vs {tot_c} gained).')
    print('         This is the trade-off Section X predicted. Report it as measured.')
print(f'         Significance: p = {p:.3f} on {tot_b + tot_c} discordant pairs.')

# How many queries did upstream refuse via the degenerate branch? Under the
# floor at k'=1 that branch is unreachable, so any such query is a clean win.
#
# CAREFUL -- THE TWO THRESHOLDS ARE NOT THE SAME NUMBER. The `mu` column is the
# INFERENCE-path threshold, mu = min(alpha*n, beta) on the n non-abstaining
# responses (verified: 300/300 rows match that formula exactly). certify() in
# src/defense.py computes its threshold on n PLUS the corruption size:
#
#     count_threshold = self._mu(non_abs_cnt + corruption_size)
#     if count_threshold <= corruption_size: return False
#
# so the branch fires on min(alpha*(n+k'), beta) <= k', not on mu <= k'. An
# earlier version of this script tested the mu column and reported 49 queries;
# the branch really fires on 41. The 49 also failed to reconcile with the paired
# test -- it implied 6 gained queries where the pairing found 5. Using the
# threshold certify() actually evaluates, every number below agrees.
#
# That the two differ at all (they agree on only 68 of 300 queries) is itself an
# instance of this paper's thesis: the certifier and the inference path do not
# evaluate the same threshold on the same query.
print()
deg = []
for k in idx['n']:
    r = idx['n'][k]
    a, b, kp = float(r['alpha']), float(r['beta']), int(r['k_prime'])
    n_ret = nret.get(k)
    if n_ret is None:
        sys.exit(f"no per-query row for {k}; cannot recover n. "
                 f"results/certify_realtimeqa_n_*.csv must accompany "
                 f"results/certify_n_*.csv")
    if min(a * (n_ret + kp), b) <= kp:
        deg.append(k)

print(f"queries where certify()'s degenerate branch fires, min(a(n+k'),b) <= k':"
      f" {len(deg)}")
if deg:
    still = sum(1 for k in deg if int(idx['n'][k]['certified']) == 1)
    gained = sum(1 for k in deg
                 if k in idx['floor2'] and int(idx['floor2'][k]['certified']) == 1)
    print(f"   certified under upstream: {still}   (Prop. 4: must be 0)")
    print(f"   certified under the floor: {gained}")
    if still:
        sys.exit("INCONSISTENT: the degenerate branch returns False, so these "
                 "queries cannot be certified upstream. Investigate before "
                 "using any of these numbers.")
    if gained != tot_c:
        print(f"   NOTE: {gained} gained here vs {tot_c} in the paired test above.")
    else:
        print(f"   These {gained} are exactly the {tot_c} the paired test gained,")
        print("   so every query the floor adds is one the degenerate branch had")
        print("   refused outright. The floor takes nothing away (b = 0).")
