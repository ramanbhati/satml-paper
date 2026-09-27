#!/usr/bin/env python
"""
Significance tests for the omega-ladder and threshold-fix comparisons.

    python3 analyze_significance.py results/omega_realtimeqa_k10_all.csv
    python3 analyze_significance.py results/fix_realtimeqa_k10_all.csv --fix

WHY THIS FILE EXISTS. Every z value quoted for this project so far was computed
ad hoc, outside the repository. Nothing recomputed them, nothing reviewed them,
and nothing would catch it if one were wrong. Numbers headed for a paper need to
come from code that can be re-run. This is that code.

THE STATISTICAL POINT, which is not a formality.

The omega conditions are measured on THE SAME 100 queries. q_idx 7 at omega=1
and q_idx 7 at omega=4 are the same question, the same retrieved passages, the
same injected keyword -- only the grouping differs. Those observations are
PAIRED, and they are strongly positively correlated: a query whose corpus makes
the attack easy tends to be easy at every omega.

A two-proportion z-test assumes the two samples are INDEPENDENT. Applied to
paired data it uses the wrong standard error. With positive correlation between
conditions -- which is what we have -- the independent-samples SE is too LARGE,
so that test is conservative for the difference itself; but it is answering a
different question than the one we mean ("do these two populations differ?"
rather than "does changing omega change the outcome for a given query?"). The
correct test for a paired binary outcome is McNemar's, which conditions on the
queries that DISAGREED between the two conditions and ignores the rest.

Both are reported below, side by side, precisely so the difference is visible
and no one has to take a claim on trust. Quote McNemar. If the two ever point
different directions, that is a finding about the data, not a rounding issue.

McNemar with continuity correction:  chi2 = (|b - c| - 1)^2 / (b + c)
  b = queries where condition A survived and B did not
  c = queries where condition B survived and A did not
  Reported as z = sign * sqrt(chi2) so it is comparable to the z values already
  in circulation. Exact binomial p is used when b + c < 25, where the chi-square
  approximation is unreliable -- which happens often here.
"""
import argparse
import csv
import math
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument('path')
p.add_argument('--metric', default='attacker_kw_survived',
               help='binary per-query column to test (default: attacker_kw_survived)')
p.add_argument('--fix', action='store_true',
               help='compare threshold_mode variants instead of omega values')
p.add_argument('--arm', default='1',
               help="k_prime value selecting the arm (default '1' = attack; '0' = clean)")
args = p.parse_args()


def _phi(z):
    """Standard normal CDF via erf; avoids a scipy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - _phi(abs(z)))


def exact_binom_two_sided(b, c):
    """Exact two-sided binomial p for McNemar under H0: p=0.5 on b+c trials."""
    n = b + c
    if n == 0:
        return 1.0
    # P(X <= min) * 2, capped at 1 -- the standard two-sided exact convention
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(pairs):
    """pairs: list of (a_outcome, b_outcome) as 0/1. Returns dict."""
    b = sum(1 for a, x in pairs if a == 1 and x == 0)   # A only
    c = sum(1 for a, x in pairs if a == 0 and x == 1)   # B only
    n_disc = b + c
    if n_disc == 0:
        return dict(b=b, c=c, z=0.0, p=1.0, test='no discordant pairs')
    chi2 = (abs(b - c) - 1) ** 2 / n_disc                # continuity-corrected
    z = math.copysign(math.sqrt(chi2), c - b)            # + = B higher than A
    if n_disc < 25:
        return dict(b=b, c=c, z=z, p=exact_binom_two_sided(b, c), test='exact')
    return dict(b=b, c=c, z=z, p=two_sided_p_from_z(z), test='mcnemar')


def two_prop_z(x1, n1, x2, n2):
    """The INDEPENDENT-samples test, reported only for comparison."""
    if n1 == 0 or n2 == 0:
        return None, None
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return None, None
    z = (p2 - p1) / se
    return z, two_sided_p_from_z(z)


rows = []
with open(args.path) as fh:
    for r in csv.DictReader(fh):
        v = str(r.get(args.metric, '')).strip()
        if v == '' or str(r.get('k_prime', '')).strip() != args.arm:
            continue
        try:
            rows.append(dict(model=r.get('model', '?'),
                             cond=(r['threshold_mode'] if args.fix
                                   else int(r.get('group_size', 1) or 1)),
                             omega=int(r.get('group_size', 1) or 1),
                             q=int(r['q_idx']), y=int(v)))
        except (ValueError, KeyError):
            continue
if not rows:
    raise SystemExit(f'no usable rows (metric={args.metric}, k_prime={args.arm})')

label = 'threshold_mode' if args.fix else 'omega'
print(f'\n{len(rows)} rows | metric={args.metric} | arm k_prime={args.arm} | by {label}')
print(f'file: {args.path}')
if args.fix:
    print('NOTE: --fix compares threshold modes WITHIN each omega, since the '
          'fix was tested at omega=1 and omega=4 separately.\n')
else:
    print()

# Every comparison below is one of a FAMILY -- the full omega ladder across
# every model -- so an uncorrected 0.05 is the wrong bar. Holm-Bonferroni is
# used rather than plain Bonferroni: it is uniformly more powerful and makes no
# independence assumption, which matters because these tests share queries and
# are therefore correlated. Results are collected first, corrected, then printed.
_results = []

groups = defaultdict(dict)
for r in rows:
    groups[(r['model'], r['omega'] if args.fix else None)].setdefault(r['cond'], {})[r['q']] = r['y']

for (model, omega), by_cond in sorted(groups.items(), key=lambda kv: str(kv[0])):
    conds = sorted(by_cond, key=str)
    if len(conds) < 2:
        continue
    base = conds[0]
    for c in conds[1:]:
        qa, qb = by_cond[base], by_cond[c]
        common = sorted(set(qa) & set(qb))
        if not common:
            print(f'  {base} -> {c}: no shared queries'); continue
        pairs = [(qa[q], qb[q]) for q in common]
        m = mcnemar(pairs)
        x1 = sum(qa[q] for q in common); x2 = sum(qb[q] for q in common)
        n = len(common)
        zi, pi = two_prop_z(x1, n, x2, n)
        _results.append(dict(model=model, omega=omega, base=base, cond=c,
                             p1=x1 / n, p2=x2 / n, n=n, m=m, zi=zi, pi=pi))


# ---- Holm-Bonferroni over the whole family ----
_results.sort(key=lambda d: d['m']['p'])
M = len(_results)
_prev = 0.0
for i, d in enumerate(_results):
    adj = min(1.0, (M - i) * d['m']['p'])
    adj = max(adj, _prev)          # Holm adjusted p values are monotone
    _prev = adj
    d['p_adj'] = adj

_by = defaultdict(list)
for d in _results:
    _by[(d['model'], d['omega'])].append(d)

for (model, omega), ds in sorted(_by.items(), key=lambda kv: str(kv[0])):
    print(f'=== {model}' + (f'  omega={omega}' if omega is not None else '') + ' ===')
    for d in sorted(ds, key=lambda d: str(d['cond'])):
        m = d['m']
        star = ('***' if d['p_adj'] < .001 else '**' if d['p_adj'] < .01
                else '*' if d['p_adj'] < .05 else 'ns')
        print(f'  {label} {d["base"]} -> {d["cond"]}:  '
              f'{d["p1"]:>5.0%} -> {d["p2"]:>5.0%}  (n={d["n"]} paired)')
        print(f'     McNemar   b={m["b"]:<3} c={m["c"]:<3} z={m["z"]:+6.2f}  '
              f'p={m["p"]:.2e}  ->  Holm p={d["p_adj"]:.2e}  {star}   [{m["test"]}]')
        if d['zi'] is not None:
            print(f'     indep. z  {d["zi"]:+6.2f}  p={d["pi"]:.2e}   '
                  f'<- WRONG TEST for paired data; comparison only')
    print()

print(f'Holm-Bonferroni applied across all {M} comparisons in this file.')

print('Quote the McNemar row. b and c are the discordant counts it is built on:')
print('when b+c is small the comparison rests on very few queries, whatever the')
print('percentages look like.\n')
