#!/usr/bin/env python
"""
Why does the low-n regime exist at all, and why only in some corpora?

    python3 analyze_abstention_correlation.py \
        results/lown.csv results/lown_opennq_pilot.csv

THE QUESTION. RobustRAG's keyword filter collapses when n -- the number of
isolated groups that gave a substantive answer -- is small. The obvious reading
is that low n is a random tail event: each passage independently fails to
answer, and occasionally enough of them fail at once. If that were true, low n
would be vanishingly rare and the finding would be a curiosity.

It is not true, and this quantifies by how much.

THE TEST. Take the measured per-passage abstention rate p for a dataset, assume
passages abstain INDEPENDENTLY, and compute P(n <= 3) = P(at least k-3 of k
groups abstain) from the binomial. Compare with the observed rate.

WHAT WE FOUND (mistral7b, k=10, alpha=0.3, k'=1):

    dataset      p        P(n<=3) independent    observed    excess
    realtimeqa   23.4%    0.23%                  8.0%        34x
    open_nq      18.6%    0.05%                  2.0%        37x

Both are ~35x above the independence prediction, so abstentions are strongly
CORRELATED: when the corpus does not cover a question, every retrieved passage
fails to answer it together. Low n is therefore not a random tail -- it is a
signature of corpus/query mismatch.

WHY THIS MATTERS FOR THE THREAT MODEL. The vulnerable regime is not distributed
uniformly over queries. It concentrates on questions the corpus does not cover:
recent events, long-tail entities, gaps in a proprietary index. An attacker does
not need to induce the condition (which the composite experiment showed is a
dominated strategy anyway) -- they need only SELECT queries that already have
it, which is cheap and requires no access to the system.

It also explains why open_nq cannot be used to study this: Natural Questions is
Wikipedia-backed and answerable by construction, so it has ~4x less absolute
low-n mass than news QA, and a 500-query run there yields fewer low-n queries
than the 100-query realtimeqa run already in hand.
"""
import csv
import sys
from collections import Counter
from math import comb

paths = sys.argv[1:] or ["results/lown.csv", "results/lown_opennq_pilot.csv"]
LOW_N = 3
TOPK = 10


def load(path):
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                r["n"] = int(r["n_retained"])
                r["ab"] = int(r["n_abstained"])
                r["groups"] = int(r["n_groups"])
                r["k_prime"] = int(r["k_prime"])
                r["alpha"] = float(r["alpha"])
                r["model"] = r.get("model", "?")
            except (ValueError, KeyError):
                continue
            out.append(r)
    return out


def p_low_n_independent(p, k=TOPK, low_n=LOW_N):
    """P(n <= low_n) = P(at least k-low_n of k groups abstain), independent."""
    need = k - low_n
    return sum(comb(k, i) * p ** i * (1 - p) ** (k - i) for i in range(need, k + 1))


print(f"\n{'dataset (file)':<34} {'model':<20} {'N':>5} {'abstain p':>10} "
      f"{'P(n<={0}) indep'.format(LOW_N):>15} {'observed':>9} {'excess':>8}")
print("-" * 106)

for path in paths:
    rows = load(path)
    if not rows:
        print(f"{path:<34} (no usable rows)")
        continue
    # Hold the configuration fixed: one budget, one alpha, full-length contexts.
    # alpha does not affect n (it only sets the filter threshold), but pinning it
    # avoids double-counting the same query once per alpha in a swept file.
    alpha0 = sorted({r["alpha"] for r in rows})[0]
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model and r["k_prime"] == 1
               and r["alpha"] == alpha0 and r["groups"] == TOPK]
        if not sub:
            continue
        p = sum(r["ab"] for r in sub) / (TOPK * len(sub))
        obs = sum(1 for r in sub if r["n"] <= LOW_N) / len(sub)
        pred = p_low_n_independent(p)
        excess = (obs / pred) if pred > 0 else float("inf")
        name = path.split("/")[-1]
        print(f"{name:<34} {model:<20} {len(sub):>5} {p:>9.1%} "
              f"{pred:>14.3%} {obs:>8.1%} {excess:>7.0f}x")

print()
print("An excess far above 1x means abstentions are correlated across the")
print("passages retrieved for a query -- i.e. the corpus either covers the")
print("question or it does not. Low n is a corpus-coverage signature, not a")
print("random tail, which is why it is selectable by an attacker and why it")
print("is dataset-dependent.")
print()

# n distribution, for the paper's figure
for path in paths:
    rows = load(path)
    if not rows:
        continue
    alpha0 = sorted({r["alpha"] for r in rows})[0]
    sub = [r for r in rows if r["k_prime"] == 1 and r["alpha"] == alpha0
           and r["groups"] == TOPK]
    if not sub:
        continue
    c = Counter(r["n"] for r in sub)
    print(f"n distribution -- {path.split('/')[-1]} (k'=1, alpha={alpha0}, N={len(sub)})")
    for n in range(0, TOPK + 1):
        cnt = c.get(n, 0)
        bar = "#" * int(round(40 * cnt / max(c.values()))) if cnt else ""
        flag = "  <- vulnerable" if n <= LOW_N else ""
        print(f"   n={n:<3d} {cnt:>4d} {bar}{flag}")
    print()
