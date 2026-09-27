#!/usr/bin/env python
"""
Settle the accuracy confound: compare attacked vs clean accuracy, bucketed by n.

    python3 analyze_clean.py results/clean.csv results/lown.csv

THE CONFOUND. Under attack, accuracy is ~20% at n<=3 and ~57% at n>=4. Low-n
queries are the ones whose passages lacked the answer, so they would score badly
with no attacker at all. The attacked numbers alone cannot separate "the attack
destroyed utility" from "these questions were hard".

WHAT THIS PRINTS. Clean accuracy in the same n buckets, and the DELTA. Three
readings, and the paper should state whichever the data supports:

  (a) clean low-n accuracy is also low, and the delta is small
      -> the attack does not destroy much utility that existed; its damage is
         that it replaces an honest failure with a confident wrong answer.
         Headline becomes the ASR swing, not the accuracy drop. This is still a
         strong claim -- arguably stronger, because a system that says nothing
         is safer than one that lies.

  (b) clean low-n accuracy is high and the delta is large
      -> the attack destroys real utility at low n. Headline can include the
         accuracy drop directly.

  (c) clean accuracy is high everywhere and n is rarely low without an attacker
      -> low n is largely attacker-induced, which argues for the composite
         suppress-then-inject framing.

NOTE ON n. n is measured per run. A clean run and an attacked run can assign the
SAME query different n, because the attack changes the passages. Bucketing is
therefore by each run's own n, and the query-level join below reports how often
n moved. Do not assume the buckets contain the same queries.
"""
import csv
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                r["n"] = int(r["n_retained"])
                r["k_prime"] = int(r["k_prime"])
                r["q_idx"] = int(r["q_idx"])
                r["model"] = r.get("model", "?")
                c = r.get("defended_correct", "")
                r["correct"] = None if str(c).strip() == "" else int(c)
                a = r.get("defended_asr", "")
                r["asr"] = None if str(a).strip() == "" else int(a)
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def mean(grp, key):
    v = [r[key] for r in grp if r.get(key) is not None]
    return (sum(v) / len(v)) if v else None


def pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


clean_path = sys.argv[1] if len(sys.argv) > 1 else "results/clean.csv"
atk_path = sys.argv[2] if len(sys.argv) > 2 else "results/lown.csv"

clean = load(clean_path)
atk = [r for r in load(atk_path) if r["k_prime"] == 1]   # k'=1 is the headline

if not clean:
    sys.exit(f"no usable rows in {clean_path}")
if not atk:
    sys.exit(f"no k'=1 rows in {atk_path}")

print(f"\nclean: {len(clean)} rows from {clean_path}")
print(f"attacked (k'=1): {len(atk)} rows from {atk_path}\n")

LO = lambda r: r["n"] <= 3
HI = lambda r: r["n"] >= 4

print("=== accuracy by n bucket ===")
print(f"  {'bucket':<12} {'clean':>12} {'attacked':>12} {'delta':>10}")
for label, pred in (("n <= 3", LO), ("n >= 4", HI)):
    c = [r for r in clean if pred(r)]
    a = [r for r in atk if pred(r)]
    ca, aa = mean(c, "correct"), mean(a, "correct")
    d = f"{aa - ca:+.0%}" if (ca is not None and aa is not None) else "n/a"
    print(f"  {label:<12} {pct(ca):>12} {pct(aa):>12} {d:>10}"
          f"   (clean N={len(c)}, atk N={len(a)})")

print("\n=== how often is n low WITHOUT an attacker? ===")
lo_c = sum(1 for r in clean if LO(r))
lo_a = sum(1 for r in atk if LO(r))
print(f"  clean:    {lo_c}/{len(clean)} ({lo_c/len(clean):.0%}) have n <= 3")
print(f"  attacked: {lo_a}/{len(atk)} ({lo_a/len(atk):.0%}) have n <= 3")
if lo_a > lo_c:
    print(f"  => even ONE injected passage pushes {lo_a - lo_c} more queries into the")
    print(f"     vulnerable regime. Suppression should push more.")

print("\n=== per-query: did n move when the attack was applied? ===")
cmap = {(r["model"], r["q_idx"]): r for r in clean}
moved = same = missing = 0
drops = defaultdict(int)
for r in atk:
    c = cmap.get((r["model"], r["q_idx"]))
    if c is None:
        missing += 1
        continue
    if c["n"] == r["n"]:
        same += 1
    else:
        moved += 1
        drops[r["n"] - c["n"]] += 1
print(f"  matched {same + moved} queries ({missing} unmatched)")
print(f"  n unchanged: {same}   n changed: {moved}")
if drops:
    print("  change in n (attacked - clean):")
    for d in sorted(drops):
        print(f"    {d:+d}: {drops[d]} queries")

print("\n=== per-model accuracy (sanity: should match the earlier clean numbers) ===")
bym = defaultdict(list)
for r in clean:
    bym[r["model"]].append(r)
for m in sorted(bym):
    print(f"  {m:<24} clean accuracy {pct(mean(bym[m], 'correct'))}  (N={len(bym[m])})")
print()
