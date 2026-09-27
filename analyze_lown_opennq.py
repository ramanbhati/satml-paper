#!/usr/bin/env python
"""
Is there ANY alpha that makes the low-n regime both safe and useful?

    python3 analyze_lown_opennq.py results/lown_opennq_pilot.csv

TWO CONFOUNDS ARE EXCLUDED BY DEFAULT. Both were found by inspecting
data/open_nq.json before the run, and neither is visible in the aggregate
numbers, so leaving them in would quietly bias exactly the bucket under study:

  1. SHORT-CONTEXT ITEMS. 10 of the 500 open_nq items have fewer than 10
     usable passages (6-8 after the title/text filter). Those queries land in
     the low-n bucket for a STRUCTURAL reason -- there were never 10 groups to
     answer -- not because the corpus failed to answer them. They would inflate
     the low-n population and make the defense look worse than it is.
     Detected via n_groups, which the defense records per query.

  2. ANSWER-OVERLAP ITEMS. 12 of 500 have an 'incorrect answer' that overlaps a
     correct answer ('Austin' vs ['Austin', 'Pflugerville']). The attacker's
     keyword then survives on HONEST votes, scoring a false success. This is the
     same artifact that accounted for the 1.6% of formula-contradicting rows in
     the realtimeqa sweep. Detected by joining back to the dataset on q_idx.

Pass --keep-confounds to see the unfiltered numbers for comparison.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("-")]
KEEP = "--keep-confounds" in sys.argv
path = args[0] if args else "results/lown_opennq_pilot.csv"
DATA = "data/open_nq.json"
LOW_N = 3          # the regime: mu = min(alpha*n, beta) < 1 at alpha=0.3
TOPK = 10


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# ---- identify answer-overlap items from the dataset itself ----
overlap = set()
if os.path.exists(DATA):
    d = json.load(open(DATA))
    for i, it in enumerate(d):
        ia = norm(it.get("incorrect answer", ""))
        cands = [norm(x) for x in it.get("correct answer", [])] + \
                [norm(x) for x in it.get("expanded answer", [])]
        if ia and any(c and (ia in c or c in ia) for c in cands):
            overlap.add(i + 1)          # q_idx is 1-based (eval_cnt)
else:
    print(f"!! {DATA} not found -- cannot exclude answer-overlap items")

rows, dropped = [], 0
n_short = n_overlap = 0
with open(path) as fh:
    for r in csv.DictReader(fh):
        try:
            r["alpha"] = float(r["alpha"])
            r["k_prime"] = int(r["k_prime"])
            r["n"] = int(r["n_retained"])
            r["n_groups"] = int(r["n_groups"])
            r["q_idx"] = int(r["q_idx"])
            r["model"] = r.get("model", "?")
            r["surv"] = int(r["attacker_kw_survived"])
            c = r.get("defended_correct", "")
            r["correct"] = None if str(c).strip() == "" else int(c)
            a = r.get("defended_asr", "")
            r["asr"] = None if str(a).strip() == "" else int(a)
        except (ValueError, KeyError):
            dropped += 1
            continue
        # expected group count: clean arm has TOPK, an attacked arm still has
        # TOPK because poisoned passages REPLACE clean ones at full context.
        if r["n_groups"] < TOPK:
            n_short += 1
            if not KEEP:
                continue
        if r["q_idx"] in overlap:
            n_overlap += 1
            if not KEEP:
                continue
        rows.append(r)

if dropped:
    print(f"note: {dropped} malformed rows dropped")
print(f"\nexcluded {n_short} short-context rows and {n_overlap} answer-overlap rows"
      + (" (KEPT: --keep-confounds)" if KEEP else ""))
if not rows:
    sys.exit(f"no usable rows in {path}")

models = sorted({r["model"] for r in rows})
alphas = sorted({r["alpha"] for r in rows})
print(f"{len(rows)} rows | models {models} | alphas {alphas}\n")


def mean(g, k):
    v = [r[k] for r in g if r.get(k) is not None]
    return (sum(v) / len(v)) if v else None


def pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


# ---- STAGE-1 GO/NO-GO: does open_nq even have a low-n population? ----
print("=== GO/NO-GO: is there enough low-n mass in open_nq? ===")
for m in models:
    base = [r for r in rows if r["model"] == m
            and r["k_prime"] == 1 and r["alpha"] == alphas[0]]
    if not base:
        continue
    lo = sum(1 for r in base if r["n"] <= LOW_N)
    rate = lo / len(base)
    proj = rate * 500
    print(f"  {m:<24} {lo}/{len(base)} queries at n<={LOW_N}  ({rate:.0%})"
          f"   -> projected {proj:.0f} at N=500")
    if proj < 40:
        print(f"      !! PROJECTED YIELD TOO LOW. open_nq cannot support the claim.")
        print(f"         Do not run the full 500. (realtimeqa already gives 41.)")
    else:
        print(f"      OK -- the full run would give a usable sample.")
print()

# ---- THE TEST ----
# NOT a binary "safe AND useful". Low-n queries are intrinsically hard (on
# realtimeqa, clean accuracy is 27% at n<=3 vs 64% at n>=4), so a binary test
# would always score them 'not useful' regardless of the defense, and would be
# measuring question difficulty rather than the defense.
#
# The real claim is a DIFFERENTIAL: raising alpha is the defender's only lever
# against the low-n hole, and it costs far more utility at low n than at high n
# while buying security only at low n. On realtimeqa/mistral, alpha 0.3->0.7
# costs 18 points of clean accuracy at low n (27%->9%) and 3 points at high n
# (64%->61%). If that asymmetry holds at scale, the defender cannot tune their
# way out: the setting that closes the hole is the setting that destroys the
# queries the hole applies to.
for m in models:
    mr = [r for r in rows if r["model"] == m]

    def cell(a, kp, low):
        return [r for r in mr if r["alpha"] == a and r["k_prime"] == kp
                and ((r["n"] <= LOW_N) if low else (r["n"] > LOW_N))]

    print(f"=== {m}: what raising alpha costs, by regime ===")
    print(f"  {'':>6} | {'--- low n (<=%d) ---' % LOW_N:^34} | {'--- high n ---':^22}")
    print(f"  {'alpha':>6} | {'N':>4} {'clean':>7} {'attacked':>9} {'kw surv':>8} "
          f"| {'N':>4} {'clean':>7} {'kw surv':>8}")
    base = {}
    for a in alphas:
        lc, la = cell(a, 0, True), cell(a, 1, True)
        hc, ha = cell(a, 0, False), cell(a, 1, False)
        if not la:
            continue
        vals = (mean(lc, "correct"), mean(la, "correct"), mean(la, "surv"),
                mean(hc, "correct"), mean(ha, "surv"))
        base[a] = vals
        print(f"  {a:>6} | {len(la):>4} {pct(vals[0]):>7} {pct(vals[1]):>9} "
              f"{pct(vals[2]):>8} | {len(ha):>4} {pct(vals[3]):>7} {pct(vals[4]):>8}")

    if len(base) >= 2:
        a0, aN = alphas[0], alphas[-1]
        lc0, lcN = cell(a0, 0, True), cell(aN, 0, True)
        hc0, hcN = cell(a0, 0, False), cell(aN, 0, False)
        la0, laN = cell(a0, 1, True), cell(aN, 1, True)

        def k_n(g, key="correct"):
            v = [r[key] for r in g if r.get(key) is not None]
            return sum(v), len(v)

        # Difference-in-differences with a normal-approximation z test.
        #
        # WHY NOT A RATIO. An earlier version reported lo_cost / hi_cost. That
        # is a ratio of two noisy differences: on synthetic data drawn from a
        # TRUE 18-point low-n drop it swung between 3x and 0.7x on sampling
        # noise alone, i.e. it flipped the verdict. At N~70 the low-n accuracy
        # estimate carries about +/-5 points of standard error, so any statistic
        # that divides by a small number is unusable here.
        def prop(g):
            k, n = k_n(g)
            return (k / n if n else None), n

        p_l0, n_l0 = prop(lc0)
        p_lN, n_lN = prop(lcN)
        p_h0, n_h0 = prop(hc0)
        p_hN, n_hN = prop(hcN)
        s_l0, _ = prop_s0 = (mean(la0, "surv"), len(la0))
        s_lN, _ = (mean(laN, "surv"), len(laN))

        if None not in (p_l0, p_lN, p_h0, p_hN) and min(n_l0, n_lN, n_h0, n_hN) > 0:
            lo_cost = p_l0 - p_lN
            hi_cost = p_h0 - p_hN
            did = lo_cost - hi_cost
            var = sum(p * (1 - p) / n for p, n in
                      ((p_l0, n_l0), (p_lN, n_lN), (p_h0, n_h0), (p_hN, n_hN)))
            se = var ** 0.5
            z = did / se if se > 0 else 0.0
            print(f"\n  --- alpha {a0} -> {aN} (clean arm) ---")
            print(f"    accuracy lost at LOW  n : {lo_cost:+.1%}  (N={n_l0}/{n_lN})")
            print(f"    accuracy lost at HIGH n : {hi_cost:+.1%}  (N={n_h0}/{n_hN})")
            print(f"    difference-in-differences: {did:+.1%}  (SE {se:.1%},  z = {z:.2f})")
            if s_l0 is not None and s_lN is not None:
                print(f"    attacker survival at low n: {s_l0:.0%} -> {s_lN:.0%}"
                      f"  (removed {s_l0 - s_lN:+.0%})")
            sec_gain = (s_l0 - s_lN) if (s_l0 is not None and s_lN is not None) else 0
            print()
            if p_l0 < 0.10:
                print("    !! VACUOUS: clean accuracy at low n is already near zero at the")
                print("       default alpha. There is no utility for the attack to destroy,")
                print("       so 'the defender cannot tune out the hole' is not a meaningful")
                print("       claim on this dataset. Report it as a null result.")
            elif z >= 2.0 and did > 0.08 and sec_gain > 0.2:
                print("    => CLAIM SUPPORTED (z >= 2): raising alpha costs significantly")
                print("       more utility in the low-n regime than elsewhere, while the")
                print("       security it buys applies only there. The defender cannot")
                print("       tune their way out.")
            elif z < 1.0:
                print("    => CLAIM NOT SUPPORTED: the extra cost at low n is not")
                print("       distinguishable from the cost everywhere else. The defender")
                print("       CAN just raise alpha. No paper.")
            else:
                print(f"    => UNDERPOWERED (z = {z:.2f}). The point estimate leans the right")
                print(f"       way but the sample cannot support it. Need roughly "
                      f"{int(n_l0 * (2.0 / max(z, 0.01)) ** 2)} low-n queries for z=2.")
    print()

print("=== how to read this ===")
print(f"  Small N is the main hazard: realtimeqa gave only 11 low-n queries per")
print(f"  alpha. Treat any row with N < 30 as directional only.")
print("  The vacuity check matters as much as the claim: if clean low-n accuracy")
print("  is already ~0, the defense is not losing utility, the corpus never had")
print("  the answer, and the honest finding is a null result.")
print()
