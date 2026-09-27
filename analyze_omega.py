#!/usr/bin/env python
"""
Does RobustRAG's own utility knob (passage group size omega) remove its filter?

    python3 analyze_omega.py results/omega.csv

BACKGROUND. The paper recommends omega as the knob for benign performance
("with omega=3, we reduce the benign performance drop from 7% to 0%"), and
upstream's code implements omega=1 only. But omega caps the group count at
ceil(k/omega), and mu = min(alpha*n, beta) with n <= ceil(k/omega). At k=10,
alpha=0.3, beta=3:

    omega  groups  mu(max)  mu(one abstains)   k'=1 survives?
      1      10     3.00         2.70          needs 7 abstentions
      2       5     1.50         1.20          needs 2 abstentions
      3       4     1.20         0.90          ONE abstention
      4       3     0.90         0.60          ALWAYS  (mu <= 1)
      5       2     0.60         0.30          ALWAYS  (mu <= 1)

So this is a LADDER with a hard step at omega=4, where mu drops below 1 with no
abstentions at all. Unlike the corpus-coverage result, this prediction does not
depend on the dataset -- it is forced by the configuration.

WHAT TO LOOK FOR.
  - survival at omega>=4 should be ~1.00 on essentially every query;
  - survival at omega=3 should track the fraction of queries with >=1 abstaining
    group;
  - survival at omega=1 should be low (~12%, our earlier measurement);
  - clean accuracy should RISE with omega (that is what omega is for), so the
    two curves cross. The crossing point is the figure.

A failure to see the step at omega=4 would mean mu is not the operative
mechanism at all, and would contradict the alpha/beta sweep. Treat that as a
reason to stop and debug, not to write up.
"""
import csv
import math
import sys
from collections import defaultdict

# "results/omega.csv" was a placeholder that never existed; name a real file so
# the script runs with no arguments on the released data.
path = sys.argv[1] if len(sys.argv) > 1 else "results/omega_realtimeqa_k10_all.csv"
# TOPK is read from the data, not hardcoded: the k=12 control exists precisely
# because group composition depends on k mod omega, so an analyser that assumes
# k=10 would mis-predict every boundary in that run.
TOPK = None

rows, dropped = [], 0
with open(path) as fh:
    for r in csv.DictReader(fh):
        try:
            r["omega"] = int(r.get("group_size", 1) or 1)
            r["top_k"] = int(r.get("top_k") or 0) or None
            r["alpha"] = float(r["alpha"])
            r["beta"] = float(r["beta"])
            r["k_prime"] = int(r["k_prime"])
            r["n"] = int(r["n_retained"])
            r["groups"] = int(r["n_groups"])
            r["mu"] = float(r["mu"])
            r["surv"] = int(r["attacker_kw_survived"])
            r["model"] = r.get("model", "?")
            c = r.get("defended_correct", "")
            r["correct"] = None if str(c).strip() == "" else int(c)
            a = r.get("defended_asr", "")
            r["asr"] = None if str(a).strip() == "" else int(a)
        except (ValueError, KeyError):
            dropped += 1
            continue
        rows.append(r)

if dropped:
    print(f"note: {dropped} malformed rows dropped")
if not rows:
    sys.exit(f"no usable rows in {path}")


def mean(g, k):
    v = [r[k] for r in g if r.get(k) is not None]
    return (sum(v) / len(v)) if v else None


def pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


omegas = sorted({r["omega"] for r in rows})
models = sorted({r["model"] for r in rows})
# Every prediction below is a function of alpha, and it is read from the first
# row. A file holding two alphas would therefore be analysed entirely under one
# of them. Refuse, exactly as we refuse mixed k.
_alphas = {r["alpha"] for r in rows}
if len(_alphas) > 1:
    sys.exit(f"mixed alpha values {sorted(_alphas)} in one file -- split before analysing")
alpha = rows[0]["alpha"]
beta = rows[0]["beta"]

# k must be constant within a file: every prediction below is a function of
# ceil(k/omega), so mixing k=10 and k=12 rows would silently compare different
# group structures under one label.
ks = {r["top_k"] for r in rows if r["top_k"]}
if len(ks) > 1:
    sys.exit(f"mixed top_k values {sorted(ks)} in one file -- split before analysing")
if ks:
    TOPK = ks.pop()
else:
    # older CSVs predate the top_k column; recover it from max_groups at omega=1
    _m1 = [r for r in rows if r["omega"] == 1 and r.get("max_groups")]
    TOPK = int(_m1[0]["max_groups"]) if _m1 else 10
    print(f"note: no top_k column; inferred k={TOPK} from max_groups at omega=1")

print(f"\n{len(rows)} rows | models {models} | omega {omegas} | "
      f"k={TOPK} alpha={alpha} beta={beta}\n")

# Group composition matters as much as group count: when omega does not divide k
# the last group is short, and at k=10/omega=3 that leaves the poisoned passage
# ALONE (10 = 3+3+3+1) with no in-group contest. Flag it so the ladder is read
# correctly rather than as a pure threshold effect.
print("=== group structure (poison at the last slot, backward placement) ===")
for w in omegas:
    grp = [list(range(i, min(i + w, TOPK))) for i in range(0, TOPK, w)]
    pg = [g for g in grp if (TOPK - 1) in g][0]
    even = len({len(g) for g in grp}) == 1
    note = "" if even else "  <-- RAGGED"
    if len(pg) == 1 and w > 1:
        note = "  <-- poison ALONE: no in-group contest, pure threshold effect"
    print(f"  omega={w:<2} {len(grp):>2} groups, sizes {sorted({len(g) for g in grp})}, "
          f"poison shares with {len(pg)-1}{note}")
print()

# ---- 1. sanity: did grouping actually happen? ----
print("=== sanity: group count matches ceil(k/omega) ===")
bad = 0
for w in omegas:
    sub = [r for r in rows if r["omega"] == w]
    expected = math.ceil(TOPK / w)
    seen = sorted({r["groups"] for r in sub})
    ok = all(g <= expected for g in seen)
    bad += (not ok)
    print(f"  omega={w:<2} expected <= {expected:<3} observed {seen}  "
          f"{'ok' if ok else '<-- MISMATCH: grouping did not take effect'}")
if bad:
    sys.exit("\nGrouping is not working. Stop and debug before interpreting anything.")

# ---- 2. the ladder ----
for m in models:
    mr = [r for r in rows if r["model"] == m]
    print(f"\n=== {m}: what omega buys and what it costs ===")
    print(f"  {'omega':>6} {'groups':>7} {'mu':>6} | {'clean acc':>10} "
          f"{'atk acc':>8} | {'kw surv':>8} {'ASR':>6} | predicted")
    for w in omegas:
        clean = [r for r in mr if r["omega"] == w and r["k_prime"] == 0]
        atk = [r for r in mr if r["omega"] == w and r["k_prime"] == 1]
        if not atk and not clean:
            continue
        g_max = math.ceil(TOPK / w)
        mu_max = min(alpha * g_max, beta)
        # prediction for k'=1
        if mu_max <= 1:
            pred = "ALWAYS (mu<=1)"
        elif min(alpha * (g_max - 1), beta) <= 1:
            pred = "if >=1 abstains"
        else:
            pred = f"needs n<={math.floor(1 / alpha)}"
        print(f"  {w:>6} {g_max:>7} {mu_max:>6.2f} | {pct(mean(clean,'correct')):>10} "
              f"{pct(mean(atk,'correct')):>8} | {pct(mean(atk,'surv')):>8} "
              f"{pct(mean(atk,'asr')):>6} | {pred}")

    # ---- 3. the crossing ----
    print(f"\n  --- the tradeoff omega actually makes ---")
    base_clean = mean([r for r in mr if r["omega"] == 1 and r["k_prime"] == 0], "correct")
    base_surv = mean([r for r in mr if r["omega"] == 1 and r["k_prime"] == 1], "surv")
    for w in omegas:
        if w == 1:
            continue
        c = mean([r for r in mr if r["omega"] == w and r["k_prime"] == 0], "correct")
        s = mean([r for r in mr if r["omega"] == w and r["k_prime"] == 1], "surv")
        if None in (c, s, base_clean, base_surv):
            continue
        print(f"    omega 1 -> {w}:  clean accuracy {c - base_clean:+.1%}   "
              f"attacker survival {s - base_surv:+.1%}")

    # ---- 4. verdict ----
    hi = [w for w in omegas if min(alpha * math.ceil(TOPK / w), beta) <= 1]
    if hi:
        s_hi = mean([r for r in mr if r["omega"] in hi and r["k_prime"] == 1], "surv")
        s_lo = mean([r for r in mr if r["omega"] == 1 and r["k_prime"] == 1], "surv")
        print(f"\n  --- verdict ---")
        print(f"    omega with mu<=1 unconditionally: {hi}")
        print(f"    attacker survival there: {pct(s_hi)}   at omega=1: {pct(s_lo)}")
        if s_hi is not None and s_lo is not None:
            if s_hi > 0.9 and s_lo < 0.3:
                print("    => CONFIRMED: once mu <= 1, a single poisoned passage survives on")
                print("       essentially every query. The configuration recommended for")
                print("       benign performance removes the keyword filter entirely.")
            elif s_hi - s_lo > 0.3:
                print("    => Directionally confirmed but not absolute; inspect per-query rows.")
            else:
                print("    => NOT CONFIRMED. mu<=1 did not produce survival. Something other")
                print("       than the threshold is filtering the keyword -- debug before")
                print("       writing anything.")
print()
