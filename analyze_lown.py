#!/usr/bin/env python
"""
Analyse the low-n study.

Tests the claim that RobustRAG's filter mu = min(alpha*n, beta) collapses when
few groups answer, letting a single poisoned passage through.

    python3 analyze_lown.py results/lown.csv
"""
import csv
import sys
from collections import defaultdict

ALPHAS, BETAS = set(), set()
path = sys.argv[1] if len(sys.argv) > 1 else "results/lown.csv"

def _opt_int(v):
    """Blank means 'not measured' (e.g. a --filter_only run has no final answer
    to score), which is NOT the same as zero. Returning None keeps those rows in
    the analysis for the keyword-survival question while excluding them from the
    ASR/accuracy means. An earlier version int()'d these, raised ValueError and
    `continue`d -- silently DROPPING every filter-only row."""
    if v is None or str(v).strip() == "":
        return None
    return int(v)


rows = []
dropped = 0
with open(path) as fh:
    for r in csv.DictReader(fh):
        try:
            r["n_retained"] = int(r["n_retained"])
            r["k_prime"] = int(r["k_prime"])
            r["mu"] = float(r["mu"])
            r["attacker_kw_survived"] = int(r["attacker_kw_survived"])
            r["defended_asr"] = _opt_int(r.get("defended_asr"))
            r["defended_correct"] = _opt_int(r.get("defended_correct"))
            r["early_abstain"] = int(r.get("early_abstain",0) or 0)
            r["attacker_kw_partial"] = int(r.get("attacker_kw_partial",0) or 0)
            r["filter_only"] = int(r.get("filter_only",0) or 0)
            r["k_suppress"] = int(r.get("k_suppress",0) or 0)
            r["k_inject"] = int(r.get("k_inject", r["k_prime"]) or 0)
            r["attack"] = r.get("attack","") or ""
            if r.get("alpha"): ALPHAS.add(float(r["alpha"]))
            if r.get("beta"):  BETAS.add(float(r["beta"]))
        except (ValueError, KeyError):
            dropped += 1
            continue
        rows.append(r)

if dropped:
    print(f"note: {dropped} malformed rows dropped from {path}")

if not rows:
    sys.exit(f"no usable rows in {path}")


def _mean(grp, key):
    """Mean over rows where the metric was actually measured; None if none were."""
    vals = [r[key] for r in grp if r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(x, width=8):
    return f"{x:>{width}.2f}" if x is not None else f"{'n/a':>{width}}"

# alpha/beta come from the data, not hardcoded
if len(ALPHAS) > 1 or len(BETAS) > 1:
    sys.exit(f"mixed alpha/beta in one file ({ALPHAS}, {BETAS}) -- split before analysing")
ALPHA = ALPHAS.pop() if ALPHAS else 0.3
BETA = BETAS.pop() if BETAS else 3.0
print(f"alpha={ALPHA}, beta={BETA} (read from the data)")

_ea = sum(r["early_abstain"] for r in rows)
if _ea:
    print(f"note: {_ea} queries tripped the abstention threshold and returned early "
          f"(no keyword filtering ran); they are kept but have empty keyword sets.")

print(f"\n{len(rows)} query-rows from {len({r['run'] for r in rows})} runs\n")

# ---- 1. does mu match the formula? (sanity) ----
bad = [r for r in rows if abs(r["mu"] - min(ALPHA * r["n_retained"], BETA)) > 0.01]
print(f"[{'ok' if not bad else 'MISMATCH'}] mu == min(alpha*n, beta) on all rows"
      + (f"  ({len(bad)} mismatches)" if bad else ""))

# ---- 2. the headline table: survival vs n, for k'=1 ----
for kp in sorted({r["k_prime"] for r in rows}):
    sub = [r for r in rows if r["k_prime"] == kp]
    if not sub:
        continue
    print(f"\n=== k' = {kp}  ({len(sub)} queries) ===")
    print(f"  {'n':>3} {'mu':>6} {'queries':>8} {'atk kw survived':>17} {'ASR':>8} {'accuracy':>9}   predicted")
    buckets = defaultdict(list)
    for r in sub:
        buckets[r["n_retained"]].append(r)
    for n in sorted(buckets):
        b = buckets[n]
        mu = min(ALPHA * n, BETA)
        surv = sum(x["attacker_kw_survived"] for x in b) / len(b)
        part = sum(x["attacker_kw_partial"] for x in b) / len(b)
        asr = _mean(b, "defended_asr")
        acc = _mean(b, "defended_correct")
        pred = "SURVIVES" if kp >= mu else "filtered"
        print(f"  {n:>3} {mu:>6.2f} {len(b):>8} {surv:>17.2f} {_fmt(asr)} {_fmt(acc,9)}   {pred}"
              + (f"   (partial {part:.2f})" if abs(part-surv)>0.01 else ""))

# ---- 3. the crisp test: low-n vs high-n, k'=1 ----
k1 = [r for r in rows if r["k_prime"] == 1]
if k1:
    lo = [r for r in k1 if r["n_retained"] <= 3]
    hi = [r for r in k1 if r["n_retained"] >= 4]
    print("\n=== THE TEST (k'=1) ===")
    for label, grp in (("n <= 3  (mu < 1, attacker predicted to survive)", lo),
                       ("n >= 4  (mu > 1, attacker predicted filtered)", hi)):
        if not grp:
            print(f"  {label}: no queries")
            continue
        surv = sum(x["attacker_kw_survived"] for x in grp) / len(grp)
        asr = _mean(grp, "defended_asr")
        acc = _mean(grp, "defended_correct")
        pct = lambda v: f"{v:.0%}" if v is not None else "n/a"
        print(f"  {label}")
        print(f"      n={len(grp):<4} attacker keyword survived {surv:.0%} | ASR {pct(asr)} | accuracy {pct(acc)}")
    if lo and hi:
        s_lo = sum(x["attacker_kw_survived"] for x in lo) / len(lo)
        s_hi = sum(x["attacker_kw_survived"] for x in hi) / len(hi)
        gap = s_lo - s_hi
        print(f"\n  gap = {gap:+.0%}")
        if gap > 0.3:
            print("  => CLAIM SUPPORTED: one poisoned passage gets through when few groups answer.")
        elif gap > 0.1:
            print("  => weak support; needs more low-n queries. Consider open_nq (500 queries).")
        else:
            print("  => CLAIM NOT SUPPORTED. mu is not the operative mechanism here; do not write it up.")

# ---- 4. how common is low n? (is the attack practical?) ----
print("\n=== how often does n fall low on its own? ===")
allq = [r for r in rows if r["k_prime"] == 1]
if allq:
    lo = sum(1 for r in allq if r["n_retained"] <= 3)
    print(f"  {lo}/{len(allq)} queries ({lo/len(allq):.0%}) had n <= 3 with only ONE poisoned passage.")
    print("  These are queries the corpus answers poorly -- an attacker can select for them.")
