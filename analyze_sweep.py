#!/usr/bin/env python
"""
Analyse the alpha/beta sweep.

THE QUESTION. The low-n study showed the attacker's keyword survives when
n <= 3 at (alpha=0.3, beta=3). On its own that is one data point, and a
reviewer can read it as a RealtimeQA quirk. This asks the sharper question:

    does the collapse boundary MOVE when alpha and beta move,
    and does it move to where mu = min(alpha*n, beta) says it should?

A keyword with count c survives iff c >= mu = min(beta, alpha*n). For an
attacker holding k_inject passages all asserting the same answer, c = k_inject,
so the prediction is purely arithmetic:

    survives  <=>  k_inject >= min(beta, alpha * n)

Equivalently: the attacker wins iff  k_inject >= beta  (wins at EVERY n), or
n <= k_inject / alpha. Note the boundary is inclusive -- at alpha=0.5, n=2 gives
mu = 1.0 and k_inject = 1 >= 1.0, so n=2 is still a WIN. Predicted boundaries at
k_inject = 1, over the observable range n = 1..10:

    alpha=0.5, beta=3  ->  survives for n <= 2
    alpha=0.3, beta=3  ->  survives for n <= 3    (3.33 floored)
    alpha=0.2, beta=3  ->  survives for n <= 5
    alpha=0.1, beta=3  ->  survives at every n in range (boundary is n=10)
    beta=2, k_inject=2 ->  survives at EVERY n (k_inject >= beta)

If the measured boundary tracks that line across configurations, the mechanism
is the formula and not the dataset. If it does not, the low-n result is an
artifact and should not be written up as a mechanism claim.

    python3 analyze_sweep.py results/sweep.csv
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/sweep.csv"

rows = []
dropped = 0
with open(path) as fh:
    for r in csv.DictReader(fh):
        try:
            r["n"] = int(r["n_retained"])
            r["mu"] = float(r["mu"])
            r["alpha"] = float(r["alpha"])
            r["beta"] = float(r["beta"])
            r["k_prime"] = int(r["k_prime"])
            r["k_inject"] = int(r.get("k_inject") or r["k_prime"])
            r["surv"] = int(r["attacker_kw_survived"])
            r["early"] = int(r.get("early_abstain", 0) or 0)
            r["model"] = r.get("model", "?")
            r["attack"] = r.get("attack", "?")
        except (ValueError, KeyError):
            dropped += 1
            continue
        rows.append(r)

if dropped:
    print(f"note: {dropped} malformed rows dropped")
if not rows:
    sys.exit(f"no usable rows in {path}")

configs = sorted({(r["alpha"], r["beta"], r["k_inject"]) for r in rows})
print(f"\n{len(rows)} rows across {len(configs)} (alpha, beta, k_inject) configurations\n")

# ---- 1. mu must equal the formula on every row, in every configuration ----
bad = [r for r in rows
       if abs(r["mu"] - min(r["alpha"] * r["n"], r["beta"])) > 0.01]
if bad:
    print(f"[MISMATCH] mu != min(alpha*n, beta) on {len(bad)} rows -- "
          f"the run did not use the alpha/beta it recorded. STOP.")
    for r in bad[:3]:
        print(f"    alpha={r['alpha']} beta={r['beta']} n={r['n']} "
              f"mu={r['mu']} expected={min(r['alpha']*r['n'], r['beta']):.2f}")
    sys.exit(1)
print("[ok] mu == min(alpha*n, beta) on every row")

# ---- 2. the prediction table ----
print("\n=== does the boundary move where the formula says? ===")
print("  'predicted boundary' = largest n with k_inject >= min(beta, alpha*n);")
print("  'measured boundary'  = largest n where survival is still >= 50%.")
print()
hdr = f"  {'alpha':>6} {'beta':>5} {'k_inj':>6} {'rows':>6} {'predicted':>10} {'measured':>9}   verdict"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

summary = []
for (a, b, ki) in configs:
    sub = [r for r in rows if (r["alpha"], r["beta"], r["k_inject"]) == (a, b, ki)]
    by_n = defaultdict(list)
    for r in sub:
        by_n[r["n"]].append(r)

    # predicted: the largest n for which the attacker still wins
    ns = sorted(by_n)
    pred_ns = [n for n in ns if ki >= min(b, a * n)]
    predicted = max(pred_ns) if pred_ns else None
    if pred_ns and len(pred_ns) == len(ns):
        predicted_s = "all n"
    elif predicted is None:
        predicted_s = "none"
    else:
        predicted_s = f"n <= {predicted}"

    # measured: largest n whose survival rate is still >= 0.5
    meas_ns = [n for n in ns if sum(x["surv"] for x in by_n[n]) / len(by_n[n]) >= 0.5]
    measured = max(meas_ns) if meas_ns else None
    if meas_ns and len(meas_ns) == len(ns):
        measured_s = "all n"
    elif measured is None:
        measured_s = "none"
    else:
        measured_s = f"n <= {measured}"

    ok = (predicted_s == measured_s)
    summary.append((a, b, ki, ok, predicted_s, measured_s))
    print(f"  {a:>6} {b:>5} {ki:>6} {len(sub):>6} {predicted_s:>10} {measured_s:>9}   "
          f"{'MATCH' if ok else '<-- MISMATCH'}")

# ---- 3. per-configuration detail ----
for (a, b, ki) in configs:
    sub = [r for r in rows if (r["alpha"], r["beta"], r["k_inject"]) == (a, b, ki)]
    by_n = defaultdict(list)
    for r in sub:
        by_n[r["n"]].append(r)
    print(f"\n=== alpha={a}, beta={b}, k_inject={ki} ===")
    print(f"  {'n':>3} {'mu':>6} {'queries':>8} {'survived':>9}   predicted")
    for n in sorted(by_n):
        grp = by_n[n]
        mu = min(b, a * n)
        surv = sum(x["surv"] for x in grp) / len(grp)
        pred = "SURVIVES" if ki >= mu else "filtered"
        flag = ""
        if (surv >= 0.5) != (ki >= mu):
            flag = "   <-- disagrees with prediction"
        print(f"  {n:>3} {mu:>6.2f} {len(grp):>8} {surv:>9.2f}   {pred}{flag}")

# ---- 4. verdict ----
matches = sum(1 for s in summary if s[3])
print(f"\n=== VERDICT: {matches}/{len(summary)} configurations match the formula ===")
if matches == len(summary):
    print("  The collapse boundary tracks mu = min(alpha*n, beta) across every")
    print("  configuration tested. This is a mechanism result, not a dataset quirk:")
    print("  no choice of abstention detector changes it, because the threshold")
    print("  itself is the vulnerability.")
elif matches >= len(summary) * 0.7:
    print("  Mostly tracks the formula. Inspect the mismatching rows above before")
    print("  writing this up -- a partial match may mean keyword extraction, not")
    print("  the threshold, is deciding some configurations.")
else:
    print("  Does NOT track the formula. The low-n result is not explained by mu.")
    print("  Do not write the mechanism claim; find what is actually driving it.")

_early = sum(r["early"] for r in rows)
if _early:
    print(f"\nnote: {_early} rows tripped the abstention threshold and returned before"
          f" filtering; their keyword sets are empty by construction.")
