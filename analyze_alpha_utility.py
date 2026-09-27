#!/usr/bin/env python
"""
Is there an alpha that is both secure and useful?

    python3 analyze_alpha_utility.py results/alpha_utility.csv

The alpha/beta sweep established that raising alpha shrinks the attacker's
window (boundary moves n<=5 -> n<=3 -> n<=2 as alpha goes 0.2 -> 0.3 -> 0.5).
That is only half an argument. mu = min(alpha*n, beta) filters HONEST keywords
by the same rule, so a higher threshold should also cost accuracy.

This prints both curves against alpha, and the security-per-utility exchange
rate between adjacent settings. Three possible readings:

  (a) utility falls as fast as (or faster than) security rises
      -> the defender has NO good setting; alpha trades one failure for
         another. This is a claim about the DEFENSE, not about an attack, and
         it is the strongest thing this experiment can show.

  (b) there is a knee -- security rises sharply while utility is still flat
      -> upstream's alpha=0.3 may simply be mistuned, and the paper's
         contribution is a better default. Weaker, but actionable and honest.

  (c) utility is flat everywhere
      -> raising alpha is free, the defense should just do it, and the low-n
         finding is a tuning bug rather than a structural one. Report it that
         way; do not dress it up.

Which of these holds is an empirical question. Do not decide it in advance.
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/alpha_utility.csv"

rows = []
with open(path) as fh:
    for r in csv.DictReader(fh):
        try:
            r["alpha"] = float(r["alpha"])
            r["k_prime"] = int(r["k_prime"])
            r["n"] = int(r["n_retained"])
            r["attack"] = r.get("attack", "") or ("none" if r["k_prime"] == 0 else "?")
            r["model"] = r.get("model", "?")
            c = r.get("defended_correct", "")
            r["correct"] = None if str(c).strip() == "" else int(c)
            a = r.get("defended_asr", "")
            r["asr"] = None if str(a).strip() == "" else int(a)
            r["surv"] = int(r["attacker_kw_survived"])
            r["nkw"] = int(r.get("n_surviving_keywords", 0) or 0)
        except (ValueError, KeyError):
            continue
        rows.append(r)

if not rows:
    sys.exit(f"no usable rows in {path}")


def mean(g, k):
    v = [r[k] for r in g if r.get(k) is not None]
    return (sum(v) / len(v)) if v else None


def pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


def num(x, fmt=".1f"):
    """Format a possibly-missing number. f'{None:>9}' raises TypeError, so a
    single failed run (throttling, retry exhaustion) would otherwise crash the
    analyser AFTER the money was already spent."""
    return format(x, fmt) if x is not None else "n/a"


models = sorted({r["model"] for r in rows})
alphas = sorted({r["alpha"] for r in rows})

# beta must be constant: this experiment varies alpha only. A mixed-beta file
# would silently average two different defenses together.
betas = sorted({float(r["beta"]) for r in rows if r.get("beta")})
if len(betas) > 1:
    sys.exit(f"mixed beta values {betas} in one file -- this experiment varies "
             f"alpha only; split the file before analysing")

# Every (arm, alpha) cell should have the same number of queries. A short cell
# means a run failed and the comparison across alpha is not like-for-like.
cells = defaultdict(int)
for r in rows:
    cells[(r["model"], r["alpha"], r["k_prime"])] += 1
sizes = sorted(set(cells.values()))
if len(sizes) > 1:
    print(f"!! WARNING: uneven cell sizes {sizes} -- at least one run did not "
          f"complete. Rows are NOT comparable across alpha until it is re-run.")
    for k, v in sorted(cells.items()):
        if v != max(sizes):
            print(f"     short: model={k[0]} alpha={k[1]} k'={k[2]}  {v}/{max(sizes)} queries")

print(f"\n{len(rows)} rows | models {models} | alphas {alphas} | beta {betas}\n")

for m in models:
    mr = [r for r in rows if r["model"] == m]
    print(f"=== {m} ===")
    print(f"  {'alpha':>6} {'clean acc':>10} {'kw/query':>9} "
          f"{'surv k=1':>9} {'surv k=2':>9} {'ASR k=1':>8} {'ASR k=2':>8} {'acc k=1':>8}")
    prev = None
    table = []
    for a in alphas:
        clean = [r for r in mr if r["alpha"] == a and r["k_prime"] == 0]
        k1 = [r for r in mr if r["alpha"] == a and r["k_prime"] == 1]
        k2 = [r for r in mr if r["alpha"] == a and r["k_prime"] == 2]
        if not clean and not k1:
            continue
        cacc = mean(clean, "correct")
        nkw = mean(clean, "nkw")
        s1 = mean(k1, "surv")
        s2 = mean(k2, "surv")
        a1 = mean(k1, "asr")
        a2 = mean(k2, "asr")
        acc1 = mean(k1, "correct")
        table.append((a, cacc, s1, s2))
        missing = "  <- INCOMPLETE" if (cacc is None or s1 is None) else ""
        print(f"  {a:>6} {pct(cacc):>10} {num(nkw):>9} "
              f"{pct(s1):>9} {pct(s2):>9} {pct(a1):>8} {pct(a2):>8} {pct(acc1):>8}{missing}")

    # exchange rate between adjacent alpha settings
    print(f"\n  --- what each step in alpha buys and costs ---")
    print(f"  {'step':>14} {'utility lost':>13} {'security gained':>16} {'ratio':>8}")
    for (a0, c0, s0, _), (a1_, c1, s1_, _) in zip(table, table[1:]):
        if None in (c0, c1, s0, s1_):
            continue
        lost = c0 - c1          # clean accuracy given up
        gained = s0 - s1_       # attacker survival removed
        ratio = (gained / lost) if abs(lost) > 1e-9 else float('inf')
        verdict = ""
        if lost <= 0.005 and gained > 0.05:
            verdict = "  <- free security"
        elif gained <= 0.005 and lost > 0.05:
            verdict = "  <- pure loss"
        elif ratio < 1:
            verdict = "  <- costs more than it buys"
        print(f"  {a0:>5} -> {a1_:<5} {lost:>+12.1%} {gained:>+15.1%} "
              f"{ratio:>8.2f}{verdict}")
    print()

print("=== reading the result ===")
print("  ratio > 1 and utility flat  -> upstream's alpha=0.3 is mistuned; report a better default.")
print("  ratio ~ 1 or utility fallingasfast -> no good setting exists; alpha trades")
print("                                         integrity failure for utility failure.")
print("  Note k'>=beta (=3) is excluded by design: it wins at every alpha, so it")
print("  says nothing about this tradeoff.")
print()
