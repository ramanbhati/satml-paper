#!/usr/bin/env bash
# FUTURE-STATE tests. Offline, no AWS, no cost.
#
#     ./test_scenarios.sh
#
# WHY THIS EXISTS, AND WHY IT IS SEPARATE FROM test_run_final.sh.
#
# test_run_final.sh checks the control flow against the state currently on disk.
# That is not enough on its own: three bugs in run_final.sh and
# make_paper_assets.py were found while that suite was passing 22 of 22, because
# each one lived in a state the repository had not reached yet.
#
#   1. resume trusted certify_<mode>_<model>.csv alone, but main.py writes that
#      file BEFORE the per-query CSV. Only a half-written pair reveals it.
#   2. make_paper_assets silently skipped queries whose n_retained was missing,
#      understating the degenerate-branch count. Requires state (1) to exist.
#   3. Section VIII asserts "The floor costs nothing" and "A zero on the losing
#      side" in PROSE, which no generated macro can correct. Only a six-model run
#      with b > 0 contradicts it, which three certified models cannot produce.
#
# The generalisation is that a suite testing the present state cannot catch a bug
# that needs a future one. This file simulates the states a pending run will
# produce, so add a scenario here whenever a run is about to create a state the
# repository has not been in before.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
[ -x .venv/bin/python ] && PY=.venv/bin/python

PASS=0; FAIL=0
ok  () { echo "  [ ok ] $*"; PASS=$((PASS+1)); }
bad () { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
SB=$(mktemp -d); trap 'rm -rf "$SB"' EXIT

echo "=================================================================="
echo "S1. six models certified, with b > 0 on the new ones"
echo "=================================================================="
# The paper's prose says the floor costs nothing. Three models say so. Six might
# not. If the data ever contradicts the sentence, generation must refuse rather
# than print a table the prose denies.
cp -r results "$SB/r1"
$PY - "$SB/r1" <<'MK'
import csv, os, sys, random
R = sys.argv[1]; random.seed(11)
for mode in ('n', 'floor2'):
    for tgt in ('nova-micro-bedrock', 'nova-lite-bedrock', 'nova-pro-bedrock'):
        for t in ('certify_{m}_{s}.csv', 'certify_realtimeqa_{m}_{s}.csv'):
            src = os.path.join(R, t.format(m=mode, s='mistral7b-bedrock'))
            rows = list(csv.DictReader(open(src)))
            for r in rows:
                r['model'] = tgt
                # flip some floor2 certificates OFF -> the floor now costs
                if mode == 'floor2' and 'certified' in r and random.random() < 0.25:
                    r['certified'] = '0'
            with open(os.path.join(R, t.format(m=mode, s=tgt)), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
MK
OUT=$($PY make_paper_assets.py --results "$SB/r1" --out "$SB/g1" 2>&1)
if echo "$OUT" | grep -q "Section VIII"; then
  ok "b>0 refuses and names the prose that must change"
else
  bad "b>0 generated silently; Section VIII would contradict its own table"
  echo "$OUT" | tail -3 | sed 's/^/        /'
fi

echo
echo "=================================================================="
echo "S2. six models certified, b = 0 (the good case) still generates"
echo "=================================================================="
cp -r results "$SB/r2"
$PY - "$SB/r2" <<'MK'
import csv, os, sys
R = sys.argv[1]
for mode in ('n', 'floor2'):
    for tgt in ('nova-micro-bedrock', 'nova-lite-bedrock', 'nova-pro-bedrock'):
        for t in ('certify_{m}_{s}.csv', 'certify_realtimeqa_{m}_{s}.csv'):
            src = os.path.join(R, t.format(m=mode, s='llama8b-bedrock'))
            rows = list(csv.DictReader(open(src)))
            for r in rows:
                r['model'] = tgt
            with open(os.path.join(R, t.format(m=mode, s=tgt)), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
MK
OUT=$($PY make_paper_assets.py --results "$SB/r2" --out "$SB/g2" 2>&1)
if echo "$OUT" | grep -q "\[certify\] 6 models"; then
  ok "six-model table generates: $(echo "$OUT" | grep -o '\[certify\].*')"
  N=$(grep -c '^    Nova' "$SB/g2/tab_certify.tex" 2>/dev/null || echo 0)
  [ "$N" = "3" ] && ok "all three Nova rows present in the table" \
    || bad "$N Nova rows in the table, expected 3"
  grep -q 'certmodels}{6}' "$SB/g2/certify_facts.tex" \
    && ok "\\certmodels updates to 6, so the prose follows the data" \
    || bad "\\certmodels did not update"
else
  bad "six-model b=0 case failed to generate"; echo "$OUT" | tail -3 | sed 's/^/        /'
fi

echo
echo "=================================================================="
echo "S3. a certify run interrupted between its two CSV writes"
echo "=================================================================="
# main.py writes certify_csv first, per_query_csv second. Resume must not treat
# the first alone as a finished run, and assets must not silently undercount.
cp -r results "$SB/r3"
rm -f "$SB/r3/certify_realtimeqa_n_llama8b-bedrock.csv"
OUT=$($PY make_paper_assets.py --results "$SB/r3" --out "$SB/g3" 2>&1)
echo "$OUT" | grep -q "n_retained cannot be recovered" \
  && ok "missing per-query file refuses instead of understating the count" \
  || bad "missing per-query file was tolerated; degenerate count would be wrong"

echo
echo "=================================================================="
echo "S4. an incomplete their-attacks file"
echo "=================================================================="
cp -r results "$SB/r4"
$PY - "$SB/r4" <<'MK'
import csv, os, sys
R = sys.argv[1]
p = os.path.join(R, 'theirattacks_realtimeqa_k10.csv')
rows = [r for r in csv.DictReader(open(p)) if 'nova' not in r['model']]
with open(p, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
MK
OUT=$($PY make_paper_assets.py --results "$SB/r4" --out "$SB/g4" 2>&1)
echo "$OUT" | grep -q "tab_theirs.*incomplete" \
  && ok "partial their-attacks file refuses rather than shrinking the table" \
  || bad "partial their-attacks file produced a quiet 3-model table"

echo
echo "=================================================================="
if [ "$FAIL" -gt 0 ]; then
  echo "$FAIL SCENARIO(S) FAILED"
  exit 1
fi
echo "ALL $PASS SCENARIO CHECKS PASSED"
