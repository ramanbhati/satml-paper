#!/usr/bin/env bash
# Everything still missing, in one unattended pass.
#
#     caffeinate -i ./run_overnight.sh          # macOS: keeps the machine awake
#     ./run_overnight.sh                        # if the machine will not sleep
#
# Runs, in order, skipping anything already complete:
#   0. gate.sh              offline checks; ABORTS the whole run if not green
#   1. certify              Nova Lite and Nova Pro (Nova Micro already done)
#   2. nq                   the full 500-item Natural Questions sweep
#   3. make_paper_assets    regenerate every table and figure
#   4. audit_numbers        re-derive every headline number
#
# Then writes OVERNIGHT_REPORT.md with what happened and what needs a decision.
#
# DESIGN NOTES, because this runs with nobody watching:
#
#  - The gate runs FIRST and aborts everything if it fails. Spending money on a
#    repo that cannot regenerate its own tables is pointless.
#  - Experiments 1 and 2 are INDEPENDENT. A failure in one does not stop the
#    other; both are recorded and the report says which succeeded.
#  - Every block already resumes, so re-running this after a failure re-issues
#    only what is missing. Nothing clean is ever paid for twice.
#  - make_paper_assets is EXPECTED to fail in one specific case: if the floor
#    turns out to cost a certificate (b > 0), it refuses to build because
#    Section VIII asserts in prose that it costs nothing. That is the guard
#    working, not a crash, and the report says so.
#  - audit_numbers is EXPECTED to fail if the NQ sweep lands, because the paper
#    currently states the Natural Questions coverage figure over 100 queries and
#    the new file has 500. That failure is the audit telling you which sentence
#    to update. Again, not a crash.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-}"
if [ -z "$PY" ]; then PY=.venv/bin/python; [ -x "$PY" ] || PY=python3; fi

STAMP=$(date '+%Y-%m-%d_%H%M')
LOGDIR="logs/overnight_$STAMP"
mkdir -p "$LOGDIR"
REPORT="OVERNIGHT_REPORT.md"
START=$(date '+%s')

# Concurrency 2 is safe now that src/models.py has the shared rate-limit
# breaker; the canary ran clean at this setting. Drop to 1 if the guard trips.
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-2}"
# CERTIFY_APPROVED is deliberately NOT exported here. The gate runs first and
# includes a check that certify REFUSES when unattended without approval;
# exporting the approval before the gate made that check pass vacuously and the
# gate fail. Approval belongs to the experiment, not to the checks that decide
# whether the experiment should start. It is set just before certify below.

say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/console.txt"; }
STATUS=()

say "=== overnight run starting; logs in $LOGDIR ==="
say "throttle: workers=$BEDROCK_MAX_WORKERS"

# ---------------------------------------------------------------- 0. the gate
say ">> gate (offline, no cost)"
if ./gate.sh > "$LOGDIR/00_gate.txt" 2>&1; then
  say "   gate PASSED"
  STATUS+=("PASS  gate")
else
  say "   gate FAILED -- aborting before spending anything"
  say "   see $LOGDIR/00_gate.txt"
  STATUS+=("FAIL  gate (nothing was run)")
  {
    echo "# Overnight run $STAMP"
    echo
    echo "**Aborted before spending.** The offline gate failed, so no experiment ran."
    echo
    echo '```'
    tail -30 "$LOGDIR/00_gate.txt"
    echo '```'
  } > "$REPORT"
  exit 1
fi

# ------------------------------------------------------------- 1. certification
say ">> certify (Nova Lite, Nova Pro; everything else resumes)"
# approval scoped to this one invocation, after the gate has had its say
CERTIFY_APPROVED=yes ./run_final.sh certify > "$LOGDIR/01_certify.txt" 2>&1; RC=$?
if [ "$RC" -eq 0 ]; then
  say "   certify OK"
  STATUS+=("PASS  certify")
else
  say "   certify FAILED (exit $RC) -- continuing to nq, which is independent"
  STATUS+=("FAIL  certify -- see 01_certify.txt")
fi
CERT_MODELS=$(ls results/certify_n_*.csv 2>/dev/null | wc -l | tr -d ' ')
say "   certify_n_*.csv present for $CERT_MODELS of 6 models"

# ------------------------------------------------------------------- 2. the NQ
# WHY THIS IS IN THE OVERNIGHT RUN AND NOT DEFERRED. The abstract freezes at
# registration and quotes the Natural Questions coverage figure. That figure is
# currently over 100 queries. If the 500-item sweep is run AFTER registration
# and the number moves, the abstract is wrong and cannot be corrected. Running
# it now means whatever goes into the frozen abstract is the final number.
say ">> nq (500-item Natural Questions sweep, 5 group sizes)"
./run_final.sh nq > "$LOGDIR/02_nq.txt" 2>&1; RC=$?
if [ "$RC" -eq 0 ]; then
  say "   nq OK"
  STATUS+=("PASS  nq")
else
  say "   nq FAILED (exit $RC)"
  STATUS+=("FAIL  nq -- see 02_nq.txt")
fi
NQ_ROWS=$([ -s results/omega_open_nq_k10_full.csv ] \
  && echo $(( $(wc -l < results/omega_open_nq_k10_full.csv) - 1 )) || echo 0)
say "   omega_open_nq_k10_full.csv rows: $NQ_ROWS"

# ------------------------------------------------------------------- 3. assets
say ">> regenerate tables and figures"
if $PY make_paper_assets.py > "$LOGDIR/03_assets.txt" 2>&1; then
  say "   assets OK"
  STATUS+=("PASS  assets")
  ASSET_NOTE=""
else
  say "   assets REFUSED -- this is usually the b>0 guard, not a crash"
  STATUS+=("STOP  assets refused -- see 03_assets.txt")
  ASSET_NOTE=$(tail -6 "$LOGDIR/03_assets.txt")
fi

# -------------------------------------------------------------------- 4. audit
say ">> re-derive every headline number"
if $PY audit_numbers.py > "$LOGDIR/04_audit.txt" 2>&1; then
  say "   audit OK"
  STATUS+=("PASS  audit")
else
  say "   audit reported mismatches (expected if NQ landed)"
  STATUS+=("CHECK audit mismatches -- see 04_audit.txt")
fi

# ------------------------------------------------------------------- 5. report
ELAPSED=$(( ($(date '+%s') - START) / 60 ))
{
  echo "# Overnight run $STAMP"
  echo
  echo "Finished $(date '+%Y-%m-%d %H:%M'), ${ELAPSED} minutes. Logs in \`$LOGDIR/\`."
  echo
  echo "## Outcome"
  echo
  printf '%s\n' "${STATUS[@]}" | sed 's/^/    /'
  echo
  echo "## State"
  echo
  echo "- certification models on disk: **$CERT_MODELS of 6**"
  echo "- Natural Questions 500-item file: **$NQ_ROWS rows** (expect 2500 when complete)"
  echo
  if [ -f ../../../paper-satml2027/generated/certify_facts.tex ]; then
    echo "## Certification, as currently derived"
    echo
    echo '```'
    grep -E 'certb|certc|certp|certmodels|certtotal|certdeg' \
      ../../../paper-satml2027/generated/certify_facts.tex 2>/dev/null
    echo '```'
    echo
  fi
  echo "## Contamination"
  echo
  TOTE=0
  for L in logs/certify_*.txt logs/nq500_*.txt; do
    [ -e "$L" ] || continue
    E=$(grep -oE "!! [0-9]+ EMPTY" "$L" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
    TOTE=$((TOTE + ${E:-0}))
  done
  echo "Total empty responses across all certify and nq logs: **$TOTE**"
  echo "(Anything above zero means some run was throttled and its data discarded.)"
  echo
  if [ -n "$ASSET_NOTE" ]; then
    echo "## Asset generation refused"
    echo
    echo '```'
    echo "$ASSET_NOTE"
    echo '```'
    echo
    echo "If this is the \`b > 0\` guard, the floor now costs at least one query"
    echo "its certificate. That is a real result, not a failure. Section VIII"
    echo "says \"The floor costs nothing\" and \"A zero on the losing side\" in"
    echo "prose, and those sentences need rewriting before anything regenerates."
    echo
  fi
  echo "## Audit"
  echo
  echo '```'
  tail -12 "$LOGDIR/04_audit.txt" 2>/dev/null
  echo '```'
  echo
  echo "If the Natural Questions checks fail, that is expected: the paper states"
  echo "that coverage figure over 100 queries and the sweep now has 500. The"
  echo "number in Section 3, Section 5 and **the abstract** has to be updated to"
  echo "whatever the 500-item run gives, before Tuesday's registration."
  echo
  echo "## Next"
  echo
  echo "1. Read this file and \`$LOGDIR/04_audit.txt\`."
  echo "2. If the NQ figure moved, update it everywhere including the abstract."
  echo "3. Decide the abstract ending: one option claims the certification"
  echo "   result, the other claims only what the propositions prove."
  echo "4. Build on this machine to confirm 12 pages; IEEEtran is not available"
  echo "   in the assistant's environment."
} > "$REPORT"

say "=== done in ${ELAPSED} min; read $REPORT ==="
printf '%s\n' "${STATUS[@]}"
