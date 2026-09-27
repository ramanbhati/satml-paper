#!/usr/bin/env bash
# =============================================================================
# THE MITIGATION — does changing the threshold rule close both holes?
#
# THE PROBLEM. The filter keeps a keyword iff count >= mu, so a keyword seen
# ONCE survives whenever mu <= 1. Upstream uses mu = min(alpha*n, beta) where n
# is the number of non-abstaining groups, and that is <= 1 whenever
#
#       ceil(k/omega) <= 1/alpha
#
# which is reachable two independent ways, NEITHER of which the certificate
# mentions:
#   (a) thin corpus coverage      -> n falls   (measured: 14% of RealtimeQA)
#   (b) large passage group size  -> the group count itself falls (omega >= 4)
#
# THE CANDIDATES.
#   n       upstream baseline, mu = min(alpha*n, beta)
#   k       mu = min(alpha*top_k, beta). top_k is a CONFIGURED constant, so
#           neither the corpus nor omega can move it. Blunt: discards the
#           adaptivity entirely (at k=10, alpha=0.3 it is just mu = beta = 3).
#   floor2  mu = max(2, min(alpha*n, beta)). Minimal and targeted: keeps
#           upstream's adaptive threshold but never lets it fall below the
#           smallest value that means anything -- a keyword must be corroborated
#           by at least TWO independent groups. (Not 1: count >= 1 admits
#           everything, which is exactly the bug.)
#
# THE TEST. Both failure regimes, on the same queries:
#   omega=1  -> the LOW-COVERAGE hole (mu <= 1 only when n <= 3, ~14% of queries)
#   omega=4  -> the GROUP-SIZE hole   (mu = 0.9 <= 1 on EVERY query)
# A fix has to close both. And the clean arm measures what it costs: demanding
# corroboration a thin corpus cannot supply converts a wrong answer into an
# abstention, which is better but not free.
#
# COST. Nearly free. threshold_mode changes only the FINAL aggregation prompt
# (via the surviving-keyword hints); the per-passage isolated calls are
# identical and already cached by the omega runs at these exact omega values.
# Expect ~1 new call per query per cell.
#
# Output: results/fix.csv  ->  analyze_fix.py
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
N="${N:-100}"
K="${K:-10}"
DATASET="${DATASET:-realtimeqa}"
MODELS="${MODELS:-mistral7b-bedrock}"
CSV="${CSV:-results/fix_${DATASET}_k${K}.csv}"

MODES=( "n" "k" "floor2" )
# omega=1 is the coverage hole, omega=4 is the group-size hole. A fix must
# close BOTH; testing only one would let a partial fix look complete.
OMEGAS=( "1" "4" )
ARMS=( "none 0" "KeywordInjection 1" )

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED"; exit 1; }
echo "    passed"

echo
echo "### cache coverage (isolated calls should already be warm from run_omega)"
for M in $MODELS; do
  echo "--- $M"
  if ! $PY check_cache_coverage.py --model_name "$M" --dataset_name "$DATASET" \
        --top_k "$K" --attack_method KeywordInjection --corruption_size 1 \
        --max_samples "$N" --no_vanilla | tail -6; then
    echo "!! cost estimate FAILED -- refusing to continue blind"
    exit 1
  fi
done
echo
echo "threshold_mode changes ONLY the final aggregation prompt, so expect"
echo "roughly $N new calls per cell: ${#MODES[@]} modes x ${#OMEGAS[@]} omegas x ${#ARMS[@]} arms."
echo "dataset=$DATASET k=$K -> $CSV"
read -r -p "proceed and spend on Bedrock? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

mkdir -p results
# Do NOT rotate the CSV aside. Rotating before the run means an aborted run,
# or a run covering only some models, strands the earlier data under a .bak
# nobody looks at -- that happened three times. Instead APPEND, and refuse to
# append into a file whose rows would be silently mixed with a different k.
# (Model and threshold_mode legitimately differ between runs and are columns,
# so the analysers can separate them; k cannot be, because every prediction is
# a function of ceil(k/omega).)
if [ -f "$CSV" ]; then
  EXIST_K=$($PY csv_top_k.py "$CSV")
  if [ -n "$EXIST_K" ] && [ "$EXIST_K" != "$K" ]; then
    echo "!! $CSV already holds top_k=$EXIST_K but this run is k=$K."
    echo "   Refusing to mix. Move it aside yourself or set CSV=..."
    exit 1
  fi
  echo "### appending to existing $CSV ($(wc -l < "$CSV") lines)"
fi

FAILED_CELLS=0
ROWS_BEFORE=$( [ -f "$CSV" ] && wc -l < "$CSV" || echo 0 )

for M in $MODELS; do
  for MODE in "${MODES[@]}"; do
    for W in "${OMEGAS[@]}"; do
      for ARM in "${ARMS[@]}"; do
        set -- $ARM; A="$1"; KP="$2"
        echo
        echo "############ $M  mode=$MODE  omega=$W  $A k'=$KP"
        $PY main.py --model_name "$M" --dataset_name "$DATASET" --top_k "$K" \
          --defense_method keyword --alpha 0.3 --beta 3 --abstention_threshold 1 \
          --threshold_mode "$MODE" --group_size "$W" \
          --attack_method "$A" --corruption_size "$KP" \
          --max_samples "$N" --per_query_csv "$CSV" --no_vanilla --use_cache \
          || { echo "!! failed (continuing)"; FAILED_CELLS=$((FAILED_CELLS+1)); }
      done
    done
  done
done

echo
ROWS_AFTER=$( [ -f "$CSV" ] && wc -l < "$CSV" || echo 0 )
NEW_ROWS=$(( ROWS_AFTER - ROWS_BEFORE ))
echo "### $FAILED_CELLS cell(s) failed; $NEW_ROWS new rows written"
# A failed cell means the CSV is MISSING that condition. The analyser cannot
# tell a missing condition from one that was never requested -- it would print
# a clean-looking table built from whatever rows happen to be in the file,
# including rows from EARLIER runs. That is how a completely broken run gets
# read as a successful one. Refuse, and say what to do.
if [ "$FAILED_CELLS" -gt 0 ]; then
  echo
  echo "!! REFUSING TO ANALYSE: $FAILED_CELLS cell(s) failed."
  echo "   The table would silently mix this run's partial rows with older ones."
  echo "   Fix the failures and re-run; completed cells are cached and will not"
  echo "   be re-billed. To look anyway, run the analyser by hand:"
  echo "       $PY $(basename "$0" .sh | sed 's/^run_/analyze_/').py $CSV"
  exit 1
fi
if [ "$NEW_ROWS" -eq 0 ]; then
  echo "!! No new rows were written. Nothing to analyse."
  exit 1
fi
echo "### analysing"
$PY analyze_fix.py "$CSV"
