#!/usr/bin/env bash
# =============================================================================
# THE VANILLA (UNDEFENDED) BASELINE — the arm the omega runs deliberately skipped
#
# WHY IT IS NEEDED. RobustRAG states its utility claim RELATIVE TO VANILLA RAG:
# "with omega=3, we reduce the benign performance drop from 7% to 0%". The drop
# is measured against undefended RAG, not against omega=1. Every omega run so far
# used --no_vanilla, so results/omega_*.csv contains no vanilla column and CANNOT
# verify or refute that claim -- it can only show accuracy varies little across
# omega. This run supplies the missing reference line.
#
# IT IS OMEGA-INDEPENDENT, SO DO NOT SWEEP OMEGA. query_undefended() reads
# data_item, and defense.py groups a COPY, leaving data_item ungrouped. The
# vanilla prompt therefore concatenates all k passages no matter what omega is.
# One run per (model, arm) covers the entire ladder. Sweeping omega here would
# pay five times for five identical numbers.
#
# --defense_method none: skips the defended arm entirely. Those calls are already
# cached from the omega runs, but skipping them avoids re-deriving numbers we
# already have and keeps this run to exactly one new call per query.
#
# TWO ARMS, both worth having:
#   none 0            benign accuracy -> the reference for the "performance drop"
#   KeywordInjection 1  undefended ASR -> the gate: does the attack even work
#                       without a defense? A defense that "reduces" an attack
#                       that never worked is not evidence of anything.
#
# COST. 1 new call per query per cell: 6 models x 2 arms x 100 = 1200 calls.
# Vanilla prompts are LONG (all k passages, ~2.5k tokens) so this is pricier per
# call than an isolated one, but still well under a dollar in total.
#
# Output: results/vanilla_<dataset>_k<K>.csv  ->  analyze_vanilla.py
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
MODELS="${MODELS:-mistral7b-bedrock llama8b-bedrock llama70b-bedrock nova-micro-bedrock nova-lite-bedrock nova-pro-bedrock}"
CSV="${CSV:-results/vanilla_${DATASET}_k${K}.csv}"

ARMS=( "none 0" "KeywordInjection 1" )
[ -n "${ARMS_OVERRIDE:-}" ] && ARMS=( "$ARMS_OVERRIDE" )

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED -- run it directly to see why"; exit 1; }
echo "    passed"

echo
echo "### plan"
echo "  models : $MODELS"
echo "  arms   : ${#ARMS[@]}  (${ARMS[*]})"
echo "  dataset: $DATASET  k=$K  N=$N"
echo "  omega  : NOT swept -- vanilla is omega-independent by construction"
echo "  -> $(echo $MODELS | wc -w) models x ${#ARMS[@]} arms x $N = $(( $(echo $MODELS | wc -w) * ${#ARMS[@]} * N )) new calls"
echo "  -> $CSV"

# COST GATE. check_cache_coverage.py is NOT used here: it models the DEFENDED
# path (top_k isolated calls + aggregation) and would report a number that has
# nothing to do with this run. Vanilla is exactly one call per query, so the
# volume is known exactly; the only unknown is prompt length, which is measured
# from the caches rather than guessed. Aborts if the estimate cannot be produced.
if ! $PY vanilla_cost.py --models "$MODELS" --arms "${#ARMS[@]}" --n "$N" \
        --dataset "$DATASET" --top_k "$K"; then
  echo "!! cost estimate FAILED -- refusing to continue blind"
  exit 1
fi

read -r -p "proceed and spend on Bedrock? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

mkdir -p results
FAILED_CELLS=0
ROWS_BEFORE=$( [ -f "$CSV" ] && wc -l < "$CSV" || echo 0 )

for M in $MODELS; do
  for ARM in "${ARMS[@]}"; do
    set -- $ARM; A="$1"; KP="$2"
    echo
    echo "############ $M  $A k'=$KP  (vanilla only)"
    $PY main.py --model_name "$M" --dataset_name "$DATASET" --top_k "$K" \
      --defense_method none \
      --attack_method "$A" --corruption_size "$KP" \
      --max_samples "$N" --vanilla_csv "$CSV" --use_cache \
      || { echo "!! failed (continuing)"; FAILED_CELLS=$((FAILED_CELLS+1)); }
  done
done

echo
ROWS_AFTER=$( [ -f "$CSV" ] && wc -l < "$CSV" || echo 0 )
NEW_ROWS=$(( ROWS_AFTER - ROWS_BEFORE ))
echo "### $FAILED_CELLS cell(s) failed; $NEW_ROWS new rows written"
if [ "$FAILED_CELLS" -gt 0 ]; then
  echo
  echo "!! REFUSING TO ANALYSE: $FAILED_CELLS cell(s) failed."
  echo "   A missing model would silently drop out of the comparison table."
  echo "   Fix and re-run; completed cells are cached and will not be re-billed."
  exit 1
fi
if [ "$NEW_ROWS" -eq 0 ]; then
  echo "!! No new rows were written. Nothing to analyse."
  exit 1
fi

echo "### analysing"
$PY analyze_vanilla.py "$CSV" results/omega_${DATASET}_k${K}_all.csv
