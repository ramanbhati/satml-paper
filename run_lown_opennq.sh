#!/usr/bin/env bash
# =============================================================================
# OPEN_NQ LOW-n STUDY — the one experiment that could still make SaTML viable.
#
# THE CLAIM UNDER TEST (not derivable from RobustRAG's formula, which is why it
# is worth running):
#
#     There is a non-trivial population of real queries for which NO setting of
#     alpha makes the defense both SAFE and USEFUL. Raising alpha shrinks the
#     attacker's window but destroys the accuracy on exactly those queries, so
#     the defender trades an integrity failure for a utility failure. The
#     certificate does not cover this because it is stated over k', not over n.
#
# The realtimeqa version of this hint had N=8 per cell (alpha=0.7 -> 38% keyword
# survival AND 0% accuracy). Far too small. open_nq has 500 queries, ~5x the
# mass, and is a second, independent dataset.
#
# THE BIGGEST RISK, AND WHY THIS RUNS A PILOT FIRST.
# RealtimeQA is news QA: hard, time-sensitive, 14% of queries land at n<=3.
# open_nq is Natural Questions, Wikipedia-backed and generally answerable, so
# its group-abstention rate may be far lower. If open_nq only produces ~3% low-n
# queries, 500 queries yields ~15 -- barely better than what we already have,
# and the full run is wasted. The old open_nq logs predate the group-level
# instrumentation, so this CANNOT be predicted from existing data. It has to be
# measured, cheaply, before committing.
#
#   STAGE 1 (default):  PILOT=1  N=100   ~$0.16, ~10 min   <- run this first
#   STAGE 2:            PILOT=0  N=500   ~$1.10, ~1.3 hr   mistral only
#   STAGE 3:            PILOT=0 N=500 MODELS="llama8b-bedrock llama70b-bedrock"
#
# GO/NO-GO AFTER STAGE 1: the analyser prints the observed low-n rate and
# extrapolates the yield at N=500. If the projected low-n count is under ~40 per
# model, STOP -- open_nq cannot support the claim and the money is better spent
# elsewhere.
#
# WHY THE CLEAN ARM IS INCLUDED. Accuracy at low n is meaningless without it:
# low-n queries are the ones whose passages lacked the answer, so they would
# score badly with no attacker at all. The clean arm at the SAME alpha on the
# SAME queries is the only way to separate "the attack destroyed utility" from
# "these questions were unanswerable". This is the confound that has been open
# since the first low-n run.
#
# COST STRUCTURE. The 10 isolated per-passage calls do NOT depend on alpha, so
# they are paid once per (model, arm) and reused across the whole alpha grid.
# Only the final aggregation call varies with alpha. For mistral the first 100
# open_nq CLEAN isolated calls are already cached from July, which is why the
# pilot is cheaper than it looks.
#
# Output: results/lown_opennq.csv  ->  analyze_lown_opennq.py
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
PILOT="${PILOT:-1}"
if [ "$PILOT" = "1" ]; then
  N="${N:-100}"
  CSV="results/lown_opennq_pilot.csv"
else
  N="${N:-500}"
  CSV="results/lown_opennq.csv"
fi
MODELS="${MODELS:-mistral7b-bedrock}"

# alpha grid: 0.3 is upstream's default; 0.5 and 0.7 are the settings a defender
# would reach for to close the low-n hole. The question is what they cost.
ALPHAS=( "0.3" "0.5" "0.7" )
# arms: clean (utility control) and a single injected passage (the low-n attack).
# k'=1 only: at beta=3, k'>=3 wins at every alpha and says nothing about this.
ARMS=( "none 0" "KeywordInjection 1" )

echo "###############################################################"
if [ "$PILOT" = "1" ]; then
  echo "### STAGE 1: PILOT  (N=$N)  -- measuring whether open_nq HAS a low-n population"
  echo "### If the projected yield is too low, STOP. Do not run stage 2."
else
  echo "### FULL RUN (N=$N) -- only do this if the pilot projected >=40 low-n queries"
fi
echo "###############################################################"

echo
echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED -- run it directly to see why"; exit 1; }
echo "    passed"

echo
echo "### predicted cost (no calls made)"
for M in $MODELS; do
  for ARM in "${ARMS[@]}"; do
    set -- $ARM; A="$1"; KP="$2"
    echo "--- $M  arm=$A k'=$KP"
    # errors NOT suppressed; pipefail makes this test python's exit status
    if ! $PY check_cache_coverage.py --model_name "$M" --dataset_name open_nq \
          --top_k 10 --attack_method "$A" --corruption_size "$KP" \
          --max_samples "$N" --no_vanilla | tail -6; then
      echo "!! cost estimate FAILED -- refusing to continue blind"
      exit 1
    fi
  done
done
echo
echo "Above is per ARM, and the isolated calls are alpha-independent (paid once)."
echo "Each of the ${#ALPHAS[@]} alphas then adds ~$N aggregation calls per arm."
read -r -p "proceed and spend on Bedrock? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

mkdir -p results
[ -f "$CSV" ] && { echo "### moving existing $CSV aside"; mv "$CSV" "$CSV.$(date +%s).bak"; }

# Order matters for cost: run all alphas of one arm together so the arm's
# isolated calls are cached by the first alpha and reused by the rest.
for M in $MODELS; do
  for ARM in "${ARMS[@]}"; do
    set -- $ARM; A="$1"; KP="$2"
    for AL in "${ALPHAS[@]}"; do
      echo
      echo "############ $M  $A k'=$KP  alpha=$AL"
      $PY main.py --model_name "$M" --dataset_name open_nq --top_k 10 \
        --defense_method keyword --alpha "$AL" --beta 3 --abstention_threshold 1 \
        --attack_method "$A" --corruption_size "$KP" \
        --max_samples "$N" --per_query_csv "$CSV" --no_vanilla --use_cache \
        || echo "!! failed (continuing)"
    done
  done
done

echo
echo "### analysing"
$PY analyze_lown_opennq.py "$CSV"
