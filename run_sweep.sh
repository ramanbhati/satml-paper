#!/usr/bin/env bash
# =============================================================================
# ALPHA / BETA SWEEP — turns one data point into a mechanism result.
#
# The low-n study showed the attacker's keyword survives at n <= 3 with
# (alpha=0.3, beta=3). Alone, that is one configuration and a reviewer can call
# it a RealtimeQA quirk. This moves alpha and beta and asks whether the collapse
# boundary moves to where mu = min(alpha*n, beta) predicts:
#
#     attacker wins  <=>  k_inject >= min(beta, alpha * n)
#
# At k'=1 (boundary is INCLUSIVE: n = k'/alpha still wins, since mu = 1.0 there):
#     alpha=0.5 -> n <= 2      alpha=0.3 -> n <= 3   (3.33 floored)
#     alpha=0.2 -> n <= 5      alpha=0.1 -> every n in 1..10
#     beta=2 with k'=2 -> survives at EVERY n (k' >= beta)
#
# If the measured boundary tracks that line, the threshold IS the vulnerability
# and no abstention detector can fix it. That is the claim worth publishing.
#
# COST: ~$0. The isolated per-passage calls do NOT depend on alpha or beta, and
# the KeywordInjection k'=1,2,3 runs are already cached from run_lown.sh.
# --filter_only skips the ONLY call that would vary (final aggregation), so
# every call in this sweep should be a cache hit. The script verifies that with
# check_cache_coverage.py before running, and main.py refuses --filter_only
# without --use_cache.
#
# TRADE-OFF: filter_only means no ASR and no accuracy -- those columns are
# written blank. This sweep answers ONE question: did the keyword survive the
# filter. Use run_lown.sh / run_composite.sh for the end-to-end metrics.
#
# Output: results/sweep.csv  ->  analyze_sweep.py
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
N="${N:-100}"
CSV="results/sweep.csv"
MODELS="${MODELS:-mistral7b-bedrock llama8b-bedrock llama70b-bedrock}"

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED"; exit 1; }
echo "    passed"

echo
echo "### cache coverage (expect ZERO billed isolated calls)"
for M in $MODELS; do
  echo "--- $M"
  $PY check_cache_coverage.py --model_name "$M" --dataset_name realtimeqa \
    --top_k 10 --attack_method KeywordInjection --corruption_size 1 \
    --max_samples "$N" --no_vanilla 2>/dev/null \
    | grep -E "BILLED isolated|prompt-level misses"
done
echo
echo "If 'BILLED isolated calls' is not 0 above, the k'=1..3 KeywordInjection"
echo "runs are not fully cached and this sweep will NOT be free."
read -r -p "proceed? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

mkdir -p results
[ -f "$CSV" ] && { echo "### moving existing $CSV aside"; mv "$CSV" "$CSV.$(date +%s).bak"; }

# (alpha, beta, k') grid, stated literally so test_preflight.py can read the
# real values statically rather than a shell variable. beta=3 is upstream's
# default; beta=2 tests the OTHER branch of the min(), where a budget of 2 wins
# at every n.
#
# predicted survival boundary (k' >= min(beta, alpha*n)), at k'=1:
#   alpha=0.5 -> n<=2    alpha=0.3 -> n<=3    alpha=0.2 -> n<=5
#   alpha=0.1 -> all n   beta=2 k'=2 -> all n (k'>=beta)
GRID=(
  "0.5 3" "0.3 3" "0.2 3" "0.1 3" "0.3 2" "0.3 5"
)
for M in $MODELS; do
  for AB in "${GRID[@]}"; do
    set -- $AB; A="$1"; B="$2"
    for KP in 1 2 3; do
      echo
      echo "############ $M  alpha=$A beta=$B  k'=$KP"
      $PY main.py --model_name "$M" --dataset_name realtimeqa --top_k 10 \
        --defense_method keyword --alpha "$A" --beta "$B" --abstention_threshold 1 \
        --attack_method KeywordInjection --corruption_size "$KP" \
        --max_samples "$N" --per_query_csv "$CSV" \
        --no_vanilla --use_cache --filter_only \
        || echo "!! failed (continuing)"
    done
  done
done

echo
echo "### analysing"
$PY analyze_sweep.py "$CSV"
