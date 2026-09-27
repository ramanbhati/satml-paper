#!/usr/bin/env bash
# =============================================================================
# ALPHA: SECURITY vs UTILITY — the open question the filter_only sweep left.
#
# The alpha/beta sweep showed alpha=0.5 is strictly more secure than alpha=0.3
# (attacker boundary moves from n<=3 down to n<=2). The obvious reviewer
# question follows immediately: then why not just raise alpha?
#
# Presumably because mu = min(alpha*n, beta) filters HONEST keywords too, so a
# higher threshold costs accuracy. That is a hypothesis. This measures it.
#
# WHY IT NEEDS REAL MONEY. The filter_only sweep was free precisely because it
# skipped the final aggregation call. Accuracy only exists AFTER that call, and
# its prompt embeds the surviving keyword set, which changes with alpha. So one
# aggregation call per (query, alpha) is unavoidable.
#
# WHAT MAKES IT CHEAP ANYWAY. The 10 isolated per-passage calls do NOT depend on
# alpha, so they are paid at most once per (model, attack, k') and then cached.
# The clean arm's isolated calls are NOT yet cached (the old corr0 runs used
# use_cache=False), so the first clean alpha pays for them; every later alpha
# reuses them.
#
# THE FIGURE THIS PRODUCES. Two curves against alpha:
#     clean accuracy      (utility the defender keeps)
#     attacker survival   (security the defender buys)
# If utility falls as fast as security rises, the defender has no good setting
# -- which is a substantive claim about the defense, not just about an attack.
#
# Output: results/alpha_utility.csv  ->  analyze_alpha_utility.py
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
N="${N:-100}"
CSV="results/alpha_utility.csv"
MODELS="${MODELS:-mistral7b-bedrock}"

# alpha grid. 0.3 is upstream's default. 0.7 is included to show the far end of
# the utility cost even though its security gain is small.
ALPHAS=( "0.1" "0.2" "0.3" "0.5" "0.7" )
# attack arms: 'none' measures utility, k'=1..2 measure security.
# k'=3 is omitted: at beta=3 it wins at every alpha (k'>=beta short-circuits the
# min), so it carries no information about alpha and would be wasted money.
ARMS=( "none 0" "KeywordInjection 1" "KeywordInjection 2" )

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED"; exit 1; }
echo "    passed"

echo
echo "### predicted cost (no calls made)"
for M in $MODELS; do
  echo "--- $M  (clean arm, the expensive one: isolated calls not yet cached)"
  # NOTE: errors are NOT suppressed. If the cost estimator cannot run, that is
  # itself a reason to stop -- a silent failure here would leave you approving
  # a spend with no estimate at all.
  if ! $PY check_cache_coverage.py --model_name "$M" --dataset_name realtimeqa \
        --top_k 10 --attack_method none --corruption_size 0 \
        --max_samples "$N" --no_vanilla | tail -7; then
    echo "!! cost estimate FAILED for $M -- refusing to continue blind"
    exit 1
  fi
done
echo
echo "NOTE: the figure above is for ONE alpha. Later alphas reuse the isolated"
echo "calls and add only ~$N aggregation calls each (short prompts, ~1/4 the cost)."
echo "Arms: ${#ARMS[@]}  x  alphas: ${#ALPHAS[@]}  x  models: $(echo $MODELS | wc -w)"
read -r -p "proceed and spend on Bedrock? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

mkdir -p results
[ -f "$CSV" ] && { echo "### moving existing $CSV aside"; mv "$CSV" "$CSV.$(date +%s).bak"; }

for M in $MODELS; do
  for ARM in "${ARMS[@]}"; do
    set -- $ARM; A="$1"; KP="$2"
    for AL in "${ALPHAS[@]}"; do
      echo
      echo "############ $M  attack=$A k'=$KP  alpha=$AL"
      $PY main.py --model_name "$M" --dataset_name realtimeqa --top_k 10 \
        --defense_method keyword --alpha "$AL" --beta 3 --abstention_threshold 1 \
        --attack_method "$A" --corruption_size "$KP" \
        --max_samples "$N" --per_query_csv "$CSV" --no_vanilla --use_cache \
        || echo "!! failed (continuing)"
    done
  done
done

echo
echo "### analysing"
$PY analyze_alpha_utility.py "$CSV"
