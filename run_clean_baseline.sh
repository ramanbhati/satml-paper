#!/usr/bin/env bash
# =============================================================================
# CLEAN (NO-ATTACK) BASELINE — resolves the accuracy confound.
#
# WHY. In the low-n study, accuracy is 20% at n<=3 and 57% at n>=4. That gap is
# uninterpretable on its own: low-n queries are exactly the ones whose retrieved
# passages lacked the answer, so the system would score badly there with NO
# attacker present. Without this baseline we cannot say whether the attack
# destroyed utility or the questions were simply hard.
#
# WHAT IT SETTLES. With clean per-query accuracy bucketed by the same n, the
# claim becomes precise: if clean accuracy at n<=3 is also ~20%, the attack
# converts an UNHELPFUL answer into a WRONG one -- the system stops failing
# visibly and starts failing convincingly. That is a sharper statement than the
# raw accuracy drop, and it is the one the paper should make.
#
# COST -- READ THIS. This is NOT free, despite earlier assumptions. The existing
# corr0-attacknone runs in log/ were all executed with use_cache=False, so none
# of their responses were ever written to the cache. Verify with:
#     python3 check_cache_coverage.py --model_name mistral7b-bedrock \
#         --attack_method none --no_vanilla
# Expect ~11 billed calls/query (10 isolated + 1 aggregation).
#
# --use_cache is MANDATORY here: it writes the responses to disk so that any
# re-analysis, and the clean arm of any later sweep, is free. Omitting it is
# what created this situation in the first place.
#
# Output: results/clean.csv  ->  analyze_clean.py
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
N="${N:-100}"
CSV="results/clean.csv"
MODELS="${MODELS:-mistral7b-bedrock llama8b-bedrock llama70b-bedrock}"

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED"; exit 1; }
echo "    passed"

echo
echo "### predicted cost (no calls made)"
for M in $MODELS; do
  echo "--- $M"
  # errors deliberately NOT suppressed; pipefail makes this test python's status
  if ! $PY check_cache_coverage.py --model_name "$M" --dataset_name realtimeqa \
        --top_k 10 --attack_method none --corruption_size 0 \
        --max_samples "$N" --no_vanilla | tail -7; then
    echo "!! cost estimate FAILED for $M -- refusing to continue blind"
    exit 1
  fi
done
echo
read -r -p "proceed and spend on Bedrock? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

mkdir -p results
[ -f "$CSV" ] && { echo "### moving existing $CSV aside"; mv "$CSV" "$CSV.$(date +%s).bak"; }

for M in $MODELS; do
  echo
  echo "############ $M  clean (no attack)"
  $PY main.py --model_name "$M" --dataset_name realtimeqa --top_k 10 \
    --defense_method keyword --alpha 0.3 --beta 3 --abstention_threshold 1 \
    --attack_method none --corruption_size 0 \
    --max_samples "$N" --per_query_csv "$CSV" --no_vanilla --use_cache \
    || echo "!! failed (continuing)"
done

echo
echo "### analysing"
$PY analyze_clean.py "$CSV" results/lown.csv
