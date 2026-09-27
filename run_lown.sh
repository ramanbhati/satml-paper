#!/usr/bin/env bash
# =============================================================================
# LOW-n STUDY — the core experiment.
#
# CLAIM: RobustRAG's keyword filter is mu = min(alpha*n, beta), where n is the
# number of groups that gave a substantive answer. When n is small the alpha*n
# branch drops mu below 1, so ANY keyword seen once survives. With alpha=0.3,
# beta=3 this means a SINGLE poisoned passage (k'=1) gets through whenever
# n <= 3 -- regardless of how abstention is detected.
#
# PREDICTION: attacker_kw_survived ~ 1 for n <= 3, then falls sharply at n >= 4.
# The transition should sit exactly where mu crosses 1.
#
# This uses KeywordInjection, which already asserts the attacker's answer in
# k' passages -- no new attack needed. The earlier runs are cached, so this is
# a replay and costs ~$0.
#
# Output: results/lown.csv  (one row per query)   ->  analyze_lown.py
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
N="${N:-100}"
CSV="results/lown.csv"

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED"; exit 1; }
echo "    passed"

mkdir -p results
[ -f "$CSV" ] && { echo "### moving existing $CSV aside"; mv "$CSV" "$CSV.$(date +%s).bak"; }

for M in mistral7b-bedrock llama8b-bedrock llama70b-bedrock; do
  for KP in 1 2 3; do
    echo
    echo "############ $M  k'=$KP"
    $PY main.py --model_name "$M" --dataset_name realtimeqa --top_k 10 \
      --defense_method keyword --alpha 0.3 --beta 3 --abstention_threshold 1 \
      --attack_method KeywordInjection --corruption_size "$KP" \
      --max_samples "$N" --per_query_csv "$CSV" --no_vanilla --use_cache \
      || echo "!! failed (continuing)"
  done
done

echo
echo "### analysing"
$PY analyze_lown.py "$CSV"
