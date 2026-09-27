#!/usr/bin/env bash
# =============================================================================
# SUPPRESS-THEN-INJECT — the experiment that decides whether there is a paper.
#
# CLAIM. RobustRAG's keyword filter threshold is mu = min(alpha*n, beta), where
# n is the number of isolated groups that gave a substantive answer. n is not a
# constant of the system -- it is a property of the responses, so an attacker
# who can silence honest groups can DRIVE mu DOWN. Below n = 1/alpha (n <= 3 at
# alpha=0.3) the threshold falls under 1 and a keyword seen ONCE survives.
#
# The low-n study only observed this where n happened to be small (~14% of
# RealtimeQA queries). This spends part of the budget MANUFACTURING that
# condition with guardrail-triggering passages, then injects with the rest.
#
# PREDICTION. At a FIXED total budget k', splitting it (suppress + inject)
# beats spending it all on injection, because the suppression passages buy a
# lower mu and the filter admits the injected keyword at a count that would
# otherwise be rejected. Specifically at k'=3, alpha=0.3, beta=3:
#     KeywordInjection k'=3        -> k'>=beta, survives anyway (ceiling)
#     KeywordInjection k'=2        -> needs n<=6
#     SuppressThenInject k'=3,ki=1 -> needs n<=3 UNAIDED, but 2 suppression
#                                     passages should push n there
#
# NEGATIVE CONTROL (built in, not optional). Every SuppressThenInject row has a
# KeywordInjection row at the SAME total k'. If the composite wins only because
# it perturbs more passages, the control matches it and the claim is dead.
#
# COST. These payload combinations are NEW, so the isolated calls are NOT
# cached. BaseModel.batch_query_from_cache is all-or-nothing, so every query
# re-issues all top_k calls. Budget ~11 calls/query. This script runs ONE model
# by default -- prove the mechanism cheaply before paying for three.
#
# Output: results/composite.csv  ->  analyze_lown.py (per-model) and by hand
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
N="${N:-100}"
CSV="results/composite.csv"
# default: cheapest model only. Override to go wide:
#   MODELS="mistral7b-bedrock llama8b-bedrock llama70b-bedrock" ./run_composite.sh
MODELS="${MODELS:-mistral7b-bedrock}"

# The experiment grid, stated literally as "<attack> <k'> <k_inject|->" so that
# test_preflight.py can read it statically and verify the control pairing. Do
# NOT hide these behind shell variables: a preflight that cannot see the real
# arguments gives false confidence. Declared here, before the cost estimate,
# because that estimate reports the arm count.
CONFIGS=(
  # --- the composite: same total budget, different splits ---
  "SuppressThenInject 2 1"   # 1 suppress + 1 inject
  "SuppressThenInject 3 1"   # 2 suppress + 1 inject
  "SuppressThenInject 3 2"   # 1 suppress + 2 inject
  # --- budget-matched controls (MANDATORY: one per composite budget) ---
  "KeywordInjection 2 -"     # whole budget on injection
  "KeywordInjection 3 -"
  "GuardrailTrigger 2 -"     # whole budget on suppression -> pure denial
  "GuardrailTrigger 3 -"
)

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED -- run it directly to see why"; exit 1; }
echo "    passed"

echo
echo "### predicted cost (no calls made)"
for M in $MODELS; do
  echo "--- $M"
  # errors deliberately NOT suppressed: a silent failure here would leave you
  # approving a spend with no estimate. pipefail (set above) makes the `if`
  # test python's exit status rather than tail's.
  if ! $PY check_cache_coverage.py --model_name "$M" --dataset_name realtimeqa \
        --top_k 10 --attack_method GuardrailTrigger --corruption_size 2 \
        --max_samples "$N" --no_vanilla | tail -7; then
    echo "!! cost estimate FAILED for $M -- refusing to continue blind"
    exit 1
  fi
done
echo
echo "NOTE: the figure above is for ONE arm. This script runs ${#CONFIGS[@]} arms"
echo "per model, and each arm's payloads are distinct, so none of them are cached."
echo
read -r -p "proceed and spend on Bedrock? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

mkdir -p results
[ -f "$CSV" ] && { echo "### moving existing $CSV aside"; mv "$CSV" "$CSV.$(date +%s).bak"; }

for M in $MODELS; do
  for CFG in "${CONFIGS[@]}"; do
    set -- $CFG; A="$1"; KP="$2"; KI="$3"
    echo
    echo "############ $M  $A  k'=$KP  k_inject=$KI"
    # shellcheck disable=SC2086
    $PY main.py --model_name "$M" --dataset_name realtimeqa --top_k 10 \
      --defense_method keyword --alpha 0.3 --beta 3 --abstention_threshold 1 \
      --attack_method "$A" --corruption_size "$KP" \
      $( [ "$KI" != "-" ] && echo "--k_inject $KI" ) \
      --max_samples "$N" --per_query_csv "$CSV" --no_vanilla --use_cache \
      || echo "!! failed (continuing)"
  done
done

echo
echo "### done. rows:"
wc -l "$CSV"
echo
echo "Compare, at each total k', the attacker_kw_survived rate for"
echo "SuppressThenInject against KeywordInjection at the SAME k'."
echo "The composite must win, and GuardrailTrigger alone must show ~0 ASR."
