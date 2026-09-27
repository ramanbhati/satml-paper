#!/usr/bin/env bash
# =============================================================================
# RE-SCORE — replay every availability run with the FIXED detectors.
#
# All four D1/D2 runs were scored with the buggy detectors (curly apostrophe
# missed, "can't answer" absent), so their refusal and leakage counts are
# undercounts. --use_cache replays the identical cached model responses, so
# this costs $0 in Bedrock calls and only re-applies the scoring.
#
# Run test_preflight.py first.
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
N="${N:-20}"
TOP_K=10; ALPHA=0.3; BETA=3

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED -- run it directly and fix before continuing"; exit 1; }
echo "    passed"

for STYLE in robustrag mutedrag; do
  for M in llama8b-bedrock llama70b-bedrock mistral7b-bedrock; do
    echo
    echo "############ $M / prompt_style=$STYLE"
    $PY main.py --model_name "$M" --dataset_name realtimeqa --top_k $TOP_K \
      --defense_method keyword --alpha $ALPHA --beta $BETA --abstention_threshold 1 \
      --attack_method GuardrailTrigger --corruption_size 1 \
      --prompt_style "$STYLE" --max_samples "$N" --dump_responses 3 --use_cache \
      || echo "!! failed (continuing)"
  done
done

cat <<'EOF'

############################################################
Corrected numbers are in the LAST summary block of each log.
Collect into the R1 sheet of SaTML_Experiment_Tables.xlsx:
   undefended_refusal_cnt   <- does the attack work, per model/prompt
   defended_refusal_cnt     <- does isolation contain it
   group_refusal_leaked_avg <- refusals the upstream filter misses
Any run that hits Bedrock rather than cache will print slowly; that means a
prompt changed and it is NOT a pure re-score.
EOF
