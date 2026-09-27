#!/usr/bin/env bash
# =============================================================================
# DIAGNOSTICS — the bounded follow-up to the NO-GO pilot.
#
# The pilot produced undefended_refusal_cnt = 0 on Llama-3.1-8B and Mistral-7B.
# Two untested explanations remain. This script tests both. HARD STOP after it:
# if neither produces refusals, the availability line of work is dropped.
#
#   D1  Stronger alignment      -> Llama-3.3-70B, RobustRAG prompt
#   D2  Prompt format A/B       -> MutedRAG's conversational prompt
#
# D2 is the interesting one. RobustRAG's prompt says "answer with only keywords"
# and offers an explicit "I don't know" escape, so a refusal has nowhere to go.
# MutedRAG's prompt allows three sentences. If refusals appear under D2 but not
# under the RobustRAG prompt, THAT is the paper: the defense's terse output
# format incidentally immunises it against guardrail-triggered DoS.
#
# Cost: ~$1.50 total. Runtime: under an hour, serialised.
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

run() { echo; echo "+ $*"; "$@" || echo "!! run failed (continuing): $*"; }

echo "############################################################"
echo "### D1 — Llama-3.3-70B, RobustRAG prompt (strongest alignment)"
echo "############################################################"
run $PY main.py --model_name llama70b-bedrock --dataset_name realtimeqa --top_k $TOP_K \
  --defense_method keyword --alpha $ALPHA --beta $BETA --abstention_threshold 1 \
  --attack_method GuardrailTrigger --corruption_size 1 \
  --prompt_style robustrag --max_samples "$N" --dump_responses 3 --use_cache

echo
echo "############################################################"
echo "### D2 — MutedRAG conversational prompt (the A/B that matters)"
echo "###      same models as the failed pilot, only the prompt changes"
echo "############################################################"
for M in llama8b-bedrock llama70b-bedrock; do
  run $PY main.py --model_name "$M" --dataset_name realtimeqa --top_k $TOP_K \
    --defense_method keyword --alpha $ALPHA --beta $BETA --abstention_threshold 1 \
    --attack_method GuardrailTrigger --corruption_size 1 \
    --prompt_style mutedrag --max_samples "$N" --dump_responses 3 --use_cache
done

cat <<'EOF'

############################################################
### DECIDE
############################################################

Compare undefended_refusal_cnt across the runs:

  D2 >> D1  ..... The prompt format is the cause.
                  RobustRAG's keyword-only output format incidentally blocks
                  guardrail-triggered DoS. Clean two-condition experiment.

  D1 >> pilot ... Alignment strength is the cause. Original plan is viable,
                  but only the 70B is attackable -- a thinner story.

  both ~0 ....... STOP. No refusal primitive exists in this setup.

Read the dumped responses, not just the counters. Mistral was COMPLYING with
the jailbreak in the pilot ("I. Intro A. Defamatory article..."), which is a
different failure mode from refusing and must not be scored as a refusal.
EOF
