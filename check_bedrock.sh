#!/usr/bin/env bash
# =============================================================================
# Bedrock preflight — tells you WHICH piece is missing, in order.
#   1. AWS CLI present?
#   2. Credentials valid on the standard chain?
#   3. Region reachable and Bedrock responding?
#   4. The specific model ids this repo uses, accessible?
#   5. End-to-end: can main.py actually complete a call?
#
# Usage:  ./check_bedrock.sh
# =============================================================================
set -uo pipefail

REGION="${AWS_REGION_NAME:-${AWS_REGION:-us-east-1}}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
ok()   { printf "  \033[32mOK\033[0m    %s\n" "$1"; }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; }
info() { printf "        %s\n" "$1"; }

echo "=== 1. AWS CLI ==="
if command -v aws >/dev/null 2>&1; then
  ok "$(aws --version 2>&1)"
else
  bad "aws CLI not found"
  info "macOS:  brew install awscli"
  info "(the CLI is only needed for these checks; boto3 does the real work)"
fi

echo
echo "=== 2. Credentials (standard chain) ==="
if IDENT=$(aws sts get-caller-identity --output json 2>&1); then
  ok "authenticated"
  echo "$IDENT" | sed 's/^/        /'
else
  bad "no valid credentials"
  info "$IDENT"
  info ""
  info "Pick ONE:"
  info "  a) IAM access keys:      aws configure"
  info "  b) AWS SSO / Identity Center:"
  info "         aws configure sso        # first time"
  info "         aws sso login            # each session (tokens expire!)"
  info "  c) Temporary env vars:   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=..."
  info ""
  info "If this worked before and stopped: SSO tokens expire (often 8-12h)."
  info "Run 'aws sso login' and re-check."
  exit 1
fi

echo
echo "=== 3. Bedrock reachable in $REGION ==="
if aws bedrock list-foundation-models --region "$REGION" >/dev/null 2>&1; then
  ok "bedrock API responding in $REGION"
else
  bad "cannot call bedrock in $REGION"
  info "Check the region is one where Bedrock is available (us-east-1, us-west-2),"
  info "and that your IAM principal has bedrock:ListFoundationModels + bedrock:InvokeModel."
fi

echo
echo "=== 4. Model ids used by this repo ==="
for MID in "us.meta.llama3-1-8b-instruct-v1:0" "us.meta.llama3-3-70b-instruct-v1:0" "mistral.mistral-7b-instruct-v0:2"; do
  BASE="${MID#us.}"
  if aws bedrock list-foundation-models --region "$REGION" \
       --query "modelSummaries[?modelId=='${BASE}'].modelId" --output text 2>/dev/null | grep -q .; then
    ok "$MID  (base model visible)"
  else
    bad "$MID  (base model NOT visible in $REGION)"
  fi
done
echo
info "The 'us.*' ids are CROSS-REGION INFERENCE PROFILES. List yours with:"
info "  aws bedrock list-inference-profiles --region $REGION \\"
info "     --query 'inferenceProfileSummaries[].inferenceProfileId' --output table"
info ""
info "Visibility is NOT the same as access. Model access is granted per-account in"
info "the console:  Bedrock -> Model access -> Manage model access -> enable"
info "Meta (Llama) and Mistral. Approval is usually instant."

echo
echo "=== 5. End-to-end through main.py (2 queries, ~\$0.01) ==="
if [ ! -x "$PY" ]; then
  bad "$PY not found — set PY=/path/to/python"
  exit 1
fi
AWS_REGION_NAME="$REGION" BEDROCK_MAX_WORKERS=1 \
  "$PY" main.py --model_name llama8b-bedrock --dataset_name realtimeqa \
  --top_k 6 --defense_method none --attack_method none --max_samples 2 \
  && ok "end-to-end call succeeded — you are ready to run the pilot" \
  || bad "end-to-end call failed; read the traceback above"
