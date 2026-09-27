#!/usr/bin/env bash
# The three remaining experiments for the SaTML 2027 submission.
#
#   ./run_final.sh profile     STEP 0 -- run this first. Dry run of cert cost.
#   ./run_final.sh certify      Experiment 1 -- certified accuracy, floor vs upstream
#   ./run_final.sh attacks      Experiment 2 -- RobustRAG's own PIA and Poison
#   ./run_final.sh nq           Experiment 3 -- the full 500 Natural Questions
#
# STAGED WRITES. attacks and nq used to point every run at ONE shared CSV. The
# contamination guard deletes the file it is given, so a twelfth attack run
# coming back dirty takes the eleven clean ones with it. That is not theoretical:
# it is how an earlier certification run lost its data. Each run now writes its
# own file under results/.staging/ and the canonical CSV is assembled only when
# every part is present and clean. A rerun skips parts already staged, so no
# clean result is ever paid for twice. make_paper_assets.py globs results/*.csv
# non-recursively, so staged parts are invisible to it until they are merged.
#
# RUN `profile` FIRST AND READ ITS OUTPUT. Certification enumerates a keyword
# powerset per query, capped at 2^14 = 16384 aggregation calls. The profiler
# reports the exact count without issuing a single CERTIFICATION call.
#
# What `profile` does still cost: it runs ordinary inference (isolated calls
# plus one aggregation call per query). Every one of those prompts is already
# in the cache -- results/fix_realtimeqa_k10_all.csv holds clean-arm, omega=1
# rows for both threshold modes on the three Llama/Mistral models, and the
# omega ladder populated the RealtimeQA k=10 cache for all six -- so with
# --use_cache the expected marginal spend is zero. If you have purged the cache
# it is roughly 100 queries x 11 calls x N models x 2 modes (the second mode
# reuses the first mode's isolated responses).
set -uo pipefail
cd "$(dirname "$0")"
# PY is overridable so the control flow can be exercised end to end against a
# stub runner (test_run_final.sh) without issuing a single Bedrock call.
if [ -z "${PY:-}" ]; then
  PY=.venv/bin/python
  [ -x "$PY" ] || PY=python3
fi
STAGE="results/.staging"
mkdir -p logs results "$STAGE"

# All six models now carry a certification arm. Every other table in the paper
# has six rows; a three-row certification table invites the obvious question.
# Models already measured are skipped by the resume check below, so extending
# this list only costs the runs that are actually missing.
# Override to narrow a rerun, e.g. MODELS_CERT="nova-pro-bedrock" ./run_final.sh certify
MODELS_CERT="${MODELS_CERT:-mistral7b-bedrock llama8b-bedrock llama70b-bedrock nova-micro-bedrock nova-lite-bedrock nova-pro-bedrock}"
MODELS_ATK="${MODELS_ATK:-mistral7b-bedrock llama8b-bedrock llama70b-bedrock nova-micro-bedrock nova-lite-bedrock nova-pro-bedrock}"

# Assemble a canonical CSV from staged parts. Refuses on a header mismatch and
# refuses to overwrite an existing canonical file, so this can never silently
# replace good data with a partial rerun.
#   merge_stage <outfile> <part> [<part>...]
merge_stage () {
  local out="$1"; shift
  local parts=("$@") hdr="" p
  if [ -e "$out" ]; then
    echo "   !! $out already exists -- refusing to overwrite. Move it aside first."
    return 1
  fi
  for p in "${parts[@]}"; do
    [ -s "$p" ] || { echo "   !! missing staged part $p -- not merging"; return 1; }
    if [ -z "$hdr" ]; then hdr=$(head -1 "$p")
    elif [ "$(head -1 "$p")" != "$hdr" ]; then
      echo "   !! header mismatch in $p -- not merging"; return 1
    fi
  done
  { echo "$hdr"; for p in "${parts[@]}"; do tail -n +2 "$p"; done; } > "$out"
  echo "   merged ${#parts[@]} parts -> $out ($(( $(wc -l < "$out") - 1 )) rows)"
}

preflight () {
  echo ">> preflight"
  $PY test_preflight.py || { echo "PREFLIGHT FAILED -- not spending money"; exit 1; }
}

# --------------------------------------------------------------------------
# THROTTLE SETTINGS. These are deliberately conservative, because a less
# conservative set (workers=2, delay=0.15s, retries=3) lost 222 of roughly 1,100
# calls on the first cache-cold run. That configuration looked safe only because
# the runs that preceded it were served from cache and barely touched the API, so
# the throttle had never met real load. The cold run failed from query 14 onward
# and never recovered: an account bucket held shut by continued pressure.
#
# It could not recover because back-off was PER THREAD. src/models.py now has a
# breaker shared by all workers: one 429 parks every worker until a common
# deadline, so the bucket gets a quiet window. With that in front, MORE retries
# is the safe direction -- attempts are spaced out instead of hammering.
# test_throttle.py exercises this offline against a limiter that punishes
# hammering: the old scheme loses 103 of 120 calls, this one loses none, and
# finishes sooner for not fighting the limiter.
#
#   MAX_WORKERS   2 -> 1     serialise; concurrency was never the throughput win
#   REQUEST_DELAY 0.15 -> 0.2s
#   NUM_RETRIES   3 -> 8     safe now that each attempt waits behind the breaker
#   BACKOFF_BASE  2s, doubling per attempt, full jitter, capped at 60s
#   ABORT_AFTER   25 hard failures ends the run instead of finishing and being
#                 discarded -- the old behaviour spent six minutes to produce
#                 data the guard then threw away
# Override on the command line if your account limits differ.
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
export BEDROCK_REQUEST_DELAY="${BEDROCK_REQUEST_DELAY:-0.2}"
export BEDROCK_NUM_RETRIES="${BEDROCK_NUM_RETRIES:-8}"
export BEDROCK_BACKOFF_BASE="${BEDROCK_BACKOFF_BASE:-2.0}"
export BEDROCK_BACKOFF_MAX="${BEDROCK_BACKOFF_MAX:-60.0}"
export BEDROCK_ABORT_AFTER_EMPTY="${BEDROCK_ABORT_AFTER_EMPTY:-25}"
COOLDOWN="${COOLDOWN:-30}"   # seconds between models, to let the limit reset

echo ">> throttle: workers=$BEDROCK_MAX_WORKERS delay=${BEDROCK_REQUEST_DELAY}s" \
     "retries=$BEDROCK_NUM_RETRIES backoff=${BEDROCK_BACKOFF_BASE}s-${BEDROCK_BACKOFF_MAX}s" \
     "abort-after=$BEDROCK_ABORT_AFTER_EMPTY cooldown=${COOLDOWN}s"
echo ">> uncached calls are paced; a cache-cold run is slow ON PURPOSE."

# HARD GUARD. main.py already detects empty responses and prints
# "THIS RUN IS CONTAMINATED". Until now nothing acted on it, so a contaminated
# run wrote a full-looking CSV and exited 0. This aborts instead, and removes
# the CSVs the bad run wrote so they cannot be mistaken for good data.
# $1 = log file, $2.. = result files to delete on contamination
assert_clean () {
  local log="$1"; shift
  local n_empty n_rate
  n_empty=$(grep -oE "!! [0-9]+ EMPTY RESPONSES" "$log" | grep -oE "[0-9]+" | tail -1)
  n_rate=$(grep -c "RateLimitError" "$log" || true)
  if grep -q "THIS RUN IS CONTAMINATED" "$log" || [ "${n_empty:-0}" -gt 0 ]; then
    echo
    echo "   !!!! CONTAMINATED RUN -- ABORTING !!!!"
    echo "   ${n_empty:-?} empty responses, ${n_rate} rate-limit errors in $log"
    echo "   Empty responses are counted as NON-ABSTAINING, so they inflate n,"
    echo "   raise mu, and bias certification. This data is not usable."
    echo
    for f in "$@"; do
      [ -e "$f" ] && { echo "   removing $f"; rm -f "$f"; }
    done
    echo
    echo "   Successful responses ARE cached, so a rerun re-issues only the"
    echo "   missing calls -- you are not paying again for what already landed."
    echo "   Retry with heavier throttling, e.g."
    echo "     BEDROCK_REQUEST_DELAY=1.0 BEDROCK_BACKOFF_BASE=4 COOLDOWN=120 $0 $CMD"
    exit 1
  fi
  echo "   clean: 0 empty responses, ${n_rate} rate-limit errors (retried OK)"
}

CMD="${1:-}"
case "$CMD" in

profile)
  # No CERTIFICATION call is issued. Ordinary inference still runs, but every
  # prompt it needs is already cached (see the header), so expect zero spend.
  preflight
  FAILED=0
  NRUNS=$(( $(echo $MODELS_CERT | wc -w) * 2 ))
  echo ">> profiling $NRUNS runs: $(echo $MODELS_CERT | wc -w) model(s) x 2 modes"
  for M in $MODELS_CERT; do
    for MODE in n floor2; do
      echo ">> [dry run] $M mode=$MODE"
      LOG="logs/profile_${M}_${MODE}.txt"
      $PY main.py --model_name "$M" --dataset_name realtimeqa --defense_method keyword \
        --attack_method none --corruption_size 1 --top_k 10 \
        --alpha 0.3 --beta 3.0 --group_size 1 --threshold_mode "$MODE" \
        --profile_certify --use_cache --no_vanilla \
        --per_query_csv "$STAGE/PROFILE_DISCARD.csv" > "$LOG" 2>&1
      RC=$?
      if [ "$RC" -ne 0 ]; then
        echo "   !! FAILED (exit $RC). Last lines of $LOG:"
        tail -5 "$LOG" | sed 's/^/      /'
        FAILED=$((FAILED+1))
      elif ! grep -qE '\[PROFILE\]' "$LOG"; then
        echo "   !! ran OK but produced NO [PROFILE] lines -- certification never"
        echo "      fired. Check --corruption_size and --attack_method in $LOG."
        FAILED=$((FAILED+1))
      elif grep -q "THIS RUN IS CONTAMINATED" "$LOG"; then
        # A throttled profile run is not merely incomplete, it is WRONG: empty
        # responses inflate n, which raises mu, which changes added_list, which
        # is the number the cost estimate is built from.
        echo "   !! CONTAMINATED -- the cost estimate from this run is invalid."
        echo "      $(grep -oE '!! [0-9]+ EMPTY RESPONSES' "$LOG" | tail -1)"
        FAILED=$((FAILED+1))
      else
        grep -E '\[PROFILE\]' "$LOG" | sed 's/^/   /'
      fi
      mv -f certify_cost_profile.json "logs/certify_cost_${M}_${MODE}.json" 2>/dev/null || true
    done
  done
  rm -f "$STAGE/PROFILE_DISCARD.csv"
  echo
  echo "=============================================================="
  if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED of $NRUNS profile runs did NOT produce a cost estimate."
    echo "DO NOT run 'certify' until every one of them reports [PROFILE]."
    exit 1
  fi
  echo "All $NRUNS profile runs reported. Sum the 'would issue N aggregation"
  echo "calls' lines: that total is what 'certify' will actually spend."
  echo "Do not run 'certify' until you have read and approved it."
  echo "=============================================================="
  ;;

certify)
  # Experiment 1. Certified accuracy under the upstream rule vs the floor.
  # This is what turns Prop. 3 from 'the algorithm carries over' into a measured
  # trade-off, which Section X currently lists as the open question.
  preflight
  # The prompt exists so nobody starts a five-figure run by accident. It also
  # blocks forever in an unattended run, so allow approval to be given up front
  # instead. CERTIFY_APPROVED=yes is an explicit decision recorded in the
  # command, not a way to skip the check.
  if [ "${CERTIFY_APPROVED:-}" = "yes" ]; then
    echo ">> call count approved in advance (CERTIFY_APPROVED=yes)"
  elif [ ! -t 0 ]; then
    echo "!! stdin is not a terminal and CERTIFY_APPROVED is not set."
    echo "   Refusing to spend money without an approval. Re-run with"
    echo "   CERTIFY_APPROVED=yes once you have read the profiler output."
    exit 1
  else
    read -r -p "Have you read the profiler output and approved the call count? [yes/N] " ok
    [ "$ok" = "yes" ] || { echo "aborted"; exit 1; }
  fi
  # PER-MODEL OUTPUT FILES. Earlier every model appended to one shared CSV, so
  # when llama70b came back contaminated the guard deleted the file -- taking
  # the already-clean mistral and llama8b rows with it. One file per model means
  # a bad model loses only its own data, and a rerun SKIPS models already on
  # disk, so nothing clean is ever paid for twice.
  for M in $MODELS_CERT; do
    DID_WORK=0
    for MODE in n floor2; do
      OUT="results/certify_realtimeqa_${MODE}_${M}.csv"
      CERT="results/certify_${MODE}_${M}.csv"
      # RESUME NEEDS BOTH FILES, NOT JUST $CERT. main.py writes certify_csv
      # FIRST and per_query_csv SECOND, so an interruption between the two
      # leaves $CERT present and $OUT missing. Skipping on $CERT alone would
      # then make that gap permanent: n_retained lives only in $OUT, and
      # make_paper_assets recovers the degenerate-branch count from it, so the
      # paper would quietly report a smaller number with nothing to flag it.
      # A half-written pair is unusable and cheap to redo (responses are
      # cached), so delete it and run again rather than trusting it.
      NC=-1; NO=-1
      [ -s "$CERT" ] && NC=$(( $(wc -l < "$CERT") - 1 ))
      [ -s "$OUT" ]  && NO=$(( $(wc -l < "$OUT") - 1 ))
      if [ "$NC" -gt 0 ] && [ "$NC" -eq "$NO" ]; then
        echo ">> $M mode=$MODE -- already present and complete, skipping ($NC rows)"
        continue
      fi
      if [ "$NC" -ge 0 ] || [ "$NO" -ge 0 ]; then
        echo ">> $M mode=$MODE -- INCOMPLETE pair (cert=$NC per-query=$NO);"
        echo "   discarding both and redoing; cached responses make this cheap"
        rm -f "$CERT" "$OUT"
      fi
      DID_WORK=1
      echo ">> $M mode=$MODE -> $CERT"
      : > "logs/certify_${M}_${MODE}.txt"   # fresh log, so the guard cannot read a stale verdict
      $PY main.py --model_name "$M" --dataset_name realtimeqa --defense_method keyword \
        --attack_method none --corruption_size 1 --top_k 10 \
        --alpha 0.3 --beta 3.0 --group_size 1 --threshold_mode "$MODE" \
        --use_cache --no_vanilla \
        --per_query_csv "$OUT" \
        --certify_csv "$CERT" >> "logs/certify_${M}_${MODE}.txt" 2>&1 \
        || { echo "   !! FAILED -- stopping so a partial CSV is not mistaken for a complete one"; tail -5 "logs/certify_${M}_${MODE}.txt"; exit 1; }
      assert_clean "logs/certify_${M}_${MODE}.txt" "$OUT" "$CERT"
      tail -2 "logs/certify_${M}_${MODE}.txt"
    done
    if [ "$DID_WORK" -eq 1 ]; then
      echo ">> cooldown ${COOLDOWN}s before the next model"; sleep "$COOLDOWN"
    fi
  done
  echo
  echo "certify complete. merge and analyse with:  $PY analyze_certify.py"
  ;;

attacks)
  # Experiment 2. RobustRAG's OWN attacks, in their harness, answering the
  # "you used your own injection, not theirs" objection directly.
  preflight
  PARTS=()
  for M in $MODELS_ATK; do
    DID_WORK=0
    for A in PIA Poison; do
      OUT="$STAGE/theirattacks_${M}_${A}.csv"
      PARTS+=("$OUT")
      if [ -s "$OUT" ]; then
        echo ">> $M attack=$A -- already staged and clean, skipping ($(( $(wc -l < "$OUT") - 1 )) rows)"
        continue
      fi
      DID_WORK=1
      LOG="logs/atk_${M}_${A}.txt"
      : > "$LOG"   # fresh log: the guard must not read a previous attempt's verdict
      echo ">> $M attack=$A -> $OUT"
      $PY main.py --model_name "$M" --dataset_name realtimeqa --defense_method keyword \
        --attack_method "$A" --corruption_size 1 --top_k 10 \
        --alpha 0.3 --beta 3.0 --group_size 1 --threshold_mode n \
        --use_cache --no_vanilla \
        --per_query_csv "$OUT" >> "$LOG" 2>&1 \
        || { echo "   !! FAILED -- stopping so a partial CSV is not mistaken for a complete one"; tail -5 "$LOG"; exit 1; }
      assert_clean "$LOG" "$OUT"
      tail -2 "$LOG"
    done
    if [ "$DID_WORK" -eq 1 ]; then
      echo ">> cooldown ${COOLDOWN}s before the next model"; sleep "$COOLDOWN"
    fi
  done
  echo
  merge_stage "results/theirattacks_realtimeqa_k10.csv" "${PARTS[@]}" \
    || { echo "   parts are staged; rerun to finish, nothing was lost"; exit 1; }
  ;;

nq)
  # Experiment 3. The remaining 300 Natural Questions items. open_nq.json holds
  # 500 and all 500 carry attacker targets; we previously used the first 200.
  preflight
  # data/open_nq.json holds 500 items and all 500 carry attacker targets. The
  # existing file covers q_idx 1..100 -- ONE HUNDRED distinct questions, not 200:
  # it has 1,000 rows because each query appears in both the k'=0 and k'=1 arms.
  # Reading that row count as a question count is an easy mistake and doubles the
  # apparent corpus size. So this run adds the remaining 400.
  PARTS=()
  for M in mistral7b-bedrock; do
    # MUST cover the same omega set as the file this supersedes
    # (omega_open_nq_k10.csv has 1..5). Dropping omega=5 here would silently
    # delete those rows from every asset once the 100-item file is superseded.
    for W in 1 2 3 4 5; do
      OUT="$STAGE/nq500_${M}_w${W}.csv"
      PARTS+=("$OUT")
      if [ -s "$OUT" ]; then
        echo ">> $M open_nq omega=$W -- already staged and clean, skipping ($(( $(wc -l < "$OUT") - 1 )) rows)"
        continue
      fi
      LOG="logs/nq500_${M}_w${W}.txt"
      : > "$LOG"
      echo ">> $M open_nq omega=$W (500 items) -> $OUT"
      $PY main.py --model_name "$M" --dataset_name open_nq --defense_method keyword \
        --attack_method KeywordInjection --corruption_size 1 --top_k 10 \
        --alpha 0.3 --beta 3.0 --group_size "$W" --threshold_mode n \
        --max_samples 500 --use_cache --no_vanilla \
        --per_query_csv "$OUT" >> "$LOG" 2>&1 \
        || { echo "   !! FAILED -- stopping so a partial CSV is not mistaken for a complete one"; tail -5 "$LOG"; exit 1; }
      assert_clean "$LOG" "$OUT"
      tail -2 "$LOG"
      echo ">> cooldown ${COOLDOWN}s"; sleep "$COOLDOWN"
    done
  done
  echo
  merge_stage "results/omega_open_nq_k10_full.csv" "${PARTS[@]}" \
    || { echo "   parts are staged; rerun to finish, nothing was lost"; exit 1; }
  ;;

*)
  sed -n '2,14p' "$0"
  exit 1
  ;;
esac

echo
echo "done. regenerate paper assets with:"
echo "  $PY make_paper_assets.py"
