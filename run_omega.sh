#!/usr/bin/env bash
# =============================================================================
# PASSAGE GROUP SIZE omega  —  does the defense's own utility knob break it?
#
# THE CLAIM. RobustRAG's paper studies passage group size omega (Fig. 6) and
# recommends it as the knob for benign performance: "with omega=3, we reduce the
# benign performance drop from 7% to 0% while maintaining non-trivial certifiable
# robustness". Upstream's released code implements omega=1 only.
#
# But omega caps the number of isolated groups at ceil(k/omega), and the keyword
# threshold is mu = min(alpha*n, beta) where n <= ceil(k/omega). At k=10,
# alpha=0.3, beta=3 the arithmetic is:
#
#   omega  groups  mu(no abstentions)  mu(one group abstains)  k'=1 survives?
#     1      10          3.00                  2.70            needs 7 abstentions
#     2       5          1.50                  1.20            needs 2 abstentions
#     3       4          1.20                  0.90            ONE abstention
#     4       3          0.90                  0.60            ALWAYS (mu<1)
#     5       2          0.60                  0.30            ALWAYS (mu<1)
#
# So the prediction is a ladder, not a single point:
#   - at omega>=4, mu < 1 unconditionally -> a keyword seen ONCE always survives,
#     and a single poisoned passage flips the output on every query;
#   - at omega=3 (the RECOMMENDED setting) one abstaining group is enough.
#
# If this holds, the knob the authors recommend for benign performance is the
# knob that removes the filter. That is a sharper claim than our corpus-coverage
# result, because it does not depend on the corpus at all -- it is forced by the
# configuration.
#
# WHAT THIS MEASURES. Two curves against omega, on the same queries:
#     clean accuracy         (the utility omega is supposed to buy)
#     attacker survival k'=1 (the security it costs)
# The figure is the pair crossing.
#
# COST. Grouped prompts are new (not cached), but there are FEWER of them:
# ceil(10/omega) isolated calls per query instead of 10. Roughly $1 for one
# model over the whole omega grid. omega=1 is already cached from earlier runs.
#
# Output: results/omega.csv  ->  analyze_omega.py
# =============================================================================
set -uo pipefail
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export BEDROCK_MAX_WORKERS="${BEDROCK_MAX_WORKERS:-1}"
# Prefer the project virtualenv, but fall back so a copy without one still
# runs. Without the fallback this is "no such file: .venv/bin/python".
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
N="${N:-100}"
K="${K:-10}"
DATASET="${DATASET:-realtimeqa}"
MODELS="${MODELS:-mistral7b-bedrock}"
# alpha is a PARAMETER, not a constant. RobustRAG's paper states
# alpha=0.2 for short-answer QA while its released code defaults to 0.3,
# and the vacuity boundary ceil(k/omega) <= 1/alpha moves a long way
# between them: at alpha=0.2, k=10, omega=2 already gives mu=1.00.
ALPHA="${ALPHA:-0.3}"
# k goes in the filename: the cache is keyed by top_k, the analyser refuses to
# mix k values, and the whole point of the k=12 control is that group structure
# differs. Never let two k's share a CSV.
# The default filename keeps its historical form at alpha=0.3 so existing
# result files stay reachable; any other alpha gets its own file, because
# analyze_omega.py applies the FIRST row's alpha to the whole file.
if [ "$ALPHA" = "0.3" ]; then
  CSV="${CSV:-results/omega_${DATASET}_k${K}.csv}"
else
  CSV="${CSV:-results/omega_${DATASET}_k${K}_a${ALPHA}.csv}"
fi

# omega grid. Defaults suit k=10: 1 = upstream's implemented setting,
# 3 = the paper's recommended setting, 4 and 5 cross the mu<1 line.
#
# For the k=12 CONTROL use OMEGAS="1 2 3 4 6": 12 divides evenly by all of
# those, so every group is the same size and the poisoned passage is never
# left alone in a short remainder group. That isolates the threshold effect
# from the in-group contest, which k=10 cannot do (10 = 3+3+3+1 strands the
# poison by itself at omega=3).
OMEGAS=( ${OMEGAS_OVERRIDE:-1 2 3 4 5} )
# arms: clean measures the utility omega buys, k'=1 measures what it costs.
# k'=1 specifically: it is BELOW beta=3, so under omega=1 it should be filtered
# on almost every query. Any survival at larger omega is caused by omega alone.
#
# ARMS_OVERRIDE="none 0" runs the CLEAN ARM ONLY. That is the right setting for
# hotpotqa, which ships `incorrect answer` == [] on every item and therefore has
# no attacker target at all -- an attack arm there would score zero by
# construction. main.py refuses such a run outright; this is the supported way
# to get hotpotqa's utility half.
ARMS=( "none 0" "KeywordInjection 1" )
[ -n "${ARMS_OVERRIDE:-}" ] && ARMS=( "$ARMS_OVERRIDE" )

echo "### preflight"
$PY test_preflight.py >/dev/null 2>&1 || { echo "!! preflight FAILED -- run it directly to see why"; exit 1; }
echo "    passed"

echo
echo "### predicted cost (no calls made)"
for M in $MODELS; do
  echo "--- $M  (omega=1 reference; larger omega makes FEWER, longer calls)"
  if ! $PY check_cache_coverage.py --model_name "$M" --dataset_name "$DATASET" \
        --top_k "$K" --attack_method KeywordInjection --corruption_size 1 \
        --max_samples "$N" --no_vanilla | tail -6; then
    echo "!! cost estimate FAILED -- refusing to continue blind"
    exit 1
  fi
done
echo
echo "NOTE: the estimator models omega=1. For omega>1 the isolated calls are"
echo "NEW (grouped prompts are not in the cache) but fewer per query:"
for W in "${OMEGAS[@]}"; do
  printf "  omega=%-2s -> %s isolated calls/query\n" "$W" "$(( (K + W - 1) / W ))"
done
echo "alpha=$ALPHA  (vacuous for all queries when ceil(k/omega) <= $(python3 -c "print(1/$ALPHA)"))"
echo "dataset=$DATASET  k=$K  omegas: ${OMEGAS[*]}  arms: ${#ARMS[@]}  models: $(echo $MODELS | wc -w)  -> $CSV"
read -r -p "proceed and spend on Bedrock? [y/N] " reply
[ "$reply" = "y" ] || { echo "aborted"; exit 0; }

mkdir -p results
# Do NOT rotate the CSV aside. Rotating before the run means an aborted run,
# or a run covering only some models, strands the earlier data under a .bak
# nobody looks at -- that happened three times. Instead APPEND, and refuse to
# append into a file whose rows would be silently mixed with a different k.
# (Model and threshold_mode legitimately differ between runs and are columns,
# so the analysers can separate them; k cannot be, because every prediction is
# a function of ceil(k/omega).)
if [ -f "$CSV" ]; then
  EXIST_K=$($PY csv_top_k.py "$CSV")
  if [ -n "$EXIST_K" ] && [ "$EXIST_K" != "$K" ]; then
    echo "!! $CSV already holds top_k=$EXIST_K but this run is k=$K."
    echo "   Refusing to mix. Move it aside yourself or set CSV=..."
    exit 1
  fi
  EXIST_A=$($PY csv_alpha.py "$CSV")
  if [ -n "$EXIST_A" ] && [ "$EXIST_A" != "$ALPHA" ]; then
    echo "!! $CSV already holds alpha=$EXIST_A but this run is alpha=$ALPHA."
    echo "   analyze_omega.py applies the first row's alpha to the whole file,"
    echo "   so mixing would mispredict every boundary. Refusing."
    exit 1
  fi
  echo "### appending to existing $CSV ($(wc -l < "$CSV") lines)"
fi

FAILED_CELLS=0
ROWS_BEFORE=$( [ -f "$CSV" ] && wc -l < "$CSV" || echo 0 )

for M in $MODELS; do
  for ARM in "${ARMS[@]}"; do
    set -- $ARM; A="$1"; KP="$2"
    for W in "${OMEGAS[@]}"; do
      echo
      echo "############ $M  $A k'=$KP  omega=$W"
      $PY main.py --model_name "$M" --dataset_name "$DATASET" --top_k "$K" \
        --defense_method keyword --alpha "$ALPHA" --beta 3 --abstention_threshold 1 \
        --group_size "$W" \
        --attack_method "$A" --corruption_size "$KP" \
        --max_samples "$N" --per_query_csv "$CSV" --no_vanilla --use_cache \
        || { echo "!! failed (continuing)"; FAILED_CELLS=$((FAILED_CELLS+1)); }
    done
  done
done

echo
ROWS_AFTER=$( [ -f "$CSV" ] && wc -l < "$CSV" || echo 0 )
NEW_ROWS=$(( ROWS_AFTER - ROWS_BEFORE ))
echo "### $FAILED_CELLS cell(s) failed; $NEW_ROWS new rows written"
# A failed cell means the CSV is MISSING that condition. The analyser cannot
# tell a missing condition from one that was never requested -- it would print
# a clean-looking table built from whatever rows happen to be in the file,
# including rows from EARLIER runs. That is how a completely broken run gets
# read as a successful one. Refuse, and say what to do.
if [ "$FAILED_CELLS" -gt 0 ]; then
  echo
  echo "!! REFUSING TO ANALYSE: $FAILED_CELLS cell(s) failed."
  echo "   The table would silently mix this run's partial rows with older ones."
  echo "   Fix the failures and re-run; completed cells are cached and will not"
  echo "   be re-billed. To look anyway, run the analyser by hand:"
  echo "       $PY $(basename "$0" .sh | sed 's/^run_/analyze_/').py $CSV"
  exit 1
fi
if [ "$NEW_ROWS" -eq 0 ]; then
  echo "!! No new rows were written. Nothing to analyse."
  exit 1
fi
echo "### analysing"
$PY analyze_omega.py "$CSV"
