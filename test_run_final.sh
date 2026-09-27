#!/usr/bin/env bash
# End-to-end dry run of run_final.sh against a STUB runner. Issues no Bedrock
# calls and costs nothing. Run this before every paid run.
#
#     ./test_run_final.sh
#
# WHY. Two bugs of this kind have already cost either money or data here:
#   - the certify block passed --attack_method KeywordInjection, and main.py
#     zeroes corruption_size when an attack is set, so certification never ran
#     and every `certified` value came back 0;
#   - every model appended to ONE shared CSV, so the contamination guard, firing
#     on the last model, deleted the earlier models' clean rows.
# Both are control-flow bugs. Neither is visible by reading the script, and both
# surface only after the money is spent. This exercises the control flow instead:
# resume, the guard, the merge, and the exact flags each run receives.
set -uo pipefail
cd "$(dirname "$0")"

SANDBOX=$(mktemp -d)
trap 'rm -rf "$SANDBOX"' EXIT
PASS=0; FAIL=0
ok   () { echo "  [ ok ] $*"; PASS=$((PASS+1)); }
bad  () { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

# BSD/macOS `wc -l` PADS ITS OUTPUT WITH SPACES: $(wc -l < f) is "      12",
# not "12", so every string comparison against a count silently fails there
# while passing on GNU. Arithmetic contexts $(( )) strip whitespace and were
# always fine; these were not. nlines/npipe normalise.
nlines () { wc -l < "$1" | tr -d '[:space:]'; }
npipe  () { tr -d '[:space:]'; }

# ---------------------------------------------------------------- stub runner
# Records the argv it was called with, then writes a plausible CSV and log so
# the calling block proceeds exactly as it would in a real run.
cat > "$SANDBOX/stub.py" <<'STUB'
import sys, os, csv
argv = sys.argv[1:]
def get(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default
os.makedirs(os.path.dirname(os.environ['ARGLOG']) or '.', exist_ok=True)
with open(os.environ['ARGLOG'], 'a') as fh:
    fh.write(' '.join(argv) + '\n')

# emulate main.py: an attack forces corruption_size to 0
attack = get('--attack_method', 'none')
csz = '0' if attack != 'none' else get('--corruption_size', '0')

PER = ['run','model','prompt_style','abstention_threshold','q_idx','n_retained',
       'n_abstained','n_groups','mu','alpha','beta','group_size','threshold_mode',
       'top_k','max_groups','early_abstain','k_prime','k_suppress','k_inject',
       'attack','n_surviving_keywords','attacker_kw_survived','attacker_kw_partial',
       'filter_only','defended_asr','defended_correct','defended_refusal',
       'defended_abstain','incorrect_answer','sample_keywords']
CERT = ['run','model','dataset','top_k','q_idx','attack','k_prime','alpha','beta',
        'group_size','threshold_mode','mu','certified','defended_correct']

def emit(path, cols, n):
    if not path: return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(cols)
        for i in range(1, n + 1):
            row = {c: '' for c in cols}
            row.update(model=get('--model_name'), q_idx=str(i),
                       alpha=get('--alpha'), beta=get('--beta'),
                       group_size=get('--group_size'), top_k=get('--top_k'),
                       threshold_mode=get('--threshold_mode'), attack=attack,
                       k_prime=csz, mu='2.0', certified='1',
                       dataset=get('--dataset_name'), n_retained='7',
                       defended_correct='1')
            w.writerow([row[c] for c in cols])

N = int(get('--max_samples', '100'))
if '--profile_certify' in argv:
    print('[PROFILE] DRY RUN: certification will make NO LLM calls')
    print('[PROFILE]   would issue 500 aggregation calls (0 queries skipped)')
    open('certify_cost_profile.json', 'w').write('[]')
emit(get('--per_query_csv'), PER, N)
emit(get('--certify_csv'), CERT, N)
print(f'wrote {N} per-query rows')
if os.environ.get('STUB_CONTAMINATE') == '1':
    print('!! 9 EMPTY RESPONSES')
    print('THIS RUN IS CONTAMINATED')
sys.exit(int(os.environ.get('STUB_RC', '0')))
STUB

# preflight is a separate real script; stub it out so this test stays offline
run_block () {   # run_block <cmd> [env assignments...]
  local cmd="$1"; shift
  rm -rf "$SANDBOX/wd"; mkdir -p "$SANDBOX/wd"
  cp run_final.sh "$SANDBOX/wd/"
  printf '#!/usr/bin/env python\n' > "$SANDBOX/wd/test_preflight.py"
  cp "$SANDBOX/stub.py" "$SANDBOX/wd/main.py"
  : > "$SANDBOX/args.txt"
  ( cd "$SANDBOX/wd" && env PY="$(command -v python3)" ARGLOG="$SANDBOX/args.txt" \
      COOLDOWN=0 "$@" bash run_final.sh "$cmd" ) > "$SANDBOX/out.txt" 2>&1
  echo $?
}

echo "=================================================================="
echo "A. every run gets the flags the paper claims"
echo "=================================================================="
RC=$(run_block attacks)
[ "$RC" = "0" ] && ok "attacks block exits 0" || { bad "attacks exited $RC"; tail -15 "$SANDBOX/out.txt"; }

NRUNS=$(nlines "$SANDBOX/args.txt")
[ "$NRUNS" = "12" ] && ok "attacks issued 12 runs (6 models x 2 attacks)" \
  || bad "attacks issued $NRUNS runs, expected 12"

# the paper says alpha=0.3 beta=3 k=10 omega=1 mode=n for the their-attacks arm
BADP=$(awk '!/--alpha 0.3/ || !/--beta 3.0/ || !/--top_k 10/ || !/--group_size 1/ \
            || !/--threshold_mode n/ || !/--dataset_name realtimeqa/' "$SANDBOX/args.txt" | wc -l | npipe)
[ "$BADP" = "0" ] && ok "all 12 carry alpha=0.3 beta=3.0 k=10 omega=1 mode=n realtimeqa" \
  || { bad "$BADP run(s) have wrong params"; grep -vE -- '--alpha 0.3' "$SANDBOX/args.txt" | head -3; }

PIA=$(grep -c -- '--attack_method PIA' "$SANDBOX/args.txt")
POI=$(grep -c -- '--attack_method Poison' "$SANDBOX/args.txt")
[ "$PIA" = "6" ] && [ "$POI" = "6" ] && ok "6 PIA + 6 Poison" || bad "PIA=$PIA Poison=$POI"

MERGED=$(( $(wc -l < "$SANDBOX/wd/results/theirattacks_realtimeqa_k10.csv") - 1 ))
[ "$MERGED" = "1200" ] && ok "merged canonical CSV has 1200 rows (12 x 100)" \
  || bad "merged CSV has $MERGED rows, expected 1200"
HDRS=$(head -1 "$SANDBOX/wd/results/theirattacks_realtimeqa_k10.csv" | tr ',' '\n' | wc -l | npipe)
DUPH=$(grep -c '^run,model,prompt_style' "$SANDBOX/wd/results/theirattacks_realtimeqa_k10.csv")
[ "$DUPH" = "1" ] && ok "exactly one header row in the merged CSV" \
  || bad "$DUPH header rows in the merged CSV"

echo
echo "=================================================================="
echo "B. certify never runs under an attack (the 20 Sep bug)"
echo "=================================================================="
RC=$(run_block certify STUB_YES=1 < /dev/null)
# certify prompts for confirmation; feed it 'yes'
rm -rf "$SANDBOX/wd"; mkdir -p "$SANDBOX/wd"
cp run_final.sh "$SANDBOX/wd/"; printf '#!/usr/bin/env python\n' > "$SANDBOX/wd/test_preflight.py"
cp "$SANDBOX/stub.py" "$SANDBOX/wd/main.py"; : > "$SANDBOX/args.txt"
( cd "$SANDBOX/wd" && env CERTIFY_APPROVED=yes PY="$(command -v python3)" ARGLOG="$SANDBOX/args.txt" \
    COOLDOWN=0 bash run_final.sh certify ) > "$SANDBOX/out.txt" 2>&1
CRC=$?
[ "$CRC" = "0" ] && ok "certify block exits 0" || { bad "certify exited $CRC"; tail -15 "$SANDBOX/out.txt"; }
ATK=$(grep -c -- '--attack_method none' "$SANDBOX/args.txt")
TOT=$(nlines "$SANDBOX/args.txt")
[ "$ATK" = "$TOT" ] && ok "all $TOT certify runs use --attack_method none" \
  || bad "only $ATK of $TOT certify runs use attack_method none"
CS=$(grep -c -- '--corruption_size 1' "$SANDBOX/args.txt")
[ "$CS" = "$TOT" ] && ok "all $TOT carry --corruption_size 1" || bad "corruption_size set on $CS of $TOT"
[ "$TOT" = "12" ] && ok "certify covers 6 models x 2 modes = 12 runs" || bad "certify issued $TOT runs"
CERTF=$(ls "$SANDBOX/wd"/results/certify_n_*.csv 2>/dev/null | wc -l | npipe)
[ "$CERTF" = "6" ] && ok "one certify_n_<model>.csv per model, not a shared file" \
  || bad "found $CERTF certify_n_*.csv files"

echo
echo "=================================================================="
echo "C. resume: a second run pays for nothing already on disk"
echo "=================================================================="
: > "$SANDBOX/args.txt"
( cd "$SANDBOX/wd" && env CERTIFY_APPROVED=yes PY="$(command -v python3)" ARGLOG="$SANDBOX/args.txt" \
    COOLDOWN=0 bash run_final.sh certify ) > "$SANDBOX/out2.txt" 2>&1
AGAIN=$(nlines "$SANDBOX/args.txt")
[ "$AGAIN" = "0" ] && ok "rerun issued ZERO runs (everything resumed)" \
  || bad "rerun issued $AGAIN runs -- that is paying twice"
# match on the durable part of the message, not its adjective: an earlier
# version of this line pinned "already present and clean" and broke when the
# resume check was tightened and the wording changed to "complete".
SKIPPED=$(grep -c "skipping" "$SANDBOX/out2.txt")
[ "$SKIPPED" = "12" ] && ok "all 12 reported as skipped" || bad "$SKIPPED reported skipped"

echo
echo "=================================================================="
echo "B2. unattended with no approval refuses to spend"
echo "=================================================================="
# The prompt is the last thing standing between a stray cron entry and a
# five-figure call count. When stdin is not a terminal it must refuse, not
# fall through to a default.
rm -rf "$SANDBOX/wd"; mkdir -p "$SANDBOX/wd"
cp run_final.sh "$SANDBOX/wd/"; printf '#!/usr/bin/env python\n' > "$SANDBOX/wd/test_preflight.py"
cp "$SANDBOX/stub.py" "$SANDBOX/wd/main.py"; : > "$SANDBOX/args.txt"
# `env -u` so the check holds even when the CALLER exported an approval.
# run_overnight.sh exports CERTIFY_APPROVED=yes before invoking the gate, which
# this test runs under; inheriting it made the refusal never fire and the gate
# failed only from inside the overnight script, never standalone.
( cd "$SANDBOX/wd" && env -u CERTIFY_APPROVED PY="$(command -v python3)" \
    ARGLOG="$SANDBOX/args.txt" COOLDOWN=0 \
    bash run_final.sh certify < /dev/null ) > "$SANDBOX/outb2.txt" 2>&1
B2RC=$?
[ "$B2RC" != "0" ] && ok "refuses without approval when stdin is not a terminal" \
  || bad "proceeded unattended with no approval"
[ "$(nlines "$SANDBOX/args.txt")" = "0" ] && ok "issued zero runs while refusing" \
  || bad "issued runs despite refusing"

echo
echo "=================================================================="
echo "C2. a half-written certify pair is redone, not trusted"
echo "=================================================================="
# main.py writes certify_csv BEFORE per_query_csv, so an interruption between
# them leaves the first present and the second missing. Resuming on the first
# alone would make that permanent, and n_retained (only in the second) is what
# make_paper_assets uses for the degenerate-branch count.
#
# SELF-CONTAINED ON PURPOSE. This block used to inherit the working directory
# from section C. Inserting a new section between them silently removed the
# state it depended on and the check started failing for the wrong reason, so
# it now builds what it needs.
rm -rf "$SANDBOX/wd"; mkdir -p "$SANDBOX/wd"
cp run_final.sh "$SANDBOX/wd/"; printf '#!/usr/bin/env python\n' > "$SANDBOX/wd/test_preflight.py"
cp "$SANDBOX/stub.py" "$SANDBOX/wd/main.py"
( cd "$SANDBOX/wd" && env CERTIFY_APPROVED=yes PY="$(command -v python3)" \
    ARGLOG="$SANDBOX/args.txt" COOLDOWN=0 MODELS_CERT="mistral7b-bedrock" \
    bash run_final.sh certify ) > /dev/null 2>&1
rm -f "$SANDBOX/wd/results/certify_realtimeqa_n_mistral7b-bedrock.csv"
: > "$SANDBOX/args.txt"
( cd "$SANDBOX/wd" && env CERTIFY_APPROVED=yes PY="$(command -v python3)" ARGLOG="$SANDBOX/args.txt" \
    COOLDOWN=0 MODELS_CERT="mistral7b-bedrock" bash run_final.sh certify ) > "$SANDBOX/outc2.txt" 2>&1
grep -q "INCOMPLETE pair" "$SANDBOX/outc2.txt" \
  && ok "half-written pair detected and redone" || bad "half-written pair was trusted"
REDONE=$(grep -c . "$SANDBOX/args.txt" | npipe)
[ "$REDONE" = "1" ] && ok "exactly the broken mode was rerun ($REDONE run)" \
  || bad "$REDONE runs issued, expected 1"
[ -s "$SANDBOX/wd/results/certify_realtimeqa_n_mistral7b-bedrock.csv" ] \
  && ok "the missing per-query file now exists" || bad "per-query file still missing"

echo
echo "=================================================================="
echo "D. the guard destroys only the contaminated run's own data"
echo "=================================================================="
rm -rf "$SANDBOX/wd"; mkdir -p "$SANDBOX/wd"
cp run_final.sh "$SANDBOX/wd/"; printf '#!/usr/bin/env python\n' > "$SANDBOX/wd/test_preflight.py"
cp "$SANDBOX/stub.py" "$SANDBOX/wd/main.py"; : > "$SANDBOX/args.txt"
( cd "$SANDBOX/wd" && env PY="$(command -v python3)" ARGLOG="$SANDBOX/args.txt" \
    COOLDOWN=0 STUB_CONTAMINATE=1 bash run_final.sh attacks ) > "$SANDBOX/out3.txt" 2>&1
GRC=$?
[ "$GRC" = "1" ] && ok "contaminated run aborts with exit 1" || bad "exit was $GRC, expected 1"
grep -q "CONTAMINATED RUN -- ABORTING" "$SANDBOX/out3.txt" \
  && ok "guard fired and said so" || bad "guard did not fire"
[ ! -e "$SANDBOX/wd/results/theirattacks_realtimeqa_k10.csv" ] \
  && ok "no canonical CSV written from a contaminated run" \
  || bad "a contaminated run produced a canonical CSV"

# now the important half: earlier CLEAN parts must survive a later dirty run
rm -rf "$SANDBOX/wd"; mkdir -p "$SANDBOX/wd"
cp run_final.sh "$SANDBOX/wd/"; printf '#!/usr/bin/env python\n' > "$SANDBOX/wd/test_preflight.py"
cp "$SANDBOX/stub.py" "$SANDBOX/wd/main.py"
( cd "$SANDBOX/wd" && env PY="$(command -v python3)" ARGLOG="$SANDBOX/args.txt" COOLDOWN=0 \
    MODELS_ATK="mistral7b-bedrock" bash run_final.sh attacks ) > /dev/null 2>&1
CLEAN_PARTS=$(ls "$SANDBOX/wd/results/.staging"/theirattacks_* 2>/dev/null | wc -l | npipe)
mv "$SANDBOX/wd/results/theirattacks_realtimeqa_k10.csv" "$SANDBOX/wd/results/.keep" 2>/dev/null
( cd "$SANDBOX/wd" && env PY="$(command -v python3)" ARGLOG="$SANDBOX/args.txt" COOLDOWN=0 \
    STUB_CONTAMINATE=1 MODELS_ATK="mistral7b-bedrock llama8b-bedrock" \
    bash run_final.sh attacks ) > "$SANDBOX/out4.txt" 2>&1
STILL=$(ls "$SANDBOX/wd/results/.staging"/theirattacks_mistral7b-bedrock_* 2>/dev/null | wc -l | npipe)
[ "$STILL" = "$CLEAN_PARTS" ] && [ "$STILL" -gt 0 ] \
  && ok "earlier clean parts ($STILL) survived a later contaminated run" \
  || bad "clean parts went from $CLEAN_PARTS to $STILL -- the guard ate good data"

echo
echo "=================================================================="
echo "E. nq covers all five omegas and the full 500"
echo "=================================================================="
rm -rf "$SANDBOX/wd"; mkdir -p "$SANDBOX/wd"
cp run_final.sh "$SANDBOX/wd/"; printf '#!/usr/bin/env python\n' > "$SANDBOX/wd/test_preflight.py"
cp "$SANDBOX/stub.py" "$SANDBOX/wd/main.py"; : > "$SANDBOX/args.txt"
( cd "$SANDBOX/wd" && env PY="$(command -v python3)" ARGLOG="$SANDBOX/args.txt" \
    COOLDOWN=0 bash run_final.sh nq ) > "$SANDBOX/out5.txt" 2>&1
NRC=$?
[ "$NRC" = "0" ] && ok "nq block exits 0" || { bad "nq exited $NRC"; tail -15 "$SANDBOX/out5.txt"; }
for W in 1 2 3 4 5; do
  grep -q -- "--group_size $W" "$SANDBOX/args.txt" || bad "omega=$W missing from nq"
done
grep -q -- "--group_size 5" "$SANDBOX/args.txt" && ok "omega 1..5 all present (omega=5 is the one that was droppable)"
MS=$(grep -c -- '--max_samples 500' "$SANDBOX/args.txt")
[ "$MS" = "5" ] && ok "all five nq runs request 500 samples" || bad "max_samples 500 on $MS of 5"
NQROWS=$(( $(wc -l < "$SANDBOX/wd/results/omega_open_nq_k10_full.csv") - 1 ))
[ "$NQROWS" = "2500" ] && ok "merged nq CSV has 2500 rows (5 omegas x 500)" \
  || bad "merged nq CSV has $NQROWS rows, expected 2500"

echo
echo "=================================================================="
echo "F. staged parts stay invisible to asset generation"
echo "=================================================================="
python3 - "$SANDBOX/wd/results" <<'CHK'
import glob, os, sys
R = sys.argv[1]
seen = glob.glob(os.path.join(R, '*.csv'))
staged = glob.glob(os.path.join(R, '.staging', '*.csv'))
assert staged, 'test bug: no staged files to check against'
leaked = [p for p in seen if '.staging' in p]
print(f"  [{'ok' if not leaked else 'FAIL'}] results/*.csv sees "
      f"{len(seen)} file(s), none from .staging ({len(staged)} staged)")
sys.exit(1 if leaked else 0)
CHK
[ $? = 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo
echo "=================================================================="
if [ "$FAIL" -gt 0 ]; then
  echo "$FAIL CHECK(S) FAILED -- do not spend money until these pass"
  exit 1
fi
echo "ALL $PASS CHECKS PASSED -- control flow is safe to run for real"
