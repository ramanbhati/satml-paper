#!/usr/bin/env bash
# The gate. Everything that can be checked without spending money, in one place.
#
#     ./gate.sh
#
# Exit 0 means every offline check passes and a paid run is safe to start.
# Exit non-zero means it is not. There is no partial credit.
#
# WHY IT IS ONE COMMAND. The suites below can each be run separately, but a
# check that has to be remembered is a check that gets skipped. Bundling them
# means the precondition for spending money is a single exit status.
#
# WHAT IT DOES NOT ESTABLISH. Passing here is necessary, not sufficient. The
# suites cover the states the repository has already been in. A bug that only
# appears in a state a new run will CREATE passes the gate and then fires, which
# is what test_scenarios.sh exists to cover: it simulates prospective states
# rather than the current one. A green gate means the known failure modes are
# covered and says nothing about the rest.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-}"
if [ -z "$PY" ]; then PY=.venv/bin/python; [ -x "$PY" ] || PY=python3; fi

RESULTS=(); FAILED=0
GATE_TMP=$(mktemp -d); trap 'rm -rf "$GATE_TMP"' EXIT

# PER-CHECK TIMEOUT, PORTABLY. A hung check is worse than a failing one:
# run_overnight.sh gates on this, so a hang burns the whole unattended window.
#
# `timeout` is GNU coreutils and is absent from a stock macOS, so hard-coding it
# makes every check fail with "timeout: command not found" there while passing on
# Linux. Detect what is available and degrade to running unbounded rather than
# failing the whole gate: no timeout is still better than no checks.
GATE_TIMEOUT="${GATE_TIMEOUT:-600}"
TIMEOUT_CMD=""
if command -v timeout  >/dev/null 2>&1; then TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_CMD="gtimeout"   # brew coreutils
fi
if [ -z "$TIMEOUT_CMD" ]; then
  echo "note: no timeout(1) available, so checks run unbounded."
  echo "      'brew install coreutils' provides gtimeout if you want the limit."
fi

run () {                       # run <label> <command...>
  local label="$1"; shift
  # portable slug: bash substitution, no tr -c (which differs on BSD)
  local slug="${label//[^a-zA-Z0-9]/_}"
  local out="$GATE_TMP/$slug.txt"
  local rc=0
  printf '\n\033[1m>> %s\033[0m\n' "$label"
  if [ -n "$TIMEOUT_CMD" ]; then
    "$TIMEOUT_CMD" "$GATE_TIMEOUT" "$@" > "$out" 2>&1 || rc=$?
  else
    "$@" > "$out" 2>&1 || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    tail -1 "$out" | sed 's/^/   /'
    RESULTS+=("  PASS  $label")
  elif [ "$rc" -eq 124 ]; then
    echo "   TIMED OUT after ${GATE_TIMEOUT}s"
    RESULTS+=("  HUNG  $label (timed out)")
    FAILED=$((FAILED+1))
  else
    echo "   FAILED:"
    tail -12 "$out" | sed 's/^/      /'
    RESULTS+=("  FAIL  $label")
    FAILED=$((FAILED+1))
  fi
}

echo "======================================================================"
echo " OFFLINE GATE -- no AWS calls, no cost"
echo "======================================================================"

run "syntax: run_final.sh"        bash -n run_final.sh
run "syntax: test_run_final.sh"   bash -n test_run_final.sh
run "syntax: test_scenarios.sh"   bash -n test_scenarios.sh
run "buildable (requirements parse, downloads documented)" $PY check_buildable.py
run "preflight (data, defenses, thresholds)"   $PY test_preflight.py
run "control flow (resume, guard, merge, flags)" ./test_run_final.sh
run "future states (six models, half-written pairs, partial files)" ./test_scenarios.sh
run "rate-limit handling converges"            $PY test_throttle.py
run "regenerate every table and figure"        $PY make_paper_assets.py
run "every headline number re-derived"         $PY audit_numbers.py

# Paper-side checks that need no LaTeX toolchain.
#
# SKIPPED WHEN THE PAPER IS NOT THERE. A released copy of this repository does
# not carry the paper tree, so this check has nothing to read and a hard failure
# here would make the gate red for every reader on a clean checkout. Nothing in
# the code depends on it; it guards the manuscript against the code, not the
# reverse.
if [ ! -d ../../../paper-satml2027 ]; then
  printf '\n\033[1m>> paper structure\033[0m\n'
  echo "   skipped: the paper tree is not present beside this repository."
  echo "            This check validates the manuscript against the results and"
  echo "            is only meaningful where both are checked out together."
  RESULTS+=("  SKIP  paper structure (paper tree absent)")
else
run "paper structure (refs, inputs, macros, em-dashes)" $PY - <<'PYEOF'
import re, os, glob, sys
P = '../../../paper-satml2027'
# ONLY THE SECTIONS main.tex ACTUALLY INPUTS. Globbing sections/*.tex would also
# collect labels from files commented out of the build, so a \ref pointing at a
# disabled section resolves here and then fails at LaTeX time. 07-deadends.tex is
# commented out and still defines \label{sec:deadends}, which is exactly the case
# this avoids.
_main = open(os.path.join(P, 'main.tex')).read()
_live = [l.split('%', 1)[0] for l in _main.splitlines()]
secs = [os.path.join(P, m + '.tex')
        for m in re.findall(r'\\input\{(sections/[^}]*)\}', '\n'.join(_live))]
secs = [f for f in secs if os.path.exists(f)] + [os.path.join(P, 'main.tex')]
alls = '\n'.join('\n'.join(l for l in open(f).read().splitlines()
                 if not l.lstrip().startswith('%')) for f in secs)
allg = '\n'.join(open(f).read() for f in glob.glob(os.path.join(P, 'generated', '*.tex')))
blob = alls + '\n' + allg
bad = []
missing = set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]*)\}', blob)) \
    - set(re.findall(r'\\label\{([^}]*)\}', blob))
if missing: bad.append(f'broken refs: {sorted(missing)}')
cites = {c.strip() for g in re.findall(r'\\cite[a-z]*\{([^}]*)\}', blob) for c in g.split(',')}
bib = set(re.findall(r'@\w+\{([^,]+),', open(os.path.join(P, 'refs.bib')).read()))
if cites - bib: bad.append(f'broken cites: {sorted(cites - bib)}')
for i in set(re.findall(r'\\input\{([^}]*)\}', blob)):
    if not os.path.exists(os.path.join(P, i + '.tex')): bad.append(f'missing input: {i}')
defd = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', blob))
used = {u for u in re.findall(r'\\([a-z]+)(?![A-Za-z])', alls)
        if u.startswith(('cert', 'theirs', 'nobs'))}
if used - defd: bad.append(f'undefined macros: {sorted(used - defd)}')
if '---' in blob: bad.append(f'em-dashes present: {blob.count("---")}')
abst = open(os.path.join(P, 'sections', '00-abstract.tex')).read()
body = '\n'.join(l for l in abst.splitlines() if not l.lstrip().startswith('%'))
if '\\' + 'nobs' in body: bad.append('abstract uses \\nobs but is frozen at registration')
# An earlier framing of this paper held that the certificate was stated over the
# wrong quantity. That is false: RobustRAG's certifier does compute mu and does
# decline when mu <= k'. Section I now says "The gap is not in the analysis" and
# the conclusion grants that the certificate is sound, so any return of the old
# wording contradicts the paper on its own front page.
_retired = ['wrong variable', 'does not govern its failure',
            'stated over a variable that does not govern']
# STRIP COMMENTS FIRST. This scans for retired framing that would reach a
# reader, so it must read what renders. main.tex deliberately RECORDS the
# retired titles in a comment so they are not reintroduced by accident, and
# reading the file raw turned that record into a failure.
_mainsrc = '\n'.join(l.split('%', 1)[0] for l in
                     open(os.path.join(P, 'main.tex')).read().splitlines())
_front = body + '\n' + _mainsrc
for _r in _retired:
    if _r.lower() in _front.lower():
        bad.append('retired "certificate is misdirected" framing is back: %r' % _r)
# THE ABSTRACT IS FROZEN AS SUBMITTED. The CfP allows no substantial change to
# the registered abstract, so the useful guard is no longer a list of phrases to
# avoid but a fingerprint of the registered text. Normalised on whitespace, so
# reflowing lines is fine and changing a word is not. If this fires, either the
# abstract was edited (revert it) or the registered text genuinely changed at the
# venue (update the constant, and say so in the file header).
# The TITLE is frozen at registration for the same reason as the abstract, so it
# gets the same treatment. Compared on collapsed whitespace, since \\ and a line
# break inside \title{} are formatting.
_REGISTERED_TITLE = ('One Vote Is Enough: When One Agreement Suffices in '
                     'Certified RAG Defenses')
_tm = re.search(r'\\title\{(.+?)\}', _mainsrc, re.S)
if not _tm:
    bad.append('no \\title{} found in main.tex')
else:
    _got_title = re.sub(r'\s+', ' ', _tm.group(1).replace('\\\\', ' ')).strip()
    if _got_title != _REGISTERED_TITLE:
        bad.append('title differs from the registered one.\n'
                   '        got:      %s\n        expected: %s'
                   % (_got_title, _REGISTERED_TITLE))

_FROZEN = 'b325eed2997604ab'
import hashlib
_norm = re.sub(r'\s+', ' ', body).strip()
_got = hashlib.sha256(_norm.encode()).hexdigest()[:16]
if _got != _FROZEN:
    bad.append('the registered abstract has been edited: fingerprint %s, expected %s'
               % (_got, _FROZEN))

# Checked against pristine upstream, git HEAD~1:src/defense.py. Inference
# filters with min(alpha*n,beta); certify() tests min(alpha*(n+k'),beta).
# Different expressions, strictly different on 476 of 600 queries, and
# Section VIII states as much. So the abstract must not say inference applies
# "the same filter". Section VIII's own "not describing the same filter" is the
# negation and has to stay legal, hence a check scoped to the abstract that
# matches only the affirmative form. Upstream also does emit the threshold
# (logger.debug 'count_threshold'), so "reports nothing" is refutable by one
# grep where "raises no warning" is not.
for _p in ['the same filter anyway', 'applies the same filter',
           'reports nothing']:
    if _p in body.lower():
        bad.append('abstract clause refuted by upstream defense.py: %r '
                   '(see the header comment in 00-abstract.tex)' % _p)
if bad:
    for b in bad: print('  !!', b)
    sys.exit(1)
print('paper structure clean: refs, cites, inputs, macros, no em-dashes')
PYEOF
fi

echo
echo "======================================================================"
printf '%s\n' "${RESULTS[@]}"
echo "======================================================================"
if [ "$FAILED" -gt 0 ]; then
  echo " $FAILED CHECK(S) FAILED -- do not start a paid run"
  exit 1
fi
echo " ALL CHECKS PASSED"
echo
echo " Necessary, not sufficient: these suites cover states the repository has"
echo " already been in. A run that creates a new state needs that state added to"
echo " test_scenarios.sh before it is trusted."
