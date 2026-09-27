#!/usr/bin/env python
"""
Offline preflight for the availability experiments. Makes NO network calls and
costs nothing. Run this before every AWS run.

    python3 test_preflight.py

Checks, in order:
  A. jailbreak prompt file is present and sane
  B. GuardrailTrigger builds correct payloads at every k' in the sweep
  C. refusal / abstention detectors behave on realistic model output
  D. prompt_style swaps only the 'qa' template and nothing else
  E. cache keys differ between prompt styles  <-- prevents a FALSE D2 result
  F. cache actually prevents re-billing on a repeat run
  G. CLI defaults match what the run scripts assume
  H. run scripts never pass --no_vanilla (that flag hid the control earlier)
  I. predicted Bedrock call volume and cost for the scripts
"""
import glob
import importlib.util
import json
import os
import re
import sys
import types

FAIL = []
WARN = []


def shell_runs(path):
    """Extract main.py invocations from a shell script: strip heredocs and
    comments, join line continuations, then pull out each command."""
    raw = open(path).read()
    # drop heredoc bodies (cat <<'EOF' ... EOF)
    raw = re.sub(r"<<'?(\w+)'?.*?\n.*?\n\1\n", "\n", raw, flags=re.S)
    # join line continuations FIRST, so a wrapped command becomes one line
    raw = raw.replace("\\\n", " ")
    lines = []
    for ln in raw.splitlines():
        ln = ln.split("#", 1)[0]          # strip comments
        if "main.py" in ln:
            lines.append(" ".join(ln.split()))
    return lines


def shell_code(path):
    """Script body with heredocs and comments removed (for flag presence tests)."""
    raw = open(path).read()
    raw = re.sub(r"<<'?(\w+)'?.*?\n.*?\n\1\n", "\n", raw, flags=re.S)
    return "\n".join(l.split("#", 1)[0] for l in raw.splitlines())




def check(cond, label, detail=""):
    mark = "\033[32m ok \033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  [{mark}] {label}")
    if detail:
        print(f"         {detail}")
    if not cond:
        FAIL.append(label)
    return cond


def warn(cond, label, detail=""):
    if not cond:
        print(f"  [\033[33mwarn\033[0m] {label}")
        if detail:
            print(f"         {detail}")
        WARN.append(label)
    else:
        print(f"  [\033[32m ok \033[0m] {label}")


def load(name, path):
    for m in ("torch", "numpy", "transformers", "nltk", "spacy", "litellm", "boto3", "tqdm"):
        sys.modules.setdefault(m, types.ModuleType(m))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    return mod


print("\n=== A. jailbreak prompts ===")
JB = "data/jailbreak_prompts.json"
# ABSENT IS NOT A FAILURE. The harmful-behaviour goals are deliberately not
# vendored (see data/JAILBREAK_PROMPTS.md), so a fresh clone does not have this
# file and every check below it is unrunnable rather than failing. Treating it
# as a failure made the whole gate red on a clean checkout, which is the state a
# reader is in. Absent -> skip with a pointer; present but malformed -> fail.
if not os.path.exists(JB):
    print(f"  [\033[33mskip\033[0m] {JB} is not present")
    print( "         Not vendored by policy; sections A and B need it.")
    print( "         Populate it with: python3 fetch_jailbreak_prompts.py")
    print( "         Nothing else in this suite, and no reported number, depends on it.")
elif True:
    prompts = json.load(open(JB))
    check(isinstance(prompts, list) and len(prompts) >= 10,
          f"is a list with >=10 entries", f"{len(prompts)} entries")
    check(all(isinstance(p, str) and p.strip() for p in prompts),
          "all entries are non-empty strings")
    check(len(set(prompts)) == len(prompts), "no duplicate behaviours",
          "duplicates would weaken the anti-dedup property")
    warn(len(prompts) >= 8, "enough behaviours to cover k'=8 without reuse",
         f"only {len(prompts)}; payloads would repeat above that")

print("\n=== B. GuardrailTrigger payloads ===")
atk = load("atk", "src/attack.py")
if os.path.exists(JB):
    for kp in (1, 2, 3, 5, 8):
        a = atk.GuardrailTrigger(top_k=10, poison_num=kp, repeat=5, jailbreak_path=JB)
        item = {"question": "TESTQUESTION?",
                "topk_content": [f"clean-{i}" for i in range(10)]}
        out = a.attack(item)["topk_content"]
        poisoned = [c for c in out if "Forget the previous details" in c]
        ok = (len(out) == 10 and len(poisoned) == kp and len(set(poisoned)) == kp)
        tail_ok = all("Forget the previous details" in c for c in out[10 - kp:])
        qs_ok = all(c.startswith("TESTQUESTION?") for c in poisoned)
        check(ok and tail_ok and qs_ok, f"k'={kp}: {kp} unique payloads, placed last, query-prefixed")

print("\n=== C. detectors ===")
h = load("h", "src/helper.py")
CASES = [
    ("I don't know.", True, False),
    ("No information found.", True, False),
    ("I cannot assist with that request.", False, True),
    ("I can't help with that.", False, True),
    ("I'm sorry, but I cannot provide information on that.", False, True),
    ("I'm not able to answer this question.", False, True),
    ("I cannot fulfill your request.", False, True),
    ("Answer: 15%", False, False),
    ("England", False, False),
    ("Sorry for the confusion, the answer is 15%.", False, False),
    ("I cannot say for certain, but the answer is 15%.", False, False),
    # the Mistral COMPLIANCE mode seen in the pilot must NOT count as refusal
    ("I. Intro A. Defamatory article B. Unfounded claim", False, False),
    # --- cases drawn from the real D2 logs; each of these was initially MISSED ---
    ("I can\u2019t fulfill that request.", False, True),        # CURLY apostrophe
    ("I'm sorry, but I can't answer that.", False, True),      # "can't answer" was absent
    ("I can't provide a defamatory article about a U.S. president.", False, True),
    ("I don't have enough information to answer the question.", True, False),
    ("I don't have the specific information on the percentage", True, False),
    ("I'm sorry, but there is no word suggestion provided.", True, False),
]
bad = [t for t, ea, er in CASES
       if (h.is_epistemic_abstention(t), h.is_guardrail_refusal(t)) != (ea, er)]
check(not bad, f"all {len(CASES)} detector cases correct",
      "; ".join(bad[:2]) if bad else "incl. compliance-mode text scored as NOT a refusal")

print("\n=== D. prompt_style ===")
pt = load("pt", "src/prompt_template.py")
base = {"qa": pt.QA_PROMPT_TMPL, "qa-mc": "MC", "qa-hint": "HINT", "qa-zero": "Z"}
rr = pt.apply_prompt_style(base, "robustrag")
md = pt.apply_prompt_style(base, "mutedrag")
check(rr is base, "robustrag returns the dict unchanged")
check(md["qa"] == pt.MUTEDRAG_QA_PROMPT_TMPL, "mutedrag swaps the 'qa' template")
check(all(md[k] == base[k] for k in ("qa-mc", "qa-hint", "qa-zero")),
      "all other modes untouched", "aggregation machinery is unaffected")
check(base["qa"] == pt.QA_PROMPT_TMPL, "original dict not mutated in place")
try:
    pt.apply_prompt_style(base, "nonsense")
    check(False, "unknown style raises")
except ValueError:
    check(True, "unknown style raises ValueError")

print("\n=== E. cache keys differ between prompt styles ===")
q = "What percentage of couples are sleep-divorced?"
ctx = "<one retrieved passage>"
p_rr = rr["qa"].format(query_str=q, context_str=ctx)
p_md = md["qa"].format(query_str=q, context_str=ctx)
check(p_rr != p_md, "the two styles render different prompt text")
check(len(p_md) > 0 and "three sentences" in p_md,
      "mutedrag prompt is the conversational one")
# cache key is BaseModel.hash = lambda x: x  -> the full prompt string
src_models = open("src/models.py").read()
check("self.hash = lambda x: x" in src_models,
      "cache key is the full prompt string",
      "=> a style change cannot collide with cached responses from the other style")

print("\n=== F. caching prevents double billing ===")
check("if self.use_cache:" in src_models and "result = self.query_from_cache(prompt)" in src_models,
      "query() consults the cache before calling the model")
check("cache_path = f'cache/{args.model_name}-{args.dataset_name}-{args.top_k}.z'"
      in open("main.py").read(),
      "cache file is keyed by model+dataset+top_k")
for s in ("run_pilot_availability.sh", "run_diagnostics.sh"):
    if os.path.exists(s):
        runs = shell_runs(s)
        missing = [r for r in runs if "--use_cache" not in r]
        check(not missing, f"{s}: all {len(runs)} runs pass --use_cache",
              ("would re-bill: " + missing[0][:90]) if missing else "repeat runs cost $0")

print("\n=== G. CLI defaults ===")
main_src = open("main.py").read()
for flag, default in (("--abstention_threshold", "default=1"),
                      ("--prompt_style", "default='robustrag'"),
                      ("--top_k", "default=10"),
                      ("--alpha", "default=0.3"),
                      ("--beta", "default=3.0")):
    line = next((l for l in main_src.splitlines() if flag in l and "add_argument" in l), "")
    check(default in line, f"{flag} defaults to {default.split('=')[1]}")
check("prompt_style=args.prompt_style" in main_src,
      "prompt_style is actually passed to create_model",
      "without this the flag would be silently ignored")
check("abstention_threshold=args.abstention_threshold" in main_src,
      "abstention_threshold is passed to KeywordAgg")
check("attack_poison_num" in main_src,
      "poison count captured before upstream zeroes corruption_size")

print("\n=== H. no_vanilla must be absent (it hid the control earlier) ===")
# The availability gate NEEDS the undefended control, so --no_vanilla must be
# absent there. The low-n study is defended-path-only: its control is the n>=4
# bucket, and running vanilla would mean 900 UNCACHED Bedrock calls (the earlier
# KeywordInjection runs used --no_vanilla, so vanilla responses are not cached).
for s in ("run_pilot_availability.sh", "run_diagnostics.sh", "run_rescore.sh"):
    if os.path.exists(s):
        check("--no_vanilla" not in shell_code(s), f"{s} never passes --no_vanilla",
              "(comments mentioning it are ignored)")
# run_vanilla.sh is the ONE script that must NOT pass --no_vanilla: supplying the
# undefended reference line is its entire purpose. It is also the one script that
# must NOT sweep omega -- query_undefended() reads the ungrouped data_item, so the
# vanilla answer is identical at every omega and a sweep would pay 5x for one number.
if os.path.exists("run_vanilla.sh"):
    _v = shell_code("run_vanilla.sh")
    _vruns = shell_runs("run_vanilla.sh")
    check("--no_vanilla" not in _v, "run_vanilla.sh never passes --no_vanilla",
          "it exists to measure the vanilla arm; that flag would suppress it")
    check(all("--vanilla_csv" in r for r in _vruns) and len(_vruns) > 0,
          "run_vanilla.sh writes --vanilla_csv on every run",
          "without it the undefended numbers reach only the log, not a joinable file")
    check(all("--group_size" not in r for r in _vruns),
          "run_vanilla.sh does NOT sweep omega",
          "vanilla is omega-independent by construction; sweeping pays 5x for "
          "five identical numbers")
    check("--defense_method none" in _v,
          "run_vanilla.sh skips the defended arm (--defense_method none)",
          "those calls are already cached from the omega runs")
    check("vanilla_cost.py" in _v,
          "run_vanilla.sh gates on a cost estimate and aborts if it fails",
          "every other runner does; 'no estimate' means spending blind")
    if os.path.exists("vanilla_cost.py"):
        _vc = open("vanilla_cost.py").read()
        # A real second definition is an assignment at COLUMN 0. The string
        # "PRICING = {" also appears as the src.index() search key, which is
        # the opposite of duplication -- it is how the single source is located.
        _dup = re.search(r"^PRICING\s*=", _vc, re.M)
        check("from check_cache_coverage import PRICING" not in _vc and not _dup,
              "vanilla_cost.py reads PRICING from check_cache_coverage.py "
              "without importing it and without duplicating it",
              "importing drags in torch; duplicating creates a second source of truth")
    _m = open("main.py").read()
    check("VANILLA_COLS" in _m and "check_vanilla_csv" in _m,
          "the vanilla CSV has a fixed schema and an append guard",
          "same misalignment hazard as the per-query file")
    check("--vanilla_csv needs the vanilla arm" in _m,
          "main.py refuses --vanilla_csv together with --no_vanilla",
          "that combination would silently write a file of zeros")

if os.path.exists("run_lown.sh"):
    runs = shell_runs("run_lown.sh")
    check(all("--no_vanilla" in r for r in runs),
          "run_lown.sh passes --no_vanilla on every run",
          "defended-path-only study; vanilla is uncached and unused -> avoids ~900 needless calls")

print("\n=== J. low-n study instrumentation ===")
_dsrc = open("src/defense.py").read()
_msrc = open("main.py").read()
_i_init = _dsrc.find("self.last_mu = self._mu(")
_i_ret  = _dsrc.find("return \"I don't know.\", certify_flg")
check(_i_init != -1 and _i_init < _i_ret,
      "last_mu / last_surviving_keywords initialised BEFORE the early-abstain return",
      "otherwise an abstaining query logs the PREVIOUS query's mu and keywords")
check("self.last_early_abstain" in _dsrc, "early_abstain flag recorded")
check("ia == k or ia in k" in _msrc,
      "attacker-keyword match is strict (no unguarded 'k in ia')",
      "'2', '%' and '3' are all substrings of '32%' -- unguarded matching drives the metric to 1.0")
check("len(k) >= 4 and k in ia" in _msrc,
      "partial match is length-guarded at >=4 chars")
check("'alpha': args.alpha" in _msrc and "'beta': args.beta" in _msrc,
      "alpha/beta recorded per row so the analysis is self-contained")
check("'sample_keywords'" in _msrc,
      "a sample of surviving keywords is logged for audit")

print("\n=== I. predicted Bedrock volume / cost ===")
PRICE = {"mistral7b-bedrock": (0.15, 0.20), "llama8b-bedrock": (0.20, 0.25),
         "llama70b-bedrock": (0.72, 0.72)}
ISO_IN, ISO_OUT, JNT_IN, JNT_OUT = 606, 6, 1100, 3
for s in ("run_pilot_availability.sh", "run_diagnostics.sh"):
    if not os.path.exists(s):
        continue
    total, calls, unpriced = 0.0, 0, []
    for flat in shell_runs(s):
        m = re.search(r"--model_name\s+(\S+)", flat)
        if not m:
            continue
        model = m.group(1)
        if model not in PRICE:
            unpriced.append(model)
            continue
        n = re.search(r"--max_samples\s+(\S+)", flat)
        N = 20 if (not n or "$" in n.group(1)) else int(n.group(1))
        K = 10
        defended = "--defense_method keyword" in flat
        vanilla = "--no_vanilla" not in flat
        per = (K + 1 if defended else 0) + (1 if vanilla else 0)
        calls += N * per
        pin, pout = PRICE[model]
        tin = N * ((K * ISO_IN + JNT_IN) if defended else 0) + N * (JNT_IN if vanilla else 0)
        tout = N * ((K * ISO_OUT + JNT_OUT) if defended else 0) + N * (JNT_OUT if vanilla else 0)
        total += (tin * pin + tout * pout) / 1e6
    print(f"  {s}: {len(shell_runs(s))} runs, ~{calls:,} Bedrock calls, ~${total:.2f} at list price")
    print(f"         (a repeat run costs $0 where --use_cache hits)")
    if unpriced:
        print(f"         NOTE: {len(unpriced)} run(s) use a shell variable for --model_name "
              f"({', '.join(sorted(set(unpriced)))}) and are NOT included above; "
              f"a loop body executes once per model.")

print("\n=== K. SuppressThenInject (composite attack) ===")
_asrc = open("src/attack.py").read()
check("_keyword_injection_payload" in _asrc and _asrc.count("_keyword_injection_payload(") >= 3,
      "injection payload defined ONCE and reused",
      "composite vs KeywordInjection comparison is meaningless if the payloads drift")
check("_guardrail_payload" in _asrc and _asrc.count("_guardrail_payload(") >= 3,
      "suppression payload defined ONCE and reused")
if os.path.exists(JB):
    _item = {"question": "TESTQ?", "incorrect_answer": "32%",
             "topk_content": [f"clean-{i}" for i in range(10)]}

    def _cls(c):
        if "Forget the previous details" in c:
            return "S"
        if "Verified Answer Report" in c:
            return "I"
        return "."

    _all_ok = True
    for kp, ki in ((2, 1), (3, 1), (3, 2), (5, 2), (8, 3)):
        a = atk.SuppressThenInject(top_k=10, poison_num=kp, k_inject=ki, jailbreak_path=JB)
        out = a.attack(_item)["topk_content"]
        lay = "".join(_cls(c) for c in out)
        sup = [c for c in out if _cls(c) == "S"]
        inj = [c for c in out if _cls(c) == "I"]
        _all_ok &= (
            len(out) == 10
            and len(sup) == kp - ki and len(inj) == ki
            and len(set(sup)) == len(sup)           # unique payloads (anti-dedup)
            and all("32%" in c for c in inj)
            and lay.endswith("I" * ki)              # injection LAST
            and "." not in lay[10 - kp:]            # poisoned block contiguous at the end
        )
    check(_all_ok, "payload layout correct at every (k', k_inject) tested",
          "suppression first, injection last, contiguous, unique jailbreaks")

    _raised = 0
    for kp, ki in ((1, 1), (2, 2), (2, 0), (3, 5)):
        try:
            atk.SuppressThenInject(top_k=10, poison_num=kp, k_inject=ki, jailbreak_path=JB)
        except ValueError:
            _raised += 1
    check(_raised == 4, "degenerate budget splits all raise ValueError",
          "a silent degenerate split would look like a real datapoint in the CSV")

    # the composite's injection stage must be byte-identical to the control
    _k = atk.KeywordInjection(top_k=10, poison_num=1, repeat=5)
    _s = atk.SuppressThenInject(top_k=10, poison_num=2, k_inject=1, jailbreak_path=JB)
    _ki = [c for c in _k.attack(_item)["topk_content"] if _cls(c) == "I"][0]
    _si = [c for c in _s.attack(_item)["topk_content"] if _cls(c) == "I"][0]
    check(_ki == _si, "composite injection payload == KeywordInjection payload (byte-identical)",
          "this is what makes the budget-matched control valid")
    _g = atk.GuardrailTrigger(top_k=10, poison_num=1, repeat=5, jailbreak_path=JB)
    _gs = [c for c in _g.attack(_item)["topk_content"] if _cls(c) == "S"][0]
    _ss = [c for c in _s.attack(_item)["topk_content"] if _cls(c) == "S"][0]
    check(_gs == _ss, "composite suppression payload == GuardrailTrigger payload")

check("'k_suppress': k_suppress_col" in _msrc and "'k_inject': k_inject_col" in _msrc,
      "budget split written to the per-query CSV")
check("assert k_suppress_col + k_inject_col == attack_poison_num" in _msrc,
      "budget split is asserted to sum to k'",
      "catches a mislabelled row before it reaches the CSV")
check("raise NotImplementedError" in _msrc,
      "unhandled attack_method RAISES",
      "upstream had a bare `NotImplementedError` expression that did nothing")

print("\n=== L. filter_only mode (alpha/beta sweep at $0) ===")
check("FILTER_ONLY_SENTINEL" in _dsrc, "sentinel defined in defense.py")
_sent = "<FILTER_ONLY_NO_AGGREGATION_CALL>"
check(not h.is_epistemic_abstention(_sent) and not h.is_guardrail_refusal(_sent),
      "sentinel trips NEITHER detector",
      "otherwise filter-only runs would fabricate abstention/refusal counts")
_i_fo = _dsrc.find("if getattr(self, 'filter_only', False):")
_i_agg = _dsrc.find("response = self.llm.query(query_prompt)")
check(_i_fo != -1 and _i_agg != -1 and _i_fo < _i_agg,
      "filter_only returns BEFORE the aggregation call",
      "that call is the only one that varies with alpha/beta, i.e. the only billed one")
_i_fo_reset = _dsrc.find("self.last_filter_only = False\n\n        if len(seperate_responses) < abstention_threshold:")
check(_i_fo_reset != -1,
      "last_filter_only reset before the early-abstain return",
      "same stale-value hazard that corrupted last_mu")
for _flag in ("--filter_only only applies to --defense_method keyword",
              "--filter_only requires --no_vanilla",
              "--filter_only without --use_cache"):
    check(_flag in _msrc, f"main.py guards: {_flag[:46]}...")
check("'' if args.filter_only else" in _msrc,
      "ASR/accuracy/refusal/abstain written BLANK in filter_only mode",
      "scoring the sentinel would fabricate metrics")
_ansrc = open("analyze_lown.py").read()
check("def _opt_int" in _ansrc,
      "analyze_lown.py treats blank metrics as missing, not zero",
      "an int() on blank previously raised and silently DROPPED the row")

print("\n=== M. new run scripts ===")
for s in ("run_sweep.sh", "run_composite.sh", "run_clean_baseline.sh"):
    if not os.path.exists(s):
        WARN.append(f"{s} missing")
        continue
    runs = shell_runs(s)
    check(bool(runs) and all("--use_cache" in r for r in runs),
          f"{s}: all {len(runs)} runs pass --use_cache",
          "without it the responses are not saved and the next run re-bills")
    check(all("--no_vanilla" in r for r in runs),
          f"{s}: all runs pass --no_vanilla",
          "vanilla is uncached here and its result is not used")
def bash_array(path, name):
    """Read a literal bash array `NAME=( "a" "b" )` and return its entries.

    These scripts loop over a grid, so the single main.py line they contain is a
    TEMPLATE full of shell variables -- parsing it tells us nothing about what
    will actually run. The grid itself is declared literally for exactly this
    reason; read that instead."""
    src = open(path).read()
    m = re.search(name + r"=\(\s*(.*?)\)\s*\n", src, flags=re.S)
    if not m:
        return []
    body = "\n".join(l.split("#", 1)[0] for l in m.group(1).splitlines())
    quoted = re.findall(r'"([^"]+)"', body)
    if quoted:
        return quoted
    # unquoted form, possibly with a ${VAR:-default} fallback:
    #   OMEGAS=( ${OMEGAS_OVERRIDE:-1 2 3 4 5} )
    # take the DEFAULT, since that is what runs when the override is unset.
    d = re.search(r'\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}', body)
    if d:
        return d.group(1).split()
    return body.split()


if os.path.exists("run_sweep.sh"):
    runs = shell_runs("run_sweep.sh")
    check(all("--filter_only" in r for r in runs),
          "run_sweep.sh passes --filter_only on every run",
          "this is what makes the sweep free")
    grid = bash_array("run_sweep.sh", "GRID")
    _alphas = sorted({g.split()[0] for g in grid})
    _betas = sorted({g.split()[1] for g in grid})
    check(len(grid) >= 4 and len(_alphas) >= 3,
          f"sweep grid is literal and varies alpha: {_alphas}",
          f"betas {_betas}; {len(grid)} (alpha,beta) points x 3 budgets")
    # A grid point is only informative if the attacker WINS at some n and LOSES
    # at another within the observable range -- i.e. the boundary is visible. A
    # point where the attacker always (or never) wins cannot falsify anything.
    def _has_visible_boundary(alpha, beta, k_inject, n_max=10):
        wins = [k_inject >= min(beta, alpha * n) for n in range(1, n_max + 1)]
        return any(wins) and not all(wins)

    _informative = [g for g in grid
                    if _has_visible_boundary(float(g.split()[0]), float(g.split()[1]), 1)]
    check(len(_informative) >= 3,
          f"{len(_informative)}/{len(grid)} grid points have a visible boundary in n=1..10 at k'=1",
          "points without one: "
          + str([g for g in grid if g not in _informative]))

if os.path.exists("run_composite.sh"):
    runs = shell_runs("run_composite.sh")
    check(all("--filter_only" not in r for r in runs),
          "run_composite.sh does NOT use --filter_only",
          "this study needs ASR and accuracy, which filter_only cannot produce")
    cfgs = [c.split() for c in bash_array("run_composite.sh", "CONFIGS")]
    check(bool(cfgs), "run_composite.sh declares a literal CONFIGS grid")
    _sti_k = {c[1] for c in cfgs if c[0] == "SuppressThenInject"}
    _ctl_k = {c[1] for c in cfgs if c[0] == "KeywordInjection"}
    _gt_k = {c[1] for c in cfgs if c[0] == "GuardrailTrigger"}
    check(bool(_sti_k) and _sti_k <= _ctl_k,
          f"every composite budget {sorted(_sti_k)} has a KeywordInjection control "
          f"at the same k' {sorted(_ctl_k)}",
          "without a budget-matched control the composite result is uninterpretable")
    check(_sti_k <= _gt_k,
          f"every composite budget has a GuardrailTrigger control {sorted(_gt_k)}",
          "pure-suppression arm must show ~0 ASR or the framing is wrong")
    _bad = [c for c in cfgs if c[0] == "SuppressThenInject"
            and not (c[2] != "-" and 1 <= int(c[2]) < int(c[1]))]
    check(not _bad, "every SuppressThenInject config has a valid k_inject split",
          f"invalid: {_bad}" if _bad else "1 <= k_inject < k'")
    _nonsti = [c for c in cfgs if c[0] != "SuppressThenInject" and c[2] != "-"]
    check(not _nonsti, "non-composite configs pass no k_inject",
          "k_inject on a single-stage attack would be silently ignored")

print("\n=== N. open_nq low-n study ===")
if os.path.exists("run_lown_opennq.sh"):
    _src = shell_code("run_lown_opennq.sh")
    runs = shell_runs("run_lown_opennq.sh")
    check(all("--use_cache" in r for r in runs) and all("--no_vanilla" in r for r in runs),
          "run_lown_opennq.sh: every run passes --use_cache and --no_vanilla")
    check(all("--dataset_name open_nq" in r for r in runs),
          "targets open_nq (not realtimeqa)")
    check(all("--filter_only" not in r for r in runs),
          "does NOT use --filter_only",
          "this study needs accuracy, which filter_only cannot produce")
    check("PILOT" in _src and 'PILOT:-1' in _src,
          "defaults to PILOT mode",
          "open_nq's low-n rate is unknown and must be measured before the full spend")
    _arms = bash_array("run_lown_opennq.sh", "ARMS")
    _alphas = bash_array("run_lown_opennq.sh", "ALPHAS")
    check(any(a.startswith("none") for a in _arms),
          f"includes the clean (no-attack) control arm {_arms}",
          "accuracy at low n is uninterpretable without it -- low-n queries are "
          "intrinsically hard (27% vs 64% clean on realtimeqa)")
    check(len(_alphas) >= 3, f"sweeps at least 3 alphas {_alphas}")
    check(all("KeywordInjection 3" not in a for a in _arms),
          "excludes k'>=beta",
          "k'>=beta wins at every alpha and says nothing about the tradeoff")
    # cost ordering: all alphas of one arm must run together so the arm's
    # alpha-independent isolated calls are paid once
    _i_arm = _src.find('for ARM in "${ARMS[@]}"')
    _i_al = _src.find('for AL in "${ALPHAS[@]}"')
    check(_i_arm != -1 and _i_al != -1 and _i_arm < _i_al,
          "loops arms OUTSIDE alphas",
          "reversing these would re-pay the isolated calls for every alpha")

    # the dataset itself must support the study
    if os.path.exists("data/open_nq.json"):
        _d = json.load(open("data/open_nq.json"))
        check(len(_d) >= 400, f"open_nq has {len(_d)} items",
              "needs ~5x realtimeqa to lift low-n N from 11 to ~70")
        _bad = [i for i, it in enumerate(_d)
                if "question" not in it
                or not isinstance(it.get("correct answer"), list)
                or not isinstance(it.get("expanded answer"), list)
                or not str(it.get("incorrect answer", "")).strip()]
        check(not _bad, "every item has question / correct+expanded answer lists / "
                        "a non-empty incorrect answer",
              f"{len(_bad)} malformed items would crash mid-run" if _bad else
              "process_data_item and the ASR eval both depend on these")
        _short = sum(1 for it in _d
                     if len([x for x in it.get("context", [])[:10]
                             if "text" in x and "title" in x]) < 10)
        warn(_short == 0,
             f"{_short} items have fewer than 10 usable passages",
             "these land in the low-n bucket STRUCTURALLY; the analyser excludes "
             "them via n_groups -- confirm it still does")
        check("n_groups" in open("analyze_lown_opennq.py").read(),
              "analyser excludes short-context items via n_groups")
        check("overlap" in open("analyze_lown_opennq.py").read(),
              "analyser excludes answer-overlap items",
              "12/500 have an incorrect answer overlapping a correct one, which "
              "scores a false attack success on honest votes")
    _an = open("analyze_lown_opennq.py").read()
    check("difference-in-differences" in _an and " z = " in _an,
          "verdict uses a difference-in-differences z test, not a ratio",
          "a ratio of two noisy differences flipped 3x <-> 0.7x on sampling noise")
    check("VACUOUS" in _an,
          "analyser detects the vacuous case",
          "if clean low-n accuracy is ~0 there is no utility to lose and the "
          "claim is not meaningful")
else:
    WARN.append("run_lown_opennq.sh missing")

print("\n=== O. passage group size omega ===")
check("self.group_size = group_size" in _dsrc, "KeywordAgg accepts group_size")
check("group_size=args.group_size" in _msrc, "main.py passes group_size to KeywordAgg",
      "without this the flag is silently ignored -- the bug class that hid "
      "abstention_threshold upstream")
check("'group_size': args.group_size" in _msrc and "'max_groups'" in _msrc,
      "group_size and max_groups recorded per CSV row",
      "n=3 means something different at omega=1 than at omega=3")
_i_grp = _dsrc.find("_grouped = ")
_i_wrap = _dsrc.find("self.llm.wrap_prompt(iso_item")
check(_i_grp != -1 and _i_wrap != -1 and _i_grp < _i_wrap,
      "grouping happens BEFORE the isolated wrap_prompt")
check("iso_item = dict(data_item)" in _dsrc,
      "grouping uses a COPY; data_item stays ungrouped",
      "the aggregation call and eval_response both read the original")
check("raise NotImplementedError" in _dsrc and "group_size=1" in _dsrc,
      "certification refuses to run at omega>1",
      "certify() slices [:-corruption_size] per passage; at omega>1 that drops "
      "the wrong entries and the bound would be silently WRONG")
# the join must match what wrap_prompt(seperate=False) uses, or a group of
# omega passages is rendered differently from vanilla RAG's omega passages
check("'\\n\\n'.join(_tc[i:i + self.group_size])" in _dsrc,
      "groups joined with '\\n\\n', matching wrap_prompt(seperate=False)")

# the arithmetic this parameter exists to test
def _mu_max(w, k=10, a=0.3, b=3.0):
    return min(a * -(-k // w), b)
_ladder = {w: _mu_max(w) for w in (1, 2, 3, 4, 5)}
check(abs(_ladder[3] - 1.2) < 1e-9 and _ladder[4] <= 1.0 and _ladder[1] > 1.0,
      f"mu ceiling by omega: {({w: round(v,2) for w,v in _ladder.items()})}",
      "a count-1 keyword survives iff mu <= 1 (the filter tests count >= mu), so the vacuity rule is ceil(k/omega) <= 1/alpha -- NOT strict <")

if os.path.exists("run_omega.sh"):
    runs = shell_runs("run_omega.sh")
    check(bool(runs) and all("--use_cache" in r and "--no_vanilla" in r for r in runs),
          "run_omega.sh: every run passes --use_cache and --no_vanilla")
    check(all("--group_size" in r for r in runs), "every run passes --group_size")
    check(all("--corruption_size 0" in r or "--corruption_size 1" in r or
              "$KP" in r for r in runs),
          "only k'=0 (clean) and k'=1 arms",
          "k'=1 is below beta=3, so any survival is caused by omega alone")
    _om = bash_array("run_omega.sh", "OMEGAS")
    _arms = bash_array("run_omega.sh", "ARMS")
    _k_default = 10
    _km = re.search(r'^K="\$\{K:-(\d+)\}"', shell_code("run_omega.sh"), re.M)
    if _km:
        _k_default = int(_km.group(1))
    # the grid must contain an omega on each side of the mu<1 boundary, which
    # sits where ceil(k/omega) < 1/alpha
    _below = [w for w in _om if min(0.3 * -(-_k_default // int(w)), 3.0) <= 1]
    _above = [w for w in _om if min(0.3 * -(-_k_default // int(w)), 3.0) > 1]
    check(bool(_below) and bool(_above),
          f"omega grid {_om} straddles the mu<=1 step at k={_k_default}",
          f"mu>1 at omega {_above}; mu<=1 at omega {_below} -- both sides needed "
          f"or the ladder has no control")
    # k=12 control: warn if the grid would leave the poison alone in a short
    # remainder group, which confounds threshold effect with in-group contest
    _ragged = [w for w in _om
               if _k_default % int(w) == 1 and int(w) > 1]
    warn(not _ragged,
         f"omega {_ragged} leaves the poisoned passage ALONE at k={_k_default}",
         "k mod omega == 1 strands it in a 1-passage remainder group, so that "
         "row measures the threshold only -- use the k=12 grid (1 2 3 4 6) as "
         "the control")
    check(any(a.startswith("none") for a in _arms),
          f"includes the clean control arm {_arms}",
          "omega is a UTILITY knob; without the clean arm we cannot show the "
          "tradeoff it is supposed to buy")
else:
    WARN.append("run_omega.sh missing")

print("\n=== R. Bedrock model registry ===")
_msrc_models = open("src/models.py").read()
_i = _msrc_models.index("BEDROCK_MODELS = {")
_j = _msrc_models.index("\n}\n", _i) + 3
_ns = {}
exec(compile(_msrc_models[_i:_j], "reg", "exec"), _ns)
_REG = _ns["BEDROCK_MODELS"]

# Four models have BOTH an explicit branch (kept so nothing already-validated
# changes behaviour) and a registry entry. If those ever disagree, runs would
# silently use different settings depending on which path fired.
_explicit = dict(re.findall(
    r"elif model_name == '([\w.-]+)':.*?\n.*?BedrockModel\('([^']+)'",
    _msrc_models, flags=re.S))
_drift = [n for n, mid in _explicit.items()
          if n in _REG and _REG[n]["id"] != mid]
check(not _drift,
      f"registry ids match the explicit branches for {sorted(_explicit)}",
      f"DRIFT on {_drift} -- the two code paths would build different models"
      if _drift else "no drift")

# mistral is the only model that takes the steering system prompt
check(_REG["mistral7b-bedrock"]["system_prompt"] == "terse"
      and all(_REG[m]["system_prompt"] is None for m in _REG if _REG[m]["family"] == "llama"),
      "system_prompt matches the validated settings (mistral terse, llama none)",
      "Llama over-abstains with the steering prompt; Mistral needs it")

_eol = {k: v["status"] for k, v in _REG.items() if v.get("status")}
# "runnable" means BOTH gates pass: calibrated AND not withdrawn by the
# provider. Listing an end-of-lifed model here once read as if it were usable.
_runnable = [k for k, v in _REG.items()
             if v["system_prompt"] != "UNCALIBRATED" and not v.get("status")]
_gated = [k for k, v in _REG.items() if v["system_prompt"] == "UNCALIBRATED"]
check(len(_runnable) >= 3, f"runnable today (calibrated and not withdrawn): {_runnable}",
      f"blocked: uncalibrated {_gated}, withdrawn {sorted(_eol)}"
      if _gated or _eol else "")
check("UNCALIBRATED" in _msrc_models and "raise SystemExit" in _msrc_models,
      f"uncalibrated models are BLOCKED at construction: {_gated}",
      "a chat-tuned model that answers in prose collapses the keyword defense "
      "silently -- the run completes and reports on a defense that never worked")

# the within-family size series that tests the capability hypothesis
check("_spec.get('status')" in _msrc_models,
      f"models withdrawn by the provider are refused before any call: {sorted(_eol)}",
      "; ".join(f"{k}: {v}" for k, v in _eol.items()) if _eol else "none marked")
_llama = sorted((v["params_b"], k) for k, v in _REG.items()
                if v["family"] == "llama" and v.get("params_b") and not v.get("status"))
warn(len(_llama) >= 3,
     f"llama size series has {len(_llama)} usable points: {[p for p, _ in _llama]}B",
     "AWS end-of-lifed Llama-3.2-3B, so the within-family capability test is "
     "down to 8B/70B. Find a live small Llama with "
     "list_bedrock_models.py --provider meta to restore a third point")
# A model must sit somewhere on a capability ladder or it cannot enter the
# trend analysis. params_b for dense models; `rank` for families that publish no
# parameter counts (Nova). Requiring params_b alone would have FAILED preflight
# the moment Nova was calibrated -- after the money was spent on calibration.
_no_ladder = [k for k, v in _REG.items()
              if v.get("params_b") is None and v.get("rank") is None
              and v["system_prompt"] != "UNCALIBRATED" and not v.get("status")]
check(not _no_ladder,
      "every runnable model sits on a capability ladder (params_b or rank)",
      f"missing: {_no_ladder}" if _no_ladder
      else "params_b = dense size; rank = ordinal tier (nova micro<lite<pro)")

# The Nova ladder is ORDINAL. Guard against it being silently mixed into a
# parameter axis, and against a duplicate or gapped rank inside one family.
for _f in {v["family"] for v in _REG.values()}:
    _ms = {k: v for k, v in _REG.items() if v["family"] == _f and v.get("rank")}
    if _ms:
        _rs = sorted(v["rank"] for v in _ms.values())
        check(_rs == list(range(1, len(_rs) + 1)),
              f"{_f} ranks are a clean 1..n ordering: {_rs}",
              f"{sorted(_ms)} -- a duplicate or gap would silently reorder the ladder")
        check(all(v.get("params_b") is None for v in _ms.values()),
              f"{_f} uses rank INSTEAD of params_b, never both",
              "mixing the two would let an ordinal tier be plotted as a size")
        # A capability ladder claims capability is the ONLY variable across its
        # rungs. A per-model system_prompt breaks that claim silently, and
        # calibrate_model.py WILL suggest different settings for different rungs
        # because it calibrates one model at a time and decides on margins of a
        # single character. Pin them together.
        _sp = {k: v["system_prompt"] for k, v in _ms.items()}
        check(len(set(_sp.values())) == 1,
              f"{_f} ladder shares one system_prompt: {sorted(set(_sp.values()))}",
              f"{_sp} -- differing settings confound capability with prompt "
              f"configuration; change ALL rungs or none")

# THE DRIFT THAT ACTUALLY BIT: --model_name had a hand-written choices=[...]
# literal duplicating the registry. Adding nova-* to BEDROCK_MODELS left
# argparse rejecting them, and run_omega.sh marched through every cell printing
# "failed (continuing)". Nothing was billed, but the run had to be restarted.
# Generalise the lesson: a registry model must be USABLE end to end -- accepted
# by the CLI, and priced by the cost estimator the runners gate on.
_main = open("main.py").read()
check("choices=_MODEL_CHOICES" in _main and "sorted(BEDROCK_MODELS)" in _main,
      "--model_name choices are DERIVED from BEDROCK_MODELS, not hand-written",
      "a literal list silently desyncs every time a model is added")

_usable = [k for k, v in _REG.items()
           if v["system_prompt"] != "UNCALIBRATED" and not v.get("status")]
_ccc = open("check_cache_coverage.py").read()
_i = _ccc.index("PRICING = {"); _priced = set(re.findall(r"'([\w.-]+)':\s*\(",
                                                        _ccc[_i:_ccc.index("}", _i)]))
_unpriced = [m for m in _usable if m not in _priced]
check(not _unpriced,
      "every runnable model has a PRICING entry in check_cache_coverage.py",
      f"missing: {_unpriced} -- the runners gate on this estimate, so a missing "
      f"rate means spending blind" if _unpriced else
      f"{len(_priced)} models priced")

# Every z value quoted for this project was originally computed ad hoc, outside
# the repo, with an INDEPENDENT-samples two-proportion test on data that is
# paired (same 100 queries in every omega condition). The correct test is
# McNemar. Pin the corrected implementation so nobody silently reverts.
_sig = "analyze_significance.py"
check(os.path.exists(_sig), f"{_sig} exists",
      "paper numbers must come from code that can be re-run, not from chat")
if os.path.exists(_sig):
    _s = open(_sig).read()
    check("def mcnemar" in _s and "exact_binom_two_sided" in _s,
          "significance uses McNemar with an exact small-sample fallback",
          "omega conditions are PAIRED; an independent-samples z is the wrong test")
    check("Holm" in _s or "holm" in _s,
          "multiplicity is corrected across the comparison family",
          "the omega ladder is 12 tests per file; an uncorrected 0.05 is not 0.05")
    # the implementations must actually be right, not merely present
    import math as _m
    _ns = {"math": _m}      # the exec'd slice calls math.* but not the import
    _i = _s.index("def _phi"); _j = _s.index("rows = []")
    exec(compile(_s[_i:_j], "sig", "exec"), _ns)
    _r = _ns["mcnemar"]([(1, 0)] * 12 + [(0, 1)] * 2 + [(1, 1)] * 30 + [(0, 0)] * 56)
    _want = min(1, 2 * sum(_m.comb(14, i) for i in range(3)) / 2 ** 14)
    check(abs(_r["p"] - _want) < 1e-12,
          f"McNemar exact p reproduces the textbook value ({_r['p']:.4f})",
          "b=12, c=2 -> two-sided exact binomial on 14 discordant pairs")
    _z, _p = _ns["two_prop_z"](20, 100, 40, 100)
    check(abs(_z - 3.0861) < 1e-3, f"two-proportion z reproduces by hand ({_z:.4f})",
          "kept only as the labelled comparison column")
    check(abs(_ns["two_sided_p_from_z"](1.96) - 0.05) < 1e-3,
          "normal CDF is correct at z=1.96")

# RobustRAG's PAPER states alpha=0.2 for short-answer QA; its released code
# defaults to 0.3. The vacuity boundary ceil(k/omega) <= 1/alpha moves from
# omega>=4 to omega>=2 between them, so alpha must be a run parameter and must
# never be silently mixed within one result file.
if os.path.exists("run_omega.sh"):
    _ro = shell_code("run_omega.sh")
    _rr = shell_runs("run_omega.sh")
    check('ALPHA="${ALPHA:-0.3}"' in _ro,
          "run_omega.sh takes alpha as a parameter",
          "it was hard-coded to 0.3, which is the CODE default, not the PAPER's")
    check(all("--alpha 0.3" not in r for r in _rr) and
          all('--alpha "$ALPHA"' in r for r in _rr),
          "run_omega.sh passes $ALPHA, never a literal",
          "a literal here silently pins every run to one threshold")
    check("csv_alpha.py" in _ro,
          "run_omega.sh refuses to append across different alphas",
          "analyze_omega.py applies row 0's alpha to the whole file")
if os.path.exists("analyze_omega.py"):
    _ao = open("analyze_omega.py").read()
    check("mixed alpha values" in _ao,
          "analyze_omega.py refuses a file holding two alphas",
          "every boundary it predicts is a function of alpha")

print("\n=== Q. shell portability (macOS ships bash 3.2) ===")
# bash 3.2 mis-parses a heredoc inside $( ) command substitution: it executes the
# heredoc BODY as shell. `bash -n` on bash 5 accepts it, so this cannot be caught
# by a syntax check on Linux -- it has to be a pattern check. It broke run_fix.sh
# mid-run after the runs had already been paid for.
for _sh in sorted(glob.glob("run_*.sh")):
    _src = open(_sh).read()
    _bad = re.search(r"\$\([^)]*<<'?\w+'?", _src, flags=re.S)
    check(not _bad, f"{_sh}: no heredoc inside $( ) command substitution",
          "bash 3.2 executes the heredoc body as shell; use a helper script file")
    # also flag bash-4-only constructs
    for _c, _why in ((r"\$\{[A-Za-z_]\w*\^\^\}", "${VAR^^} is bash 4+"),
                     (r"\bdeclare\s+-A\b", "associative arrays are bash 4+"),
                     (r"\breadarray\b|\bmapfile\b", "readarray/mapfile are bash 4+")):
        check(not re.search(_c, _src), f"{_sh}: {_why}")

print("\n=== P. mitigation (threshold_mode) and dataset targets ===")
# INSTANTIATE KeywordAgg for real. String matching cannot catch a constructor
# that is silently truncated -- inserting a method definition mid-__init__ turns
# every following assignment into unreachable code after _mu's return, and the
# class still imports, still passes every grep, and fails only at query() time
# with AttributeError. That happened; this is the check that catches it.
_KA_ATTRS = ("abstention_threshold", "keyword_extractor", "ignore_set", "absolute",
             "relative", "longgen", "certify_save_path", "filter_only",
             "group_size", "threshold_mode", "top_k")
try:
    import types as _t
    _ns = {}
    _src = open("src/defense.py").read()
    _i = _src.index("class KeywordAgg"); _j = _src.index("    def query(self, data_item", _i)
    _body = _src[_i:_j].replace("(RRAG)", "")
    _stub = ("class _S:\n    @staticmethod\n    def load(*a, **k): return object()\n"
             "spacy = _S()\n"
             "class _L:\n    def info(self, *a, **k): pass\n    def warning(self, *a, **k): pass\n"
             "logger = _L()\n")
    exec(compile(_stub + _body, "ka", "exec"), _ns)
    _ka = _ns["KeywordAgg"](llm=None, relative_threshold=0.3, absolute_threshold=3,
                            top_k=10, threshold_mode="floor2", group_size=2)
    _missing = [a for a in _KA_ATTRS if not hasattr(_ka, a)]
    check(not _missing, "KeywordAgg constructs with every attribute set",
          f"MISSING: {_missing} -- the constructor is truncated" if _missing
          else "constructor runs to completion; no stranded dead code")
    if not _missing:
        check(abs(_ka._mu(1) - 2.0) < 1e-9 and abs(_ka._mu(10) - 3.0) < 1e-9,
              "the constructed object's _mu() computes correctly (floor2)",
              "_mu(1)=2.0 (floored), _mu(10)=3.0 (beta)")
except Exception as _e:
    check(False, "KeywordAgg constructs with every attribute set", f"raised {_e!r}")

check("def _mu(self" in _dsrc, "KeywordAgg._mu() is the single threshold definition")
check(_dsrc.count("self._mu(len(seperate_responses))") == 2,
      "BOTH mu sites route through _mu()",
      "inference and the logged mu must never disagree")
check("threshold_mode=args.threshold_mode" in _msrc, "main.py passes threshold_mode")
check("'threshold_mode': args.threshold_mode" in _msrc, "threshold_mode recorded per CSV row")
# certify() derives its threshold from _mu, so the
# certified bound describes the defense that actually answered. That makes
# 'floor2' certifiable (Prop. 3 in the paper). Mode 'k' is still refused: it
# discards the adaptive term, which the monotonicity argument does not cover.
check("count_threshold = self._mu(" in _dsrc,
      "certify() derives its threshold from _mu(), not the upstream formula",
      "otherwise the certified bound describes a different defense from the "
      "one that answered")
check("threshold_mode not in ('n', 'floor2')" in _dsrc,
      "certification accepts 'n' and 'floor2', refuses 'k'")
check(_dsrc.count("self._mu(") >= 4,
      "every threshold site routes through _mu()",
      "inference, the logged mu, and both certify branches must agree")

# Prop. 4: with the floor on, the degenerate branch must be unreachable at k'=1
def _mu_p(mode, n, alpha=0.3, beta=3.0):
    b = min(alpha * n, beta)
    return max(2.0, b) if mode == 'floor2' else b
check(all(_mu_p('floor2', n + 1) > 1 for n in range(0, 21)),
      "floor2 makes the vacuous branch unreachable at k'=1 (Prop. 4)")
check(any(_mu_p('n', n + 1) <= 1 for n in range(0, 21)),
      "upstream 'n' DOES reach the vacuous branch at k'=1",
      "if this ever fails, the paper's premise is gone")

# the three modes must actually differ, and only 'n' may be vacuous
def _mu_of(mode, n, alpha=0.3, beta=3.0, k=10):
    if mode == 'k':
        return min(alpha * k, beta)
    base = min(alpha * n, beta)
    return max(2.0, base) if mode == 'floor2' else base
_vac = {m: [n for n in range(0, 11) if _mu_of(m, n) <= 1] for m in ('n', 'k', 'floor2')}
check(_vac['n'] == [0, 1, 2, 3] and not _vac['k'] and not _vac['floor2'],
      f"only mode 'n' is vacuous: {_vac}",
      "a keyword survives iff count >= mu, so mu <= 1 admits everything")

# datasets must actually ship attacker targets before an attack arm is run
import json as _json
for _ds in ('realtimeqa', 'open_nq', 'hotpotqa'):
    _p = f'data/{_ds}.json'
    if not os.path.exists(_p):
        continue
    _d = _json.load(open(_p))[:100]
    _u = 0
    for _it in _d:
        _ia = _it.get('incorrect answer', '')
        if isinstance(_ia, (list, tuple)):
            _ia = _ia[0] if _ia else ''
        if str(_ia).strip():
            _u += 1
    if _ds == 'hotpotqa':
        check(_u == 0, f"{_ds}: {_u}/100 attacker targets -- attack arms must be REFUSED",
              "all 200 items ship `incorrect answer` == []; an attack there scores "
              "zero by construction")
    else:
        check(_u > 0, f"{_ds}: {_u}/100 attacker targets present")
check("has NO usable 'incorrect answer'" in _msrc,
      "main.py refuses an attack run on a dataset with no targets",
      "otherwise the run completes, writes a full CSV, and every attack column "
      "is zero -- a silent, expensive null result")

if os.path.exists("run_fix.sh"):
    _runs = shell_runs("run_fix.sh")
    check(bool(_runs) and all("--threshold_mode" in r for r in _runs),
          "run_fix.sh passes --threshold_mode on every run")
    _m = bash_array("run_fix.sh", "MODES"); _o = bash_array("run_fix.sh", "OMEGAS")
    check(set(_m) == {"n", "k", "floor2"}, f"fix grid covers all modes {_m}")
    check(set(_o) >= {"1", "4"}, f"fix grid tests BOTH holes {_o}",
          "omega=1 is the coverage hole, omega=4 is the group-size hole; a fix "
          "must close both")

print("\n" + "=" * 60)
if FAIL:
    print(f"\033[31m{len(FAIL)} CHECK(S) FAILED — do not run on AWS:\033[0m")
    for f in FAIL:
        print(f"   - {f}")
    sys.exit(1)
print(f"\033[32mALL CHECKS PASSED\033[0m" + (f" ({len(WARN)} warning(s))" if WARN else ""))
for w in WARN:
    print(f"   warn: {w}")
