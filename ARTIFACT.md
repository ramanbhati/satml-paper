# Artifact guide

This repository is a fork of RobustRAG (Xiang et al.), instrumented to measure
the keyword-aggregation threshold `mu = min(alpha*n, beta)` per query and to
evaluate a floored variant of it. `README.md` is the upstream README, unchanged;
this file describes what was added and how to reproduce the paper.

Everything the paper reports can be re-derived **offline, at no cost**, from the
result files already in `results/`. Only new measurements need AWS credentials.

## 0. Setup: there are two tiers, and most readers need only the first

**To check the paper's numbers: install nothing.** Every table, figure and
headline number is re-derived from `results/*.csv` using the standard library
alone. Verified on a clean checkout with Python 3.9 and 3.14 and no virtualenv:

```bash
python3 audit_numbers.py        # re-derives every headline number
python3 make_paper_assets.py    # regenerates all 21 tables and figures
./gate.sh                       # the above plus every test suite
```

**To run new measurements: install the full stack, plus two non-pip downloads.**

```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('stopwords')"
```

Both downloads are required, not optional: `src/defense.py` evaluates
`stopwords.words('english')` at module level, so a missing corpus is an
`ImportError` at import rather than a failure at first use. `requirements.txt`
leaves `torch` unpinned, so pip resolves a current build and on Linux pulls the
CUDA wheels with it -- roughly 2 GB that the first tier does not need. Then AWS
credentials on the standard chain, and Bedrock model access enabled in your
region; `./check_bedrock.sh` reports which prerequisite is missing, in order.

---

## 1. Why code sits both at the root and in `src/`

This is upstream's layout, kept deliberately:

- **`src/`** is the importable library — the defense, the attacks, the model
  wrappers, the prompt templates. Every file in it is upstream's.
- **The root** holds executables: upstream's `main.py` and `llm_eval.py`, plus
  the runner, analysis and test scripts added for this work.

Paths inside `src/` were **not** changed, so that

```bash
git diff HEAD~1 HEAD -- src/defense.py
```

shows exactly what was modified in the defense under study, against the
upstream tree in the first commit. That diff is the most direct way to check the
paper's claims about the released implementation, and moving files would
obscure it. If you are browsing without git history, `MODIFICATIONS.md`
summarises the same diff file by file.

None of the added root-level scripts import one another; the only import edges
are root into `src`.

---

## 2. Upstream code, and what changed in it

The first commit of this repository is upstream's tree at its last commit,
`9bc35b2`, unmodified. All seven files below are upstream's and were modified:

| File | Added |
|---|---|
| `src/defense.py` | `_mu()` as the single threshold definition, `threshold_mode` (`n` / `floor2` / `k`), per-query recording of the realised `mu` and surviving keyword set |
| `src/models.py` | Bedrock backend, response cache with a parameter fingerprint, and a process-wide rate-limit circuit breaker |
| `src/attack.py` | `KeywordInjection`, `Blocker`, `BlockerLONG`, `SuppressThenInject`, `GuardrailTrigger`, `SemanticSteering` |
| `main.py` | Per-query CSV output, contamination guards (`check_certify_csv`, `check_per_query_csv`, `check_vanilla_csv`), abstention classification |
| `src/helper.py` | Refusal and no-answer detectors |
| `src/prompt_template.py` | Templates for the added model families |
| `src/dataset_utils.py` | Loader tolerance for the added corpora |

The upstream threshold rule is unchanged under `threshold_mode=n`, which is the
mode every "upstream" column in the paper reports.

---

## 3. Reproducing the tables and figures

`make_paper_assets.py` generates **all 21** LaTeX assets from `results/*.csv`.
It writes nothing else and reads no network.

```bash
python3 make_paper_assets.py     # regenerate every table and figure
python3 audit_numbers.py         # re-derive every headline number (32 checks; see below)
./gate.sh                        # both of the above, plus every test suite
```

`audit_numbers.py` is the one to run first if you only run one thing: it
re-derives each number the paper states directly from the CSVs and reports
`N ok, N failed` rather than regenerating anything. On a clean checkout it
reports `31 ok, 1 skipped`: the skipped check compares the certification call
count against run logs, which are not shipped (Section 8).

### Four shared inputs

Most assets are built from four frames loaded once at the top of
`make_paper_assets.py`:

| Frame | File |
|---|---|
| `om` | `results/omega_realtimeqa_k10_all.csv` — the omega ladder, six models |
| `van` | `results/vanilla_realtimeqa_k10.csv` — the undefended control |
| `sw` | `results/sweep.csv` — the alpha-beta surface |
| `fx` | `results/fix_realtimeqa_k10_all.csv` — upstream rule vs the floor |

### Asset to data

| Generated asset | Shows | Built from |
|---|---|---|
| `tab_omega`, `fig_ladder`, `fig_asr` | keyword survival up the omega ladder | `om` |
| `tab_vanilla`, `tab_benefit`, `fig_benefit` | defended vs undefended, per model | `om` + `van` |
| `tab_surface` | vacuity across alpha and beta | `sw` |
| `tab_lown` | the low-`n` split at omega=1 | `om` |
| `tab_fix` | the floor in both regimes | `fx` |
| `tab_coverage`, `fig_ndist` | vulnerable population per corpus | `omega_{realtimeqa_k10_all,open_nq_k10,hotpotqa_k10}.csv` |
| `tab_generalize` | k=10 vs k=12, and a second corpus | `omega_realtimeqa_k10_all.csv`, `omega_realtimeqa_k12.csv`, `omega_open_nq_k10.csv` |
| `tab_alpha`, `fig_alpha` | alpha=0.2, and the utility optimum | `omega_realtimeqa_k10_a0.2.csv`, `alpha_utility.csv` |
| `tab_composite` | the composite and availability attacks | `composite.csv` |
| `tab_certify`, `certify_facts` | certified accuracy, upstream rule vs floor | `certify_{n,floor2}_*.csv`, `certify_realtimeqa_n_*.csv` |
| `tab_theirs`, `theirs_facts` | the same contrast under RobustRAG's own PIA and Poison | `theirattacks_realtimeqa_k10.csv` |
| `fig_mustep` | survival against realised `mu`, pooled | every CSV carrying a `mu` column |
| `nobs` | total distinct observations | all result files |

`fig_mustep` pools across files, so `make_paper_assets.py` classifies every file
in `results/` explicitly and **fails on any file it does not recognise**, rather
than silently folding an unreviewed file into a published figure.

---

## 4. Producing new measurements (needs AWS credentials)

`run_final.sh` is the current entry point; the other `run_*.sh` scripts produced
the earlier result files and are kept so each one is reproducible.

```bash
./check_bedrock.sh                  # which prerequisite is missing, in order
./run_final.sh profile              # cost of a certification run, no billed calls
./run_final.sh certify              # certified accuracy, both threshold rules
./run_final.sh attacks              # RobustRAG's own PIA and Poison
./gate.sh                           # offline; run before anything that spends
```

Read `profile` output before `certify`: certification enumerates a keyword
powerset per query and the profiler reports the exact call count without issuing
a single certification call.

Runners write per-run files under `results/.staging/` and assemble the canonical
CSV only when every part is present and clean, so an aborted run cannot damage
completed results and a rerun never pays twice for the same measurement.

| Script | Produces |
|---|---|
| `run_final.sh` | `fix_realtimeqa_k10_all.csv`, `theirattacks_realtimeqa_k10.csv`, `certify_*` |
| `run_omega.sh` | `omega_<dataset>_k<k>[_a<alpha>].csv` |
| `run_fix.sh` | `fix_<dataset>_k<k>.csv` |
| `run_vanilla.sh` | `vanilla_<dataset>_k<k>.csv` |
| `run_sweep.sh` | `sweep.csv` |
| `run_alpha_utility.sh` | `alpha_utility.csv` |
| `run_composite.sh` | `composite.csv` |
| `run_lown.sh` | `lown.csv` |
| `run_clean_baseline.sh` | `clean.csv`, consumed by `analyze_clean.py` only (not shipped; see below) |
| `run_lown_opennq.sh` | `lown_opennq_pilot.csv` |
| `run_pilot_availability.sh`, `run_diagnostics.sh`, `run_rescore.sh` | the availability line reported in the appendix |

`results/` ships the 38 CSVs the paper is built from. Two intermediates are not
among them: `clean.csv` (a baseline consumed only by `analyze_clean.py`) and
`lown_opennq.csv` (superseded by `lown_opennq_pilot.csv`). Neither is read by
`make_paper_assets.py` or `audit_numbers.py`, so their absence does not affect
any reported number; rerunning the script in the table regenerates them.

`results/` also holds two things no script reads: the timestamped `*.bak`
copies the runners make before overwriting a file, and
`results/CONTAMINATED_20260920/`, a certification run quarantined because rate
limiting produced empty responses.
`results/CONTAMINATED_20260920/README.txt` explains the failure; it was
rerun under throttled settings and is kept only as evidence of that failure mode.

---

## 5. Checks

All offline and free. `./gate.sh` runs every one.

| Script | Checks |
|---|---|
| `test_preflight.py` | data, defenses and thresholds agree with what the runners assume |
| `test_run_final.sh` | `run_final.sh` control flow against a stub runner: resume, guards, merge, and the exact flags each run receives |
| `test_scenarios.sh` | states the repository has not reached yet — half-written pairs, six-model runs, partial files |
| `test_throttle.py` | that the rate-limit handling converges, against a simulated endpoint |
| `audit_numbers.py` | every headline number, re-derived from the CSVs |

---

## 6. Analysis scripts

Each answers one question and prints a table; none is needed to rebuild the
paper, which `make_paper_assets.py` does on its own.

| Script | Question |
|---|---|
| `analyze_certify.py` | certified accuracy, upstream threshold vs the floor |
| `analyze_omega.py` | does the defense's own utility knob remove its protection |
| `analyze_fix.py` | does the floor close both routes, and at what cost |
| `analyze_significance.py` | McNemar tests for the omega and floor comparisons |
| `analyze_sweep.py` | the alpha-beta surface |
| `analyze_lown.py`, `analyze_lown_opennq.py` | the low-`n` regime, and whether any alpha makes it safe and useful |
| `analyze_abstention_correlation.py` | why the low-`n` regime exists, and why only in some corpora |
| `analyze_vanilla.py` | the benign cost relative to undefended RAG |
| `analyze_by_model.py` | whether in-group resistance tracks model capability |
| `analyze_clean.py` | the attacked-vs-clean accuracy confound |
| `analyze_alpha_utility.py` | whether any alpha is both secure and useful |

Utilities: `check_cache_coverage.py` (what a run would have to pay for),
`calibrate_model.py` (before a new model is used in any experiment),
`purge_empty_cache.py`, `probe_bedrock.py`, `list_bedrock_models.py`,
`vanilla_cost.py`, and `csv_top_k.py` / `csv_alpha.py`, which the runners call to
refuse appending rows recorded at one parameter value into a file recorded at
another.

---

## 7. Corpora, and which are upstream's

| File | Origin |
|---|---|
| `data/realtimeqa.json` | upstream, byte-identical |
| `data/open_nq.json` | upstream, byte-identical |
| `data/biogen.json` | upstream, byte-identical |
| `data/hotpotqa.json` | **added by this work** — upstream ships no multi-hop corpus; rebuilt by `data/build_hotpotqa.py` |

`data/hotpotqa.json` carries 200 items in the HotpotQA distractor setting, 10
passages each, of which two must be combined to answer. By `hop_type` it is 166
bridge and 34 comparison questions. It is loaded through the same short-answer
reader as Natural Questions (`src/dataset_utils.py`), so scoring is substring
match on the short answer, identical to the other corpora. The `incorrect_context`
field is present in the schema but unpopulated: no attacked arm exists for this
corpus, which is why every HotpotQA figure in the paper is a clean-arm ($k'=0$)
measurement and why the coverage table holds the arm fixed at $k'=0$ for all
three corpora rather than pooling.

**Provenance.** The 200 items are the first 200 questions, in their original
order, of the official HotpotQA development set in the distractor setting
(hotpot_dev_distractor_v1.json; on Hugging Face, `hotpotqa/hotpot_qa`, config
`distractor`, split `validation`). There is no sampling: the first item is
`5a8b57f25542995d1e6f1371` ("Were Scott Derrickson and Ed Wood of the same
nationality?") and the last is `5a75f79555429976ec32bcca`. Every item in that
split is labelled `hard`. Field mapping:

| Field here | Source field |
|---|---|
| `question` | `question`, verbatim |
| `correct answer` | `[answer]` |
| `hop_type` | `type` (`bridge` or `comparison`) |
| `level` | `level` |
| `context` | the source's paragraphs, in source order; each paragraph's sentences joined with a single space |
| `supporting_titles` | titles in `supporting_facts` |

Every item has the source's 10 paragraphs except item 27
(`5abd259d55429924427fcf1a`), which has 2 in the source as well. The paper uses
the first 100 of these 200.

`data/build_hotpotqa.py` rebuilds the file from that split using the standard
library alone and reports whether the result is byte-identical to the shipped
copy (it is; sha256 prefix `24adb39b4ebcee11`):

```bash
python3 data/build_hotpotqa.py            # fetch, rebuild, compare
python3 data/build_hotpotqa.py --source hotpot_dev_distractor_v1.json   # offline
```

It fetches from the Hugging Face datasets server by default, because the
original download host has been unreachable; `--source` takes a local copy of
the original file instead.

## 8. Not included

- **`data/jailbreak_prompts.json`** — the harmful-behaviour goals used by
  `GuardrailTrigger` are not vendored. `fetch_jailbreak_prompts.py` retrieves
  them from JailbreakBench; see `data/JAILBREAK_PROMPTS.md`.
- **`logs/`** and **`log/`** — run logs are not tracked. `audit_numbers.py`
  reads `logs/certify_cost_*.json` for the certification call counts and skips
  those checks when the directory is absent; every other check is unaffected.
- **`.venv/`** — rebuild from `requirements.txt`. Scripts fall back to
  `python3` when it is absent.
- **Response cache** — `cache/*.fingerprint.json` records the parameters each
  cache was built under, so a cache cannot be reused across a parameter change.
  The cached responses themselves are not tracked.
