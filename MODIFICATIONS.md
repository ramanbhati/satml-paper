# What was modified, and what was not

This repository has exactly two commits.

```
25059a6  2026-09-27  Instrumentation, experiments and analysis for this submission
c1f8b64  2024-09-15  RobustRAG as released by its authors (upstream commit 9bc35b2)
```

The first is the upstream tree, unmodified. The second is the present
work. So the complete and authoritative account of what changed is:

```bash
git diff HEAD~1 HEAD            # everything
git diff HEAD~1 HEAD -- src/    # just the defense and attacks
```

This file is generated from that diff and summarises it.

---

## Upstream files modified (10)

| File | +/- | What was added |
|---|---|---|
| `.gitignore` | +25 / -0 | exclude virtualenvs, caches and per-run staging scratch |
| `README.md` | +10 / -0 | banner pointing at the artifact guide; upstream text otherwise unchanged |
| `main.py` | +575 / -15 | per-query CSV output; contamination guards; abstention classification; flags for the added parameters |
| `requirements.txt` | +38 / -11 | pin `transformers` at 4.44.x (4.40 misparses Llama-3.2 `rope_scaling`; >=4.45 drops `_crop_past_key_values`, which the decoding defense needs), plus `boto3`/`tenacity` for Bedrock and the Mistral v0.3 tokenizer deps |
| `src/attack.py` | +422 / -1 | `KeywordInjection`, `Blocker`, `BlockerLONG`, `SuppressThenInject`, `GuardrailTrigger`, `SemanticSteering` |
| `src/dataset_utils.py` | +4 / -0 | loader tolerance for the added corpora |
| `src/defense.py` | +265 / -18 | `_mu()` as the single threshold definition; `threshold_mode` (`n`/`floor2`/`k`); `group_size` for the omega ladder; per-query recording of the realised threshold and surviving keyword set |
| `src/helper.py` | +97 / -4 | guardrail-refusal and no-answer detectors, used for measurement only |
| `src/models.py` | +558 / -14 | Amazon Bedrock backend; response cache with a parameter fingerprint; process-wide rate-limit circuit breaker |
| `src/prompt_template.py` | +50 / -1 | prompt templates for the added model families |

## Upstream files used verbatim (7)

Byte-identical to upstream. No change of any kind.

- `data/README.md`
- `data/biogen.json`
- `data/google_search.py`
- `data/open_nq.json`
- `data/realtimeqa.json`
- `llm_eval.py`
- `run.sh`

## Files added by this work (106)


**Result data (`results/`)** — 48 files, the measurements every reported number is derived from.


**Experiment runners** (14)

- `run_alpha_utility.sh`
- `run_clean_baseline.sh`
- `run_composite.sh`
- `run_diagnostics.sh`
- `run_final.sh`
- `run_fix.sh`
- `run_lown.sh`
- `run_lown_opennq.sh`
- `run_omega.sh`
- `run_overnight.sh`
- `run_pilot_availability.sh`
- `run_rescore.sh`
- `run_sweep.sh`
- `run_vanilla.sh`

**Analysis** (17)

- `analyze_abstention_correlation.py`
- `analyze_alpha_utility.py`
- `analyze_by_model.py`
- `analyze_certify.py`
- `analyze_clean.py`
- `analyze_fix.py`
- `analyze_lown.py`
- `analyze_lown_opennq.py`
- `analyze_omega.py`
- `analyze_significance.py`
- `analyze_sweep.py`
- `analyze_vanilla.py`
- `audit_numbers.py`
- `csv_alpha.py`
- `csv_top_k.py`
- `make_paper_assets.py`
- `vanilla_cost.py`

**Tests and verification** (6)

- `gate.sh`
- `test_preflight.py`
- `test_run_final.sh`
- `test_scenarios.sh`
- `test_throttle.py`
- `verify_artifact_map.py`

**Operational utilities** (9)

- `calibrate_model.py`
- `check_anonymity.py`
- `check_bedrock.sh`
- `check_buildable.py`
- `check_cache_coverage.py`
- `fetch_jailbreak_prompts.py`
- `list_bedrock_models.py`
- `probe_bedrock.py`
- `purge_empty_cache.py`

**Documentation** (3)

- `ARTIFACT.md`
- `OVERNIGHT_REPORT.md`
- `data/JAILBREAK_PROMPTS.md`

**Cache metadata** (7)

- `cache/llama70b-bedrock-realtimeqa-10.z.fingerprint.json`
- `cache/llama8b-bedrock-realtimeqa-10.z.fingerprint.json`
- `cache/mistral7b-bedrock-open_nq-10.z.fingerprint.json`
- `cache/mistral7b-bedrock-realtimeqa-10.z.fingerprint.json`
- `cache/nova-lite-bedrock-realtimeqa-10.z.fingerprint.json`
- `cache/nova-micro-bedrock-realtimeqa-10.z.fingerprint.json`
- `cache/nova-pro-bedrock-realtimeqa-10.z.fingerprint.json`

**Other** (2)

- `data/hotpotqa.json`
- `generate_modifications.py`

---

## Is the original defense still the original defense?

At the settings upstream released — `group_size=1`, `threshold_mode='n'`,
`abstention_threshold=1` — yes. The checks below are reproducible from this
repository.

**One upstream executable line inside `KeywordAgg.query()` differs.** Stripping
comments from the method in both commits leaves a single changed line:

```
- ...wrap_prompt(data_item, as_multi_choice=False, seperate=True)
+ ...wrap_prompt(iso_item,  as_multi_choice=False, seperate=True)
```

and `iso_item` **is** `data_item` whenever `group_size <= 1`, not a copy. Every
other difference is an added line.

**The threshold is arithmetically identical.** Upstream computes
`min(self.absolute, self.relative*n)`; `_mu(n)` in mode `n` computes
`min(self.relative*n, self.absolute)`. `min` is commutative. Checked exhaustively
over 2,040 combinations of alpha, beta and n: no mismatch, and no disagreement in
the resulting keep/drop decision over every (alpha, beta, n, count).

**The filter comparison is byte-identical**, as is the `base_list` / `added_list`
partition in `certify()`.

**The abstention test is byte-identical**: `if "I don't" in x`. This matters more
than it looks, because that test fixes n, and n fixes the threshold. The added
refusal and epistemic-abstention detectors are counters: they never move a
response between the abstained and retained sets.

**The added recording is write-only.** Every `self.last_*` attribute is assigned
in `src/` and never read there; only `main.py` reads them, to write CSV columns.

**Upstream defaults are preserved.** `group_size=1` and `threshold_mode='n'`.
Upstream hardcodes `self.abstention_threshold = 1`, ignoring its own argument;
here the argument is honoured, and every runner passes exactly `1`.

### The three deliberate divergences

1. **`threshold_mode='floor2'`** is the mitigation the paper proposes, not a
   claim about upstream. Every table reporting upstream behaviour uses mode `n`,
   and each result row records which mode produced it.
2. **`group_size > 1`** is ours. Upstream fixes omega=1 and exposes no group-size
   parameter, so the omega ladder runs on IsoGroup as implemented here from the
   paper's specification. The paper discloses this in its own section and notes
   that the coverage route carries no such caveat, running at the omega=1 the
   released code implements.
3. **Empty responses are not cached.** Upstream's `batch_query` stores every
   response, including the empty string an API returns on failure. An empty
   string does not contain `"I don't"`, so upstream's abstention test counts it
   as a real answer: it raises n, raises the threshold, and makes the filter
   look stronger than it is, for every later run served from that cache. Here a
   response is cached only if it is non-empty, and the count of empties is
   recorded. On a cold run the two behave identically; they differ only in what
   a warm cache replays. This is a correction to the measurement instrument, not
   a change to the defense: the filter, the threshold and the abstention test are
   untouched.
4. **`--filter_only`** stops after the keyword filter and skips the final
   aggregation call, which makes a parameter sweep affordable against a warm
   cache. It is used only for `results/sweep.csv`, whose table reports keyword
   survival — the quantity that mode measures. Accuracy and attack success are
   written empty for those rows rather than estimated.

### What these checks do not establish

They are textual and numeric, not a runtime differential test: upstream's
`query()` has not been executed against this one on identical inputs and the
outputs compared. The threshold and filter equivalences above are exhaustive over
their inputs, and the changed lines are enumerated, but a reader wanting
end-to-end behavioural proof should run both commits against a fixed set of
responses.

