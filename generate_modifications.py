import subprocess, os, collections
def sh(*a): return subprocess.check_output(['git',*a], text=True)
UP='HEAD~1'
name_status = [l.split('\t') for l in sh('diff','--name-status',UP,'HEAD').strip().split('\n')]
stat = {}
for l in sh('diff','--numstat',UP,'HEAD').strip().split('\n'):
    a,d,f = l.split('\t'); stat[f]=(a,d)
up_files = set(sh('ls-tree','-r','--name-only',UP).split())
mod = sorted(f for s,*rest in name_status if s=='M' for f in rest)
add = sorted(f for s,*rest in name_status if s=='A' for f in rest)
dele= sorted(f for s,*rest in name_status if s=='D' for f in rest)
now = set(sh('ls-tree','-r','--name-only','HEAD').split())
unchanged = sorted(up_files & now - set(mod))
WHY = {
 'src/defense.py':'`_mu()` as the single threshold definition; `threshold_mode` (`n`/`floor2`/`k`); `group_size` for the omega ladder; per-query recording of the realised threshold and surviving keyword set',
 'src/models.py':'Amazon Bedrock backend; response cache with a parameter fingerprint; process-wide rate-limit circuit breaker',
 'src/attack.py':'`KeywordInjection`, `Blocker`, `BlockerLONG`, `SuppressThenInject`, `GuardrailTrigger`, `SemanticSteering`',
 'main.py':'per-query CSV output; contamination guards; abstention classification; flags for the added parameters',
 'src/helper.py':'guardrail-refusal and no-answer detectors, used for measurement only',
 'src/prompt_template.py':'prompt templates for the added model families',
 'src/dataset_utils.py':'loader tolerance for the added corpora',
 'README.md':'banner pointing at the artifact guide; upstream text otherwise unchanged',
 '.gitignore':'exclude virtualenvs, caches and per-run staging scratch',
 'requirements.txt':'pin `transformers` at 4.44.x (4.40 misparses Llama-3.2 `rope_scaling`; >=4.45 drops `_crop_past_key_values`, which the decoding defense needs), plus `boto3`/`tenacity` for Bedrock and the Mistral v0.3 tokenizer deps',
}
L=[]
L.append('# What was modified, and what was not\n')
L.append('This repository has exactly two commits.\n')
L.append('```')
L.append(sh('log','--format=%h  %ad  %s','--date=short').strip())
L.append('```\n')
L.append('The first is the upstream tree, unmodified. The second is the present')
L.append('work. So the complete and authoritative account of what changed is:\n')
L.append('```bash\ngit diff HEAD~1 HEAD            # everything\ngit diff HEAD~1 HEAD -- src/    # just the defense and attacks\n```\n')
L.append('This file is generated from that diff and summarises it.\n')
L.append('---\n')
L.append(f'## Upstream files modified ({len(mod)})\n')
L.append('| File | +/- | What was added |')
L.append('|---|---|---|')
for f in mod:
    a,d = stat.get(f,('?','?'))
    L.append(f'| `{f}` | +{a} / -{d} | {WHY.get(f,"")} |')
L.append(f'\n## Upstream files used verbatim ({len(unchanged)})\n')
L.append('Byte-identical to upstream. No change of any kind.\n')
for f in unchanged: L.append(f'- `{f}`')
if dele:
    L.append(f'\n## Upstream files removed ({len(dele)})\n')
    for f in dele: L.append(f'- `{f}`')
L.append(f'\n## Files added by this work ({len(add)})\n')
groups = collections.OrderedDict([
 ('Result data (`results/`)', lambda f: f.startswith('results/')),
 ('Experiment runners', lambda f: f.startswith('run_')),
 ('Analysis', lambda f: f.startswith(('analyze_','audit_','make_paper','csv_','vanilla_cost'))),
 ('Tests and verification', lambda f: f.startswith(('test_','gate.sh','verify_'))),
 ('Operational utilities', lambda f: f.startswith(('check_','calibrate','probe_','purge_','list_','fetch_'))),
 ('Documentation', lambda f: f.endswith('.md')),
 ('Cache metadata', lambda f: f.startswith('cache/')),
])
seen=set()
for title, pred in groups.items():
    hit=[f for f in add if pred(f) and f not in seen]
    seen |= set(hit)
    if not hit: continue
    if title.startswith('Result'):
        L.append(f'\n**{title}** — {len(hit)} files, the measurements every reported number is derived from.\n')
    else:
        L.append(f'\n**{title}** ({len(hit)})\n')
        for f in hit: L.append(f'- `{f}`')
rest=[f for f in add if f not in seen]
if rest:
    L.append(f'\n**Other** ({len(rest)})\n')
    for f in rest: L.append(f'- `{f}`')
L.append("""
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
""")
open('MODIFICATIONS.md','w').write('\n'.join(L)+'\n')
print(f"modified={len(mod)} unchanged={len(unchanged)} added={len(add)} deleted={len(dele)}")
