#!/usr/bin/env python
"""
Calibrate a new Bedrock model before it is used in any experiment.

    python3 calibrate_model.py claude-haiku-bedrock
    python3 calibrate_model.py nova-lite-bedrock --n 30

WHAT THIS IS FOR. RobustRAG's per-passage prompt is a few-shot COMPLETION that
expects a terse continuation after "Answer:". A chat-tuned model instead answers
conversationally -- "Based on the provided context, the percentage appears to
be..." -- and that verbosity silently destroys the defense: the spaCy extractor
pulls keywords out of prose, the keyword set explodes, and clean accuracy
collapses (60% -> 10% in earlier testing on this codebase).

The obvious fix is a steering system prompt. But it is NOT universally right:
Mistral needs it, Llama OVER-ABSTAINS with it and does better with none. So for
each new family the setting has to be measured, not guessed. This runs a small
sample BOTH ways and reports the numbers that decide it.

WHAT TO LOOK FOR, in priority order:
  1. response length   -- terse is the point. Long answers are the failure mode.
  2. abstention rate   -- if the system prompt pushes this up, it is hurting
                          (that is exactly what happened to Llama).
  3. keywords/query    -- prose inflates this; the reference models sit near 3-7.
  4. clean accuracy    -- the bottom line, but read it WITH the three above,
                          since a model can score well while behaving wrongly.

Choose the setting with terse answers AND an abstention rate comparable to the
reference models. If neither setting looks like the references, say so in the
paper rather than forcing the model in.

COST. 2 settings x --n queries x (top_k isolated + 1 aggregation) calls.
At the default n=25 and k=10 that is ~550 calls, a few cents on a small model.
Responses ARE cached, so a later full run reuses them.
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--dataset', default='realtimeqa')
p.add_argument('--top_k', type=int, default=10)
p.add_argument('--n', type=int, default=25, help='queries to sample')
p.add_argument('--alpha', type=float, default=0.3)
p.add_argument('--beta', type=float, default=3.0)
args = p.parse_args()

os.environ.setdefault('AWS_REGION_NAME', 'us-east-1')
os.environ.setdefault('BEDROCK_MAX_WORKERS', '1')

from src.dataset_utils import load_data          # noqa: E402
from src.models import BedrockModel, BEDROCK_MODELS, create_model  # noqa: E402
from src.defense import KeywordAgg               # noqa: E402

if args.model not in BEDROCK_MODELS:
    sys.exit(f'{args.model} is not in BEDROCK_MODELS (src/models.py)')
spec = BEDROCK_MODELS[args.model]

# Reference behaviour from the models already used in every published run, so a
# new family can be judged against something rather than in the abstract.
REFERENCE = {
    'mistral7b-bedrock (terse)': dict(chars=27, abstain=0.23, kw=7.1, acc=0.60),
    'llama70b-bedrock (none)':   dict(chars=38, abstain=0.32, kw=3.9, acc=0.62),
}

data = load_data(args.dataset, args.top_k)
items = [data.process_data_item(x) for x in data.data[:args.n]]

print(f'\nmodel   : {args.model}')
print(f'id      : {spec["id"]}')
print(f'family  : {spec["family"]}')
print(f'sample  : {len(items)} queries from {args.dataset}, k={args.top_k}\n')


def run(system_prompt, label):
    # cache_path=None: calibration must measure THIS setting, and the cache is
    # keyed by prompt text -- which is identical between the two settings when
    # the system turn is not part of the key. Bypass it to avoid cross-talk.
    llm = BedrockModel(spec['id'],
                       {'MISTRAL_TMPL': __import__('src.prompt_template', fromlist=['x']).MISTRAL_TMPL,
                        'LLAMA_TMPL': __import__('src.prompt_template', fromlist=['x']).LLAMA_TMPL}[spec['template']],
                       cache_path=None, system_prompt=system_prompt)
    dfn = KeywordAgg(llm, relative_threshold=args.alpha, absolute_threshold=args.beta,
                     abstention_threshold=1, top_k=args.top_k, threshold_mode='n')
    lens, abst, kws, corr, empties = [], 0, [], 0, 0
    samples = []
    for it in items:
        iso = llm.batch_query(llm.wrap_prompt(it, as_multi_choice=False, seperate=True))
        for r in iso:
            r = str(r)
            if not r.strip():
                empties += 1
                continue
            lens.append(len(r))
            if "I don't" in r:
                abst += 1
            if len(samples) < 6:
                samples.append(r)
        resp, _ = dfn.query(it, corruption_size=0)
        kws.append(len(getattr(dfn, 'last_surviving_keywords', set())))
        corr += bool(data.eval_response(resp, it))
    n_iso = len(lens) + empties
    return dict(label=label,
                chars=st.median(lens) if lens else float('nan'),
                p95=sorted(lens)[int(0.95 * len(lens))] if lens else float('nan'),
                abstain=abst / max(n_iso, 1),
                kw=sum(kws) / max(len(kws), 1),
                acc=corr / max(len(items), 1),
                empties=empties,
                samples=samples)


results = []
for sp, label in ((None, 'no system prompt'),
                  (BedrockModel.SYSTEM_PROMPT, 'terse system prompt')):
    print(f'--- running: {label} ...')
    try:
        results.append(run(sp, label))
    except Exception as e:
        print(f'    FAILED: {type(e).__name__}: {str(e)[:200]}')
        print(f'    If this is an access or model-id error, run list_bedrock_models.py')
        print(f'    and correct the id in BEDROCK_MODELS.')
        sys.exit(1)

print()
print(f"{'setting':<24} {'med chars':>10} {'p95':>6} {'abstain':>8} {'kw/query':>9} {'clean acc':>10} {'empty':>6}")
print('-' * 80)
for r in results:
    print(f"{r['label']:<24} {r['chars']:>10.0f} {r['p95']:>6.0f} {r['abstain']:>7.0%} "
          f"{r['kw']:>9.1f} {r['acc']:>10.0%} {r['empties']:>6}")
print()
print('reference (already-validated models, same metrics):')
for k, v in REFERENCE.items():
    print(f"  {k:<28} chars~{v['chars']:<4} abstain {v['abstain']:.0%}  kw {v['kw']:.1f}  acc {v['acc']:.0%}")

print()
best = min(results, key=lambda r: (r['chars'], r['abstain']))
print(f'TERSER setting: {best["label"]}')
warn = []
if best['chars'] > 120:
    warn.append('answers are long even at the better setting -- the extractor will '
                'pull keywords from prose; this family may not be usable as-is')
if best['abstain'] > 0.5:
    warn.append('abstention above 50% -- n will be small on most queries, which '
                'CONFOUNDS the low-n study (the regime would be model-induced, '
                'not corpus-induced)')
if best['empties'] > 0:
    warn.append(f'{best["empties"]} empty responses -- throttling or a parse failure')
for w in warn:
    print(f'  !! {w}')
if not warn:
    setting = "'terse'" if 'terse' in best['label'] else 'None'
    print(f'  looks usable. Set system_prompt in BEDROCK_MODELS to {setting} '
          f'and re-run the preflight.')

print()
print('sample isolated responses at the better setting (eyeball these):')
for s in best['samples']:
    print('   ' + ' '.join(s.split())[:110])
print()
