#!/usr/bin/env python
"""Cost estimate for run_vanilla.sh. Makes no network calls.

Vanilla issues exactly ONE call per query, so the call count is exact. The only
estimated quantity is prompt length, and that is MEASURED from the existing
caches (the longest grouped prompts, scaled to k passages) rather than guessed.
Exits non-zero if it cannot produce an estimate, so the runner can refuse to
continue blind.
"""
import argparse, os, statistics as st, sys

p = argparse.ArgumentParser()
p.add_argument('--models', required=True)
p.add_argument('--arms', type=int, required=True)
p.add_argument('--n', type=int, required=True)
p.add_argument('--dataset', required=True)
p.add_argument('--top_k', type=int, required=True)
a = p.parse_args()

def _pricing():
    """Read PRICING out of check_cache_coverage.py WITHOUT importing it.

    Importing that module pulls in src.attack -> torch/transformers/spacy, which
    is slow and makes this estimate fail anywhere the full stack is absent --
    exactly when you most want a cheap sanity check. The dict is a plain
    literal, so exec just that slice. One source of truth, no import cost."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, 'check_cache_coverage.py')).read()
    i = src.index('PRICING = {')
    j = src.index('\n}\n', i) + 3
    ns = {}
    exec(compile(src[i:j], 'pricing', 'exec'), ns)
    return ns['PRICING']


try:
    PRICING = _pricing()
except Exception as e:
    sys.exit(f'could not read PRICING from check_cache_coverage.py: {e}')

models = a.models.split()
missing = [m for m in models if m not in PRICING]
if missing:
    sys.exit(f'!! no PRICING entry for {missing} -- add it before running blind')

# measure a vanilla-sized prompt from whatever cache exists
chars = None
try:
    import joblib
    for m in models:
        f = f'cache/{m}-{a.dataset}-{a.top_k}.z'
        if os.path.exists(f):
            c = joblib.load(f)
            longest = sorted(len(k) for k in c)[-200:]
            chars = st.median(longest) * 2 - 400   # ~2x the omega=5 grouped prompt
            break
except Exception:
    pass
if chars is None:
    chars = 9000.0
    note = ' (no cache to measure; using a 9000-char default)'
else:
    note = ' (measured from the longest cached prompts)'

IN, OUT = chars / 4, 10
calls_per_cell = a.n
print(f'\n  vanilla prompt ~{chars:.0f} chars ~= {IN:.0f} input tokens{note}')
print(f'  {len(models)} models x {a.arms} arms x {a.n} queries = '
      f'{len(models) * a.arms * a.n} calls (exactly 1 per query)\n')
total = 0.0
for m in models:
    i, o = PRICING[m]
    c = a.arms * calls_per_cell * (IN * i + OUT * o) / 1000
    total += c
    print(f'    {m:<22}{c:>8.3f}$')
print(f'    {"TOTAL":<22}{total:>8.2f}$\n')
