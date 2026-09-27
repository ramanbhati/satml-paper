#!/usr/bin/env python
"""
Offline cache-coverage check. Makes NO network calls and costs nothing.

Answers, for a planned run, the only question that matters before spending:
    how many Bedrock calls will ACTUALLY be issued?

It rebuilds the exact prompt strings main.py would send (same data loader, same
attack, same wrap_prompt) and tests each one for membership in the on-disk
response cache. The cache key is the full prompt string (BaseModel.hash is the
identity function), so membership here is exactly the hit/miss the real run gets.

IMPORTANT -- batch semantics. BaseModel.batch_query_from_cache is ALL-OR-NOTHING:

    if all([h in self.cache for h in h_list]):   # src/models.py
        return [self.cache[h] for h in h_list]

so a single missing passage in a query re-issues ALL top_k isolated calls for
that query, not just the missing one. This script reports both the naive
per-prompt miss count and the real billed count under batch semantics. The
difference is large and is the number you should budget against.

The final aggregation call is NOT counted: its prompt contains the surviving
keyword set, which cannot be known without running the filter over real
responses. Treat it as up to +1 call per query (an upper bound).

    python3 check_cache_coverage.py --model_name mistral7b-bedrock \
        --dataset_name realtimeqa --top_k 10 --attack_method none

Exit code is 0 always; this is a reporting tool, not a gate.
"""
import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset_utils import load_data
from src.models import create_model
from src.attack import (Attack, Blocker, GuardrailTrigger, KeywordInjection,
                        PIA, Poison, SemanticSteering, SuppressThenInject)

# Bedrock on-demand pricing, USD per 1K tokens, (input, output).
#
# SOURCE QUALITY WARNING: these were taken from secondary pricing summaries
# (Sept 2026), not from the AWS console for this account's region. Bedrock rates
# change and vary by region. The numbers below are good enough to answer "is
# this run cents or hundreds of dollars"; they are NOT good enough to quote in a
# paper. Confirm in the AWS console before committing real budget:
#   https://aws.amazon.com/bedrock/pricing/
PRICING = {
    'mistral7b-bedrock': (0.00015, 0.0002),
    'llama8b-bedrock':   (0.0002,  0.0002),
    'llama70b-bedrock':  (0.00035, 0.00035),
    'llama3b-bedrock':   (0.00015, 0.00015),
    # Nova rates read from litellm 1.83.0's bundled model cost map rather than a
    # pricing blog, so these are better sourced than the four above. Still not
    # account/region-confirmed -- same caveat applies before quoting anywhere.
    'nova-micro-bedrock': (0.000035, 0.00014),
    'nova-lite-bedrock':  (0.00006,  0.00024),
    'nova-pro-bedrock':   (0.0008,   0.0032),
}


def build_attacker(name, top_k, k_prime, jailbreak_path=None, k_inject=1):
    if name == 'none' or k_prime <= 0:
        return None
    kw = dict(top_k=top_k, poison_num=k_prime, poison_order='backward')
    if name == 'SuppressThenInject':
        return SuppressThenInject(repeat=5, k_inject=k_inject,
                                  jailbreak_path=jailbreak_path, **kw)
    if name == 'KeywordInjection':
        return KeywordInjection(repeat=5, **kw)
    if name == 'SemanticSteering':
        return SemanticSteering(repeat=5, **kw)
    if name == 'GuardrailTrigger':
        return GuardrailTrigger(repeat=5, jailbreak_path=jailbreak_path, **kw)
    if name == 'Blocker':
        return Blocker(repeat=10, **kw)
    if name == 'PIA':
        return PIA(repeat=10, **kw)
    if name == 'Poison':
        return Poison(repeat=10, **kw)
    raise ValueError(f'unhandled attack_method {name!r}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', required=True)
    p.add_argument('--dataset_name', default='realtimeqa')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--attack_method', default='none')
    p.add_argument('--corruption_size', type=int, default=0)
    p.add_argument('--max_samples', type=int, default=100)
    p.add_argument('--jailbreak_path', default=None)
    p.add_argument('--k_inject', type=int, default=1,
                   help='SuppressThenInject only; changes the payloads and so the cache keys')
    p.add_argument('--prompt_style', default='robustrag')
    p.add_argument('--no_vanilla', action='store_true',
                   help='match the run script: skip the undefended single-prompt call')
    args = p.parse_args()

    cache_path = f'cache/{args.model_name}-{args.dataset_name}-{args.top_k}.z'
    if os.path.exists(cache_path):
        cache = joblib.load(cache_path)
    else:
        cache = {}
        print(f'!! no cache at {cache_path} -- every call will be billed')

    data_tool = load_data(args.dataset_name, args.top_k)
    # cache_path=None so create_model does not touch the cache file we just read
    llm = create_model(args.model_name, prompt_style=args.prompt_style,
                       cache_path=None)
    attacker = build_attacker(args.attack_method, args.top_k,
                              args.corruption_size, args.jailbreak_path,
                              args.k_inject)

    n_queries = 0
    iso_prompts = 0
    iso_miss = 0
    billed_iso = 0          # under all-or-nothing batch semantics
    queries_with_any_miss = 0
    van_miss = 0
    miss_chars = 0

    for data_item in data_tool.data[:args.max_samples]:
        n_queries += 1
        data_item = data_tool.process_data_item(data_item)
        if attacker is not None:
            data_item = attacker.attack(data_item)

        prompts = llm.wrap_prompt(data_item, as_multi_choice=False, seperate=True)
        misses = [q for q in prompts if q not in cache]
        iso_prompts += len(prompts)
        iso_miss += len(misses)
        if misses:
            queries_with_any_miss += 1
            billed_iso += len(prompts)      # the WHOLE batch is re-issued
            miss_chars += sum(len(q) for q in prompts)

        if not args.no_vanilla:
            vp = llm.wrap_prompt(data_item, as_multi_choice=False, seperate=False)
            if vp not in cache:
                van_miss += 1
                miss_chars += len(vp)

    print()
    print(f'  cache file          {cache_path}'
          f'  ({len(cache)} entries)' if cache else f'  cache file          <none>')
    print(f'  queries             {n_queries}')
    print(f'  isolated prompts    {iso_prompts}')
    print(f'  prompt-level misses {iso_miss}'
          f'   ({iso_miss / max(iso_prompts, 1):.0%})')
    print(f'  queries w/ a miss   {queries_with_any_miss}')
    print()
    print(f'  BILLED isolated calls (all-or-nothing batching): {billed_iso}')
    if not args.no_vanilla:
        print(f'  BILLED vanilla calls:                           {van_miss}')
    print(f'  final aggregation calls (upper bound, 1/query):  {n_queries}')
    total = billed_iso + van_miss + n_queries
    print(f'  ------------------------------------------------')
    print(f'  TOTAL upper-bound billed calls:                 {total}')

    rate = PRICING.get(args.model_name)
    if rate:
        # ~4 chars/token; assume output is short (keyword answers) -> 40 tokens
        in_tok = miss_chars / 4 + (n_queries * 400)   # + agg prompts, rough
        out_tok = total * 40
        cost = in_tok / 1000 * rate[0] + out_tok / 1000 * rate[1]
        print(f'  rough cost estimate:                            ${cost:.2f}')
        print(f'    (in~{in_tok/1000:.0f}K tok @ ${rate[0]}/1K, '
              f'out~{out_tok/1000:.0f}K tok @ ${rate[1]}/1K)')
    else:
        print(f'  !! NO PRICING ENTRY for {args.model_name} -- cost NOT estimated.')
        print(f'     You would be spending blind. Add it to PRICING in')
        print(f'     check_cache_coverage.py (rates are in litellm\'s model cost map,')
        print(f'     .venv/lib/*/site-packages/litellm/model_prices_and_context_window_backup.json).')
    print()


if __name__ == '__main__':
    main()
