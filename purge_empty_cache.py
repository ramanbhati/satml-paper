#!/usr/bin/env python
"""
Remove empty responses from the LLM response cache.

    python3 purge_empty_cache.py                 # report only
    python3 purge_empty_cache.py --apply         # actually purge

WHY THIS EXISTS. BedrockModel._query() returns "" when a call fails after all
retries (throttling / quota). Until the fix in src/models.py, that empty string
was written to the cache, which made the failure PERMANENT:

  - every later run hits the cache, gets "", and treats it as a real answer;
  - an empty response does NOT contain "I don't", so the keyword defense counts
    it as a NON-ABSTAINING group -- inflating n, and therefore inflating
    mu = min(alpha*n, beta), which makes the attacker look WEAKER than it is;
  - the contamination is invisible in the CSV: the row looks ordinary.

src/models.py no longer caches empty responses, but entries written before that
fix are still on disk. This removes them so the next run re-queries them.

Purging is always safe: a missing cache entry costs one re-query. Note that
BaseModel.batch_query_from_cache is all-or-nothing, so removing one passage's
entry means that query's whole top-k batch is re-issued -- the report below
counts that properly.
"""
import argparse
import glob
import os
import shutil
import time

import joblib

p = argparse.ArgumentParser()
p.add_argument('--apply', action='store_true',
               help='actually rewrite the cache files (default: report only)')
p.add_argument('--cache_dir', default='cache')
args = p.parse_args()

paths = sorted(glob.glob(os.path.join(args.cache_dir, '*.z')))
if not paths:
    raise SystemExit(f'no cache files in {args.cache_dir}/')

total_entries = total_empty = 0
print()
print(f"{'cache file':<44} {'entries':>9} {'empty':>7} {'pct':>7}")
print('-' * 72)
for path in paths:
    try:
        cache = joblib.load(path)
    except Exception as e:
        print(f'{os.path.basename(path):<44} !! unreadable: {e}')
        continue
    empty = [k for k, v in cache.items() if not str(v).strip()]
    total_entries += len(cache)
    total_empty += len(empty)
    pct = (100.0 * len(empty) / len(cache)) if cache else 0.0
    flag = '  <-- POISONED' if empty else ''
    print(f'{os.path.basename(path):<44} {len(cache):>9} {len(empty):>7} {pct:>6.1f}%{flag}')

    if empty and args.apply:
        backup = f'{path}.{int(time.time())}.bak'
        shutil.copy2(path, backup)
        for k in empty:
            del cache[k]
        joblib.dump(cache, path)
        print(f'{"":<44} purged; backup at {os.path.basename(backup)}')

print('-' * 72)
print(f'{"TOTAL":<44} {total_entries:>9} {total_empty:>7} '
      f'{(100.0*total_empty/total_entries if total_entries else 0):>6.1f}%')
print()
if total_empty == 0:
    print('Clean -- no empty responses cached.')
elif args.apply:
    print(f'Purged {total_empty} empty entries. Re-run the affected configs;')
    print('those calls will be re-issued and (with the models.py fix) a failure')
    print('will no longer be written back.')
else:
    print(f'{total_empty} empty entries found. Re-run with --apply to purge them.')
    print('Until then, any run touching those prompts silently reuses a failure.')
print()
