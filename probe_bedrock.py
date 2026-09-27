#!/usr/bin/env python
"""
Is Bedrock reachable right now, and at what rate? One tiny call at a time.

    python3 probe_bedrock.py                          # 1 call
    python3 probe_bedrock.py --repeat 20 --delay 0.5  # sustainable-rate test
    python3 probe_bedrock.py --model llama70b-bedrock

WHY. When a 100-query run dies at 0/100 with "Too many connections", the useful
question is not "how do I make the run work" but "can this account make ONE call
right now". Starting a full run to find out is slow and can deepen the block,
because every failed query retries up to num_retries times and each retry is
another connection.

This sends one ~20-token request per iteration and reports exactly what comes
back. Cost is negligible (a few thousand tokens even at --repeat 50).

IMPORTANT: run this with BEDROCK_NUM_RETRIES=0 (the default here) so a failure
is reported immediately instead of being retried into a connection storm.

Reading the output:
  - all succeed at --delay 0        -> the block has cleared; rerun normally
  - succeed only at a larger delay  -> use that as BEDROCK_REQUEST_DELAY
  - first call fails at any delay   -> not a rate problem. The account is
                                       blocked or credentials/region are wrong.
                                       Wait, and check for orphaned processes
                                       still holding connections.
"""
import argparse
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--model', default='mistral7b-bedrock')
p.add_argument('--repeat', type=int, default=1)
p.add_argument('--delay', type=float, default=0.0, help='seconds between calls')
p.add_argument('--retries', type=int, default=0,
               help='per-call retries; keep at 0 so failures are reported, not amplified')
args = p.parse_args()

# set BEFORE importing the model so the constructor picks them up
os.environ['BEDROCK_NUM_RETRIES'] = str(args.retries)
os.environ['BEDROCK_REQUEST_DELAY'] = '0'   # this script does its own pacing
os.environ.setdefault('AWS_REGION_NAME', 'us-east-1')
os.environ.setdefault('BEDROCK_MAX_WORKERS', '1')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import create_model  # noqa: E402

print(f"\nmodel={args.model}  region={os.environ['AWS_REGION_NAME']}  "
      f"retries={args.retries}  delay={args.delay}s  repeat={args.repeat}\n")

llm = create_model(args.model, cache_path=None, max_output_tokens=20)
PROMPT = "Answer with one word. What is the capital of France?"

ok = fail = 0
lat = []
for i in range(1, args.repeat + 1):
    if args.delay and i > 1:
        time.sleep(args.delay)
    t0 = time.time()
    try:
        r = llm._query(PROMPT)          # bypass the cache deliberately
    except Exception as e:              # _query swallows, but be safe
        r = ''
        print(f"  [{i:>3}] EXCEPTION {type(e).__name__}: {str(e)[:110]}")
    dt = time.time() - t0
    if str(r).strip():
        ok += 1
        lat.append(dt)
        print(f"  [{i:>3}] ok    {dt:>6.2f}s  {str(r).strip()[:60]!r}")
    else:
        fail += 1
        print(f"  [{i:>3}] FAIL  {dt:>6.2f}s  (empty response -- see the warning line above)")

print()
print(f"  succeeded {ok}/{args.repeat}   failed {fail}/{args.repeat}")
if lat:
    print(f"  latency: min {min(lat):.2f}s  median {sorted(lat)[len(lat)//2]:.2f}s  max {max(lat):.2f}s")
print()
if fail == 0:
    print("  => Bedrock is answering. If a full run still fails, the run's own")
    print("     concurrency is the problem: keep BEDROCK_MAX_WORKERS=1 and set")
    print(f"     BEDROCK_REQUEST_DELAY at or above {args.delay or 0.3}.")
elif ok == 0:
    print("  => EVERY call failed, including the first. This is not a rate problem.")
    print("     Check, in order:")
    print("       1. orphaned runs still holding connections:")
    print("            ps aux | grep -i '[m]ain.py'      (kill any you find)")
    print("       2. credentials / region:  ./check_bedrock.sh")
    print("       3. an account-level block that needs time to clear -- wait")
    print("          10-15 minutes and probe again before starting any run.")
else:
    print(f"  => Partial: {ok} of {args.repeat} got through. The account is rate-limited")
    print(f"     rather than blocked. Re-probe with a larger --delay until it is")
    print("     clean, then use that value as BEDROCK_REQUEST_DELAY.")
print()
sys.exit(0 if fail == 0 else 1)
