#!/usr/bin/env python
"""
List the Bedrock models THIS account can actually invoke, in THIS region.

    python3 list_bedrock_models.py
    python3 list_bedrock_models.py --provider anthropic amazon google

WHY THIS EXISTS. Bedrock model IDs are not guessable and not stable: some need a
cross-region inference profile prefix ("us.meta.llama3-1-8b-instruct-v1:0"),
some do not ("mistral.mistral-7b-instruct-v0:2"), availability differs by region,
and a model present in the catalog may still be un-invokable until access is
granted in the console. Hard-coding an ID from documentation and discovering the
mistake mid-run wastes the run.

This asks the account what it has. Nothing is invoked, so it costs nothing.

Output is grouped by provider with the exact modelId string to paste into
src/models.py, plus whether an inference profile is required.
"""
import argparse
import json
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--region', default=None, help='default: AWS_REGION_NAME or us-east-1')
p.add_argument('--provider', nargs='*', default=None,
               help='filter, e.g. anthropic amazon google meta mistral')
p.add_argument('--text-only', action='store_true', default=True,
               help='only models with TEXT output (default on)')
args = p.parse_args()

import os
region = args.region or os.environ.get('AWS_REGION_NAME', 'us-east-1')


def aws(*cmd):
    r = subprocess.run(['aws'] + list(cmd) + ['--region', region, '--output', 'json'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'!! aws {" ".join(cmd)} failed:\n{r.stderr.strip()[:400]}', file=sys.stderr)
        return None
    return json.loads(r.stdout or '{}')


print(f'\nregion: {region}\n')

fms = aws('bedrock', 'list-foundation-models')
if fms is None:
    sys.exit('could not list foundation models -- check credentials with ./check_bedrock.sh')

# Cross-region inference profiles: many newer models are ONLY invokable through
# one of these, under an id prefixed us./eu./apac.
profs = aws('bedrock', 'list-inference-profiles') or {}
prof_ids = {p_.get('inferenceProfileId', '') for p_ in profs.get('inferenceProfileSummaries', [])}
prof_models = {}
for p_ in profs.get('inferenceProfileSummaries', []):
    pid = p_.get('inferenceProfileId', '')
    for m in p_.get('models', []) or []:
        arn = m.get('modelArn', '')
        if '/' in arn:
            prof_models.setdefault(arn.rsplit('/', 1)[1], set()).add(pid)

by_provider = {}
for m in fms.get('modelSummaries', []):
    if args.text_only and 'TEXT' not in (m.get('outputModalities') or []):
        continue
    prov = (m.get('providerName') or '?').lower()
    if args.provider and not any(f.lower() in prov for f in args.provider):
        continue
    by_provider.setdefault(prov, []).append(m)

if not by_provider:
    print('no matching models. Try without --provider, or check region/access.')
    sys.exit(0)

for prov in sorted(by_provider):
    print(f'=== {prov} ===')
    for m in sorted(by_provider[prov], key=lambda x: x.get('modelId', '')):
        mid = m.get('modelId', '')
        streaming = m.get('responseStreamingSupported')
        inf = m.get('inferenceTypesSupported') or []
        on_demand = 'ON_DEMAND' in inf
        prof = sorted(prof_models.get(mid, []))
        note = []
        if not on_demand:
            note.append('NO on-demand')
        if prof:
            note.append('use profile: ' + prof[0])
        print(f'  {mid:<52} {"":2}{"  ".join(note)}')
    print()

print('HOW TO USE THIS')
print('  1. Pick one model per family. Prefer the CHEAPEST capable variant --')
print('     these experiments are ~7000 calls per model.')
print('  2. If a "use profile:" note appears, the profile id is what goes in')
print('     src/models.py, not the bare modelId.')
print('  3. "NO on-demand" on the BASE id does NOT mean skip it. If a')
print('     "use profile:" note is also present, the model is invokable')
print('     through that profile -- llama8b and llama70b in BEDROCK_MODELS are')
print('     both listed NO on-demand and both run fine via their us. profile.')
print('     Only a model with NO on-demand and NO profile needs provisioned')
print('     throughput; that is the one to skip.')
print('  4. Paste the ids into BEDROCK_MODELS in src/models.py, then run')
print('     calibrate_model.py before any experiment.')
print()
