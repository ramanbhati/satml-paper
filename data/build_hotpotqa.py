#!/usr/bin/env python3
"""
Rebuild data/hotpotqa.json from the official HotpotQA development set.

    python3 data/build_hotpotqa.py            # fetch, rebuild, compare
    python3 data/build_hotpotqa.py --write    # ... and overwrite data/hotpotqa.json
    python3 data/build_hotpotqa.py --source hotpot_dev_distractor_v1.json

The corpus is the first N (default 200) questions of the dev set in the
distractor setting, in their original order, with no sampling or filtering.
Each question keeps the source's paragraphs in source order, each paragraph's
sentences joined with a single space. HotpotQA ships no attacker targets, so the
attack fields are empty and the corpus is used for clean-arm measurements only.

By default the rows are fetched from the Hugging Face datasets server
(hotpotqa/hotpot_qa, config distractor, split validation), which serves the same
split as the original hotpot_dev_distractor_v1.json. The original download host
has been unreliable, so --source also accepts a local copy of that file.

Standard library only. Without --write, the script rebuilds in memory and
reports whether the result is byte-identical to the shipped file.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

HF_ROWS = ('https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa'
           '&config=distractor&split=validation&offset={offset}&length={length}')
HF_PAGE = 100  # the datasets server returns at most 100 rows per request
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, 'hotpotqa.json')


def from_official(x):
    """A row of hotpot_dev_distractor_v1.json, in the common shape below."""
    return {'id': x['_id'], 'question': x['question'], 'answer': x['answer'],
            'type': x['type'], 'level': x['level'],
            'paragraphs': [(t, s) for t, s in x['context']],
            'support': [t for t, _ in x['supporting_facts']]}


def from_hf(x):
    """A row from the Hugging Face datasets server, in the same shape."""
    return {'id': x['id'], 'question': x['question'], 'answer': x['answer'],
            'type': x['type'], 'level': x['level'],
            'paragraphs': list(zip(x['context']['title'], x['context']['sentences'])),
            'support': list(x['supporting_facts']['title'])}


def fetch_hf(n):
    rows = []
    while len(rows) < n:
        url = HF_ROWS.format(offset=len(rows), length=min(HF_PAGE, n - len(rows)))
        with urllib.request.urlopen(url, timeout=120) as r:
            page = json.loads(r.read().decode('utf-8'))['rows']
        if not page:
            break
        rows += [from_hf(p['row']) for p in page]
    return rows


def convert(item):
    return {
        'question': item['question'],
        'correct answer': [item['answer']],
        'expanded answer': [],
        'incorrect answer': [],
        'incorrect_context': [],
        'context': [{'title': title, 'text': ' '.join(sentences)}
                    for title, sentences in item['paragraphs']],
        'supporting_titles': item['support'],
        'hop_type': item['type'],
        'level': item['level'],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--source', help='local copy of hotpot_dev_distractor_v1.json '
                                     '(default: fetch from the Hugging Face datasets server)')
    ap.add_argument('--n', type=int, default=200, help='number of questions (default 200)')
    ap.add_argument('--write', action='store_true', help='overwrite data/hotpotqa.json')
    a = ap.parse_args()

    if a.source:
        with open(a.source, encoding='utf-8') as fh:
            dev = [from_official(x) for x in json.load(fh)[:a.n]]
    else:
        print('fetching %d rows from the Hugging Face datasets server...' % a.n)
        dev = fetch_hf(a.n)
    if len(dev) < a.n:
        print('only %d rows available, expected %d' % (len(dev), a.n))
        return 1

    out = json.dumps([convert(x) for x in dev])
    digest = hashlib.sha256(out.encode('utf-8')).hexdigest()
    print('built %d items, first id %s, last id %s, sha256 %s'
          % (len(dev), dev[0]['id'], dev[-1]['id'], digest[:16]))

    if a.write:
        with open(TARGET, 'w', encoding='utf-8') as fh:
            fh.write(out)
        print('wrote %s' % TARGET)
        return 0
    if not os.path.exists(TARGET):
        print('no %s to compare against; rerun with --write' % TARGET)
        return 1
    with open(TARGET, encoding='utf-8') as fh:
        same = fh.read() == out
    print('byte-identical to data/hotpotqa.json' if same
          else 'DIFFERS from data/hotpotqa.json')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
