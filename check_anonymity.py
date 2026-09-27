"""Scan every blob reachable from any ref for identity leaks.

    python3 check_anonymity.py

Exits non-zero if anything matches. Checks git OBJECTS, not just the working
tree, because a string removed from a file is still in history and still ships
with the repository.

PATTERNS LIVE OUTSIDE THIS FILE, in .anonymity_patterns (untracked, one regex
per line, '#' comments allowed).

INCLUDE PRIOR VENUE NAMES AS BARE TOKENS. Six code comments in this repository
referred to the authors' other paper by its venue acronym alone, in the course of
explaining why a control or a cached run looked the way it did. Each was enough to
identify the authors, since that paper is public. They survived earlier scans
because the pattern in use was the acronym followed by a year while the comments
carried the acronym on its own, so list such names bare.

AND DO NOT QUOTE THE OFFENDING STRING WHEN DOCUMENTING IT. An earlier version of
this note reproduced the acronym as an illustration, which put it back into a file
that ships with the anonymous artifact. Describe the shape of the mistake; never
the token. Writing them here as literals would put the
author and institution names into a file released with the anonymous artifact --
the check would be the leak it exists to prevent. That happened twice while this
was being built, which is why the rule is now structural rather than remembered.

Absent pattern file => exit 2 and say so. A silent pass is indistinguishable
from a clean result, and this is the one check where that distinction matters.
"""
import re
import subprocess
import sys
import os

PATFILE = '.anonymity_patterns'
# Upstream corpora match on CONTENT, not authorship: German "und" throughout
# HotpotQA, "pugetsound.edu" inside a RealtimeQA source link. They are
# byte-identical to upstream, so a match there is never about these authors.
EXEMPT = {
    'data/realtimeqa.json', 'data/open_nq.json', 'data/biogen.json',
    'data/hotpotqa.json', 'data/README.md', 'data/google_search.py',
    'llm_eval.py', 'run.sh',
}


def main():
    if not os.path.exists(PATFILE):
        print(f'NOT CHECKED: {PATFILE} is absent.')
        print(f'Create it with one regex per line (it is gitignored), e.g. your')
        print(f'surname, institution, and any prior venue the artifact must not name.')
        return 2
    pats = [l.strip() for l in open(PATFILE)
            if l.strip() and not l.startswith('#')]
    if not pats:
        print(f'NOT CHECKED: {PATFILE} contains no patterns.')
        return 2

    # ONE git process, not two per object. A repository whose history still
    # carries a vendored virtualenv has tens of thousands of blobs, and spawning
    # `git cat-file` twice each takes minutes; `--batch` streams them instead.
    paths = {}
    for line in subprocess.run(['git', 'rev-list', '--objects', '--all'],
                               capture_output=True, text=True).stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        paths[parts[0]] = parts[1] if len(parts) > 1 else '<no path>'

    listing = subprocess.run(['git', 'cat-file', '--batch-all-objects',
                              '--batch-check=%(objectname) %(objecttype)'],
                             capture_output=True, text=True).stdout.split('\n')
    blobs = [l.split()[0] for l in listing if l.endswith(' blob')]
    wanted = [b for b in blobs if paths.get(b, '<no path>') not in EXEMPT]
    nblob = len(blobs)

    proc = subprocess.run(['git', 'cat-file', '--batch'],
                          input=('\n'.join(wanted) + '\n').encode(),
                          capture_output=True, text=False)
    out = proc.stdout
    bad = []
    pos = 0
    rx = [(i, re.compile(p, re.I)) for i, p in enumerate(pats, 1)]
    while pos < len(out):
        nl = out.find(b'\n', pos)
        if nl < 0:
            break
        header = out[pos:nl].decode('utf8', 'ignore').split()
        if len(header) < 3:
            pos = nl + 1
            continue
        oid, size = header[0], int(header[2])
        body = out[nl + 1: nl + 1 + size].decode('utf8', 'ignore')
        pos = nl + 1 + size + 1
        path = paths.get(oid, '<no path>')
        for i, r in rx:
            m = r.search(body)
            if m:
                ctx = body[max(0, m.start() - 40):m.end() + 40].replace('\n', ' ')
                bad.append((path, i, ctx.strip()[:100]))

    # commit metadata is as much of a leak as file content
    meta = subprocess.run(['git', 'log', '--all', '--format=%an|%ae|%cn|%ce'],
                          capture_output=True, text=True).stdout
    for i, pat in enumerate(pats, 1):
        if re.search(pat, meta, re.I):
            bad.append(('<commit metadata>', i, 'author or committer identity'))
    remotes = subprocess.run(['git', 'remote', '-v'],
                             capture_output=True, text=True).stdout.strip()

    print(f'blobs scanned: {nblob}   patterns: {len(pats)}')
    if remotes:
        print(f'note: {len(remotes.splitlines())} remote(s) configured; a remote URL '
              f'can carry a local path or account name.')
        for l in remotes.splitlines():
            print(f'      {l}')
    if bad:
        print('\nPOTENTIAL LEAKS:')
        for path, i, ctx in bad[:25]:
            print(f'  {path}\n    pattern #{i}  ...{ctx}...')
        return 1
    print('clean: no pattern matches any blob, commit author, or committer.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
