#!/usr/bin/env python
"""Print the single top_k value in a results CSV, or MIXED / empty.

Exists as a file rather than an inline heredoc because bash 3.2 (which macOS
still ships) mis-parses a heredoc inside $( ) command substitution -- it tries
to execute the Python body as shell. `bash -n` on newer bash does not catch it.
"""
import csv
import sys

try:
    with open(sys.argv[1]) as fh:
        ks = {r.get("top_k", "") for r in csv.DictReader(fh)}
    ks = sorted(k for k in ks if k)
    print(ks[0] if len(ks) == 1 else ("MIXED" if ks else ""))
except Exception:
    print("")
