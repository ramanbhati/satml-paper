#!/usr/bin/env python
"""Print the distinct alpha values in a per-query CSV, comma-separated.

Companion to csv_top_k.py. The runner uses it to refuse appending rows at one
alpha into a file recorded at another: analyze_omega.py reads alpha from the
FIRST row and applies it to every prediction in the file, so a mixed-alpha file
silently mispredicts every boundary it draws.
"""
import csv, sys
try:
    with open(sys.argv[1], newline='') as fh:
        print(','.join(sorted({r['alpha'].strip() for r in csv.DictReader(fh)
                               if r.get('alpha', '').strip()})))
except Exception:
    print('')
