#!/usr/bin/env python
"""
Fetch the JailbreakBench harmful-behaviour goals into data/jailbreak_prompts.json
for the GuardrailTrigger availability attack.

Source: JailbreakBench (Chao et al., NeurIPS Datasets & Benchmarks 2024)
        https://jailbreakbench.github.io

We do not vendor these strings in the repo -- see data/JAILBREAK_PROMPTS.md.

Usage:
    python3 fetch_jailbreak_prompts.py
    python3 fetch_jailbreak_prompts.py --limit 20     # smaller set
"""
import argparse
import json
import os
import sys

OUT = os.path.join("data", "jailbreak_prompts.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N behaviours")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Need the datasets library:  .venv/bin/pip install datasets")

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    print(f"splits available: {list(ds.keys())}")

    # the harmful behaviours live in the 'harmful' split; fall back to the first
    # split if upstream renames it
    split = "harmful" if "harmful" in ds else list(ds.keys())[0]
    rows = ds[split]
    print(f"using split '{split}' with columns {rows.column_names} ({len(rows)} rows)")

    col = next((c for c in ("Goal", "goal", "prompt", "behavior") if c in rows.column_names), None)
    if col is None:
        sys.exit(f"Could not find a goal column in {rows.column_names}. "
                 "Inspect the dataset and set it manually.")

    goals = [r[col] for r in rows if r[col]]
    if args.limit:
        goals = goals[:args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(goals, f, indent=1)

    print(f"\nwrote {len(goals)} behaviours -> {args.out}")
    print("sanity check -- these should read as refusal-triggering requests:")
    for g in goals[:3]:
        print(f"   - {g[:90]}{'...' if len(g) > 90 else ''}")
    print("\nIf they do not look like harmful requests, stop and inspect the dataset: "
          "the attack depends on these reliably triggering a refusal.")


if __name__ == "__main__":
    main()
