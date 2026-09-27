# Jailbreak prompts for the GuardrailTrigger attack

The `GuardrailTrigger` availability attack needs a set of prompts that reliably
trigger an aligned model's safety guardrails. We deliberately **do not vendor
harmful strings in this repository**. Populate them from the standard public
benchmark instead.

## What to do

Download the JailbreakBench behaviours (Chao et al., *JailbreakBench: An Open
Robustness Benchmark for Jailbreaking Large Language Models*, NeurIPS Datasets
& Benchmarks 2024) from <https://jailbreakbench.github.io> and write the goal
strings to:

    data/jailbreak_prompts.json

as a flat JSON list of strings:

```json
["<behaviour 1>", "<behaviour 2>", "..."]
```

MutedRAG (Suo et al., arXiv:2504.21680) uses the same source: 100 harmful
behaviours spanning 10 OpenAI usage-policy categories, roughly 55% original to
JailbreakBench with the remainder drawn from AdvBench and TDC/HarmBench.

## Why one prompt per poisoned passage

`GuardrailTrigger` cycles through the list so that each poisoned passage carries
a *different* behaviour. Every malicious text is therefore unique, which is what
defeats SHA-256 duplicate-text filtering (MutedRAG Sec. 5.3). Using a single
repeated prompt would make the attack trivially filterable.

## Ethics / artifact note

For the SaTML submission's Open Science artifact, point at JailbreakBench rather
than redistributing the behaviours. The attack mechanism is the contribution;
the harmful strings are an existing, citable, already-public benchmark.

`data/jailbreak_prompts.json` is gitignored for this reason — see `.gitignore`.
