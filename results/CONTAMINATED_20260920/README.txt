QUARANTINED 20 Sep 2026 -- DO NOT USE.

The certify run of 20 Sep completed and wrote full-looking CSVs, but four of
its six invocations were contaminated by Bedrock rate limiting:

  mistral7b  n       9 empty responses   (81 rate-limit errors across the run)
  mistral7b  floor2 22 empty responses
  llama8b    n        0  -- clean
  llama8b    floor2   0  -- clean
  llama70b   n       33 empty responses
  llama70b   floor2  38 empty responses

An empty response is counted as NON-ABSTAINING, so it inflates n, raises mu,
and biases certification toward looking healthier than it is. Contamination was
also UNEVEN between the two arms (9 vs 22; 33 vs 38), which corrupts exactly
the paired n-vs-floor2 comparison these files exist to support. The reported
tau gaps (+0.01, +0.02) are smaller than the contamination that produced them.

Kept only as evidence of the failure mode. Superseded by the rerun under
throttled settings (BEDROCK_MAX_WORKERS=2, REQUEST_DELAY=0.15, NUM_RETRIES=3).
