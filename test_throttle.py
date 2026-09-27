#!/usr/bin/env python
"""
Does the rate-limit handling actually converge? Offline, no AWS, no cost.

    python3 test_throttle.py

WHY THIS EXISTS. The 20 Sep attacks run produced 222 empty responses out of
~1,100 calls, and every one became a row with an inflated n. The settings that
produced it (workers=2, delay=0.15s, retries=3) had been "validated" on the
certify runs -- which were served almost entirely from cache and so never put
load on the API at all. The throttle had never actually been tested.

So test it against a simulated Bedrock with a token bucket, rather than against
the real one at $0.0002 a call. The simulator is deliberately harsher than the
observed failure: a small bucket that refills slowly, and a hard concurrency
cap, both of which reject with the same RateLimitError shape litellm raises.
"""
import os
import sys
import threading
import time
import types

os.environ.setdefault('AWS_REGION_NAME', 'us-east-1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _stub_if_absent(name, build):
    """Install a minimal stand-in ONLY when the real module is unavailable.

    src/models.py imports torch and transformers at module scope. The retry
    logic under test touches neither, and requiring a GPU stack to test a
    back-off loop means the test does not get run. Where the real package is
    installed it is used unchanged, so this never masks the real thing.
    """
    try:
        __import__(name)
        return False
    except Exception:
        pass
    for part in name.split('.'):
        pass
    sys.modules[name] = build()
    return True


def _mk_torch():
    t = types.ModuleType('torch')
    t.cuda = types.SimpleNamespace(is_available=lambda: False)
    t.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    t.LongTensor = t.FloatTensor = object
    t.nn = types.SimpleNamespace(Module=object)
    t.eq = lambda *a, **k: None
    t.device = lambda *a, **k: 'cpu'
    t.tensor = lambda *a, **k: None
    t.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda s: None, __exit__=lambda s, *a: False)
    return t


def _mk_transformers():
    m = types.ModuleType('transformers')
    for n in ('LlamaTokenizer', 'LlamaForCausalLM', 'AutoTokenizer',
              'AutoModelForCausalLM', 'StoppingCriteria'):
        setattr(m, n, type(n, (), {}))
    m.StoppingCriteriaList = list
    return m


def _mk_litellm():
    m = types.ModuleType('litellm')
    m.completion = lambda **kw: None
    m.batch_completion = lambda **kw: []
    m.suppress_debug_info = True
    m.num_retries = 0
    return m


def _mk_openai():
    m = types.ModuleType('openai')
    m.OpenAI = type('OpenAI', (), {})
    return m


def _mk_joblib():
    # src/models.py uses joblib only to persist the response cache, which the
    # retry loop under test never touches.
    m = types.ModuleType('joblib')
    m.dump = m.load = lambda *a, **k: None
    return m


def _mk_numpy():
    # src/helper.py imports numpy at module scope for helpers the retry loop
    # never calls.
    return types.ModuleType('numpy')


STUBBED = [n for n, b in (('torch', _mk_torch), ('transformers', _mk_transformers),
                          ('litellm', _mk_litellm), ('openai', _mk_openai),
                          ('joblib', _mk_joblib), ('numpy', _mk_numpy))
           if _stub_if_absent(n, b)]
if STUBBED:
    print(f"(using stand-ins for absent packages: {', '.join(STUBBED)};\n"
          f" the retry loop under test does not use them)\n")


class FakeRateLimitError(Exception):
    """Same shape litellm raises: class name contains 'RateLimit'."""


class FakeBedrock:
    """Token bucket with a PENALTY for hammering a drained bucket.

    A plain bucket is too forgiving to reproduce what we saw: it recovers the
    moment load pauses, so even a bad client eventually gets through and the
    test passes no matter what. The real run did not look like that. The first
    thirteen queries succeeded and then every single one failed -- the limiter
    stayed shut while pressure continued.

    So a rejected request costs `penalty` seconds of refill. Hammer a drained
    bucket and you keep it drained; back off and it recovers. That is the
    property the shared breaker exists to exploit, and the property a
    per-thread retry loop cannot.
    """

    def __init__(self, capacity=10, refill_per_sec=6.0, max_concurrent=3,
                 penalty=0.35):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill = refill_per_sec
        self.max_concurrent = max_concurrent
        self.penalty = penalty
        self.inflight = 0
        self.lock = threading.Lock()
        self.last = time.time()
        self.served = 0
        self.rejected = 0
        self.peak_inflight = 0

    def call(self):
        with self.lock:
            now = time.time()
            self.tokens = min(self.capacity,
                              self.tokens + (now - self.last) * self.refill)
            self.last = now
            if self.inflight >= self.max_concurrent or self.tokens < 1.0:
                self.rejected += 1
                self.tokens -= self.penalty * self.refill   # pushing back costs you
                raise FakeRateLimitError(
                    'BedrockException - {"message":"Too many requests, '
                    'please wait before trying again."}')
            self.tokens -= 1.0
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            time.sleep(0.002)
            return 'paris'
        finally:
            with self.lock:
                self.inflight -= 1
                self.served += 1


def run_case(label, n_prompts, workers, delay, retries, base, use_breaker):
    import src.models as M
    M._RL_PAUSE_UNTIL = 0.0
    M._RL_TRIPS = 0
    api = FakeBedrock()

    class Stub:
        num_retries = retries
        _request_delay = delay
        _backoff_base = base
        _backoff_max = 8.0
        _abort_after = 0
        max_output_tokens = 20
        temperature = 0
        litellm_model = 'bedrock/fake'
        aws_region_name = 'us-east-1'
        max_workers = workers

        def _messages(self, p):
            return [{'role': 'user', 'content': p}]

        def _clean_response(self, r):
            return r.strip()

        _note_failure = M.BedrockModel._note_failure
        _query = M.BedrockModel._query
        _batch_query = M.BedrockModel._batch_query

    stub = Stub()

    class _Resp:
        def __init__(self, t):
            self.choices = [type('C', (), {'message': type('M', (), {'content': t})()})()]

    def fake_completion(**kw):
        return _Resp(api.call())

    real = M.litellm.completion
    M.litellm.completion = fake_completion
    if not use_breaker:                      # emulate the OLD per-thread behaviour
        M._rl_hold = lambda: None
        M._rl_trip = lambda s: 0
    else:
        M._rl_hold = _orig_hold
        M._rl_trip = _orig_trip
    t0 = time.time()
    try:
        out = stub._batch_query([f'q{i}' for i in range(n_prompts)])
    finally:
        M.litellm.completion = real
        M._rl_hold, M._rl_trip = _orig_hold, _orig_trip
    empty = sum(1 for r in out if not str(r).strip())
    print(f"  {label:34s} empty={empty:3d}/{n_prompts}  "
          f"rejected={api.rejected:4d}  peak_inflight={api.peak_inflight}  "
          f"{time.time() - t0:5.1f}s")
    return empty


import src.models as M                                     # noqa: E402
_orig_hold, _orig_trip = M._rl_hold, M._rl_trip

print("simulated Bedrock: bucket=10, refill=6/s, max_concurrent=3, "
      "penalty=0.35s per rejection\n")

# A reproduces the OLD code: no shared breaker, and near-instant retries, which
# is what litellm's num_retries does. If this does not lose calls the test is
# not exercising anything and the later PASS means nothing.
print("A. old behaviour: per-thread retries, no breaker  (must FAIL to be a test)")
old = run_case('workers=2 delay=0.15 retries=3', 120, 2, 0.15, 3, 0.02, False)

print("\nB. identical load and retry budget, WITH the shared breaker")
new = run_case('workers=2 delay=0.15 retries=3', 120, 2, 0.15, 3, 0.5, True)

print("\nC. the conservative settings recommended for the rerun")
safe = run_case('workers=1 delay=0.2 retries=8', 120, 1, 0.2, 8, 0.5, True)

print("\n" + "=" * 66)
fails = []
if old == 0:
    fails.append("case A lost nothing, so the simulator never reproduces the "
                 "failure and cases B/C prove nothing. Make it harsher.")
if new >= old:
    fails.append(f"breaker did not reduce empties ({old} -> {new})")
if safe != 0:
    fails.append(f"conservative settings still dropped {safe} calls")
if fails:
    for f in fails:
        print("FAIL:", f)
    sys.exit(1)
print(f"PASS: the old scheme dropped {old}/120 under a limiter that punishes")
print(f"      hammering; the shared breaker dropped {new}, and the conservative")
print(f"      settings dropped {safe}.")
print("      A zero here is necessary, not sufficient -- the real limit is")
print("      account-wide and shared with anything else using it.")
