import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteriaList
from .helper import StopOnTokens

from torch import LongTensor, FloatTensor

import json
import logging
import time
logger = logging.getLogger('RRAG-main')

# ---- MPS/CUDA/CPU device auto-detection (patched for Apple Silicon) ----
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")
# -----------------------------------------------------------------------

import litellm
from litellm import batch_completion
from openai import OpenAI
import os
# os.environ["OPENAI_API_KEY"] = ""
import joblib
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from .prompt_template import *

# ---------------------------------------------------------------------------
# GLOBAL RATE-LIMIT CIRCUIT BREAKER
#
# A cache-cold run here produced 222 empty responses out of roughly 1,100 calls.
# The failures did not start at the burst: the first thirteen queries succeeded and
# everything from query fourteen on failed. That is an account token bucket
# draining and never being allowed to refill, not a momentary spike.
#
# It kept draining because back-off was PER THREAD. Each worker caught its own
# 429, slept briefly, and retried into a bucket the other workers were still
# hammering. litellm's num_retries has the same shape, which is why lowering it
# to 3 changed nothing: three fast retries into a drained bucket fail three
# times. The certify runs looked clean only because they were served from cache
# and barely touched the API, so these settings had never met real load.
#
# The fix is a breaker shared by every worker. One 429 anywhere parks ALL of
# them until a common deadline, so the bucket gets a quiet window to refill.
_RL_LOCK = threading.Lock()
_RL_PAUSE_UNTIL = 0.0
_RL_TRIPS = 0


def _rl_hold():
    """Block while the breaker is open. Called before every request."""
    while True:
        with _RL_LOCK:
            remaining = _RL_PAUSE_UNTIL - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.5))


def _rl_trip(seconds):
    """Open the breaker for at least `seconds`, for every worker."""
    global _RL_PAUSE_UNTIL, _RL_TRIPS
    with _RL_LOCK:
        _RL_PAUSE_UNTIL = max(_RL_PAUSE_UNTIL, time.time() + seconds)
        _RL_TRIPS += 1
        return _RL_TRIPS


def _rl_stats():
    with _RL_LOCK:
        return _RL_TRIPS


def _is_rate_limit(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return ('ratelimit' in name or 'throttl' in name
            or 'too many requests' in text or 'throttlingexception' in text
            or 'rate limit' in text or 'too many connections' in text)
MAX_NEW_TOKENS = 20
CONTEXT_MAX_TOKENS = {'mistralai/Mistral-7B-Instruct-v0.2': 8192,
                      'mistralai/Mistral-7B-Instruct-v0.3': 32768,
                      'meta-llama/Llama-2-7b-chat-hf': 4096,
                      'meta-llama/Llama-2-13b-chat-hf': 4096,
                      'meta-llama/Meta-Llama-3-8B-Instruct': 8192,
                      'meta-llama/Llama-3.2-3B-Instruct': 131072,
                      'mistralai/Mixtral-8x7B-Instruct-v0.1': 32000,
                      'gpt-3.5-turbo-0125':16385,
                      'gpt-4-0125-preview':128000,
                      'lmsys/vicuna-7b-v1.5': 4096,
                      'lmsys/vicuna-13b-v1.5': 4096}


def create_model(model_name,prompt_style='robustrag',**kwargs):
    # prompt_style swaps ONLY the plain 'qa' template (see prompt_template.py).
    # 'robustrag' = upstream keyword-only prompt; 'mutedrag' = conversational
    # prompt used by MutedRAG, which leaves room for a refusal.
    #
    # Patch the constructed model's own template dict rather than the module-level
    # globals: mutating globals would leak across calls within a process and could
    # not be undone by a later robustrag call.
    model = _create_model(model_name,**kwargs)
    if prompt_style != 'robustrag':
        model.prompt_template = apply_prompt_style(model.prompt_template, prompt_style)
    return model


# ---------------------------------------------------------------------------
# Bedrock model registry.
#
# One place to correct model ids. They are NOT guessable: some require a
# cross-region inference profile prefix (us.), some do not, and availability
# differs by region and by account access grants. Run list_bedrock_models.py to
# see what THIS account can actually invoke, then fix any id here.
#
# `system_prompt` is the load-bearing field, not the id. RobustRAG's templates
# are few-shot completions expecting a terse continuation after "Answer:".
# Chat-tuned models instead answer conversationally, and that verbosity collapses
# the keyword defense (clean accuracy 60% -> 10% in earlier testing) because the
# extractor pulls keywords out of prose rather than a short answer. But the fix
# is not universal: Mistral needs the steering prompt, Llama OVER-ABSTAINS with
# it. Every new family must therefore be calibrated empirically --
# calibrate_model.py runs both settings on a small sample and reports which one
# produces terse, keyword-like answers. Do not guess this field.
#
#   None                      -> raw completion, no system turn (Llama default)
#   BedrockModel.SYSTEM_PROMPT -> terse-answer steering (Mistral default)
#
# `template` selects the prompt dict. For the modes these experiments use
# ('qa' for isolated answering, 'qa-hint' for aggregation) MISTRAL_TMPL and
# LLAMA_TMPL are IDENTICAL -- they differ only in 'qa-mc', which is unused here.
# So template choice is currently immaterial; it is kept explicit so a future
# family that needs its own phrasing has somewhere to put it.
BEDROCK_MODELS = {
    # name                     bedrock id                                   family     params  system_prompt
    # --- verified working (used in all published runs) ---
    'mistral7b-bedrock': dict(
        id='mistral.mistral-7b-instruct-v0:2', family='mistral',
        params_b=7, template='MISTRAL_TMPL', system_prompt='terse'),
    'llama8b-bedrock': dict(
        id='us.meta.llama3-1-8b-instruct-v1:0', family='llama',
        params_b=8, template='LLAMA_TMPL', system_prompt=None),
    'llama70b-bedrock': dict(
        id='us.meta.llama3-3-70b-instruct-v1:0', family='llama',
        params_b=70, template='LLAMA_TMPL', system_prompt=None),

    # --- llama3b: DEPRECATED BY AWS. Bedrock now answers
    #     "This model version has reached the end of its life."
    #     The id was inherited from an older branch that worked months ago --
    #     Bedrock ids ROT, and a registry entry is only as good as the date it
    #     was last verified. Kept here, marked, so nobody re-discovers this the
    #     expensive way; create_model refuses it with a readable message instead
    #     of letting Bedrock return a cryptic NotFoundError mid-run.
    #
    #     It was intended to complete a WITHIN-FAMILY size series 3B/8B/70B.
    #     SEARCHED 2026-09-16, us-east-1: there is NO Llama under 8B in this
    #     account. The whole Meta catalog is 8B, 70B (3.0/3.1/3.3) and the two
    #     Llama-4 17B-active MoE models. So the dense size series cannot be
    #     restored inside Meta, and a 17B MoE point would confound parameter
    #     count with both architecture (MoE vs dense) and generation (L4 vs L3).
    #     Re-check with list_bedrock_models.py --provider meta before assuming
    #     this is still true; the catalog moves.
    'llama3b-bedrock': dict(
        id='us.meta.llama3-2-3b-instruct-v1:0', family='llama',
        params_b=3, template='LLAMA_TMPL', system_prompt=None,
        status='EOL 2026-09: AWS end-of-life'),

    # --- NEW FAMILIES: id must be confirmed by list_bedrock_models.py for THIS
    #     account and region, and system_prompt stays 'UNCALIBRATED' until
    #     calibrate_model.py has run. create_model refuses an uncalibrated model.
    #
    #     nova-lite id VERIFIED 2026-09-16, us-east-1. The bare
    #     'amazon.nova-lite-v1:0' is listed but must be invoked through the
    #     cross-region inference profile, hence the 'us.' prefix.
    #     CALIBRATED 2026-09-16 (logs/cal_nova_{micro,lite,pro}.txt), n=25 on
    #     realtimeqa k=10. All three are terse (10-11 median chars, TERSER than
    #     both references), abstain 20-30% (references 23%/32%), and score
    #     64-76% clean (references 60%/62%). No empty responses. Usable.
    #
    #     ALL THREE ARE SET TO 'terse' DELIBERATELY, overriding what
    #     calibrate_model.py printed for nova-lite ('None'). The script picks
    #     per-model by (chars, abstain), which is the right rule when adding one
    #     INDEPENDENT family -- but these three are a within-family LADDER whose
    #     entire purpose is that capability is the only variable across rungs.
    #     Letting the setting differ between rungs would confound capability
    #     with prompt configuration and destroy the comparison. The script had
    #     no way to know that; it calibrates one model at a time.
    #
    #     Why 'terse' as the common setting, on the measured numbers:
    #       - the margins it was deciding on are noise. lite was picked by
    #         26% vs 24% abstention and identical chars; micro by ONE character.
    #       - clean accuracy at n=25 has SE ~9pp, so NOTHING in the acc column
    #         separates the settings. (Under 'terse' acc happens to run
    #         64/68/72 micro->pro, but each step is well inside noise. Do NOT
    #         cite that as evidence of anything.)
    #       - 'terse' gives the shorter p95 tail on pro (18 vs 27 chars), i.e.
    #         less prose risk, which is the failure mode that matters.
    #     If a run shows Nova behaving oddly, flipping all three to None is the
    #     first thing to try -- but flip ALL THREE, never one.
    #
    #     WATCH ITEM: kw/query is 2.5-2.7, below llama70b (3.9) and far below
    #     mistral (7.1). Nova produces few keywords. If clean accuracy collapses
    #     in the real run, check the ZERO-surviving-keyword rate first -- that
    #     is the mechanism that cost the 'k' threshold variant 15 points.
    #
    #     THE NOVA FAMILY IS A CAPABILITY LADDER, not a parameter ladder.
    #     Amazon publishes no parameter counts, so params_b stays None and
    #     `rank` carries the ordering instead (micro < lite < pro, Amazon's own
    #     tiering). The trend test is therefore ORDINAL -- a rank correlation
    #     over three points, not a regression on model size. Say it that way in
    #     the paper; do not imply a parameter axis we do not have.
    #     All three route through the Bedrock CONVERSE API (verified against the
    #     installed litellm 1.83.0), unlike mistral7b which routes to invoke.
    'nova-micro-bedrock': dict(
        id='us.amazon.nova-micro-v1:0', family='amazon',
        params_b=None, rank=1, template='MISTRAL_TMPL', system_prompt='terse'),
    'nova-lite-bedrock': dict(
        id='us.amazon.nova-lite-v1:0', family='amazon',
        params_b=None, rank=2, template='MISTRAL_TMPL', system_prompt='terse'),
    'nova-pro-bedrock': dict(
        id='us.amazon.nova-pro-v1:0', family='amazon',
        params_b=None, rank=3, template='MISTRAL_TMPL', system_prompt='terse'),

    # --- not requested yet; left gated so they are one edit away if wanted ---
    'claude-haiku-bedrock': dict(
        id='us.anthropic.claude-3-5-haiku-20241022-v1:0', family='anthropic',
        params_b=None, template='MISTRAL_TMPL', system_prompt='UNCALIBRATED'),
    'gemma-bedrock': dict(
        id='google.gemma-3-12b-it-v1:0', family='google',
        params_b=None, template='MISTRAL_TMPL', system_prompt='UNCALIBRATED'),
}


def bedrock_families():
    """model_name -> family, for reporting how many FAMILIES a result covers."""
    return {k: v['family'] for k, v in BEDROCK_MODELS.items()}


def bedrock_params():
    """model_name -> parameter count in billions (None if not disclosed)."""
    return {k: v.get('params_b') for k, v in BEDROCK_MODELS.items()}


def _create_model(model_name,**kwargs):
    if model_name == 'mistral7b':
        return HFModel('mistralai/Mistral-7B-Instruct-v0.2',MISTRAL_TMPL,**kwargs) 
    elif model_name == 'llama7b':
        return HFModel('meta-llama/Llama-2-7b-chat-hf',LLAMA_TMPL,**kwargs)
    # locally-cached models for Apple Silicon runs (see README / defense_plan.txt)
    elif model_name == 'llama3b':
        return HFModel('meta-llama/Llama-3.2-3B-Instruct',LLAMA_TMPL,**kwargs)
    elif model_name == 'mistral7bv3':
        return HFModel('mistralai/Mistral-7B-Instruct-v0.3',MISTRAL_TMPL,**kwargs)
    # ---- Amazon Bedrock models (keyword/voting defenses only; see BedrockModel) ----
    # Mistral benefits from the terse system prompt (keyword clean 28% -> 54% on RQA); Llama-3.1
    # is hurt by it (over-abstains), so Llama runs with no system prompt (both default None).
    elif model_name == 'mistral7b-bedrock':  # paper model: Mistral-7B-Instruct-v0.2
        kwargs.setdefault('system_prompt', BedrockModel.SYSTEM_PROMPT)
        return BedrockModel('mistral.mistral-7b-instruct-v0:2', MISTRAL_TMPL, **kwargs)
    # llama3b-bedrock intentionally has NO explicit branch any more: AWS
    # end-of-lifed Llama-3.2-3B. It falls through to the registry, which
    # refuses it with a readable message. See BEDROCK_MODELS.
    elif model_name == 'llama8b-bedrock':     # Llama-3.1-8B-Instruct: stand-in for the paper's Llama-2-7B (removed from Bedrock)
        return BedrockModel('us.meta.llama3-1-8b-instruct-v1:0', LLAMA_TMPL, **kwargs)
    elif model_name == 'llama70b-bedrock':    # Llama-3.3-70B-Instruct (active inference profile)
        return BedrockModel('us.meta.llama3-3-70b-instruct-v1:0', LLAMA_TMPL, **kwargs)
    elif model_name in BEDROCK_MODELS:
        # registry path: used by the families added for the multi-family study.
        _spec = BEDROCK_MODELS[model_name]
        if _spec.get('status'):
            raise SystemExit(
                f"\n!! {model_name} is not usable: {_spec['status']}\n"
                f"   id was {_spec['id']}\n"
                f"   Bedrock model ids are not stable. Find a live replacement:\n"
                f"       python3 list_bedrock_models.py --provider "
                f"{_spec['family']}\n"
                f"   then update BEDROCK_MODELS.\n")
        if _spec['system_prompt'] == 'UNCALIBRATED':
            raise SystemExit(
                f"\n!! {model_name} has not been calibrated.\n"
                f"   RobustRAG's prompts are few-shot completions; a chat-tuned model\n"
                f"   that answers conversationally collapses the keyword defense. Which\n"
                f"   setting works is family-specific (Mistral needs the steering prompt,\n"
                f"   Llama over-abstains with it), so it must be measured:\n\n"
                f"       python3 calibrate_model.py {model_name}\n\n"
                f"   then set system_prompt in BEDROCK_MODELS accordingly.\n")
        _tmpl = {'MISTRAL_TMPL': MISTRAL_TMPL, 'LLAMA_TMPL': LLAMA_TMPL}[_spec['template']]
        # Only two settings are meaningful, and the difference between them is
        # worth 50 points of clean accuracy. A typo ('Terse', 'none', 'null')
        # would otherwise fall through this branch and run with NO system
        # prompt -- silently, at the wrong setting, for the whole run. Refuse.
        if _spec['system_prompt'] not in ('terse', None):
            raise SystemExit(
                f"\n!! {model_name} has system_prompt={_spec['system_prompt']!r} in "
                f"BEDROCK_MODELS.\n"
                f"   Only two values are valid after calibration:\n"
                f"       'terse'  -> BedrockModel.SYSTEM_PROMPT (steer to short answers)\n"
                f"       None     -> raw completion, no system turn\n")
        if _spec['system_prompt'] == 'terse':
            kwargs.setdefault('system_prompt', BedrockModel.SYSTEM_PROMPT)
        return BedrockModel(_spec['id'], _tmpl, **kwargs)
    elif model_name == 'gpt3.5':
        return GPTModel('gpt-3.5-turbo-0125', GPT_TMPL, **kwargs) 
    # some other models that are not included in the paper
    elif model_name == 'llama8b':
        return HFModel('meta-llama/Meta-Llama-3-8B-Instruct',LLAMA_TMPL,**kwargs)
    elif model_name == 'llama13b':
        return HFModel('meta-llama/Llama-2-13b-chat-hf',LLAMA_TMPL,**kwargs) 
    elif model_name == 'vicuna7b':
        return HFModel('lmsys/vicuna-7b-v1.5',VICUNA_TMPL,**kwargs)  
    elif model_name == 'vicuna13b':
        return HFModel('lmsys/vicuna-13b-v1.5',VICUNA_TMPL,**kwargs)  
    elif model_name == 'mixtral8x7b':
        return HFModel('mistralai/Mixtral-8x7B-Instruct-v0.1',MISTRAL_TMPL,**kwargs) 
    elif model_name == 'mixtral8x22b':
        return HFModel('mistralai/Mixtral-8x22B-Instruct-v0.1',MISTRAL_TMPL,**kwargs) 
    elif model_name == 'commandr':
        return HFModel('CohereForAI/c4ai-command-r-v01', MISTRAL_TMPL, **kwargs)
    elif model_name == 'commandr4':
        return HFModel('CohereForAI/c4ai-command-r-v01-4bit', MISTRAL_TMPL, **kwargs)
    elif model_name == 'gpt4':
        return GPTModel('gpt-4-0125-preview', GPT_TMPL, **kwargs) 
    else:
        raise NotImplementedError

class BaseModel:
    def cache_fingerprint(self):
        """Everything that changes the MEANING of a cached response but is not
        part of the cache key. Subclasses extend this; see _check_cache_fingerprint."""
        return {}

    def _check_cache_fingerprint(self):
        """Refuse to reuse a cache built under different generation settings.

        THE HOLE THIS CLOSES. The cache key is the prompt string
        (self.hash = lambda x: x) and the cache FILE is named
        model-dataset-topk. Neither captures the system prompt. So flipping
        system_prompt in BEDROCK_MODELS -- which is exactly what calibration
        exists to decide -- leaves every later run silently serving responses
        generated under the OLD setting. The run completes, the numbers look
        plausible, and they describe a configuration that was never tested.

        This is the same class of bug as the prompt_style/cache collision
        (preflight section E), and it is MORE likely to bite for a new family,
        because the calibrate -> inspect -> maybe-change-the-setting loop is the
        normal workflow for one.

        Behaviour: a sidecar <cache_path>.fingerprint.json is written next to
        the cache. On mismatch, exit LOUDLY rather than silently mixing. On a
        pre-existing cache with no sidecar, ADOPT the current fingerprint --
        the mistral/llama caches are large, valid, and were built at the
        settings now in the registry; failing on them would be a false alarm.
        """
        if not self.use_cache:
            return
        side = self.cache_path + '.fingerprint.json'
        try:
            # inside the try: a subclass fingerprint that raises must degrade to
            # a warning, never abort a paid run over bookkeeping.
            fp = self.cache_fingerprint()
            if not fp:
                return
            if os.path.exists(side):
                with open(side) as f:
                    old = json.load(f)
                diff = {k: (old.get(k), fp.get(k)) for k in set(old) | set(fp)
                        if old.get(k) != fp.get(k)}
                if diff:
                    lines = '\n'.join(f'      {k}: cached={o!r}  now={n!r}'
                                      for k, (o, n) in sorted(diff.items()))
                    raise SystemExit(
                        f'\n!! CACHE FINGERPRINT MISMATCH for {self.cache_path}\n'
                        f'{lines}\n\n'
                        f'   The cached responses were generated under different settings.\n'
                        f'   Reusing them would report on a configuration never run.\n\n'
                        f'   Either restore the old setting, or start a clean cache:\n'
                        f'       mv {self.cache_path} {self.cache_path}.bak\n'
                        f'       mv {side} {side}.bak\n')
            else:
                # adopt: existing cache predates this check, or cache is new
                with open(side, 'w') as f:
                    json.dump(fp, f, indent=2, sort_keys=True)
        except SystemExit:
            raise
        except Exception as e:                       # never let bookkeeping kill a run
            logger.warning(f'cache fingerprint check skipped: {e}')

    def __init__(self,cache_path=None):
        # setup the LLM response cache if cache_path is not None
        self.use_cache = cache_path is not None
        self.cache_path = cache_path
        if cache_path is not None and os.path.exists(cache_path):
            self.cache = self.load_cache()
        else:
            self.cache = {}
        self.hash = lambda x: x # for now directly string as the hash key
        # cache is a dict {hash(s):LLM response for input s}
        # 
        # end of sentence str
        self.clean_str = []

        # input prompt template
        self.prompt_template = {}

    def query(self, prompt):
        if self.use_cache: # use cache if cache hits
            result = self.query_from_cache(prompt)
            if len(result)>0:
                return result

        # otherwise, do normal LLM query
        result = self._query(prompt)

        # store the response to self.cache
        #
        # NEVER CACHE AN EMPTY RESPONSE. _query() returns "" when the Bedrock
        # call fails after all retries (throttling, quota). Caching that makes
        # the failure PERMANENT: every later run hits the cache, gets "", and
        # silently treats it as a real answer. An empty response is also not
        # inert -- it does not contain "I don't", so the keyword defense counts
        # it as a NON-ABSTAINING group and inflates n, which raises mu and makes
        # the attacker look weaker than it is.
        #
        # Skipping the write costs one re-query next run and is always correct:
        # a genuinely empty model response is indistinguishable from a failed
        # call, and neither is worth persisting.
        if self.use_cache and str(result).strip():
            self.cache[self.hash(prompt)]=result
        elif not str(result).strip():
            self.n_empty_responses = getattr(self, 'n_empty_responses', 0) + 1
        return result

    def _query(self,prompt): # will be implemented in each subclass
        raise NotImplementedError

    def batch_query(self, prompt_list):
        if self.use_cache: # use cache if cache hits
            results = self.batch_query_from_cache(prompt_list)
            if len(results)>0:
                return results

        # otherwise, do normal LLM batch query

        results = self._batch_query(prompt_list)

        # store the response to self.cache -- same rule as query(): an empty
        # result means the call failed after retries, and caching it would make
        # the failure permanent and invisible. See the comment in query().
        if self.use_cache:
            for p,r in zip(prompt_list,results):
                if str(r).strip():
                    self.cache[self.hash(p)]=r
        n_empty = sum(1 for r in results if not str(r).strip())
        if n_empty:
            self.n_empty_responses = getattr(self, 'n_empty_responses', 0) + n_empty
        return results

    def _batch_query(self, prompt_list):  # will be implemented in each subclass
        raise NotImplementedError

    def query_from_cache(self,prompt):
        # return cached responses
        h = self.hash(prompt)
        if h in self.cache:
            return self.cache[h]
        else:
            return ''

    def batch_query_from_cache(self,prompt_list):
        # return cached responses
        h_list = [self.hash(prompt) for prompt in prompt_list]
        if all([h in self.cache for h in h_list]):
            return [self.cache[h] for h in h_list]
        else:
            return []


    def dump_cache(self): # dump cache to disk
        joblib.dump(self.cache,self.cache_path)
    
    def load_cache(self): # load cache from disk
        return joblib.load(self.cache_path)
        
    def _clean_response(self,response): # clean response based on self.clean_str
        for pattern in self.clean_str:
            idx = response.find(pattern)
            if idx!=-1:
                response = response[:idx]
        return response.strip()

    def query_biogen(self, prompt):
        raise NotImplementedError

    def wrap_prompt(self,data_item,as_multi_choice=True,hints=None,seperate=False):  
        # use data_item and generate the input prompt to the LLM

        # data_item should be the output of DataUtils.process_data_item()
        # as_multi_choice: if use it as a multiple-choice QA
        # hints: if we are using hints in the last step of the keyword aggregation
        # seperate: if True, return a list of prompts for differnet passages; otherwise, concatenate all passages and return a single prompt

        # get info
        question = data_item['question']
        topk_content = data_item['topk_content']
        choices = data_item.get('choices',[])
        use_retrieval = len(topk_content) > 0 # if we use retrieved passage

        def fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints):
            filling = {'query_str': question}
            if use_retrieval:
                filling.update({'context_str': context_str})
            if as_multi_choice:
                filling.update({
                        'A': choices[0],
                        'B': choices[1],
                        'C': choices[2],
                        'D': choices[3]                
                    })
            if hints is not None:
                filling.update({'hints':hints})
            return template.format(**filling)

        # get corresponding template
        mode = 'qa'
        if as_multi_choice: mode += '-mc'
        if "long_gen" in data_item: mode += '-long'
        if not use_retrieval: mode += '-zero'
        if "decode" in data_item: mode += '-decode' 
        if "genhint" in data_item: mode += '-genhint'
        if hints is not None: mode += '-hint' 
        template = self.prompt_template[mode]

        if seperate:
            return [fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints) for context_str in topk_content]
        else:
            context_str = '\n\n'.join(topk_content)
            return fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints)




class HFModel(BaseModel):
    def __init__(self, model_name, prompt_template, cache_path = None,max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        # set max number of output tokens
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens 

        # set up tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        if "CohereForAI" in model_name:
            #'CohereForCausalLM' object has no attribute 'torch_dtype'
            self.model = AutoModelForCausalLM.from_pretrained(model_name,**kwargs) 
        else:
            # Patched: use float16 (MPS bfloat16 support is incomplete), explicit .to(DEVICE)
            # instead of device_map='auto' (which requires accelerate CUDA integration)
            _dtype = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_dtype, **kwargs).to(DEVICE)

        self.prompt_template = prompt_template
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name

        self.generation_kwargs = {
            'max_new_tokens':self.max_output_tokens,
            'pad_token_id':self.tokenizer.eos_token_id,
            'do_sample':False
        }
        if 'Llama-3' in model_name:
            self.generation_kwargs['eos_token_id']=[self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        self.clean_str = ['\n\n'] 


    def _query(self, prompt):
        # get text prompt as input and return text responses
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        token_length = inputs.input_ids.size(1)
        # check if the num of token exceeds the max context window ..
        # this only happens when we use the bio generation data without any defense (i.e., we concatenate all long passages together)
        # we did not implement this cut off for _batch_query
        if token_length >= (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens):
            prompt_length = len(prompt)
            ratio = (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens-100)/token_length # this is a rough cut off, 100 is a buffer
            ## this is left cut 
            # prompt_cut = prompt[:int(prompt_length*ratio)] + "..." + "\n\n" + prompt[prompt.rfind("Query: Tell me a bio of"):] 
            ## this is right cut 
            prompt_cut = "Context information is below.\n" + "---------------------\n ..."+ prompt[-int(prompt_length*ratio):]
            inputs = self.tokenizer(prompt_cut, return_tensors="pt").to(DEVICE)
            logger.warning(f"Prompt length exceeds the limit, cut the prompt to {inputs.input_ids.size(1)} tokens")

        outputs = self.model.generate(**inputs,**self.generation_kwargs)
        outputs = outputs[0][len(inputs[0]):]
        result = self.tokenizer.decode(outputs, skip_special_tokens=True)
        result = self._clean_response(result)
        return result
    

    def _batch_query(self, prompt_list):
        # get a list of text prompts as input and return a list text responses

        inputs = self.tokenizer(prompt_list, return_tensors="pt",
                                    padding=True, 
                                    #padding='max_length',
                                    truncation=True,
                                    #max_length=800
                                    ).to(DEVICE)
        outputs = self.model.generate(**inputs, **self.generation_kwargs)
        results = self.tokenizer.batch_decode(outputs[:, len(inputs[0]):],skip_special_tokens=True)
        results = [self._clean_response(x) for x in results]
        
        return results




class GPTModel(BaseModel):
    def __init__(self, model_name, prompt_template, cache_path=None,max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = 0
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        self.client = OpenAI()



    def _query(self, prompt): # for now, I will just make implementation to work


        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            response = completion.choices[0].message.content
        except Exception as e:
            print(e)
            response = ""
        return response
        

    def _batch_query(self,prompt_list):

        prompt_list_with_template = []
        for prompt in prompt_list:
            prompt_list_with_template.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])
        result = batch_completion(model=self.model_name, messages=prompt_list_with_template, max_tokens=self.max_output_tokens)
        response_list = [x.choices[0].message.content for x in result]

        return response_list

    def query_biogen(self, prompt):
        return self.query(prompt)


class BedrockModel(BaseModel):
    """
    Amazon Bedrock wrapper (via litellm) for the keyword/voting defenses.

    NOTE: only white-box-free defenses (keyword, voting) are supported on Bedrock.
    The decoding-based defense needs per-token logit vectors for aggregation, which
    Bedrock's InvokeModel does not expose -- run that one locally with HFModel.

    `model_name` is the Bedrock model id (or cross-region inference profile id),
    e.g. 'mistral.mistral-7b-instruct-v0:2' or 'us.meta.llama3-2-3b-instruct-v1:0'.
    litellm expects it prefixed with 'bedrock/'.
    """
    # RobustRAG's prompts are completion-style few-shot templates ending in "Answer:".
    # Chat/instruct models on Bedrock otherwise add preamble/apologies/"Source:" and refuse
    # the keyword-hint step, which collapses the keyword defense (clean acc 60% -> 10% in
    # testing). This system prompt steers them back to terse, completion-like answers so the
    # per-passage abstention instructions in each template still govern when to say I don't know.
    SYSTEM_PROMPT = (
        "Follow the template and continue after 'Answer:'. Output ONLY the direct answer, "
        "as concisely as possible (a few words or short keywords), matching the style of the "
        "examples. Do NOT add any preamble, explanation, source, apology, or meta commentary."
    )

    def __init__(self, model_name, prompt_template, cache_path=None, max_output_tokens=None,
                 aws_region_name=None, temperature=0, system_prompt=None, num_retries=10,
                 max_workers=None, **kwargs):
        super().__init__(cache_path)
        self.model_name = model_name
        self.litellm_model = f"bedrock/{model_name}"
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        # Default: NO system prompt, matching the paper's raw-completion local runs. A terse
        # system prompt was tried but it made some models (e.g. Llama-3.1) over-abstain; pass
        # system_prompt=BedrockModel.SYSTEM_PROMPT explicitly to opt in.
        self.system_prompt = system_prompt
        self.num_retries = num_retries
        # cap concurrent Bedrock requests to avoid tripping the account's rate limit on bursts
        # (batched isolated calls + certification powerset). Tunable via BEDROCK_MAX_WORKERS.
        self.max_workers = int(max_workers if max_workers is not None
                               else os.environ.get("BEDROCK_MAX_WORKERS", 4))
        # seconds to sleep before each request; see _query(). 0 = off.
        self._request_delay = float(os.environ.get("BEDROCK_REQUEST_DELAY", 0) or 0)
        # Retries can make throttling WORSE, not better. num_retries=10 means one
        # failing query opens up to 11 connections; a run with ~150 failures is
        # ~1600 connection attempts in a short window, which is itself enough to
        # trip an account-level "Too many connections" block and keep it tripped.
        # BEDROCK_NUM_RETRIES lowers it (try 2-3) when that is the situation.
        _env_retries = os.environ.get("BEDROCK_NUM_RETRIES")
        if _env_retries:
            self.num_retries = int(_env_retries)
            try:
                litellm.num_retries = self.num_retries
            except Exception:
                pass
        # Back-off schedule for the shared breaker in _query(). With a breaker
        # in front of it, MORE retries is now the safe direction: attempts are
        # spaced out rather than fired into a drained bucket, so the old
        # "retries make it worse" reasoning above no longer applies.
        self._backoff_base = float(os.environ.get("BEDROCK_BACKOFF_BASE", 2.0) or 2.0)
        self._backoff_max = float(os.environ.get("BEDROCK_BACKOFF_MAX", 60.0) or 60.0)
        # Stop a systemically throttled run instead of letting it finish and be
        # discarded. 0 disables. Kept below the smallest run (100 queries) so it
        # fires early, but above the handful of stragglers a healthy run shrugs off.
        self._abort_after = int(os.environ.get("BEDROCK_ABORT_AFTER_EMPTY", 25) or 0)
        # region: explicit arg > env var > default. litellm reads AWS creds from the standard chain.
        self.aws_region_name = aws_region_name or os.environ.get("AWS_REGION_NAME") or os.environ.get("AWS_REGION") or "us-east-1"
        self.clean_str = ['\n\n']  # match HFModel: cut generation at the first blank line
        # Bedrock throttles bursts (batched isolated calls + certification powerset). Let litellm
        # retry RateLimitError with exponential backoff instead of silently returning "" (which
        # would corrupt results by counting throttled calls as wrong/abstained). Also silence spam.
        litellm.suppress_debug_info = True
        try:
            litellm.num_retries = num_retries
        except Exception:
            pass
        # Must come AFTER system_prompt / model id / max_output_tokens are set:
        # those are precisely the fields the fingerprint covers.
        self._check_cache_fingerprint()
        logger.info(f"BedrockModel: {self.litellm_model} (region={self.aws_region_name}, num_retries={num_retries})")

    def cache_fingerprint(self):
        # The cache key is the prompt text alone, so anything here that changes
        # the response WITHOUT changing the prompt text must be fingerprinted.
        # system_prompt is the dangerous one -- it is what calibration decides.
        # temperature is included for the same reason. Region is NOT: the same
        # model id in another region is the same model.
        return {
            'bedrock_id': self.model_name,
            'system_prompt': self.system_prompt,
            'temperature': self.temperature,
            'max_output_tokens': self.max_output_tokens,
        }

    def _messages(self, prompt):
        # RobustRAG prompts are self-contained few-shot completions sent as a single user turn.
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _clean_response(self, response):
        # Bedrock returns only the assistant text (no prompt echo), and some models (Llama-3.1)
        # lead with "\n\n". Strip FIRST, then cut any trailing few-shot continuation at "\n\n" --
        # otherwise the base _clean_response would cut at a leading "\n\n" and return "".
        response = (response or "").strip()
        for pattern in self.clean_str:
            idx = response.find(pattern)
            if idx != -1:
                response = response[:idx]
        return response.strip()

    def _query(self, prompt):
        # We own the retry loop rather than delegating to litellm's num_retries,
        # because litellm retries this ONE request while the other workers keep
        # sending. On a drained account bucket that cannot converge. Here every
        # 429 opens the shared breaker, so all workers wait out the same window.
        last = None
        for attempt in range(self.num_retries + 1):
            _rl_hold()                       # respect a breaker opened by any worker
            if self._request_delay:
                time.sleep(self._request_delay)
            try:
                completion = litellm.completion(
                    model=self.litellm_model,
                    messages=self._messages(prompt),
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    aws_region_name=self.aws_region_name,
                    num_retries=0,           # this loop is the retry policy
                )
                return self._clean_response(completion.choices[0].message.content or "")
            except Exception as e:
                last = e
                if not _is_rate_limit(e) or attempt == self.num_retries:
                    break
                # Exponential with full jitter. Jitter matters: without it every
                # parked worker wakes at the same instant and re-drains the
                # bucket together, which is the thundering herd we just had.
                window = min(self._backoff_base * (2 ** attempt), self._backoff_max)
                nap = window * (0.5 + random.random() / 2.0)
                n = _rl_trip(nap)
                if n == 1 or n % 25 == 0:
                    logger.warning(
                        f"rate limited; pausing all workers {nap:.1f}s "
                        f"(breaker trip {n}, attempt {attempt + 1}/{self.num_retries})")
                time.sleep(nap)
        logger.warning(f"Bedrock _query failed after {self.num_retries + 1} attempts: {last}")
        self._note_failure()
        return ""

    def _note_failure(self):
        """Abort the run once failures are clearly systemic.

        Previously a throttled run played out to the end, wrote a full-looking
        CSV, and was only caught afterwards -- six minutes to learn the data was
        unusable. Past this threshold the outcome is already decided, so stop.
        """
        with _RL_LOCK:
            self._hard_failures = getattr(self, '_hard_failures', 0) + 1
            n = self._hard_failures
        if self._abort_after and n >= self._abort_after:
            raise SystemExit(
                f"\n!! ABORTING: {n} calls failed after full back-off.\n"
                f"   The account limit is the binding constraint, not a spike, and\n"
                f"   every further call would be billed for data that gets discarded.\n"
                f"   Successful responses so far ARE cached, so a rerun resumes\n"
                f"   from here and re-issues only what is missing. Retry with:\n"
                f"     BEDROCK_MAX_WORKERS=1 BEDROCK_REQUEST_DELAY=1.0 \\\n"
                f"     BEDROCK_NUM_RETRIES=8 BEDROCK_BACKOFF_BASE=4 ./run_final.sh <cmd>\n"
                f"   Raise BEDROCK_ABORT_AFTER_EMPTY to let a run push through.\n")

    def _batch_query(self, prompt_list):
        # Bounded-concurrency batch: cap the burst to self.max_workers concurrent requests
        # (each with retry/backoff) instead of firing all prompts at once, which tripped
        # Bedrock's rate limit. ThreadPoolExecutor.map preserves input order.
        if not prompt_list:
            return []
        workers = max(1, min(self.max_workers, len(prompt_list)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._query, prompt_list))  # _query cleans + retries

    def query_biogen(self, prompt):
        return self.query(prompt)
