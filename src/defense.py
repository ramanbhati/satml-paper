import logging

from collections import Counter,defaultdict

from transformers import StoppingCriteriaList, MaxLengthCriteria
from nltk.corpus import stopwords
#import nltk
#from nltk.util import ngrams
#from string import punctuation as PUNCTUATION
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
stopword_set = set(stopwords.words('english'))

import torch 
from itertools import combinations
from .helper import (clean_str, StopOnTokens, is_guardrail_refusal,
                     is_epistemic_abstention)
import copy
from tqdm import tqdm
from torch import LongTensor, FloatTensor
import numpy as np

# Returned by KeywordAgg.query() when filter_only is on, in place of a real
# aggregated answer. Chosen so it can never be mistaken for model output:
# it is not a plausible answer, and it must NOT contain "I don't" (which upstream
# reads as abstention), "cannot"/"can't" (which the refusal detector reads as a
# guardrail refusal), or anything a dataset's eval_response could score as
# correct. Any metric derived from this string is meaningless by construction --
# main.py blanks ASR/accuracy for filter-only runs rather than scoring it.
FILTER_ONLY_SENTINEL = '<FILTER_ONLY_NO_AGGREGATION_CALL>'

# ---- MPS/CUDA/CPU device auto-detection (patched for Apple Silicon) ----
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# -----------------------------------------------------------------------

import spacy
import os
import json
import random

logger = logging.getLogger('RRAG-main')

INJECTION = True # injection attack. if False, we consider passage modification attacks discussed in the appendix

def save_all_responses(save_path,response_list,data_item):
    all_data = []# it is a bit ugly... unnecessary read and write ; TODO: change it to jsonl instead
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        with open(save_path,'r') as f:
            all_data = json.load(f)
    all_data.append({"query":data_item['question'],
                     "answer":data_item['answer'],
                     "response":response_list})
    with open(save_path,'w') as f:
        json.dump(all_data,f,indent=4)


class RRAG:

    def __init__(self,llm):
        self.llm = llm

    def query_undefended(self,data_item):
        query_prompt = self.llm.wrap_prompt(data_item,as_multi_choice='choices' in data_item)
        #response = None 
        response =  self.llm.query(query_prompt)
        logger.debug(f'Query_prompt:\n{query_prompt}')
        logger.debug(f'Response:\n{response}')
        logger.debug(f'Answer:\n{data_item["answer"]}')
        return response

    def query(self, data_item):
        raise NotImplementedError

    def certify(self, data_item, corruption_size):
        raise NotImplementedError

    def _eval_response(self,response,data_item):
        answer = data_item['answer']
        response = clean_str(response)
        for ans in answer:
            if clean_str(ans) in response:
                return True 
        return False

class MajorityVoting(RRAG):

    def query(self, data_item, corruption_size = 0):
        # assume the prompt ask the LLM to output A., B., C., D., or E. No information found
        seperate_responses = self.llm.batch_query(self.llm.wrap_prompt(data_item,as_multi_choice=True,seperate=True))
        seperate_preds = []
        for response in seperate_responses:
            if "gpt" in self.llm.model_name: 
                if response.find('Answer') != -1:
                    response = response[(response.find('Answer')+7):].strip()
                else:
                    response = response.strip()
                if response[0] in 'ABCD':
                    seperate_preds.append(response[0]+'.')
                else:
                    seperate_preds.append('E.')
            else:
                response = response.strip()
                if len(response)>=2 and response[1]=='.' and response[0] in'ABCD':
                    seperate_preds.append(response[:2])
                else:
                    seperate_preds.append('E.')

        logger.debug(f'Seperate responses: {seperate_preds}')

        cntr = Counter(seperate_preds)
        del cntr['E.'] # do not count E. # we still append 'E.' to `seperate_preds` because it is useful for certification
        cntr = cntr.most_common(2)

        # Decide final prediction (and also get certificate at the same time)
        # Note that `certificate` only cares if the `pred` can be changed by an attacker; it does not check if `pred` is a correct answer
        if len(cntr)==0:
            pred = 'E.' # No information found.
            certificate = False 
        else:
            pred = cntr[0][0] 
            delta = cntr[0][1] if len(cntr)==1 else cntr[0][1] - cntr[1][1]
            # The certification needs to remove seperate_pred that equals to `pred` and is in `seperate_preds[-corruption_size:]` 
            # This is because after the attacker injects `corruption_size` malicious passages, 
            # the last `corruption_size` passages retrieved in benign cases will be out of topk retrieved passages. 
            if INJECTION:
                delta -= sum([pred==x for x in seperate_preds[-corruption_size:]])
                certificate = delta > corruption_size # conservatively consider tie as non-robust -- can be improved later 
            else:
                certificate = delta > 2*corruption_size
        return pred,certificate



class KeywordAgg(RRAG):

    def __init__(self,llm,relative_threshold=0.3, absolute_threshold=3, abstention_threshold=1, longgen=False, certify_save_path='', filter_only=False, group_size=1, threshold_mode='n', top_k=None):
        self.llm = llm
        # Passage group size omega (RobustRAG Sec. IV / Fig. 6). Upstream's
        # released code implements omega=1 ONLY -- every passage is answered in
        # its own isolated call. The paper, however, studies omega as a knob and
        # reports that omega=3 "reduce[s] the benign performance drop from 7% to
        # 0% while maintaining non-trivial certifiable robustness".
        #
        # This matters for the low-n study. With k retrieved passages the number
        # of isolated groups is ceil(k/omega), so omega directly caps n:
        #     omega=1, k=10 -> up to 10 groups, mu = min(0.3n, 3) up to 3
        #     omega=3, k=10 -> at most  4 groups, mu <= 0.3*4 = 1.2
        # At omega=3 a single abstaining group gives n=3 -> mu=0.9 < 1, i.e. a
        # keyword seen ONCE survives. The configuration recommended for benign
        # performance therefore sits one abstention from the collapse regime.
        # That is the hypothesis this parameter exists to test.
        if group_size < 1:
            raise ValueError(f'group_size must be >= 1, got {group_size}')
        self.group_size = group_size

        # ---- threshold mode: the proposed mitigation --------------------------
        # The filter keeps a keyword iff count >= mu, so a keyword seen ONCE
        # survives whenever mu <= 1. Upstream sets mu = min(alpha*n, beta) with n
        # = the number of NON-ABSTAINING groups, which makes mu vacuous whenever
        #     ceil(k/omega) <= 1/alpha
        # -- reachable two independent ways, neither of which the certificate
        # mentions: thin corpus coverage (n falls) and large group size (the
        # group count itself falls).
        #
        #   'n'      upstream:   mu = min(alpha*n, beta)
        #   'k'      mu = min(alpha*top_k, beta). top_k is a CONFIGURED constant,
        #            so mu cannot be moved by the corpus or by omega at all. At
        #            k=10, alpha=0.3 this is simply mu = beta = 3. Blunt: it
        #            discards the adaptivity entirely.
        #   'floor2' mu = max(2, min(alpha*n, beta)). Minimal and targeted: keeps
        #            upstream's adaptive threshold but never lets it fall below
        #            the smallest value that means anything -- a keyword must be
        #            corroborated by at least TWO independent groups. 2 rather
        #            than 1 because count >= 1 admits everything.
        #
        # Both cost utility when coverage is thin: demanding corroboration the
        # corpus cannot supply turns a wrong answer into an abstention. Measuring
        # that price is the point of evaluating them.
        if threshold_mode not in ('n', 'k', 'floor2'):
            raise ValueError(f"threshold_mode must be 'n', 'k' or 'floor2', got {threshold_mode!r}")
        self.threshold_mode = threshold_mode
        self.profile_certify = False   # set by main.py --profile_certify
        self.certify_profile = []      # dry-run cost records
        self.top_k = top_k
        if threshold_mode == 'k' and not top_k:
            raise ValueError("threshold_mode='k' needs top_k")

        # filter_only: stop after the keyword filter and skip the final
        # aggregation LLM call. See the block in query() for why. Deliberately
        # NOT the default -- a run started with this on cannot produce ASR or
        # accuracy, and a silent default would corrupt those columns everywhere.
        self.filter_only = filter_only
        self.last_filter_only = False
        # NOTE (availability study): upstream RobustRAG hardcodes this to 1 and
        # silently ignores the constructor argument, so the parameter has never
        # been swept. We restore it. The upstream value (1) remains the default
        # and is what we report as the "as-shipped" configuration.
        self.abstention_threshold = abstention_threshold
        self.keyword_extractor = spacy.load("en_core_web_sm")
        self.ignore_set = {'VERB','INTJ','ADP','AUX','CCONJ','DET','PART','PRON','SCONJ','PUNCT','SPACE'}
        self.absolute = absolute_threshold # beta in the paper
        self.relative = relative_threshold # alpha in the paper
        self.longgen = longgen # if it is long-form generation or short-form (we use slightly different prompt template)
        self.certify_save_path = certify_save_path # save all possible responses if needed
        logger.info(f'abs: {absolute_threshold}, relative: {relative_threshold}')

    def _mu(self, n):
        """The keyword-count threshold. Single definition, used by inference."""
        if self.threshold_mode == 'k':
            return min(self.relative * self.top_k, self.absolute)
        base = min(self.relative * n, self.absolute)
        if self.threshold_mode == 'floor2':
            return max(2.0, base)
        return base

    def query(self, data_item, corruption_size=0, abstention_threshold=None): 
        # if corruption_size > 0: we will do certification as well. 
        # otherwise, we only do inference. and the certification flag is always False
        certify_flg = False

        # override original threshold parameters if given
        abstention_threshold = abstention_threshold if abstention_threshold is not None else self.abstention_threshold
        if self.longgen:
            data_item['genhint'] = True # add a flag so that wrap_prompt() can retrieve the correct prompt template

        # ---- passage grouping (omega) ------------------------------------
        # Concatenate consecutive passages into groups of `group_size` before
        # the isolated call, matching RobustRAG's "generate LLM responses based
        # on the concatenated passages from each isolated group". The join is
        # '\n\n', identical to what wrap_prompt(seperate=False) uses, so a group
        # of omega passages is rendered exactly as vanilla RAG would render
        # those omega passages.
        #
        # A SEPARATE dict is used for the isolated calls: data_item itself must
        # stay ungrouped, because the final aggregation call and the evaluation
        # both read from it.
        #
        # Note the last group may be short when omega does not divide k
        # (k=10, omega=3 -> groups of 3,3,3,1). That is inherent to the scheme,
        # not a bug; it is recorded via n_groups.
        if self.group_size > 1:
            _tc = data_item['topk_content']
            _grouped = ['\n\n'.join(_tc[i:i + self.group_size])
                        for i in range(0, len(_tc), self.group_size)]
            iso_item = dict(data_item)
            iso_item['topk_content'] = _grouped
        else:
            iso_item = data_item
        self.last_group_size = self.group_size
        # ------------------------------------------------------------------

        # make seperate predictions
        seperate_responses_raw = self.llm.batch_query(self.llm.wrap_prompt(iso_item,as_multi_choice=False,seperate=True))
        abstained_idx = []
        seperate_responses = []
        logger.debug(f'Seperate responses:\n')
        for i,x in enumerate(seperate_responses_raw):
            logger.debug(f'{i}: {x}\n')
            if "I don't" in x:
                abstained_idx.append(i)
            else:
                seperate_responses.append(x)

        logger.debug(f'Number of retained responses: {len(seperate_responses)}')

        # Availability instrumentation: expose per-query group counts so the
        # caller can aggregate them. n_retained is the 'n' that drives both the
        # abstention threshold and the relative keyword threshold alpha*n.
        self.last_n_abstained = len(abstained_idx)
        self.last_n_retained = len(seperate_responses)
        self.last_n_groups = len(seperate_responses_raw)

        # MEASUREMENT ONLY -- deliberately does not change the aggregation.
        # Upstream marks a group abstained with the substring test `"I don't" in x`,
        # which catches epistemic abstentions but NOT safety refusals ("I cannot
        # assist with that"). Such a group is therefore treated as a real answer
        # and its text is passed to the keyword extractor. We count that leakage
        # here because characterising it is part of the availability study; the
        # upstream behaviour is left intact as the system under test.
        self.last_n_refusal_leaked = sum(
            1 for x in seperate_responses if is_guardrail_refusal(x)
        )
        # MEASUREMENT ONLY -- same rationale, different failure mode, and this
        # one is MODEL-SPECIFIC in a way that can invalidate a cross-model claim.
        #
        # Upstream's test is the bare substring `"I don't"`. A model that
        # abstains in other words -- "No information found", "The context does
        # not provide the answer", "Unknown" -- is NOT counted as abstaining.
        # Its non-answer is then treated as a real answer, which INFLATES n,
        # raises mu = min(alpha*n, beta), and makes the attacker look weaker
        # than it is. Every headline number here is a function of n, so a family
        # that phrases abstention differently would produce a silently
        # non-comparable column in the results table.
        #
        # The existing models were validated against this; a NEW family has not
        # been. Counting the gap costs nothing (the responses are already in
        # hand) and turns an invisible assumption into a number we can check
        # after each run. Behaviour is unchanged: upstream remains the system
        # under test.
        self.last_n_abstain_missed = sum(
            1 for x in seperate_responses if is_epistemic_abstention(x)
        )
        # keep the raw per-group responses so the caller can dump them for manual
        # inspection: (group_index, was_marked_abstained_by_upstream, raw_text)
        self.last_group_responses = [
            (i, i in abstained_idx, x) for i, x in enumerate(seperate_responses_raw)
        ]

        # last_mu / last_surviving_keywords are assigned further down,
        # AFTER the early-return below. Without initialising them here, a query
        # that trips the abstention threshold would leave the PREVIOUS query's
        # values in place and the per-query logger would record stale data --
        # corrupting precisely the low-n queries the study is about.
        # mu is well defined even when filtering never runs.
        self.last_mu = self._mu(len(seperate_responses))
        self.last_n_for_mu = len(seperate_responses)
        self.last_surviving_keywords = set()
        self.last_early_abstain = len(seperate_responses) < abstention_threshold
        # same stale-value hazard as last_mu above: the early return below skips
        # the filter_only block, so reset the flag here rather than leaving the
        # previous query's value in place
        self.last_filter_only = False

        if len(seperate_responses) < abstention_threshold:
            logger.warning('Abstain from making response...')
            return "I don't know.", certify_flg
        
        def construct_phrase(token_list):
            ret = ''
            for token in token_list:
                ret+=token.lemma_+token.whitespace_
        # extract keyword/keyphrase
        all_extracted_phrase = []
        token_counter = defaultdict(int)
        for response in seperate_responses:
            doc = self.keyword_extractor(response)
            phrase_list = [response.strip()] 
            tmp = []
            for token in doc:
                if token.pos_ in self.ignore_set:
                    if len(tmp)>0:
                        phrase = ''.join([x.lemma_+x.whitespace_ for x in tmp]).strip()
                        phrase_list.append(phrase)
                        phrase_list+=[x.lemma_ for x in tmp]
                        tmp = []
                    #if token.pos_ == 'VERB':
                    #    phrase_list.append(token.lemma_)
                else:
                    tmp.append(token)

            phrase = ''.join([x.lemma_+x.whitespace_ for x in tmp]).strip()
            phrase_list.append(phrase)
            phrase_list+=[x.lemma_ for x in tmp]
            phrase_list = set(phrase_list) # only consider unique keywords
            all_extracted_phrase.append(phrase_list)
            for phrase in phrase_list:
                token_counter[phrase]+=1

        # filtering 
        count_threshold = self._mu(len(seperate_responses))   # honours threshold_mode
        logger.debug(sorted(token_counter.items(), key=lambda x: (len(x[0]),x[0]), reverse=True))
        logger.debug(f'count_threshold,{count_threshold}')
        for token,count in list(token_counter.items()):
            if (count < count_threshold) or (token in punctuation) or (token in stopword_set)  or (self.longgen and ' ' not in token): # if it is long generation, we remove single words to reduce the size the keyword set...
                del token_counter[token]

        # --- instrumentation for the low-n study -------------------------------
        # mu = min(beta, alpha*n) is the filter threshold; when n is small the
        # alpha*n branch drops mu below 1, so ANY keyword seen once survives.
        # Expose mu and the surviving keyword set so the caller can test, per
        # query, whether the attacker's keyword got through.
        self.last_mu = count_threshold
        self.last_n_for_mu = len(seperate_responses)
        self.last_surviving_keywords = set(token_counter.keys())
        self.last_early_abstain = False
        # -----------------------------------------------------------------------

        # ---- filter-only mode (alpha/beta sweep) -------------------------------
        # The alpha/beta sweep needs ONE metric: did the attacker's keyword
        # survive mu = min(beta, alpha*n)? That is decided entirely by
        # self.last_surviving_keywords, which is already set above. Everything
        # from here on is the final aggregation call, whose prompt embeds the
        # surviving keyword set -- so it changes with every (alpha, beta) and
        # misses the cache every time, i.e. it is BILLED on every sweep point.
        #
        # The isolated per-passage calls above do NOT depend on alpha or beta at
        # all, so with a warm cache they are free. Skipping the aggregation call
        # makes the whole sweep cost ~$0.
        #
        # The cost is that ASR and accuracy are unavailable (there is no final
        # answer to score); the per-query CSV records them as empty. Do NOT use
        # a filter-only run to make any claim about ASR or accuracy.
        if getattr(self, 'filter_only', False):
            self.last_filter_only = True
            return FILTER_ONLY_SENTINEL, certify_flg
        self.last_filter_only = False
        # -----------------------------------------------------------------------

        # generate keyword hints
        sorted_tokens = sorted(token_counter.items(), key=lambda x: (len(x[0]),x[0]), reverse=True)
        hints = ', '.join([f'{token}' for token,count in sorted_tokens])
        logger.debug(sorted_tokens)
        query_prompt = self.llm.wrap_prompt(data_item,as_multi_choice=False,hints=hints)
        logger.debug(f'Hint prompt:\n{query_prompt}')
        response = self.llm.query(query_prompt)
        logger.debug(f'Keyword aggregated response:\n{response}')

        if corruption_size>0 and self.threshold_mode not in ('n', 'floor2'):
            # certify() now derives its threshold from self._mu, so 'n' and
            # 'floor2' both certify the defense that actually answered. Mode 'k'
            # discards the adaptive term entirely and is NOT covered by the
            # monotonicity argument, so it stays refused.
            raise NotImplementedError(
                f"certification is supported for threshold_mode in "
                f"('n','floor2'); got {self.threshold_mode!r}.")

        if corruption_size>0 and self.group_size > 1:
            # certify() assumes one passage per group: it indexes
            # all_extracted_phrase by passage and slices [:-corruption_size] to
            # drop the corrupted ones. With omega>1 a corrupted passage shares a
            # group with benign ones, so that slice removes the wrong entries
            # and the bound would be silently WRONG rather than merely loose.
            # Re-deriving the certificate for omega>1 is out of scope here; fail
            # loudly instead of reporting a number nobody should trust.
            raise NotImplementedError(
                f'certification (corruption_size={corruption_size}) is only valid '
                f'for group_size=1; got group_size={self.group_size}. Run attacks '
                f'with corruption_size=0 (the attack path zeroes it anyway).')

        if corruption_size>0: # only do certification when corruption is larger than zero
            if self.longgen or self._eval_response(response,data_item): # we don't need to do certification if it is a QA task (eval by correctness) and the clean response is incorrec
                for idx in abstained_idx:
                    all_extracted_phrase.insert(idx,set())                
                certify_flg = self.certify(data_item, corruption_size, all_extracted_phrase)

        return response, certify_flg


    def certify(self, data_item, corruption_size, all_extracted_phrase):
        # get the non-corrupted ones
        if INJECTION:
            all_extracted_phrase = all_extracted_phrase[:-corruption_size]
            # how many non-abstained
            non_abs_cnt = sum([len(x)>0 for x in all_extracted_phrase])

            # k'_effective == corruption_size covers any smaller k'_effective.
            # CHANGED: use self._mu, the same threshold function the inference
            # path used, instead of recomputing the upstream formula. Under
            # threshold_mode='n' this is identical to upstream. Under 'floor2'
            # it applies the floor here too, which is what makes the certified
            # bound describe the defense that actually answered.
            count_threshold = self._mu(non_abs_cnt + corruption_size)
            if count_threshold<=corruption_size: # the attacker can introduce any keyword. return false
                return False
        else:
            non_abs_cnt = sum([len(x)>0 for x in all_extracted_phrase])
            # slightly different from the paper, we can prove that k'_effective == corruption_size (k') generates keyword sets W^* that cover any k'_effective < corruption_size (k')
            count_threshold = self._mu(max(corruption_size, non_abs_cnt))
            if count_threshold<=corruption_size: # the attacker can introduce any keyword. return false
                return False

        token_counter = defaultdict(int)            
        for phrase_set in all_extracted_phrase: # get the token counter for non-corrupted responses
            for phrase in phrase_set:
                token_counter[phrase]+=1

        # delete the non-meaningful characters
        for token,count in list(token_counter.items()):
            if (token in punctuation) or (token in stopword_set) or (self.longgen and ' ' not in token):
                del token_counter[token]
        
        # construct the sets with all possible keywords senarios
        if INJECTION:
            base_list = [token for token,count in token_counter.items() if count >= count_threshold]
            added_list = [token for token,count in token_counter.items() if count_threshold - corruption_size <= count < count_threshold]
        else: 
            base_list = [token for token,count in token_counter.items() if count >= count_threshold+corruption_size]
            added_list = [token for token,count in token_counter.items() if count_threshold - corruption_size <= count < count_threshold+corruption_size]
        logger.debug(f'Base list: {base_list}')


        if getattr(self, 'profile_certify', False):
            # DRY RUN. Record what certification WOULD cost and return without
            # issuing a single LLM call, so the spend can be approved first.
            skipped = len(added_list) > 14 or (self.longgen and len(added_list) >= 10)
            self.certify_profile.append({
                'mu': float(count_threshold),
                'n_base': len(base_list),
                'n_added': len(added_list),
                'skipped': bool(skipped),
                'agg_calls': 0 if skipped else 2 ** len(added_list),
            })
            return False

        if len(added_list)>14 or (self.longgen and len(added_list)>=10):
            logger.warning('added list too long... skipped')
            response_list = ["I don't know."]
            certify_flg = False
        else: 
            added_list_powerset = []
            for i in range(len(added_list) + 1):
                for combo in combinations(added_list, i):
                    added_list_powerset.append(list(combo))

            list_powerset = [base_list + added_list for added_list in added_list_powerset]

            # sort the lists inside list_powerset by the length of the items in the list
            list_powerset = [sorted(lists, key=lambda x: (len(x),x), reverse=True) for lists in list_powerset]
            hints_list = [', '.join([f'{token}' for token in lists]) for lists in list_powerset]

            # construct the prompts
            prompt_list = [self.llm.wrap_prompt(data_item,as_multi_choice=False,hints=hints) for hints in hints_list]
            #logger.debug(f'Prompt list:\n{len(prompt_list)}')

            response_list = []
            # break into batches and query
            batch_size = 20
            #for i in tqdm(range(0, len(prompt_list), batch_size)):
            for i in range(0, len(prompt_list), batch_size):
                if i+batch_size > len(prompt_list):
                    prompt_batch = prompt_list[i:]
                else:
                    prompt_batch = prompt_list[i:i+batch_size]
                response_batch = self.llm.batch_query(prompt_batch)
                for response in response_batch:
                    if self.longgen: # we do not do correctness evaluation. we need to save all possible responses
                        response_list.append(response)
                    else:
                        if not self._eval_response(response,data_item): # as long as one possible response is incorrect, we return false.
                            return False
            certify_flg = True
            logger.debug('!!!!!!!!!!!!certify!!!!!!!!!!!!')

        if self.longgen: # we need to dump all possible responses
            save_all_responses(self.certify_save_path,response_list,data_item)


        return certify_flg

        



#######################################################################################################################
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from transformers.utils import ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import _crop_past_key_values


@torch.no_grad()
def secure_decoding(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        #ab_record = None,
        #corruption_size: Optional[int] = None,
        temperature: Optional[float] = 0.01,
        robust_agg: Optional[str] = 'mean',
        tokenizer: Optional = None,
        eta: Optional[float] = 1.0,
        **model_kwargs,
):


    # init values
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    # # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None

    max_len = stopping_criteria[0].max_length

    generated_tokens = []
    ii = 0 
    while True:
        ii+=1        
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)# this function is from huggingface model

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        new_logits_full = outputs.logits[:, -1, :]  # last token prediction
        no_retrieval_token = new_logits_full[-1,:].argmax()
        new_logits = new_logits_full[:-1,:] # remove no-retrieval prompt

        # get the softmax of the next_token_logits
        next_token_confidence = torch.nn.functional.softmax(new_logits/temperature, dim=-1)

        if robust_agg == 'mean':
            next_token_agg = torch.sum(next_token_confidence, dim=0)
        elif robust_agg == 'median':
            next_token_agg = torch.median(next_token_confidence, dim=0).values
        else:
            raise NotImplementedError
            
        if ii == 1:
            ## Here, we should reduce the probability of the "I" token
            ## We can do this by setting the probability of the "I" token to 0
            I_token = tokenizer("I", return_tensors="pt")["input_ids"][0][1].item()
            n_token = tokenizer("/n", return_tensors="pt")["input_ids"][0][1].item()
            Please_token = tokenizer("Please", return_tensors="pt")["input_ids"][0][1].item()
            next_token_agg[I_token] = 0  # set the probability of the "I" token to 0
            next_token_agg[n_token] = 0 # set the probability of the "/n" token to 0
            next_token_agg[Please_token] = 0 # set the probability of the "Please" token to 0

        top2_tokens = torch.topk(next_token_agg, 2)
        if top2_tokens.values[0] - top2_tokens.values[1]>eta:
            selected_tokens = top2_tokens.indices[0] # top 1 token
        else: # no retrieval token
            selected_tokens = no_retrieval_token

        input_ids = torch.cat((input_ids, selected_tokens.repeat(input_ids.shape[0], 1)), dim=-1)
    
        new_cur_len = input_ids.shape[-1]

        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

        model_kwargs["past_key_values"] = outputs.past_key_values
        
        model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], torch.ones_like(model_kwargs["attention_mask"][:, :1])], dim=-1)

        generated_tokens.append(selected_tokens)

        # stop if we exceed the maximum length
        if (selected_tokens == eos_token_id_tensor.item()).any():
            break
        
        if all(stopping_criteria(input_ids, scores)):
            break

    return generated_tokens


@torch.no_grad()
def secure_decoding_certify(
        self,
        initial_input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        ab_record = None,
        corruption_size: Optional[int] = None,
        temperature: Optional[float] = 0.01,
        robust_agg: Optional[str] = 'mean',
        tokenizer: Optional = None,
        data_item: Optional = None,
        eta: Optional[float] = 1.0,
        random_sample_once: Optional = False,
        no_sampling: Optional = True,
        **initial_model_kwargs,
):

    # init values
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(initial_input_ids.device) if eos_token_id is not None else None

    stopping_list = [stopping_criteria, eos_token_id_tensor.item()]

    stack = [(0, torch.tensor([], dtype=torch.long, device=initial_input_ids.device))]  # Added None for the initial cache

    all_outputs = []
    certify = True
    past_key_values = None

    while stack:
        #print(stack)
        depth, additional_tokens = stack.pop()

        input_ids = torch.cat([initial_input_ids, 
                            additional_tokens.repeat((initial_input_ids.shape[0], 1))], 
                            dim=1) if additional_tokens.numel() > 0 else initial_input_ids

        attention_mask = torch.cat([initial_model_kwargs["attention_mask"], 
                                    torch.ones_like(initial_model_kwargs["attention_mask"][:, 
                                    :additional_tokens.shape[0]])], dim=-1)

        if all(stopping_list[0](input_ids, None)) or (input_ids[:, -1] == stopping_list[1]).any():
            all_outputs.append(additional_tokens)
            continue

        # Update model inputs with past key values if available
        if past_key_values is not None:
            past_key_values = _crop_past_key_values(self, past_key_values, input_ids.shape[-1] - 1)
            model_kwargs = {'attention_mask': attention_mask, 'past_key_values': past_key_values}
        else:
            model_kwargs = {'attention_mask': attention_mask}
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs, return_dict=True, output_attentions=None, output_hidden_states=None)

        new_logits_full = outputs.logits[:, -1, :]

        if depth == 0:
            ## Here, we should reduce the probability of the "I" token
            ## We can do this by setting the probability of the "I" token to 0
            I_token = tokenizer("I", return_tensors="pt")["input_ids"][0][1].item()
            n_token = tokenizer("/n", return_tensors="pt")["input_ids"][0][1].item()
            Please_token = tokenizer("Please", return_tensors="pt")["input_ids"][0][1].item()
            new_logits_full[:, I_token] = 0  # set the probability of the "I" token to 0
            new_logits_full[:, n_token] = 0 # set the probability of the "/n" token to 0
            new_logits_full[:, Please_token] = 0 # set the probability of the "Please" token to 0

        new_logits = new_logits_full[:-1, :]
        new_logits = torch.nn.functional.softmax(new_logits / temperature, dim=-1)
        no_retrieval_token = new_logits_full[-1].argmax()
        # Update the cache with new key values
        past_key_values = outputs.past_key_values
        
        # 
        if INJECTION:
            non_abs_cnt = ab_record[:-corruption_size].sum().item()
            new_logits = new_logits[:non_abs_cnt]
        else:
            new_logits = new_logits
        confidence_sum = new_logits.sum(dim=0)
        top2_tokens = torch.topk(confidence_sum, 2)
        top1_top2_gap = (top2_tokens.values[0] - top2_tokens.values[1]).item()
        top1_index = top2_tokens.indices[0]


        if INJECTION:
            thres1 = eta + corruption_size
            thres2 = abs(eta - corruption_size)
            thres3 = eta - corruption_size
        else:

            thres1 = eta + 2*corruption_size
            thres2 = abs(eta - 2*corruption_size)
            thres3 = eta - 2*corruption_size

        selected_tokens_list = []

        if no_sampling: # generate all possible responses
            pruning_prob = 0.0
        elif random_sample_once: # generate one possible reponse
            pruning_prob = 1.1
        else: # generate a random subset of possible responses
            pruning_prob = 0.0 if depth<=12 else 0.9 # some random numbers...

        if top1_top2_gap > thres1:
            selected_tokens_list.append(top1_index)
        elif  thres1 >= top1_top2_gap > thres2:
            zero_shot_token = new_logits_full[-1,:].argmax() 
            selected_tokens_list.append(no_retrieval_token)
            if no_retrieval_token != top1_index: 
                logger.debug(f"Zero-shot: {no_retrieval_token}, top1: {top1_index}, branch: {top1_top2_gap}")
                selected_tokens_list.append(top1_index)
            if random.random()<pruning_prob:
                logger.debug('prunning!!')
                selected_tokens_list = [random.choice(selected_tokens_list)]
        elif top1_top2_gap <= thres3:
            no_retrieval_token = new_logits_full[-1,:].argmax() 
            selected_tokens_list.append(no_retrieval_token)
        else:
            return [],False

        for selected_token in selected_tokens_list:
            new_additional_tokens = torch.cat([additional_tokens, selected_token.unsqueeze(0)], dim=0) if additional_tokens.numel() > 0 else selected_token.unsqueeze(0)
            # Store updated key-value pairs for the next iteration
            stack.append((depth + 1, new_additional_tokens))

    #logger.info(f'size of outputs: {len(all_outputs)}')
    return all_outputs, True


class DecodingAgg(RRAG):
    def __init__(self,llm, args, abstention_prob=None, robust_agg = "mean",certify_save_path='',eval_certify=None):
        self.llm = llm
        self.llm.model.secure_decoding_certify = secure_decoding_certify.__get__(self.llm.model, type(self.llm.model))
        self.llm.model.secure_decoding = secure_decoding.__get__(self.llm.model, type(self.llm.model))
        self.temperature = 1.0#args.temperature
        self.robust_agg = robust_agg
        abstention_prob_list = {'mistralai/Mistral-7B-Instruct-v0.2': 0.99,
                                'mistralai/Mistral-7B-Instruct-v0.3': 0.99,
                                'meta-llama/Llama-2-7b-chat-hf': 0.99,
                                'meta-llama/Llama-3.2-3B-Instruct': 0.99,
                                'mistralai/Mixtral-8x7B-Instruct-v0.1': 0.999}
        if abstention_prob is None:
            self.abstention_prob = abstention_prob_list.get(llm.model_name, 0.99)
            logger.debug(f"Using default abstention probability: {self.abstention_prob}")

        self.args = args
        self.eta = args.eta
        self.subsample_iter = args.subsample_iter
        self.certify_save_path = certify_save_path  # we need to save all possible reponses for long-form generation certifictaion 
        self.eval_certify = eval_certify if eval_certify is not None else self.eta==0 # when eta == 0, we do not use no-retrieval token -- we are usually doing QA task

    def preprocess_input(self,data_item):
        prompt_list = self.llm.wrap_prompt(data_item,as_multi_choice=False,seperate=True)
        data_item_zero_shot = {"question": data_item["question"], "topk_content":[], "long_gen": True}
        prompt_zero_shot = self.llm.wrap_prompt(data_item_zero_shot,as_multi_choice=False,seperate=False)
        prompt_list.append(prompt_zero_shot)

        prompt_list_draft = [prompt + " I don't know" for prompt in prompt_list]

        # batched version 
        input_dict_draft = self.llm.tokenizer(prompt_list_draft, return_tensors="pt", padding=True).to(DEVICE)
        # there is some issues when certifying the model with the batched version, will fix this later
        input_ids_draft = input_dict_draft.input_ids.to(DEVICE)
        attention_mask_draft = input_dict_draft.attention_mask.to(DEVICE)

        # compute the perplexity of the prompt "I don't know"
        with torch.no_grad():
            output_token_draft = self.llm.model(input_ids_draft, attention_mask=attention_mask_draft)
            logits_draft = output_token_draft.logits

        probs = torch.softmax(logits_draft, dim=-1)
        total_probability = torch.ones(input_ids_draft.shape[0]).to(DEVICE)

        input_dict = self.llm.tokenizer(prompt_list, return_tensors="pt", padding= True)
        start_index = input_dict.input_ids.size(1)

        for i in range(start_index, input_ids_draft.size(1) - 1):  # Exclude the last token since there's no next token to predict
            # Get the probability of the actual next token
            next_token_id = input_ids_draft[0, i + 1]  # The next token in the sequence
            next_token_prob = probs[:, i, next_token_id]
            total_probability *= next_token_prob

        #print(f"total_probability: {total_probability}")
        input_ids = input_dict.input_ids.to(DEVICE)
        attention_mask = input_dict.attention_mask.to(DEVICE)

        # filter the prompt with the probability of "I don't know" is greater than 0.9
        total_probability[-1] = 0.0 # last one is the zero-shot prompt
        ab_record = total_probability < self.abstention_prob
        input_ids = input_ids[ab_record]
        attention_mask = attention_mask[ab_record]
        return input_ids,attention_mask,ab_record

    def query(self, data_item, corruption_size=1):

        input_ids,attention_mask,ab_record = self.preprocess_input(data_item)

        if input_ids.shape[0] == 1: # only the no-retrieval prediction
            return "I don't know.", False
        
        if corruption_size>0: 
            #for certification
            input_ids_copy = input_ids.clone()
            attention_mask_copy = attention_mask.clone()

        # Initialize past_key_values for caching
        past_key_values = None
        generated_outputs = []

        stop_list = ["\n#", "\n##","\n###","\n####","\n#####"] + ["\n\n"] ################ seems to work fine
        stop_token_ids = [self.llm.tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
        stop_token_ids = [LongTensor(x).to(DEVICE) for x in stop_token_ids]
        #print(stop_token_ids)
        stopping_criteria = StoppingCriteriaList([
            MaxLengthCriteria(max_length=len(input_ids[0]) + self.llm.max_output_tokens),
            StopOnTokens(stop_token_ids=stop_token_ids)
            ])
        
        generated_outputs = self.llm.model.secure_decoding(input_ids,
                            attention_mask=attention_mask,
                            stopping_criteria = stopping_criteria,
                            use_cache=True,
                            pad_token_id=self.llm.tokenizer.pad_token_id,
                            eos_token_id=self.llm.tokenizer.eos_token_id,
                            return_dict_in_generate=False,
                            #ab_record=ab_record,
                            #corruption_size=corruption_size,
                            temperature=self.temperature,
                            robust_agg = self.robust_agg,
                            tokenizer = self.llm.tokenizer,
                            eta = self.eta
                            )

        generated_output_text = self.llm.tokenizer.decode(generated_outputs, skip_special_tokens=True)
        certify = False 

        if corruption_size>0:
            if self.eval_certify and (not self._eval_response(generated_output_text,data_item)):
                pass
            else:
                certify = self.certify(data_item, input_ids_copy, attention_mask_copy, 
                                stopping_criteria, ab_record, corruption_size)

            logger.debug(f'Robust Gen Response: \n {generated_output_text} \n Certification: {certify}')
            #print(certify)

        return generated_output_text, certify

    def certify(self, data_item, input_ids, attention_mask, stopping_criteria, ab_record, corruption_size, no_sampling=True):

        iterations = self.subsample_iter
        random_sample_once  = iterations > 1
        if random_sample_once: no_sampling = False
        generated_outputs = []
        certify = True
        #print(no_sampling)
        for j in range(iterations):
            outputs_local, certify_local = self.llm.model.secure_decoding_certify(input_ids,
                                    attention_mask=attention_mask,
                                    stopping_criteria=stopping_criteria,
                                    use_cache=True,
                                    pad_token_id=self.llm.tokenizer.pad_token_id,
                                    eos_token_id=self.llm.tokenizer.eos_token_id,
                                    return_dict_in_generate=False,
                                    ab_record = ab_record,
                                    data_item = data_item, 
                                    corruption_size = corruption_size,
                                    eta = self.eta,
                                    tokenizer = self.llm.tokenizer,
                                    temperature = self.temperature,
                                    random_sample_once = random_sample_once,
                                    no_sampling = no_sampling
                                    )
            generated_outputs += outputs_local
            certify &= certify_local

        #print(certify)
        if certify:
            response_list = [self.llm.tokenizer.decode(response, skip_special_tokens=True) for response in generated_outputs]
            if len(self.certify_save_path)>0:
                save_all_responses(self.certify_save_path,response_list,data_item)

            if self.eval_certify:
                certify = all([self._eval_response(response,data_item) for response in response_list])
        return certify




