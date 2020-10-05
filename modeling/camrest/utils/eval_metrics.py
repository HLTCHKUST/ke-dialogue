"""ROUGE metric implementation.
Copy from tf_seq2seq/seq2seq/metrics/rouge.py.
This is a modified and slightly extended verison of
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import numpy as np
import numpy
import os
import re
import json
import subprocess
import tempfile
import numpy as np
from collections import Counter
from six.moves import urllib

# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BLEU metric implementation.
"""


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    multi_bleu_path = "utils/multi-bleu.perl"
    os.chmod(multi_bleu_path, 0o755)


    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()


     # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score

#pylint: disable=C0103


def _get_ngrams(n, text):
  """Calcualtes n-grams.
  Args:
    n: which n-grams to calculate
    text: An array of tokens
  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  assert len(sentences) > 0
  assert n > 0

  words = _split_into_words(sentences)
  return _get_ngrams(n, words)


def _len_lcs(x, y):
  """
  Returns the length of the Longest Common Subsequence between sequences x
  and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
  Args:
    x: sequence of words
    y: sequence of words
  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  """
  Computes the length of the longest common subsequence (lcs) between two
  strings. The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
  Args:
    x: collection of words
    y: collection of words
  Returns:
    Table of dictionary of coord and len lcs
  """
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _recon_lcs(x, y):
  """
  Returns the Longest Subsequence between x and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
  Args:
    x: sequence of words
    y: sequence of words
  Returns:
    sequence: LCS of x and y
  """
  i, j = len(x), len(y)
  table = _lcs(x, y)

  def _recon(i, j):
    """private recon calculation"""
    if i == 0 or j == 0:
      return []
    elif x[i - 1] == y[j - 1]:
      return _recon(i - 1, j - 1) + [(x[i - 1], i)]
    elif table[i - 1, j] > table[i, j - 1]:
      return _recon(i - 1, j)
    else:
      return _recon(i, j - 1)

  recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
  return recon_tuple


def rouge_n(evaluated_sentences, reference_sentences, n=2):
  """
  Computes ROUGE-N of two text collections of sentences.
  Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.
  Returns:
    A tuple (f1, precision, recall) for ROUGE-N
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  reference_count = len(reference_ngrams)
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

  # return overlapping_count / reference_count
  return f1_score, precision, recall


def _f_p_r_lcs(llcs, m, n):
  """
  Computes the LCS-based F-measure score
  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf
  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary
  Returns:
    Float. LCS-based F-measure score
  """
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs, p_lcs, r_lcs


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (sentence level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf
  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
  Returns:
    A float: F_lcs
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")
  reference_words = _split_into_words(reference_sentences)
  evaluated_words = _split_into_words(evaluated_sentences)
  m = len(reference_words)
  n = len(evaluated_words)
  lcs = _len_lcs(evaluated_words, reference_words)
  return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
  """
  Returns LCS_u(r_i, C) which is the LCS score of the union longest common
  subsequence between reference sentence ri and candidate summary C. For example
  if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
  c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
  "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5". The
  union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5" and
  LCS_u(r_i, C) = 4/5.
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries
  Returns:
    float: LCS_u(r_i, C)
  ValueError:
    Raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  lcs_union = set()
  reference_words = _split_into_words([reference_sentence])
  combined_lcs_length = 0
  for eval_s in evaluated_sentences:
    evaluated_words = _split_into_words([eval_s])
    lcs = set(_recon_lcs(reference_words, evaluated_words))
    combined_lcs_length += len(lcs)
    lcs_union = lcs_union.union(lcs)

  union_lcs_count = len(lcs_union)
  union_lcs_value = union_lcs_count / combined_lcs_length
  return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (summary level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf
  Calculated according to:
  R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
  P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
  where:
  SUM(i,u) = SUM from i through u
  u = number of sentences in reference summary
  C = Candidate summary made up of v sentences
  m = number of words in reference summary
  n = number of words in candidate summary
  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries
  Returns:
    A float: F_lcs
  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  # total number of words in reference sentences
  m = len(_split_into_words(reference_sentences))

  # total number of words in evaluated sentences
  n = len(_split_into_words(evaluated_sentences))

  union_lcs_sum_across_all_references = 0
  for ref_s in reference_sentences:
    union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences,
                                                      ref_s)
  return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


def rouge(hypotheses, references):
  """Calculates average rouge scores for a list of hypotheses and
  references"""

  # Filter out hyps that are of 0 length
  # hyps_and_refs = zip(hypotheses, references)
  # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
  # hypotheses, references = zip(*hyps_and_refs)

  # Calculate ROUGE-1 F1, precision, recall scores
  rouge_1 = [
      rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
  ]
  rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

  # Calculate ROUGE-2 F1, precision, recall scores
  rouge_2 = [
      rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
  ]
  rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

  # Calculate ROUGE-L F1, precision, recall scores
  rouge_l = [
      rouge_l_sentence_level([hyp], [ref])
      for hyp, ref in zip(hypotheses, references)
  ]
  rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))
  return rouge_1_f,rouge_2_f,rouge_l_f, np.mean([rouge_1_f,rouge_2_f,rouge_l_f])

#   return {
#       "rouge_1/f_score": rouge_1_f,
#       "rouge_1/r_score": rouge_1_r,
#       "rouge_1/p_score": rouge_1_p,
#       "rouge_2/f_score": rouge_2_f,
#       "rouge_2/r_score": rouge_2_r,
#       "rouge_2/p_score": rouge_2_p,
#       "rouge_l/f_score": rouge_l_f,
#       "rouge_l/r_score": rouge_l_r,
#       "rouge_l/p_score": rouge_l_p,
#   }

def compute_exact_match(pred,gold):
    pred = sorted(pred)
    gold = sorted(gold)
    if(" ".join(pred) == " ".join(gold)):
        return 1
    else:
        return 0

def hasNoNumbers(inputString):
  return not any(char.isdigit() for char in inputString)

def checker_global_ent(e,gold):
  nonumber = hasNoNumbers(e)
  sub_string = True
  for g in gold:
    if(e.lower() in g.lower()):
      sub_string = False
  return sub_string and nonumber

def substringSieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out

def compute_prf(pred, gold, global_entity_list, local_kb_word):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        for g in gold:
            if g.lower() in pred.lower():
                TP += 1
            else:
                FN += 1
        list_FP = []
        for e in list(set(global_entity_list+list(local_kb_word))):
          if e.lower() in pred.lower() and checker_global_ent(e,gold):
            list_FP.append(e)
        FP = len(list(set(substringSieve(list_FP))))
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        F1 = None
    return F1

def compute_prf_SMD(pred, gold, global_entity_list, local_kb_word):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        for g in gold:
            if g.lower() in pred.lower():
                TP += 1
            else:
                FN += 1
        list_FP = []
        for e in list(set(global_entity_list+list(local_kb_word))):
          if e.lower() in pred.lower():
            list_FP.append(e)
        FP = len(list(set(substringSieve(list_FP))))
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        F1 = None
    return F1


def get_global_entity_KVR():
  with open('data/SMD/kvret_entities.json') as f:
      global_entity = json.load(f)
      global_entity_list = []
      for key in global_entity.keys():
          if key != 'poi':
              global_entity_list += [str(item) for item in global_entity[key]]
          else:
              for item in global_entity['poi']:
                  global_entity_list += [str(item[k]) for k in item.keys()]
      global_entity_list = list(set(global_entity_list))
  return global_entity_list


def get_global_entity_MWOZ():
  with open('data/MultiWOZ_2.1/ontology.json') as f:
      global_entity = json.load(f)
      global_entity_list = []
      for key in global_entity.keys():
          if(key not in ["train-book-people","restaurant-book-people",
                        "hotel-semi-stars","hotel-book-stay","hotel-book-people"]): ## this because are single numeber
            global_entity_list += global_entity[key]

      global_entity_list = list(set(global_entity_list))
  return global_entity_list

def get_global_entity_DIALKG():
  with open('data/opendialkg/data/opendialkg_entities.txt') as f:
    global_entity_list = []
    for x in f:
      global_entity_list.append(x.replace("\n",""))
  return list(set(global_entity_list))

# DON'T FUCKING WASTE GPU MEMORY IF IT IS NOT NEEDED!!!!
# try:
#   bleurt_scorer = bleurt.score.BleurtScorer("bleurt-base-128")
# except:
#   bleurt_scorer = None
# def google_bleurt(predictions, references):
#   if bleurt_scorer is not None:
#     scores = bleurt_scorer.score(references, predictions)
#     return np.average(scores)
#   else:
#     return None

cached_bertscorer = {}
def huggingface_bertscore(
  predictions,
  references, 
  lang=None,
  model_type=None,
  num_layers=None,
  verbose=False,
  idf=False,
  device=None,
  batch_size=64,
  nthreads=4,
  all_layers=False,
  rescale_with_baseline=False):

  # taken from nlp repo since we can't load the function at this moment
  try:
    if model_type is None:
        model_type = bert_score.utils.lang2model[lang.lower()]

    if num_layers is None:
        num_layers = bert_score.utils.model2layers[model_type]

    hashcode = bert_score.utils.get_hash(model_type, num_layers, idf, rescale_with_baseline)
    if hashcode not in cached_bertscorer:
      cached_bertscorer[hashcode] = bert_score.BERTScorer(
          model_type=model_type,
          num_layers=num_layers,
          batch_size=batch_size,
          nthreads=nthreads,
          all_layers=all_layers,
          idf=idf,
          device=device,
          lang=lang,
          rescale_with_baseline=rescale_with_baseline,
      )

    (P, R, F) = cached_bertscorer[hashcode].score(
        cands=predictions, refs=references, verbose=verbose, batch_size=batch_size,
    )
    return np.average(F.cpu().tolist())
  except:
    return None


def test():
  ref = "I don't like dogs"
  sent = "You are mega man"
  persona = ["I hate dogs", "I like blue cars"]

  
  BLEU = moses_multi_bleu(np.array([sent]),np.array([ref]))
  print(BLEU)
  BLEU = moses_multi_bleu(np.array([ref]),np.array([ref]))
  print(BLEU)
#   BLEURT = google_bleurt([sent] * 2,[ref] * 2)
#   print(BLEURT)
  BERTScore = huggingface_bertscore([sent],[ref],lang="en",model_type="bert-base-uncased",num_layers=9, batch_size=1)
  print(BERTScore)
  ROUGE = rouge([sent],[ref])
  print(ROUGE)
  ROUGE = rouge([ref],[ref])
  print(ROUGE)
  ENT = ent_score(sent, persona)
  print(ENT)
  ENT = ent_score(ref, persona)
  print(ENT)

# test()