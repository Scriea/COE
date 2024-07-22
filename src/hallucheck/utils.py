import re
import torch
import random
import collections
import numpy as np
from scipy.stats import kstest

import json
import math
import string

# # nlp_model = spacy.load("en_core_web_sm")
# stanza.download('en')     
# nlp_ner = stanza.Pipeline(lang='en', processors='tokenize,ner')
# nlp_pos = stanza.Pipeline(lang='en', processors='tokenize,pos')
# nlp_coreref = stanza.Pipeline("en", processors="tokenize,coref")
# nlp_sent = spacy.load("en_core_web_sm")


## ----------------------------------------------------------------

"""
Squad Metrics:
    Source: https://github.com/huggingface/transformers/blob/main/src/transformers/data/metrics/squad_metrics.py
"""

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    # def lower(text):
    #     return text.lower()

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

# Exact Match
def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

# F1 Score
def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


## ----------------------------------------------------------------


def extract_text_between_double_quotes(input_string):
    pattern = r'"([^"]*)"'
    try:
        matches = re.findall(pattern, input_string)
    except:
        matches = []
    return matches

def extract_sentences(nlp_model,orig_answer):
    doc = nlp_model(orig_answer)
    sentences = [sent.text for sent in doc.sents]
    return sentences



## Temporary
# def do_coreference(sentence):
#     """
#     Description: 
#         This function takes a sentence as input and 
#         returns the sentence with coreferences resolved.
#     """

#     corerefernced_Sent = nlp_coreref(sentence)
#     for sent in corerefernced_Sent.sentences:
#         for word in sent.words:
#             # print(word.coref_chains[0].chain.representative_text)
#             try:
#                 i = 0
#                 while(i < len(word.coref_chains) and word.text in word.coref_chains[i].chain.representative_text):
#                     i = i + 1
#                 if i != len(word.coref_chains):
#                     sentence = sentence.replace(word.text, word.coref_chains[i].chain.representative_text)
#             except:
#                 i = i + 1
#                 pass
#     return sentence



def calculate_f1_score(reference_answer, answer):
    # Convert answers and reference_answer to sets of words
    answer_set = set(answer.split())
    reference_set = set(reference_answer.split())

    # Calculate precision, recall, and F1 score
    precision = len(answer_set.intersection(reference_set)) / len(answer_set)
    recall = len(answer_set.intersection(reference_set)) / len(reference_set)

    # Handle the case where precision and recall are both zero
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1
    


def find_subset_indices(orig_list, subset):
    len_orig = len(orig_list)
    len_sub = len(subset)
    for i in range(len_orig - len_sub + 1):
        if orig_list[i:i + len_sub] == subset:
            return list(range(i, i + len_sub))
    return []

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum()
