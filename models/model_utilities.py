import re
from itertools import chain

import math
import torch

import spacy
from tqdm import tqdm
import torch

nlp = spacy.load('en_core_sci_lg')

def get_sentence_tokens(text):
    doc_nlp = nlp(text)
    ret = []
    for doc_sentence in doc_nlp.sents:
        sent = []
        for tkn in doc_sentence:
            sent.append(tkn.text)
        ret.append(sent)
    return ret


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


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)



def _combine_model_inputs(model_inputs, labels, tgt_doc_ids, targets):

    paper_label_ids = {}
    for label_id, paper_id, tgt in zip(labels["input_ids"], tgt_doc_ids, targets):
        # if label_id[0] == -100:
        #     print(f'{tgt} ---- label_id: {label_id}')
        if paper_id not in paper_label_ids.keys():
            paper_label_ids[paper_id] = [label_id]
        else:
            paper_label_ids[paper_id].append(label_id)

    model_inputs["labels"] = []

    for paper_id in model_inputs['doc_ids']:
        model_inputs["labels"].append(paper_label_ids[paper_id])

    # import pdb;pdb.set_trace()
    # debugging
    # new_model_inputs = {}
    # for k in model_inputs.keys():
    #     new_model_inputs[k] = []
    # for j, val in enumerate(model_inputs['labels']):
    #     if len(val) > 1:
    #         keys = model_inputs.keys()
    #         for k in keys:
    #             if len(new_model_inputs[k]) == 0:
    #                 new_model_inputs[k] = [model_inputs[k][j]]
    #             else:
    #                 new_model_inputs[k].append(model_inputs[k][j])
    #
    # model_inputs = new_model_inputs
    # debugging done

    return model_inputs


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc, abstract, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    # abstract = sum(abstract_sent_list, [])

    abstract = list(chain.from_iterable(get_sentence_tokens(_rouge_clean(abstract))))
    doc_sents = doc

    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sents]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])
    bin_labels = []

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):

            for idx in range(len(doc_sents)):
                if idx in selected:
                    bin_labels.append(1)
                else:
                    bin_labels.append(0)

            return bin_labels
        selected.append(cur_id)
        max_rouge = cur_max_rouge


    for idx in range(len(doc_sents)):
        if idx in selected:
            bin_labels.append(1)
        else:
            bin_labels.append(0)

    return bin_labels
