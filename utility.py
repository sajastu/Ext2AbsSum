import math
import torch

def integrate_src_tgt(model_inputs, tgt_inputs):
    ret = {}


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


