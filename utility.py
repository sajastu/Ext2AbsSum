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



def _combine_model_inputs(model_inputs, labels, tgt_doc_ids, targets, is_train=False):
    paper_label_ids = {}
    q = False
    # import pdb;pdb.set_trace()
    for label_id, paper_id, tgt in zip(labels["input_ids"], tgt_doc_ids, targets):
        # if label_id[0] == -100:
        #     print(f'{tgt} ---- label_id: {label_id}')
        if paper_id=='SP:400f24337c27f8f1fbb40ba7dd6c2a7c92b7a32f--main':
            q = True
        if paper_id not in paper_label_ids.keys():
            paper_label_ids[paper_id] = [label_id]
        else:
            paper_label_ids[paper_id].append(label_id)

    # if examples[key][jpid] == 'SP:400f24337c27f8f1fbb40ba7dd6c2a7c92b7a32f':
    # if q:
        # import pdb;pdb.set_trace()
    model_inputs["labels"] = []

    for paper_id in model_inputs['doc_ids']:
        model_inputs["labels"].append(paper_label_ids[paper_id])

    # if is_train:
        # constrain each paper to have 2 summaries at most within an instance...
        model_inputs_last = {k:[] for k in model_inputs.keys()}
        # for jpid, paper_id in enumerate(model_inputs['doc_ids']):
        #     sum_sent_labels = model_inputs['sum_sents_labels'][jpid]
        #     labels = model_inputs['labels'][jpid]
        #
        #     num_summaries = len(sum_sent_labels[0])
        #     # model_inputs['doc_ids'][jpid]
        #     N_SUM=1
        #     if num_summaries > N_SUM:
        #         # split...
        #         start_idx = 0
        #         while start_idx < num_summaries - 1:
        #             end_idx = start_idx + N_SUM
        #
        #             if end_idx > num_summaries - 1:
        #                 end_idx = num_summaries - 1
        #
        #             concat_id = int(start_idx/N_SUM)
        #             if model_inputs['doc_ids'][jpid] == 'SP:6e54083a06942f2c41e1796a9f911d3dd9bab0cc':
        #                 # import pdb;pdb.set_trace()
        #                 print(f'here {concat_id}')
        #             # import pdb;pdb.set_trace()
        #             if end_idx > num_summaries:
        #                 end_idx = num_summaries
        #                 concat_id = int(start_idx/N_SUM)
        #
        #             if start_idx != end_idx:
        #                 new_sum_sent_labels = [s[start_idx:end_idx] for s in sum_sent_labels]
        #                 new_labels = labels[start_idx:end_idx]
        #                 # if model_inputs['doc_ids'][jpid] == 'SP:0cf756ba6b172f9b29e84945c093dfd89ae62803':
        #                 #     import pdb;pdb.set_trace()
        #                 for key in model_inputs.keys():
        #                     if key=='doc_ids':
        #                         model_inputs_last[key].append(model_inputs[key][jpid] + f'--{concat_id}')
        #                     elif key=='sum_sents_labels':
        #                         model_inputs_last[key].append(new_sum_sent_labels)
        #                     elif key=='labels':
        #                         model_inputs_last[key].append(new_labels)
        #                     else:
        #                         model_inputs_last[key].append(model_inputs[key][jpid])
        #             start_idx += N_SUM
        #     else:
        #         # normal
        #         # if model_inputs['doc_ids'][jpid] == 'SP:0cf756ba6b172f9b29e84945c093dfd89ae62803':
        #         #     import pdb;pdb.set_trace()
        #         for key in model_inputs.keys():
        #             model_inputs_last[key].append(model_inputs[key][jpid])

        # model_inputs = model_inputs_last
    # print(f'model_inputs {len(model_inputs["doc_ids"])}')
    return model_inputs


