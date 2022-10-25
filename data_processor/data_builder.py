# -*- coding:utf-8 -*-

import gc
import glob
import json
import os
import random
import re
from itertools import chain
from multiprocessing import Pool

import numpy as np
import torch
from os.path import join as pjoin

from collections import Counter
from rouge_score import rouge_scorer

from data_processor.prep_util import _get_word_ngrams
from transformers import LEDTokenizer
from nltk.corpus import stopwords as stop_words
sw = set(stop_words.words("english"))

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

from tqdm import tqdm

# from data_processor.others.tokenization import BertTokenizer
from data_processor.others.logging_utils import logger
from data_processor.others.vocab_wrapper import VocabWrapper



# def greedy_selection(doc, summ, summary_size):
#
#     doc_sents = list(map(lambda x: x["original_txt"], doc))
#     max_rouge = 0.0
#
#
#     # rouge = Rouge()
#     selected = []
#     while True:
#         cur_max_rouge = max_rouge
#         cur_id = -1
#         for i in range(len(doc_sents)):
#             if (i in selected):
#                 continue
#             c = selected + [i]
#             temp_txt = " ".join([doc_sents[j] for j in c])
#             if len(temp_txt.split()) > summary_size:
#                 continue
#             # rouge_score = rouge.get_scores(temp_txt, summ)
#             # rouge_1 = rouge_score[0]["rouge-1"]["r"]
#             # rouge_l = rouge_score[0]["rouge-l"]["r"]
#             scores = scorer.score(summ, temp_txt)
#             rouge_score = scores['rouge1'].recall + scores['rougeL'].recall
#             # rouge_score = 0
#             if rouge_score > cur_max_rouge:
#                 cur_max_rouge = rouge_score
#                 cur_id = i
#         if (cur_id == -1):
#             return selected
#         selected.append(cur_id)
#         max_rouge = cur_max_rouge
#
#
#     return selected


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
    selected_rg = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = []
        cur_rg_scores = []
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
            if s == 0:
                if rouge_score > cur_max_rouge:
                    # cur_max_rouge = rouge_score
                    # cur_id = i
                    cur_id.append(i)
                    selected_rg.append(rouge_score)
                    cur_rg_scores.append(rouge_score)
            else:
                if rouge_score > cur_max_rouge:
                    # cur_max_rouge = rouge_score
                    # cur_id = i

                    cur_id.append(i)
                    selected_rg.append(rouge_score)
                    cur_rg_scores.append(rouge_score)

                    if len(cur_rg_scores) > 1:
                        break
        if (len(cur_id) == 0):
            for idx in range(len(doc_sents)):
                if idx in selected:
                    bin_labels.append(1)
                else:
                    bin_labels.append(0)

            return bin_labels
        if s==0:
            # first sentence of summary, should pick top 2 somehow...
            cur_id_rg_scores = [(id, rg) for id, rg in zip(cur_id, cur_rg_scores)]
            sorted_cur_id_rg = sorted(cur_id_rg_scores, key=lambda x: x[-1], reverse=True)
            cur_id = [sorted_cur_id_rg[0][0]]
            cur_rg_scores = [sorted_cur_id_rg[0][1]]
            selected_rg = [sorted_cur_id_rg[0][1]]


        max_rouge = np.max(cur_rg_scores)
        selected.extend(cur_id)


    for idx in range(len(doc_sents)):
        if idx in selected:
            bin_labels.append(1)
        else:
            bin_labels.append(0)

    return bin_labels

class BertData():
    def __init__(self, args):
        self.args = args

        self.tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

        self.sep_token = '</s>'
        self.cls_token = '<s>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.tgt_bos = '<s>'
        self.tgt_eos = '</s>'
        # self.role_1 = '[unused3]'
        # self.role_2 = '[unused4]'

        self.sep_vid = self.tokenizer.convert_tokens_to_ids([self.sep_token])[0]
        self.cls_vid = self.tokenizer.convert_tokens_to_ids([self.cls_token])[0]
        self.pad_vid = self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]
        self.unk_vid = self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]

    def preprocess_src(self, content, info=None):
        # if_exceed_length = False
        #
        # # if not (info == "客服" or info == '客户'):
        # #     return None
        # if len(content) < self.args.min_src_ntokens_per_sent:
        #     return None
        # if len(content) > self.args.max_src_ntokens_per_sent:
        #     if_exceed_length = True
        #
        original_txt = ' '.join(content)

        if self.args.truncated:
            content = content[:self.args.max_src_ntokens_per_sent]
        content_text = ' '.join(content).lower()
        content_subtokens = self.tokenizer.tokenize(content_text)

        # [CLS] + T0 + T1 + ... + Tn
        # if info == '客服':
        #     src_subtokens = [self.cls_token, self.role_1] + content_subtokens
        # else:
        #     src_subtokens = [self.cls_token, self.role_2] + content_subtokens
        src_subtokens = [self.cls_token] + content_subtokens

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        segments_ids = len(src_subtoken_idxs) * [0]

        return src_subtoken_idxs, segments_ids, original_txt, src_subtokens

    def preprocess_summary(self, content_text):

        # original_txt = ' '.join(content)
        #
        # content_text = ' '.join(content).lower()
        content_subtokens = self.tokenizer.tokenize(content_text)

        content_subtokens = [self.tgt_bos] + content_subtokens + [self.tgt_eos]
        subtoken_idxs = self.tokenizer.convert_tokens_to_ids(content_subtokens)

        return subtoken_idxs, content_text, content_subtokens

    def integrate_dialogue(self, dialogue):
        src_tokens = [self.cls_token]
        segments_ids = [0]
        segment_id = 0
        for sent in dialogue:
            tokens = sent["src_tokens"][1:] + [self.sep_token]
            src_tokens.extend(tokens)
            segments_ids.extend([segment_id] * len(tokens))
            segment_id = 1 - segment_id
        src_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
        return {"src_id": src_ids, "segs": segments_ids}


def topic_info_generate(dialogue, section=True):
    all_counter_global = Counter()
    global_ids = chain.from_iterable(dialogue['tokenized_ids'])
    all_counter_global.update(global_ids)

    all_counter_lst_section = []

    for section_ids in dialogue['tokenized_ids']:
        all_counter = Counter()
        token_ids = section_ids
        all_counter.update(token_ids)
        all_counter_lst_section.append(all_counter)

    # import pdb;pdb.set_trace()
    # file_counter['all'].update(all_counter.keys())
    # file_counter['customer'].update(customer_counter.keys())
    # file_counter['agent'].update(agent_counter.keys())
    # file_counter['num'] += 1

    return all_counter_lst_section, all_counter_global


def topic_summ_info_generate(dialogue, ex_labels):
    all_counter = Counter()
    for i, sent in enumerate(dialogue):
        if i in ex_labels:
            token_ids = sent["tokenized_id"]
            all_counter.update(token_ids)
    return {"all": all_counter}


def format_to_lines(args, corpus_type=None, create_jsons=True):
    # write json files

    if create_jsons:
        corpus_types = ['train', 'val', 'test']
        for corpus_type in corpus_types:
            written_insts = 0
            CHUNK_SIZE = 1000

            # for corpus_type in [corpus_type]:
            instances = []
            for json_f in glob.glob(pjoin(args.raw_path + f'/{corpus_type}/' + '*.json')):
                instances.append(json.load(open(json_f)))

            print('Creating json files...')
            for iter in tqdm(range((len(instances)//CHUNK_SIZE) + 1), total=(len(instances)//CHUNK_SIZE) + 1):
                with open(f'{args.jsons_path}/{corpus_type}.{iter}.json', mode='w') as fW:
                    try:
                        json.dump(instances[iter * CHUNK_SIZE: (iter+1) * CHUNK_SIZE], fW)
                        written_insts += len(instances[iter * CHUNK_SIZE: (iter+1) * CHUNK_SIZE])
                    except:
                        json.dump(instances[iter * CHUNK_SIZE:], fW)
                        written_insts += len(instances[iter * CHUNK_SIZE:])

            print(f'Written instances for {corpus_type}: {written_insts}')

    format_to_bert(args)


def format_to_bert(args):

    # writing aggregated json files...

    a_lst = []
    # get val and train...
    for json_f in glob.glob(args.jsons_path + f'/' + '*.json'):
        real_name = json_f.split('/')[-1]
        a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

    file_counter_global = {"all": Counter(), "num": 0, "voc_size": 0}
    file_counter_section = {"all": Counter(), "num": 0, "voc_size": 0}
    voc_wrapper = VocabWrapper(args.emb_mode)
    voc_wrapper.load_emb(args.emb_path)
    file_counter_global['voc_size'] = voc_wrapper.voc_size()
    file_counter_section['voc_size'] = voc_wrapper.voc_size()

    import pandas as pd
    heading_keywords = pd.read_csv('data_processor/heading_keyword.csv')

    for d in a_lst:
        file_counter_global, file_counter_section = _format_to_bert(d, file_counter_global, file_counter_section, voc_wrapper, heading_keywords)

    # save file counter
    save_file = pjoin(args.save_path, 'idf_info_global.pt')
    logger.info('Saving global file counter to %s' % save_file)
    torch.save(file_counter_global, save_file)

    save_file = pjoin(args.save_path, 'idf_info_section.pt')
    logger.info('Saving section file counter to %s' % save_file)
    torch.save(file_counter_section, save_file)



import spacy

nlp = spacy.load('en_core_sci_lg')

def get_sentence_tokens(text):
    doc_nlp = nlp(text)
    ret = []
    for doc_sentence in doc_nlp.sents:
        sent = []
        for tkn in doc_sentence:
            sent.append(tkn.text)
        if len(sent) > 0:
            ret.append(sent)
    return ret

def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

def is_abstract_intro_or_conc(heading, heading_keywords, sect_tokens):

    while heading[0].isdigit() or heading[0] == '.':
        heading = heading[1:]
    heading = heading.strip()

    if 'abstract' in heading.lower():
        return True

    INTRO_CONC_SECTIONS = heading_keywords['intro'].dropna().tolist() + heading_keywords['conclusion'].dropna().tolist()
    one_grams = ngrams(heading.lower(), 1)
    ongrams = []
    for oneg in one_grams:
        ongrams.append(oneg[0])

    for one_gram in ongrams:
        for intro_conc_sect in INTRO_CONC_SECTIONS:
            if one_gram.strip() == intro_conc_sect.strip():
                return True

    if len(heading.split(' ')) > 1:
        two_grams = ngrams(heading.lower(), 2)
        twograms = []
        for tg in two_grams:
            twograms.append(f'{tg[0]} {tg[1]}')

        for two_gram in twograms:
            for intro_conc_sect in INTRO_CONC_SECTIONS:
                if two_gram.strip() == intro_conc_sect.strip():
                    return True

    return False


def is_bad_section(heading, sect_tokens):

    if 'related work' in heading.lower() or 'appendix' in heading.lower() or 'appendices' in heading.lower() or 'proofs' in heading.lower():
        return True

    return False

def _mp_process_instance(params):
    paper_jobj, voc_wrapper, heading_keywords, debug = params
    sections_tokens = []
    # should process section by section ... dataset
    for index, paper_info in enumerate(paper_jobj['sections_txt_tokenized']):
        sections_txt_tokenized = paper_info
        paper_id = paper_jobj['paper_id']
        summaries = paper_jobj['summaries']
        section_text, section_tokens, heading = sections_txt_tokenized['text'].replace(' </s>', '').replace(' <s>', '').replace(' <mask>', '') \
                              .replace('<s>', '').replace('</s>', '').replace('<mask>', '') \
                              .replace('\n', ' ').strip(), sections_txt_tokenized['tokenized'], sections_txt_tokenized['heading']
        sections_tokens.append((section_text, section_tokens, heading))

    b_data_dict = {"tokenized_ids": []}

    sect_idx = 0
    hyp_sections = []
    hyp_sections_sents_tokens = []
    hyp_sections_token_sents = []
    hyp_sections_txt = []
    hyp_sect_headings = []

    # performing a section filtering phase prior to the main stage.
    sections_tokens_tmp = []
    while sect_idx < len(sections_tokens):
        sect_heading = sections_tokens[sect_idx][2]
        sect_tokens = sections_tokens[sect_idx][1]

        if '</latexit>' in sections_tokens[sect_idx][0]:
            refined = re.sub('latexit.*.latexit', '', sections_tokens[sect_idx][0])
            lst = list(sections_tokens[sect_idx])
            lst[0] = refined
            lst[1] = get_sentence_tokens(refined)
            sections_tokens[sect_idx] = tuple(lst)

        if is_bad_section(sect_heading, sect_tokens):
            sect_idx += 1
            continue
        else:
            sections_tokens_tmp.append(sections_tokens[sect_idx])
            sect_idx += 1



    sections_tokens = sections_tokens_tmp
    sect_idx = 0

    while sect_idx < len(sections_tokens):

        sect_tokens = sections_tokens[sect_idx][1]

        sect_heading = sections_tokens[sect_idx][2]
        s_idx = sect_idx


        # if sect_heading is in Intro or Conclusion, take it as it is.
        # if debug:
        #     import pdb;pdb.set_trace()

        if is_abstract_intro_or_conc(sect_heading, heading_keywords, sect_tokens):
            hyp_sections.append((f'{sect_idx}', list(chain.from_iterable(sect_tokens))))
            hyp_sections_token_sents.append((f'{sect_idx}', sect_tokens))
            combined_sect_txt = sections_tokens[sect_idx][0]
            hyp_sections_txt.append(combined_sect_txt)
            hyp_sect_headings.append(sect_heading)

        else:
            if len(list(chain.from_iterable(sect_tokens))) < 512 and sect_idx+1 < len(sections_tokens):
                # check if adding next section increases it to be more than 256
                while sect_idx+1 < len(sections_tokens) and len(list(chain.from_iterable(sections_tokens[sect_idx][1]))) + len(list(chain.from_iterable(sections_tokens[sect_idx+1][1]))) < 512:
                    sect_idx += 1
                sect_idx += 1
                if sect_idx < len(sections_tokens) and sect_idx-s_idx > 0:
                    combined_sect = [list(chain.from_iterable(s[1])) for s in sections_tokens[s_idx:sect_idx+1]]
                    combined_sect_txt = [s[0] for s in sections_tokens[s_idx:sect_idx+1]]
                    combined_sect_headings = [s[2] for s in sections_tokens[s_idx:sect_idx+1]]
                    hyp_sections.append((f'{s_idx}-{sect_idx}', list(chain.from_iterable(combined_sect))))

                    new_sect = []
                    for sect_h in sections_tokens[s_idx:sect_idx + 1]:
                        for sent1 in sect_h[1]:
                            new_sect.append(sent1)

                    hyp_sections_token_sents.append((f'{s_idx}-{sect_idx}', new_sect))
                    hyp_sections_txt.append(" ".join(combined_sect_txt))
                    hyp_sect_headings.append(' <COMBINED> '.join(combined_sect_headings))

                elif sect_idx == len(sections_tokens) -1 :
                    hyp_sections.append((f'{s_idx}', list(chain.from_iterable(sections_tokens[-1][1]))))
                    # import pdb;pdb.set_trace()

                    hyp_sections_token_sents.append((f'{s_idx}', sections_tokens[s_idx][1]))

                    combined_sect_txt = sections_tokens[s_idx][0]
                    combined_sect_headings = sections_tokens[s_idx][2]
                    hyp_sections_txt.append(combined_sect_txt)
                    hyp_sect_headings.append(combined_sect_headings)

                else:
                    hyp_sections.append((f'{s_idx}', list(chain.from_iterable(sections_tokens[s_idx][1]))))

                    hyp_sections_token_sents.append((f'{s_idx}', sections_tokens[s_idx][1]))
                    combined_sect_txt = sections_tokens[s_idx][0]
                    hyp_sections_txt.append(combined_sect_txt)
                    combined_sect_headings = sections_tokens[s_idx][2]
                    hyp_sect_headings.append(combined_sect_headings)


            else:
                hyp_sections.append((f'{sect_idx}', list(chain.from_iterable(sect_tokens))))

                hyp_sections_token_sents.append((f'{sect_idx}', sect_tokens))
                combined_sect_txt = sections_tokens[sect_idx][0]
                hyp_sections_txt.append(combined_sect_txt)
                combined_sect_headings = sections_tokens[s_idx][2]
                hyp_sect_headings.append(combined_sect_headings)

        sect_idx += 1

    # import pdb;pdb.set_trace()
    sect_boundaries = [h[0] for h in hyp_sections]
    hyp_sections = [h[1] for h in hyp_sections]
    # hyp_sections_txt = [' '.join(h) for h in hyp_sections]
    # import pdb;pdb.set_trace()
    assert len(hyp_sections_txt) == len(hyp_sections), 'Discrep1: Topic section and sections'

    hyp_sections_token_sents = [h[1] for h in hyp_sections_token_sents]


    ext_labels = []
    sections_sents = []
    paper_sents = []
    sect_len = []
    masked_sections = []
    for sec_id, sect_tokens in enumerate(hyp_sections):
        # import pdb;pdb.set_trace()
        ids = [voc_wrapper.w2i(x.lower()) for x in sect_tokens]
        tokenized_id = [x for x in ids if x is not None]
        for id in tokenized_id:
            if id > voc_wrapper.voc_size():
                print('id is larger than vocab size...')

        section_sents = get_sentence_tokens(' '.join(sect_tokens))

        if len(section_sents) > 1:
            paper_sents.extend(section_sents)
            sect_len.append(len(section_sents))
            sections_sents.append([' '.join(s) for s in section_sents if len(' '.join(s).strip()) > 0].copy())
            section_ext_labels = []
            if 'sample hence no summary' in summaries[0]:
                section_ext_labels.append([0] * len(section_sents))
            else:
                for summary in summaries:
                    section_sents_tmp = []
                    for sent in section_sents:
                        sent_toks_tmp = []
                        for sent_tok in sent:
                            if sent_tok.lower() not in sw:
                                sent_toks_tmp.append(sent_tok.lower())

                        section_sents_tmp.append(sent_toks_tmp)

                    summary_tmp = summary.lower()
                    for sww in sw:
                        summary_tmp = summary_tmp.replace(f' {sww.lower()} ', ' ')

                    ext_labels_section_summary = greedy_selection(section_sents_tmp, summary_tmp, 10)
                    section_ext_labels.append(ext_labels_section_summary)
            ext_labels.append(section_ext_labels.copy())
            b_data_dict["tokenized_ids"].append(tokenized_id)
            masked_sections.append(True)
        else:
            # update hype_section
            masked_sections.append(False)

    # dropping masked sections
    # import pdb;pdb.set_trace()
    hyp_sections_txt = np.array(hyp_sections_txt)[masked_sections].tolist()
    hyp_sect_headings = np.array(hyp_sect_headings)[masked_sections].tolist()
    hyp_sections_token_sents_tmp = []
    for j, tr in enumerate(masked_sections):
        if tr == True:
            hyp_sections_token_sents_tmp.append(hyp_sections_token_sents[j])

    hyp_sections_token_sents = hyp_sections_token_sents_tmp
    # import pdb;pdb.set_trace()


    sect_len = np.concatenate(([0], np.cumsum(np.array(sect_len))))
    # get document-level ext labels...
    if 'sample hence no summary' in summaries[0]:
        # section_ext_labels.append([0] * len(section_sents))
        s=0 # do nothing
    else:

        for sum_idx, summary in enumerate(summaries):
            ext_labels_document_summary = greedy_selection(paper_sents, summary, 30)

            for sect_idx in range(len(ext_labels)):
                sect_sum_labels = ext_labels[sect_idx][sum_idx]
                ext_labels_from_document = ext_labels_document_summary[sect_len[sect_idx]: sect_len[sect_idx+1]]
                sect_sum_labels = np.array([(d+s) for d, s in zip(ext_labels_from_document, sect_sum_labels)])
                sect_sum_labels = np.where(sect_sum_labels == 2, 1, sect_sum_labels).tolist()
                ext_labels[sect_idx][sum_idx] = sect_sum_labels

    assert len(ext_labels) == len(sections_sents), 'discrep found'
    dialogue_example = {
        'paper_id': paper_id,
        'section_boundaries': sect_boundaries,
        "section_text": hyp_sections_txt,
        "section_sent_tokenized": hyp_sections_token_sents,
        "section_headings": hyp_sect_headings,
        "sections_sents": sections_sents,
        "ext_labels": ext_labels,
    }

    topic_info_section, topic_info_global = topic_info_generate(b_data_dict)

    assert len(topic_info_section) == len(dialogue_example['section_text']), 'Discrep2: Topic section and sections'

    dialogue_example["topic_info_section"] = topic_info_section
    dialogue_example["topic_info_global"] = topic_info_global


    return dialogue_example


def _format_to_bert(params, file_counter_global, file_counter_section, voc_wrapper, heading_keywords, debug=False):

    json_file, _, save_file = params

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []

    count = 0

    print(f'Processing {json_file}...')

    mp_instances = []

    if not debug:
        for paper_jobj in tqdm(jobs, total=len(jobs), desc=f'Processing...'):
            mp_instances.append((paper_jobj, voc_wrapper, heading_keywords, debug))
    else:
        mp_instances.append((jobs, voc_wrapper, heading_keywords, debug))

    pool = Pool(16)

    if debug:
        for mi in mp_instances:
            _mp_process_instance(mi)

    for dialogue_example in tqdm(pool.imap_unordered(_mp_process_instance, mp_instances), total=len(mp_instances)):
        file_counter_global['all'].update(dialogue_example['topic_info_global'].keys())
        file_counter_section['all'].update(list(chain.from_iterable([k.keys() for k in dialogue_example['topic_info_section']])))
        file_counter_global['num'] += 1
        file_counter_section['num'] += len(dialogue_example['topic_info_section'])
        datasets.append(dialogue_example)
        count += 1
        # if count % 50 == 0:
        #     print(count)

    datasets_dict = {}
    for ins in datasets:
        datasets_dict[ins['paper_id']] = ins

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets_dict, save_file)

    datasets = []
    gc.collect()
    return file_counter_global, file_counter_section

