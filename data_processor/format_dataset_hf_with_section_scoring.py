import argparse
import glob
import json
import math
import os
from multiprocessing import Pool

import numpy as np
import spacy
from tqdm import tqdm
import torch

nlp = spacy.load('en_core_sci_lg')
from rouge_score import rouge_scorer
metrics = ['rouge1', 'rouge2', 'rougeL']
scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

def get_sentence_tokens(text):
    doc_nlp = nlp(text)
    ret = []
    for doc_sentence in doc_nlp.sents:
        sent = []
        for tkn in doc_sentence:
            sent.append(tkn.text)
        ret.append(sent)
    return ret

def _parse_paper(param):
    ex = param
    example_sents = get_sentence_tokens(ex['source'])
    ex['source_sents'] = example_sents.copy()

    # new_sects = []
    # for e in ex['paper']['sections']:
    #     example_sents.extend()
        # e['tokens'] = get_sentence_tokens(e['text'])
        # new_sects.append(e)

    # ex['sentences'] = example_sents
    # ex['paper']['sections'] = new_sects

    return ex

def _cal_rg_sect(params):
    p_id, sect_sents, summaries, ext_labels, is_test = params
    input_sents = []
    for sect in sect_sents:
        sent_sects = []
        for sent in sect:
            sent_sects.append(sent.replace(' </s>', '').replace(' <s>', '').replace(' <mask>', '') \
                              .replace('<s>', '').replace('</s>', '').replace('<mask>', '') \
                              .replace('\n', ' ').strip().lower())

        SECT_SENTTS = ' <SENTTT> '.join(sent_sects.copy())
        input_sents.append(SECT_SENTTS.split(' <SENTTT> '))

    sect_sents = input_sents
    tkns = []
    for sect1 in sect_sents:
        sent_tkns_sect = []
        for sent1 in sect1:
            sent_tkns_sect.append(get_tokens(sent1))
        tkns.append(sent_tkns_sect)

    sect_sents_tokenized = tkns

    # adding section scores
    section_scores = []

    for sect_idx, sect in enumerate(sect_sents):
        sums_sect_score = []
        for summ_idx, summ in enumerate(summaries):
            sent_sect_scores = []
            for sect_sent in sect:
                if not is_test:
                    sents_scores = scorer.score(summ.strip(), sect_sent.strip())
                    sents_scores = [sents_scores['rouge1'].recall, sents_scores['rouge2'].recall,
                                    sents_scores['rougeL'].recall]
                else:
                    sents_scores = [1.0, 1.0, 1.0]
                avg_sent_score = np.average(sents_scores)
                sent_sect_scores.append(avg_sent_score)
            # try:
            sect_sum_labels = ext_labels[sect_idx][summ_idx]
            # except:
            #     sect_sum_labels = ext_labels[sect_idx][summ_idx]
            #
            #     import pdb;pdb.set_trace()
            sect_sents_ = sect_sents[sect_idx]
            sect_tokens = sum([len(s) for s in sect_sents_tokenized[sect_idx]])

            rg_avrg = (np.average(sent_sect_scores)+0.01) * ((sum(sect_sum_labels)+1) / (math.sqrt(sect_tokens / len(sect_sents_))))
            # import pdb;pdb.set_trace()
            """
            sectScoreV1.parquet: 
                -   rg_avrg = np.average(sent_sect_scores) * ((sum(sect_sum_labels)) / math.sqrt(len(sect_tokens.split())/len(sect_sents_))) 
            """
            # weighting by sect_len and number of important sentences


            sums_sect_score.append(rg_avrg)
        section_scores.append(sums_sect_score.copy())

    section_scores = np.array(section_scores)

    # section_scores / np.moveaxis(section_scores,0, -1).sum(axis=1)[None, :]
    return (p_id, section_scores, sect_sents_tokenized)

def get_tokens(text):
    doc_nlp = nlp(text)
    ret = []
    for tkn in doc_nlp:
        ret.append(tkn.text)
    return ret


def load_topic_info(args, se):
    ret = {}
    for file in glob.glob(f"{args.INP_DIR}/{se}.*.pt"):
        file_dict = torch.load(file)
        ret.update(file_dict)
    return ret

def main(args):

    CORPORA = ['train', 'val', 'test'] if not args.prep_test else ['test']

    is_test = False
    for se in CORPORA:
        if se == 'test':
            is_test = True

        json_ents_dict = {}
        with open(f"/disk1/sajad/datasets/sci/mup/{se}_complete.json") as fR:
            for l in fR:
                ent = json.loads(l)
                if ent['paper_id'] not in json_ents_dict.keys():
                    json_ents_dict[ent['paper_id']] = {}
                    source = ent['paper']['abstractText']
                    for sect_text in ent['paper']['sections']:
                        source += ' '
                        source += sect_text['text']
                    source = source.strip()

                    json_ents_dict[ent['paper_id']]['source'] = source
                    json_ents_dict[ent['paper_id']]['summary'] = [ent['summary']]

                else:
                    json_ents_dict[ent['paper_id']]['summary'].append(ent['summary'])


        topic_info_dict = load_topic_info(args, se)
        hf_df = {
            'paper_id': [],
            'source': [],
            'section_headings': [],
            'source_tokenized': [],
            'ext_labels': [],
            'section_scores': [],
            'summary': [],
            'topic_info_global': [],
            'topic_info_section': [],
        }

        count = 0
        for paper_id, paper_ent in json_ents_dict.items():

            hf_df['paper_id'].append(paper_id)

            hf_df['source'].append(topic_info_dict[paper_id]['sections_sents'])
            hf_df['section_headings'].append(topic_info_dict[paper_id]['section_headings'])

            # hf_df['source_tokenized'].append(0)
            hf_df['summary'].append(paper_ent['summary'] if paper_ent['summary'][0] is not None else ['This is official test set...!!'])
            hf_df['ext_labels'].append(topic_info_dict[paper_id]['ext_labels'])
            hf_df['topic_info_section'].append(json.dumps(topic_info_dict[paper_id]['topic_info_section']))
            hf_df['topic_info_global'].append(json.dumps(topic_info_dict[paper_id]['topic_info_global']))
            count += 1
            # if count == 12:
            #     break

        print('Calculating section scores...')
        pool = Pool(16)

        section_scores_lst = [0 for _ in range(len(hf_df['paper_id']))]
        paper_sect_tkns_lst = [0 for _ in range(len(hf_df['paper_id']))]
        mp_instances = [(p_id, src, summaries, ext_labels, is_tst) for p_id, src, summaries, ext_labels, is_tst in zip(hf_df['paper_id'], hf_df['source'], hf_df['summary'], hf_df['ext_labels'], [is_test] * len(hf_df['summary']))]
        # for m in mp_instances:
        #     _cal_rg_sect(m)
        paper_ids_indices = hf_df['paper_id'].copy()

        for ret in tqdm(pool.imap_unordered(_cal_rg_sect, mp_instances), total=len(mp_instances)):
            p_id = ret[0]
            section_scores = ret[1]
            src_tokenized = ret[2]
            # if section_scores.shape[1] > 1:
            #     import pdb;pdb.set_trace()
            paper_idx = paper_ids_indices.index(p_id)
            section_scores_lst[paper_idx] = section_scores.tolist()
            paper_sect_tkns_lst[paper_idx] = src_tokenized

        hf_df['section_scores'] = section_scores_lst
        hf_df['source_tokenized'] = paper_sect_tkns_lst
        # hf_df.pop('source_tokenized')

        print('Writing HF files...')

        try:
            os.makedirs(f'{args.WR_DIR}')
        except:
            pass

        import pandas as pd
        df = pd.DataFrame(hf_df)
        df.to_parquet(f"{args.WR_DIR}/{args.filename if len(args.filename) > 0 else se + '.parquet'}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep_test", action='store_true')
    parser.add_argument("--INP_DIR", required=True, type=str)
    parser.add_argument("--WR_DIR", required=True, type=str)
    parser.add_argument("--filename", default='', type=str)
    args = parser.parse_args()
    main(args)

