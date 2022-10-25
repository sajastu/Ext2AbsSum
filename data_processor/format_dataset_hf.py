import glob
import json
import os
from multiprocessing import Pool

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


def load_topic_info(se):
    ret = {}
    for file in glob.glob(f"/disk1/sajad/datasets/sci/mup/bert_data/{se}.*.pt"):
        file_dict = torch.load(file)
        ret.update(file_dict)
    return ret

if __name__ == '__main__':

    for se in ['train', 'val']:
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


        topic_info_dict = load_topic_info(se)
        hf_format = []
        hf_df = {
            'paper_id': [],
            'source': [],
            'summary': [],
            'topic_info_global': [],
            'topic_info_section': [],
        }

        for paper_id, paper_ent in json_ents_dict.items():

            hf_df['paper_id'].append(paper_id)
            if len(topic_info_dict[paper_id]['section_text']) != len((topic_info_dict[paper_id]['topic_info_section'])):
                import pdb;pdb.set_trace()
            hf_df['source'].append(topic_info_dict[paper_id]['section_text'])
            hf_df['summary'].append(paper_ent['summary'])
            # import pdb;pdb.set_trace()
            hf_df['topic_info_section'].append(json.dumps(topic_info_dict[paper_id]['topic_info_section']))
            hf_df['topic_info_global'].append(json.dumps(topic_info_dict[paper_id]['topic_info_global']))

        print('Writing HF files...')

        try:
            os.makedirs('/disk1/sajad/datasets/sci/mup/hf_format/')
        except:
            pass

        try:
            os.makedirs(f'/disk1/sajad/datasets/sci/mup/single_files/{se}')
        except:
            pass

        import pandas as pd
        df = pd.DataFrame(hf_df)
        df.to_parquet(f"/disk1/sajad/datasets/sci/mup/hf_format/{se}.parquet")