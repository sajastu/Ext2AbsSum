

"""

(?)
((?) (?))
[space] *
remove sentences with lower than 4 tokens.
remove sentences if no alphabet is used.

-math formula replace:
    based on equal sign (=), (+=).


"""
import argparse
import glob
import json
import os
import re
import uuid
from multiprocessing import Pool

from unidecode import unidecode


s=''
s = unidecode(s.replace('\n', ' ')).replace(' *', '')

# d = enchant.Dict("en_US")
MATH_STRINGS = ['xij', 'xi', 'xj',  'di', 'Phi', 'phi', 'd1', 'dn', 'si', 'X', 'P',
             '...', '..', 'x', 'j', 'i']
COMPANIED_SYMBS = ['(', ')', '|', '+', '-']

GENERAL_STRINGS = [('(?)', ''), ('such as.', '. '), ('such as .', '. ')]

ALL_SYMBS = []
for form in MATH_STRINGS:
    form_vars = [[f'{form} {symb}', f'{symb} {form}', f'{symb}{form}', f'{form}{symb}', f' {form} ', f'({form})']
                 for symb in COMPANIED_SYMBS]
    for form_v in form_vars:
        for form_vv in form_v:
            ALL_SYMBS.append(form_vv)




def general_preprocessing(text):
    for form in GENERAL_STRINGS:
        text = text.replace(form[0], form[1])
    return text

def general_math_preprocessing(text):
    for form_vv in ALL_SYMBS:
        text = text.replace(form_vv, ' ')
    return text


def _preprocess(text):
    new_text_split = remove_related(unidecode(text))
    new_text_split = general_preprocessing(new_text_split)
    return new_text_split

import spacy
from tqdm import tqdm

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


def _mp_process(param):
    ent, WR_DIR, file = param
    new_sect_text = ent['sections_txt_tokenized'].copy()
    for j, section in enumerate(ent['sections_txt_tokenized']):
        preprocessed_txt = _preprocess(section['text'])
        new_sect_text[j]['text'] = preprocessed_txt
        new_sect_text[j]['tokenized'] = get_sentence_tokens(preprocessed_txt)

    ent['sections_txt_tokenized'] = new_sect_text.copy()
    with open(WR_DIR + file.split('/')[-1], mode='w') as fW:
        json.dump(ent, fW)

def main():

    for se in ['train', 'val', 'test']:
        WR_DIR = f'/disk1/sajad/datasets/sci/mup/single_tokenized_my_prep/{se}/'

        try:
            os.makedirs(WR_DIR)
        except:
            pass
        mp_instances = []
        for file in tqdm(glob.glob(f'/disk1/sajad/datasets/sci/mup/single_tokenized/{se}/*'), total=len(glob.glob(f'/disk1/sajad/datasets/sci/mup/single_tokenized/{se}/*'))):
            ent = json.load(open(file))
            mp_instances.append((ent, WR_DIR, file))

        pool = Pool(17)

        for _ in tqdm(pool.imap_unordered(_mp_process, mp_instances), total=len(mp_instances)):
            pass

def pass_math_threshold(sent_tokens):
    count = 0
    count_all = 0
    for tkn in sent_tokens:
        if tkn.count('@xmath') > 1:
            count += 1
            count_all += tkn.count('@xmath')

    # if count > 1:
    #     import pdb;pdb.set_trace()

    if count_all > 4:
        return False

    return count < 3


def main_final(args):
    CORPURA = ['train', 'val', 'test'] if not args.prep_test else ['test']
    for se in CORPURA:
    # for se in ['test']:
        WR_DIR = f'{args.INP_DIR}/{se}/'
        try:
            os.makedirs(WR_DIR)
        except:
            pass

        changed_ents = 0
        changed_sents = 0
        total_sents = 0
        # read allennlp outputs...
        # paper_sent_labels = {}
        # with open(f'/home/sajad/packages/sequential_sentence_classification/mup.long20k.{se}.json') as fR:
        #     for l in fR:
        #         j_ent = json.loads(l)
        #         paper_sent_labels[j_ent['segment_id']] = j_ent['labels']

        for file in tqdm(glob.glob(f'{args.INP_DIR}/{se}/*'),
                     total=len(glob.glob(f'{args.INP_DIR}/{se}/*'))):
            ent = json.load(open(file))
            # sent_labels = paper_sent_labels[ent['paper_id']]
            # if 4 in sent_labels:
            sent_loop = 0
            new_sects_sents = []
            for sect in ent['sections_txt_tokenized']:
                new_sect_lst = []
                new_sect = {}
                for sent in sect['tokenized']:
                    # sent_label = sent_labels[sent_loop]
                    if len([s for s in sent if s.isalpha()]) > 4 and len([s for s in sent if s.isalpha()]) < 120:
                        new_sect_lst.append([s.strip() for s in sent if len(s.strip())> 0])
                    else:
                        changed_sents+=1
                    total_sents+=1
                    sent_loop += 1

                new_sect_txt = ' '.join([' '.join(s).replace('\n', ' ') for s in new_sect_lst]).strip()
                new_sect['text'] = new_sect_txt
                new_sect['tokenized'] = new_sect_lst
                new_sect['heading'] = sect['heading']
                new_sects_sents.append(new_sect)
            ent['sections_txt_tokenized'] = new_sects_sents
            changed_ents += 1

            with open(f'{WR_DIR}/{file.split("/")[-1]}', mode='w') as fW:
                json.dump(ent, fW, indent=2)

        print()
        print(f'Overall, {changed_ents} papers changed ==> {round(changed_ents / changed_ents, 4) * 100}%')
        print()
        print(f'Also, {changed_sents} sentences are removed ==> {round((changed_sents / total_sents), 4) * 100}%')

if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep_test", action='store_true')
    parser.add_argument("--INP_DIR", required=True, type=str)
    args = parser.parse_args()
    main_final(args)
