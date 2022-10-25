import argparse
import os
import json
import glob
from multiprocessing import Pool
from os.path import join as pjoin

from tqdm import tqdm

from data_processor.others.vocab_wrapper import VocabWrapper
import spacy

from data_processor.preprocess_text import WhiteSpacePreprocessingStopwords

nlp = spacy.load('en_core_sci_lg')
import nltk
from nltk.corpus import stopwords
sws = stopwords.words('english')

def get_sentence_tokens(text):
    doc_nlp = nlp(text)
    ret = []
    for doc_sentence in doc_nlp.sents:
        sent = []
        for tkn in doc_sentence:
            sent.append(tkn.text)
            ret.append(tkn.text)
        # ret.append(sent)
    return ret



def train_emb(args):
    data_dir = os.path.abspath(args.data_path)
    print("Preparing to process %s ..." % data_dir)

    ex_num = 0
    vocab_wrapper = VocabWrapper(args.mode, args.emb_size)
    vocab_wrapper.init_model()

    file_ex = []
    instances = []

    for corpus_type in ['train', 'val']:
        for json_f in glob.glob(f'{data_dir}/' + f'/{corpus_type}/' + '*.json'):
            instances.append(json.load(open(json_f)))


    print(f'All instances: {len(instances)}')
    documents = []
    for ins in instances:
        pr_instances = []
        for section in ins['sections_txt_tokenized']:
            pr_instances.append(section['text'])
        documents.extend(pr_instances)

        # for summary in ins['summaries']:
            # summ_sent_tokens = get_sentence_tokens(summary)
            # for summ_sent_token in summ_sent_tokens:
            #     pr_instances.append(summary)

        # documents.append(' '.join(pr_instances))

    sp = WhiteSpacePreprocessingStopwords(documents=documents, vocabulary_size=3000, min_words=2, stopwords_list=sws)
    preprocessed_docs, unpreprocessed_docs, vocabulary, retained_indices = sp.preprocess()
    pool = Pool(17)

    for tokens in tqdm(pool.imap_unordered(get_sentence_tokens, preprocessed_docs), total=len(preprocessed_docs)):
        file_ex.append(tokens)

    print("Training embeddings...")
    vocab_wrapper.train(file_ex)
    vocab_wrapper.report()
    print("Datasets size: %d" % ex_num)
    vocab_wrapper.save_emb(args.emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='word2vec', type=str, choices=['glove', 'word2vec'])
    parser.add_argument("-data_path", default="/disk1/sajad/datasets/sci/mup/single_tokenized_final2/", type=str)
    parser.add_argument("-emb_size", default=100, type=int)
    parser.add_argument("-emb_path", default="/disk1/sajad/w2v_embeds/w2v_mup_reduced.emb", type=str)

    args = parser.parse_args()

    train_emb(args)
