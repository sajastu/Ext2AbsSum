
"""

    This file contain codes to transform the raw data into the appropriate format
    that Transformers library processes.

"""
import argparse
import collections
import json
from multiprocessing import cpu_count, Pool
import scispacy
import spacy
from tqdm import tqdm
from datasets import load_metric

nlp = spacy.load("en_core_sci_lg")
# metric = load_metric("rouge")
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def _read_docs(file):
    out_docs = {}

    # json file
    with open(file, mode='r') as fR:
        for l in fR:

            doc = json.loads(l.strip())
            if doc['paper_id'] in out_docs.keys():
                out_docs[doc['paper_id']]['summaries'].append(
                    doc['summary']
                )

            else:
                doc['summaries'] = [doc['summary']]
                del doc['summary']
                out_docs[doc['paper_id']] = doc

    return out_docs

def get_sentences(text):
    doc = nlp(text)
    out_sents = []
    for sent in doc.sents:
        if len(sent) > 1:
            out_sents.append(sent.text)
    return out_sents

def _score_sents_core(params):
        doc_id, sects_tokenized_, sum_num, summary_txt = params
        out = {}
        out['doc_id'] = doc_id
        out['sum_num'] = sum_num
        out['section_sent_scores'] = []
        for sect_tokenized in sects_tokenized_:
            # for sent in sect_tokenized:
            out_section_scores = []
            for sent in sect_tokenized:
                # result = metric.compute(predictions=[sent], references=[summary_txt], use_stemmer=True)
                # result = {key: value.mid.fmeasure for key, value in result.items()}
                result = scorer.score(summary_txt, sent)
                out_section_scores.append(result[args.scoring_criterion].fmeasure)
            # Extract a few results from ROUGE
            out['section_sent_scores'].append(out_section_scores)

        return out


def _score_sentences(json_docs, sect_tokenized):

    def sort_sum_nums(out_scores):
        out_scores_tmp = {}
        for doc_id, scores_i in out_scores.items():
            out_scores_tmp[doc_id] = collections.OrderedDict(sorted(scores_i.items()))
        out_scores = out_scores_tmp
        return out_scores

    def add_scores(json_docs, out_scores):
        json_docs_tmp = {}
        for doc_id, doc in json_docs.items():
            doc["section_summary_scores"] = list(out_scores[doc_id].values())[0]
            json_docs_tmp[doc_id] = doc
        json_docs = json_docs_tmp
        return json_docs

    # create summary-tokenized instances...
    summary_tokenized = [(doc_id, sect_tokenized[doc_id], j, doc_sum) for doc_id, doc in json_docs.items() for j, doc_sum in enumerate(doc['summaries'])]

    if args.cpu_nodes > 1:
        out_scores = {}
        pool = Pool(args.cpu_nodes)

        # for s in summary_tokenized:
        #     _score_sents_core(s)

        for out_scored in tqdm(pool.imap_unordered(_score_sents_core, summary_tokenized), total=len(summary_tokenized), desc="Scoring sentences"):
            if out_scored['doc_id'] in out_scores.keys():
                out_scores[out_scored['doc_id']][out_scored['sum_num']] = out_scored['section_sent_scores']
            else:
                out_scores[out_scored['doc_id']] = {}
                out_scores[out_scored['doc_id']][out_scored['sum_num']] = out_scored['section_sent_scores']

        # sort sum_num
        out_scores = sort_sum_nums(out_scores)

        # now add the scores to the global dict...
        json_docs = add_scores(json_docs, out_scores)

        pool.close()
        pool.join()

    return json_docs

def _sent_extractor(doc_src):
    doc_id, sections = doc_src

    out = []
    for sect in sections:
        sents = get_sentences(sect)
        out.append(sents if len(sents)>0 else [''])

    return (doc_id, out)

def _score_sentences_on_summaries(json_docs, args):

    """
    This function will add the scores to each sentence given the summary.
    """

    def _parse_sentences(json_docs):
        docs_src = [(doc_id, [section['text'] for section in doc['paper']['sections']]) for doc_id, doc in json_docs.items()]
        out_tokenized = {}

        if args.cpu_nodes > 1:
            pool = Pool(args.cpu_nodes)
            for out in tqdm(pool.imap_unordered(_sent_extractor, docs_src), total=len(docs_src), desc="Sentence extraction"):
                out_tokenized[out[0]] = out[1]
            pool.close()
            pool.join()

        json_docs_tmp = {}
        for doc_id, doc in json_docs.items():
            doc['sent_tokenized'] = out_tokenized[doc_id]
            json_docs_tmp[doc_id] = doc

        json_docs = json_docs_tmp

        return out_tokenized, json_docs

    # tokenize the docs to output sentences... add up ''_tokenized`` key to the docs
    print()
    print("Extracting sentences...")
    out_tokenized, json_docs = _parse_sentences(json_docs)


    # score the sentences on each summary using the scoring functions
    print()
    print("Scoring sentences...")
    json_docs = _score_sentences(json_docs, out_tokenized)

    return json_docs


def write_json_ouput(json_docs, args):

    with open(args.write_file, mode='w') as fW:

        for doc_id, doc in json_docs.items():
            json.dump(doc, fW)
            fW.write('\n')

"""
    Program runs in here...
"""

def main(args):

    json_docs = _read_docs(args.dataset_file)

    # debug
    # counter = 0
    # json_docs_tmp = {}
    # for p_id, doc in json_docs.items():
    #     json_docs_tmp[p_id] = doc
    #     counter +=1
    #     if counter==10:
    #         break
    # json_docs = json_docs_tmp

    json_docs = _score_sentences_on_summaries(json_docs, args)


    print()
    print('Writing to jsonl file...')
    write_json_ouput(json_docs, args)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Getting params')

    # Defining the arguments
    my_parser.add_argument('--dataset_file',
                           type=str,
                           default="mup/validation_complete.jsonl",
                           help='Dataset\'s file directory'
   )

    my_parser.add_argument('--scoring_criterion',
                           type=str,
                           default="rougeL",
                           help='The scoring function to give sentence importance, you can also design custom scoring functions'
   )

    my_parser.add_argument('--write_file',
                           type=str,
                           default="mup/validation_processed.jsonl",
                           help='The scoring function to give sentence importance, you can also design custom scoring functions'
   )

    my_parser.add_argument('--cpu_nodes',
                           type=int,
                           default=cpu_count(),
                           help='Either to use MP or not... Nodes > 1 will be automatically done in MP setting'

   )

    args = my_parser.parse_args()

    main(args)