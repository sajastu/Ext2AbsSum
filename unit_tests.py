import glob
import random

from data_processor.data_builder import _format_to_bert
import pandas as pd

from data_processor.others.vocab_wrapper import VocabWrapper


def test_data_builder_instance(instance_path):
    heading_keywords = pd.read_csv('data_processor/heading_keyword.csv')
    vr = VocabWrapper('word2vec')
    vr.load_emb("/disk1/sajad/w2v_embeds/w2v_mup_reduced.emb")
    _format_to_bert(
        params=[
            instance_path,
            '', 'NA'],
        file_counter_global=None,
        file_counter_section=None,
        voc_wrapper= vr,
        heading_keywords=heading_keywords,
        debug=True,
    )

if __name__ == '__main__':
    # debuging
    files = glob.glob('/disk1/sajad/datasets/sci/mup/single_tokenized_final2/train/*.json')
    test_data_builder_instance(instance_path=files[random.randint(0, len(files)-1)])

