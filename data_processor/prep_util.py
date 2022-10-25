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


def generate_bow(batch_size, self, data, idf_info):
    vocab_size = idf_info["voc_size"]
    all_bow = torch.zeros([batch_size, vocab_size], dtype=torch.float)
    customer_bow = torch.zeros([batch_size, vocab_size], dtype=torch.float)
    agent_bow = torch.zeros([batch_size, vocab_size], dtype=torch.float)

    all_file_counter = idf_info["all"]
    # customer_file_counter = idf_info['customer']
    # agent_file_counter = idf_info['agent']
    file_num = idf_info["num"]

    for idx in range(self.batch_size):
        all_counter = data[idx][8]["all"]
        # customer_counter = data[idx][8]["customer"]
        # agent_counter = data[idx][8]["agent"]

        all_counter_sum = sum(all_counter.values())
        for key, value in all_counter.items():
            all_tf = value / all_counter_sum
            all_file_count = all_file_counter[key]
            if self.args.use_idf:
                all_idf = math.log(file_num / (all_file_count + 1.))
            else:
                all_idf = 0. if all_file_count > self.args.max_word_count or \
                                all_file_count < self.args.min_word_count else 1.
            all_bow[idx][key] = all_tf * all_idf

        # customer_counter_sum = sum(customer_counter.values())
        # for key, value in customer_counter.items():
        #     customer_tf = value / customer_counter_sum
        #     customer_file_count = customer_file_counter[key]
        #     if self.args.use_idf:
        #         customer_idf = math.log(file_num / (customer_file_count + 1.))
        #     else:
        #         customer_idf = 0. if customer_file_count > self.args.max_word_count or \
        #                              customer_file_count < self.args.min_word_count else 1.
        #     customer_bow[idx][key] = customer_tf * customer_idf
        #
        # agent_counter_sum = sum(agent_counter.values())
        # for key, value in agent_counter.items():
        #     agent_tf = value / agent_counter_sum
        #     agent_file_count = agent_file_counter[key]
        #     if self.args.use_idf:
        #         agent_idf = math.log(file_num / (agent_file_count + 1.))
        #     else:
        #         agent_idf = 0. if agent_file_count > self.args.max_word_count or \
        #                           agent_file_count < self.args.min_word_count else 1.
        #     agent_bow[idx][key] = agent_tf * agent_idf

    return all_bow, customer_bow, agent_bow


def generate_summ_bow(batch_size, data, idf_info):

        vocab_size = idf_info["voc_size"]
        all_bow = torch.zeros([batch_size, vocab_size], dtype=torch.float)
        customer_bow = torch.zeros([batch_size, vocab_size], dtype=torch.float)
        agent_bow = torch.zeros([batch_size, vocab_size], dtype=torch.float)

        all_file_counter = idf_info["all"]
        customer_file_counter = idf_info['customer']
        agent_file_counter = idf_info['agent']

        for idx in range(batch_size):
            all_counter = data[idx][9]["all"]
            customer_counter = data[idx][9]["customer"]
            agent_counter = data[idx][9]["agent"]

            for key in all_counter.keys():
                all_file_count = all_file_counter[key]
                if not args.use_idf:
                    if all_file_count > args.max_word_count or \
                      all_file_count < args.min_word_count:
                        continue
                all_bow[idx][key] = 1

            for key in customer_counter.keys():
                customer_file_count = customer_file_counter[key]
                if not self.args.use_idf:
                    if customer_file_count > self.args.max_word_count or \
                      customer_file_count < self.args.min_word_count:
                        continue
                customer_bow[idx][key] = 1

            for key in agent_counter.keys():
                agent_file_count = agent_file_counter[key]
                if not self.args.use_idf:
                    if agent_file_count > self.args.max_word_count or \
                      agent_file_count < self.args.min_word_count:
                        continue
                agent_bow[idx][key] = 1

        return all_bow, customer_bow, agent_bow