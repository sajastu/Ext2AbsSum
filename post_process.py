import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
metrics = ['rouge1', 'rouge2', 'rougeL']
scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

def _cal_rg(param):
    pid, pred, summary = param
    return pid,scorer.score(summary.strip(), pred.strip())


df = pd.read_csv('val_results_100_256_b3_n3.csv')

mp_list = []
for row in df.iterrows():
    paper_id = row[1][0]
    pred = (row[1][1].replace('\n', ' ').replace('of the work of wang et al. (2019)', '')) + '.'
    summaries = json.loads(row[1][2])
    for j, sum in enumerate(summaries):

        summary = sum[f'summary_{j}']
        mp_list.append((paper_id, pred, summary))


pool = Pool(16)

from tqdm import tqdm

paper_scores = {
    'rouge1_f': [],
    'rouge1_r': [],
    'rouge2_f': [],
    'rouge2_r': [],
    'rougeL_f': [],
    'rougeL_r': [],
}
paper_id_sums = {}

for ret in tqdm(pool.imap_unordered(_cal_rg, mp_list), total=len(mp_list)):
    p_id = ret[0]
    scores = ret[1]

    if p_id not in paper_id_sums.keys():
        paper_id_sums[p_id] = {'rouge1_f': [],
            'rouge1_r': [],
            'rouge2_f': [],
            'rouge2_r': [],
            'rougeL_f': [],
            'rougeL_r': [],}

    for metric in paper_scores.keys():
        paper_id_sums[p_id][metric].append(
            scores[metric.split('_')[0]].fmeasure if '_f' in metric else scores[metric.split('_')[0]].recall
        )

# now averaging
for p_id, scores in paper_id_sums.items():
    for metric in paper_scores.keys():
        paper_id_sums[p_id][metric] = np.average(scores[metric])


for metric in paper_scores.keys():
    print(f'{metric}: {np.average([k[metric] for k in paper_id_sums.values()])}')