# compute and save the counterfactual importance weights on every edge of all nb301 training architectures.
import sys
sys.path.append('../')
sys.path.append('../autodl')
sys.path.append('../autodl/lib/')

import argparse
import numpy as np
import pickle
from search_spaces.nas201 import NASBench201
from datetime import datetime
import os
from copy import deepcopy
import multiprocessing
from  tqdm import tqdm

parser = argparse.ArgumentParser('Extract motif')
# something like NATS-tss-v1_0-3ffb9-simple
parser.add_argument('--nb_path', required=True)
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120'])
parser.add_argument('--sample_fraction', default=0.05, type=float)
parser.add_argument('-us', '--uniform_sample', action='store_true')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")


# compute top 5%, bottom 5% and randomly sampled 10% from the mediocre architectures in-between.
def evaluate_single(idx):
    """Evaluate a single architecture"""
    print(f'Evaluating {idx}')
    arch = archs[idx]
    edge_graph = ss.to_networkx_edge(idx, compute_importance_weight=True, )
    arch['edge_graph'] = edge_graph
    arch['arch_str'] = ss.to_str(idx)
    return arch


print(f'Starting experiment at {date_time}')

ss = NASBench201(file_path=args.nb_path, dataset=args.dataset)
all_arch_indices = list(range(15625))     # total number of architectures contained in NB201 search space.


def _query_wrap(idx):
    return ss.query(idx, with_noise=False)

try:
    sorted_arch_idx = pickle.load(open(f'../data/nb201_{args.dataset}_index_order.pickle', 'rb'))
except:
    pool0 = multiprocessing.Pool()
    all_perfs = pool0.map(_query_wrap, all_arch_indices)
    pool0.close()

    sorted_arch_idx = np.argsort(all_perfs)
    pickle.dump(sorted_arch_idx, open(f'../data/nb201_{args.dataset}_index_order.pickle', 'wb'))

archs = {i: {} for i in all_arch_indices}
n_archs = len(archs)
print(f'Total number of architectures contained: {n_archs}')

if args.uniform_sample:
    # use multiprocessing to speed up all evals
    arch_idx = np.random.RandomState(args.seed).choice(n_archs, 1000, replace=False).tolist()
    res = []
    for idx in tqdm(arch_idx):
        res.append(evaluate_single(idx))
    pickle.dump(res, open(f'../data/nb201_{args.dataset}_uniform_sampled_arch_info_with_importance.pickle', 'wb'))
    print('Done')

else:
    n_sample = np.round(args.sample_fraction * n_archs).astype(np.int)
    top_arch_idx = sorted_arch_idx[:n_sample].tolist()
    bottom_arch_idx = sorted_arch_idx[-n_sample:].tolist()
    middle_arch_idx = np.random.RandomState(args.seed).choice(sorted_arch_idx[n_sample:-n_sample], 2 * n_sample, replace=False).tolist()
    print('Finished sorting. Starting Top archs')

    res = []
    for idx in tqdm(top_arch_idx):
        res.append(evaluate_single(idx))
    pickle.dump(res, open(f'../data/nb201_{args.dataset}_top_arch_info_with_importance.pickle', 'wb'))

    for idx in tqdm(bottom_arch_idx):
        res.append(evaluate_single(idx))
    pickle.dump(res, open(f'../data/nb201_{args.dataset}_bottom_arch_info_with_importance.pickle', 'wb'))
