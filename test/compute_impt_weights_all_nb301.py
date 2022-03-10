# compute and save the counterfactual importance weights on every edge of all nb301 training architectures.
import sys
sys.path.append('../')
import argparse
import numpy as np
import pickle
from search_spaces.nas301 import NASBench301
from datetime import datetime
from copy import deepcopy

parser = argparse.ArgumentParser('Extract motif')
parser.add_argument('-dp', '--data_path', default='../data/nb301_evaluated_arch_info.pickle')
parser.add_argument('--nb_path', help='the path to the NASBench301 data', required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('-us', '--uniform_sample', action='store_true')
args = parser.parse_args()
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")


# compute top 5%, bottom 5% and randomly sampled 10% from the mediocre architectures in-between.
def evaluate_perf(arch_idx, file_name: str):
    archs_to_save = []
    for i, idx in enumerate(arch_idx):
        if i % 10 == 0:
            print(f'Progress: {i+1} / {len(arch_idx)}')
        arch = archs[idx]
        genotype = arch['genotype']  # extract the genotype
        edge_graph_normal, edge_graph_reduce = ss.to_networkx_edge(genotype, compute_importance_weight=True, no_concat=True)
        arch['edge_graph_normal'] = edge_graph_normal
        arch['edge_graph_reduce'] = edge_graph_reduce
        archs_to_save.append(deepcopy(arch))
    pickle.dump(archs_to_save, open(f'../data/{file_name}.pickle', 'wb'))


def evaluate_single(idx):
    """Evaluate a single architecture"""
    print(f'Evaluating {idx}')
    arch = archs[idx]
    genotype = arch['genotype']  # extract the genotype
    edge_graph_normal, edge_graph_reduce = ss.to_networkx_edge(genotype, compute_importance_weight=True, no_concat=True)
    arch['edge_graph_normal'] = edge_graph_normal
    arch['edge_graph_reduce'] = edge_graph_reduce
    return arch


print(f'Starting experiment at {date_time}')

ss = NASBench301(file_path=args.nb_path, log_scale=False, negative=False)
archs = pickle.load(open(args.data_path, 'rb'))
n_archs = len(archs)
sorted_arch_idx = sorted(range(len(archs)), key=lambda k: archs[k]['final_metric_score'])
print(f'Total number of architectures contained: {n_archs}')

if args.uniform_sample:
    # use multiprocessing to speed up all evals
    import multiprocessing
    arch_idx = np.random.RandomState(args.seed).choice(n_archs, 2000, replace=False).tolist()
    pool = multiprocessing.Pool()
    res = pool.map(evaluate_single, arch_idx)
    pickle.dump(res, open('../data/nb301_uniform_sampled_arch_info_with_importance.pickle', 'wb'))
    print('Done')

else:
    n_sample = np.round(0.05 * n_archs).astype(np.int)
    top_arch_idx = sorted_arch_idx[:n_sample]
    bottom_arch_idx = sorted_arch_idx[-n_sample:]
    middle_arch_idx = np.random.RandomState(args.seed).choice(sorted_arch_idx[n_sample:-n_sample], 2 * n_sample, replace=False).tolist()

    evaluate_perf(top_arch_idx, 'nb301_top_arch_info_with_importance')
    evaluate_perf(bottom_arch_idx, 'nb301_bottom_arch_info_with_importance')
    evaluate_perf(middle_arch_idx, 'nb301_middle_arch_info_with_importance')
