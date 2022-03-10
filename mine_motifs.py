import pickle

import networkx as nx

from explain.gSpan.gspan_mining.gspan import gSpan
import argparse
import os
import numpy as np
from search_spaces.nas301 import NASBench301
from search_spaces.nas201 import NASBench201


parser = argparse.ArgumentParser('Motif mining for NB201/NB301/DARTS archs')
parser.add_argument('-ss', '--search_space', default='nb301', choices=['nb301', 'nb201'])
parser.add_argument('--normal_only', action='store_true')
parser.add_argument('--nb_path', default=None, help='data path to the NAS-Bench API')
parser.add_argument('-ms', '--min_support', default=0.05, type=float)
parser.add_argument('-fp', '--file_path', type=str, default='data/nb301_top_arch_info_with_importance.pickle')
parser.add_argument('-sp', '--save_path', type=str, default='./output/')
parser.add_argument('--min_node', type=int, default=2)
parser.add_argument('--max_node', type=int, default=6)
parser.add_argument('-wh', '--weight_threshold', type=float, default=0.001)
parser.add_argument('-c', '--condition', type=str, choices=['geq', 'leq'], default='geq')

args = parser.parse_args()
print(vars(args))

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
data_file_id = args.file_path.split('/')[1].split('.pickle')[0]
identifier = f'gSpan_{data_file_id}_minSupport_{args.min_support}_nodes_{args.min_node}_{args.max_node}_thres_{args.condition}_{args.weight_threshold}'
if args.normal_only:
    identifier += '_normalOnly_'
suffix = '.pickle'
identifier += suffix
file_name = os.path.join(args.save_path, identifier)
stored_data = pickle.load(open(args.file_path, 'rb'))
n_total = len(stored_data)
if args.min_support < 1: args.min_support = np.round(args.min_support * n_total).astype(np.int)
else: args.min_support = np.round(args.min_support).astype(np.int)


def obtain_subgraphs(g: nx.Graph, weight_threshold: float, condition: str = 'geq'):
    """Retain the important subgraphs to explain the surrogate prediction"""
    assert condition in ['geq',
                         'leq']  # the condition for subgraph selection must be greater or equal to, or less or equal to.
    edges_to_remove = []
    for i, (ni, no, data) in enumerate(g.edges(data=True)):
        if (condition == 'geq' and data['weight'] < weight_threshold) or \
                (condition == 'leq' and data['weight'] > weight_threshold):
            edges_to_remove.append((ni, no))
    g.remove_edges_from(edges_to_remove)
    g.remove_nodes_from(list(nx.isolates(g)))
    return g


def run_gspan(g_list):
    gs = gSpan(
        g_list,
        file_type='networkx',
        min_support=args.min_support,
        min_num_vertices=args.min_node,
        max_num_vertices=args.max_node,
        is_undirected=False,  # these graphs are by default directed
    )
    gs.run()
    file_to_save = gs.get_summary()
    return file_to_save


def count_motif_occurrences(motif: nx.Graph, list_of_graphs):
    """Perform subgraph isomorphism test to compute the occurrence of pattern 'motif' in the 'list_of_graphs'.
    This could be potentially time consuming, since subgraph isomorphism is in general NP hard.
    """
    isomorphic_count = 0
    for g in list_of_graphs:
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(g, motif, edge_match=lambda e1, e2: e1['op_name'] == e2['label'],
                                                           node_match=lambda n1, n2: n1['node_name'] == n2['label'])
        if matcher.subgraph_is_isomorphic():
            isomorphic_count += 1
    return isomorphic_count


normal_graphs, reduce_graphs, genotypes = [], [], []
for d in stored_data:
    if args.search_space == 'nb301':
        normal_graphs.append(d['edge_graph_normal'])
        reduce_graphs.append(d['edge_graph_reduce'])
        genotypes.append(d['genotype'])
    elif args.search_space == 'nb201':
        normal_graphs.append(d['edge_graph'])
        genotypes.append(d['arch_str'])

if args.search_space == 'nb301':
    ss = NASBench301(file_path=args.nb_path, log_scale=False, negative=False)   # no actual loading of the datasets required

    impt_normal_subgraphs = [obtain_subgraphs(g, weight_threshold=args.weight_threshold, condition=args.condition) for g in normal_graphs]
    n_edge_impt_normal_subgraphs = [nx.number_of_edges(g) for g in impt_normal_subgraphs]
    if args.normal_only:
        impt_subgraphs = impt_normal_subgraphs
    else:
        impt_reduce_subgraphs = [obtain_subgraphs(g, weight_threshold=args.weight_threshold, condition=args.condition) for g in reduce_graphs]
        impt_subgraphs = impt_normal_subgraphs + impt_reduce_subgraphs
        n_edge_impt_reduce_subgraphs = [nx.number_of_edges(g) for g in impt_reduce_subgraphs]
    # randomly sampled subgraphs as comparison
    rdn_normal_subgraphs, rdn_reduce_subgraphs = [], []
    for i, geno in enumerate(genotypes):
        n, r = ss.sample_random_subgraph(geno, n_edges_n=n_edge_impt_normal_subgraphs[i],
                                         n_edges_r=None if args.normal_only else n_edge_impt_reduce_subgraphs[i])
        rdn_normal_subgraphs.append(n)
        rdn_reduce_subgraphs.append(r)
    if args.normal_only:
        rdn_subgraphs = rdn_normal_subgraphs
    else:
        rdn_subgraphs = rdn_normal_subgraphs + rdn_reduce_subgraphs

elif args.search_space == 'nb201':
    ss = NASBench201(file_path=args.nb_path)
    impt_subgraphs = [obtain_subgraphs(g, weight_threshold=args.weight_threshold, condition=args.condition) for g in normal_graphs]
    n_edge_impt = [nx.number_of_edges(g) for g in impt_subgraphs]
    rdn_subgraphs = []
    for i, arch_str in enumerate(genotypes):
        rdn_subgraphs.append(
            ss.sample_random_subgraph(arch_str, n_edges=n_edge_impt[i])
        )
else:
    raise ValueError()
res = run_gspan(impt_subgraphs)
motifs = res['subgraphs_nx']
# compute the number of occurrences of the impt motifs in reference set
incidence_ref = []
for i, motif in enumerate(motifs):
    if i % 10 == 0:
        print(f'Processing {i+1} / {len(motifs)}')
    n_occur = count_motif_occurrences(motif, rdn_subgraphs)
    incidence_ref.append(n_occur)
res['report_df']['ref_incidence'] = np.array(incidence_ref)

pickle.dump(res, open(file_name, 'wb'))
print('Completed!')


