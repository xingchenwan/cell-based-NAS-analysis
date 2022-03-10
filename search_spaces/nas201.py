from .search_space import SearchSpace
import random
import networkx as nx
import matplotlib.pyplot as plt
import os
from nats_bench import create
import numpy as np
import logging
from copy import deepcopy

PRIMITIVES = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
OP_SPOTS = 6
NUM_OPS = len(PRIMITIVES)
OPS_dict = {i: op for i, op in enumerate(PRIMITIVES)}


class NASBench201(SearchSpace):
    _name = 'nasbench201'

    def __init__(self,
                 file_path,
                 dataset='cifar10',
                 device='cpu',
                 log_scale=False, negative=False):
        assert dataset in ['cifar10', 'cifar100', 'ImageNet16-120']
        super(NASBench201, self).__init__(file_path, dataset, device)
        self.dataset = dataset
        self.log_scale, self.negative = log_scale, negative

        self.api = create(file_path, 'tss', fast_mode=True, verbose=False)
        self.available_seeds = [777, 888, 999]
        self.ops = PRIMITIVES
        # store the optimal values
        self.optimals = {
            'cifar10': 1. - 0.9437,
            'cifar100': 1. - 0.7351,
            'ImageNet16-120': 1. - 0.4731
        }
        if self.log_scale:
            self.optimals = {k: np.log(v) for k, v in self.optimals.items()}
        if self.negative:
            self.optimals = {k: -v for k, v in self.optimals.items()}

    def query(self, arch, with_noise=True, seed=None, return_val=False):
        """Given a arch_str, return its a dictionary of results including accuracy, loss, flops, nparams and its trained
            weights loaded into a pytorch model.
        arch: an architecture str in the format of
            |nor_conv_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2| or the index
        seed: seed for the weight. If not supplied and the architecture specified is trained with multiple seeds, randomly
            select a set of weights from them.
        full_result: if True, return full set of info available
        return_val: return validation statistics in lieu of test.
        """
        query_key = 'valid' if return_val else 'test'

        if isinstance(arch, str):  arch_idx = self.api.query_index_by_arch(arch)
        else:  arch_idx = int(arch)

        # if seed is not None:
        #     if seed < len(self.available_seeds):
        #         seed = self.available_seeds[seed]
        # else:
        #     seed = int(np.random.choice(self.available_seeds))
        #
        def _single_query(arch_idx):
            # get a dictionary of accuracy, loss and etc.
            acc_results = self.api.get_more_info(arch_idx, self.dataset, hp='200', )
            acc_results[f'{query_key}-error'] = (100. - acc_results[f'{query_key}-accuracy']) / 100.
            if self.log_scale:
                acc_results[f'{query_key}-error'] = np.log(acc_results[f'{query_key}-error'])
            if self.negative:
                acc_results[f'{query_key}-error'] = -acc_results[f'{query_key}-error']
            return acc_results[f'{query_key}-error']

        if with_noise:
            return _single_query(arch_idx)
        patience = 25
        all_accs = []
        while patience > 0:
            acc = _single_query(arch_idx)
            if acc not in all_accs:
                all_accs.append(acc)
            else:
                patience -= 1
        return np.mean(all_accs)

    def get_random_arch(self, return_string=True, return_index=False):
        """Generate a random architecture in the NAS-Bench-201 search space in terms of architecture string"""
        assert return_index or return_string, "both return_string and return_index are False!"
        ops = []
        for i in range(OP_SPOTS):
            op = random.choice(PRIMITIVES)
            ops.append(op)
        string = get_string_from_ops(ops)
        index = self.api.query_index_by_arch(string)

        if return_string and not return_index:
            return string
        if return_index and not return_string:
            return index
        return string, index

    def sample_from_anasod(self, anasod_encoding, as_probability=True, return_string=True, return_index=False,):
        """Sample architecture cells from the ANASOD encoding. This can be achieved by randomly wire the connections
        amongst the operations
        unique_arch_list: if True, only return a string that is found in the unique arch list (some archs are isomorphic)
        """
        assert len(anasod_encoding) == NUM_OPS
        if as_probability:
            assert np.isclose(np.sum(anasod_encoding), 1.)
            x = np.round(anasod_encoding * OP_SPOTS).astype(int)
        else:
            x = np.round(anasod_encoding).astype(int)
        # Generate the ops from x
        ops = []
        for idx, element in enumerate(x):
            ops += [PRIMITIVES[idx]] * element
        # Randomly shuffle the order
        # if seed is None:
        np.random.shuffle(ops)
        # else:
        #     np.random.RandomState(seed).shuffle(ops)
        # print(ops)
        arch_query_string = f'|{ops[0]}~0|+' \
                            f'|{ops[1]}~0|{ops[2]}~1|+' \
                            f'|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|'
        index = self.api.query_index_by_arch(arch_query_string)

        if return_string and not return_index:
            return arch_query_string
        if return_index and not return_string:
            return index
        return arch_query_string, index

    def get_encoding(self, arch, encoding='adj'):
        """Get encoding of a NAS-Bench-201 cell. Possible choices: adjacency encoding or the ANASOD encoding"""
        assert encoding in ['adj', 'anasod']
        if not isinstance(arch, str):
            arch = self._index2arch(int(arch))
        if encoding == 'adj':
            ops = get_op_list(arch)
            encoding = []
            for op in ops:
                encoding.append(PRIMITIVES.index(op))
        elif encoding == 'anasod':
            op_list = get_op_list(arch)
            encoding = anasod_from_ops(op_list)
        return encoding

    def get_neighbours(self, arch, return_string=True, return_index=False, allowed_arch_list=None):
        """
        Get neighbours of an architecture, which is defined as the architectures with edit distance = 1 from the current
        architecture.

        allowed_arch_list: if specified, the neighbour will only be counted if it is also withint the allowed_set. Note that
            if valid_arch_list is specified, the allowed_set will be the intersection of valid_arch_list and itself.
            i.e. an architecture will only be counted if it is both within the valid_arch_list AND the allowed_set
        """
        if not isinstance(arch, str): arch = self._index2arch(int(arch))

        nbhd = []
        nbhd_idx = []
        ops = get_op_list(arch)

        allowed_archs = None

        for i in range(len(ops)):
            available = [op for op in PRIMITIVES if op != ops[i]]
            for op in available:
                new_ops = ops.copy()
                new_ops[i] = op
                arch_str = get_string_from_ops(new_ops)
                arch_idx = self.to_index(arch_str)
                if (allowed_archs is None) or (arch_idx in allowed_archs):
                    nbhd.append(arch_str)
                    nbhd_idx.append(arch_idx)
        if return_string and not return_index:
            return nbhd
        if return_index and not return_string:
            return nbhd_idx
        return nbhd, nbhd_idx

    def mutate(self, arch_str,
               mutation_rate=1.0,
               return_string=True,
               return_index=False):
        """Get a mutated architecture string different from the supplied arch_str"""
        ops = get_op_list(arch_str)
        mutation_prob = mutation_rate / (OP_SPOTS - 2)
        new_ops = []
        for i, op in enumerate(ops):
            if random.random() < mutation_prob:
                available = [o for o in PRIMITIVES if o != op]
                new_ops.append(random.choice(available))
            else:
                new_ops.append(op)

        arch_query_string = get_string_from_ops(new_ops)
        index = self.api.query_index_by_arch(arch_query_string)

        if return_string and not return_index:
            return arch_query_string
        if return_index and not return_string:
            return index
        return arch_query_string, index

    def sample_random_subgraph(self, arch, n_edges: int, numeric_features=False):
        """See documentation in the def with the same function in ../nas301.
        Note that in NB201 there is no difference between normal and reduce cells."""
        edge_graph = deepcopy(self.to_networkx_edge(arch, numeric_features=numeric_features, compute_importance_weight=False))
        all_edges = list(edge_graph.edges())
        edges_idx_to_retain = np.random.choice(len(all_edges), size=n_edges, replace=False)
        edges_to_remove = [edge for i, edge in enumerate(all_edges) if i not in edges_idx_to_retain]
        edge_graph.remove_edges_from(edges_to_remove)
        edge_graph.remove_nodes_from(list(nx.isolates(edge_graph)))
        return edge_graph

    def to_networkx_edge(self, arch, numeric_features=False, with_more_info=True, identify_ops=False, compute_importance_weight=False):
        """See documentation of the function with the same name in ../nas301.py"""

        def get_edge_importance(arch, graph):
            sensitivities = np.array([self.get_edge_perturbation(arch, o) for o in range(OP_SPOTS)]).flatten()

            for ei, eo, data in graph.edges(data=True):
                if 'edge_order' in data.keys():
                    graph.edges()[(ei, eo)]['weight'] = sensitivities[data['edge_order']]
            return graph

        if not isinstance(arch, str):
            arch = int(arch)
            arch = self._index2arch(arch)
        ops = get_op_list(arch)
        G = nx.DiGraph()
        n_nodes = 4
        G.add_nodes_from(range(n_nodes))
        # add node tags
        if identify_ops:
            for i in range(len(G)):
                G.nodes[i]['node_name'] = i
        else:
            G.nodes[0]['node_name'] = 0 if numeric_features else 'input'
            G.nodes[3]['node_name'] = 2 if numeric_features else 'output'
            G.nodes[1]['node_name'] = G.nodes[2]['node_name'] = 1 if numeric_features else 'op'

        edge_list = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]   # 6 operations
        for i, op in enumerate(ops):
            edge_to_add = edge_list[i]
            if numeric_features: G.add_edge(edge_to_add[0], edge_to_add[1], op_name=PRIMITIVES.index(op), edge_order=i)
            else: G.add_edge(edge_to_add[0], edge_to_add[1], op_name=op, edge_order=i)

        if compute_importance_weight:
            G = get_edge_importance(arch, G)

        return G

    def get_node_proximate_archs(self, arch, op_idx, ):
        """A NAS-Bench-201 cell has 6 operations. Given an op_idx \in [0, 6), this enumerates all the architectures
        induced by changing that edge by changing the wiring.
        Return a list of architecture strings in the NB201/NATS-Bench format"""
        assert 0 <= op_idx < OP_SPOTS
        proximate_archs = []
        if not isinstance(arch, str):
            arch = int(arch)
            arch = self._index2arch(arch)
        ops = get_op_list(arch)
        op = ops[op_idx]
        for option in PRIMITIVES:
            if option == op: continue
            candidate_op_list = deepcopy(ops)
            candidate_op_list[op_idx] = option
            candidate_str = get_string_from_ops(candidate_op_list)
            proximate_archs.append(candidate_str)
        return proximate_archs

    def get_edge_perturbation(self, arch, op_idx, ):
        """This computes the counterfactual change in NB201 prediction if op_idx edge is changed with another op"""
        if not isinstance(arch, str):
            arch = int(arch)
            arch = self._index2arch(arch)
        candidates = self.get_node_proximate_archs(arch, op_idx)
        ori_perf = self.query(arch, with_noise=False, )
        perf = np.array([self.query(c, with_noise=False) for c in candidates])
        perf = perf - ori_perf
        return np.mean(perf)

    def to_networkx(self, arch, node_attr='op_name'):
        """Convert the string or an architecture index into a NetworkX graph. Taken from the codebase of
        https://github.com/xingchenwan/nasbowl.

        This creates a node-attributed graphs with nodes being the operations and edges being the flow of information
        """
        if not isinstance(arch, str):
            arch = int(arch)
            arch = self._index2arch(arch)
        op_node_labelling = get_op_list(arch)
        assert len(op_node_labelling) == 6
        # the graph has 8 nodes (6 operation nodes + input + output)
        G = nx.DiGraph()
        edge_list = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7)]
        G.add_edges_from(edge_list)

        # assign node attributes and collate the information for nodes to be removed
        # (i.e. nodes with 'skip_connect' or 'none' label)
        node_labelling = ['input'] + op_node_labelling + ['output']
        nodes_to_remove_list = []
        remove_nodes_list = []
        edges_to_add_list = []
        for i, n in enumerate(node_labelling):
            G.nodes[i][node_attr] = n
            if n == 'none' or n == 'skip_connect':
                input_nodes = [edge[0] for edge in G.in_edges(i)]
                output_nodes = [edge[1] for edge in G.out_edges(i)]
                nodes_to_remove_info = {'id': i, 'input_nodes': input_nodes, 'output_nodes': output_nodes}
                nodes_to_remove_list.append(nodes_to_remove_info)
                remove_nodes_list.append(i)

                if n == 'skip_connect':
                    for n_i in input_nodes:
                        edges_to_add = [(n_i, n_o) for n_o in output_nodes]
                        edges_to_add_list += edges_to_add

        # reconnect edges for removed nodes with 'skip_connect'
        G.add_edges_from(edges_to_add_list)
        # remove nodes with 'skip_connect' or 'none' label
        G.remove_nodes_from(remove_nodes_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 7 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        G.name = arch
        return G

    def _index2arch(self, arch_idx: int):
        """Query the architecture string given an index"""
        return self.api[arch_idx]

    def to_index(self, arch):
        """Convert arch into index representations. The arch can either be the index already, or the architecture
        string """
        if isinstance(arch, int): return arch
        elif isinstance(arch, str): return self.api.query_index_by_arch(arch)
        else: raise TypeError(f'arch type {type(arch)} is not valid in search space {self._name}!')

    def to_str(self, arch):
        """Similar to above, but convert everything into arch string representation."""
        if isinstance(arch, int): return self._index2arch(arch)
        elif isinstance(arch, str): return arch
        else: raise TypeError(f'arch type {type(arch)} is not valid in search space {self._name}!')

    def adj_distance(self, arch1, arch2):
        if not isinstance(arch1, str):
            arch1 = self.to_str(arch1)
        if not isinstance(arch2, str):
            arch2 = self.to_str(arch2)
        ops1 = get_op_list(arch1)
        ops2 = get_op_list(arch2)
        return np.sum([1 for i in range(len(ops1)) if ops1[i] != ops2[i]])

    def nasbot_distance(self, arch1, arch2):
        if not isinstance(arch1, str):
            arch1 = self.to_str(arch1)
        if not isinstance(arch2, str):
            arch2 = self.to_str(arch2)

        ops1 = get_op_list(arch1)
        ops2 = get_op_list(arch2)

        cell_1_counts = [ops1.count(op) for op in PRIMITIVES]
        cell_2_counts = [ops2.count(op) for op in PRIMITIVES]
        ops_dist = np.sum(np.abs(np.subtract(cell_1_counts, cell_2_counts)))
        return ops_dist + self.adj_distance(arch1, arch2)


def get_string_from_ops(ops):
    # given a list of operations, get the string
    strings = ['|']
    nodes = [0, 0, 1, 0, 1, 2]
    for i, op in enumerate(ops):
        strings.append(op + '~{}|'.format(nodes[i]))
        if i < len(nodes) - 1 and nodes[i + 1] == 0:
            strings.append('+|')
    return ''.join(strings)


def get_op_list(arch_str):
    # given a string, get the list of operations
    tokens = arch_str.split('|')
    ops = [t.split('~')[0] for i, t in enumerate(tokens) if i not in [0, 2, 5, 9]]
    return ops


def anasod_from_ops(ops):
    """Compute the ANASOD encoding from the operations"""
    op_type, count = np.unique(ops, return_counts=True)
    op_occurrences = dict(
        zip(op_type.tolist(), count))
    res = np.zeros(len(PRIMITIVES))
    for k, v in op_occurrences.items():
        try:
            idx = PRIMITIVES.index(k)
            res[idx] = v
        except ValueError:
            continue
    return np.array(res)

