import os
import copy
import ConfigSpace
import numpy as np
import networkx as nx
from nasbench import api
from nasbench.lib import graph_util
import random
from search_spaces.search_space import SearchSpace
from search_spaces.utils import is_upper_triangular
import logging

MAX_EDGES = 9
VERTICES = 7
PRIMITIVES = [
    'input',
    'conv1x1-bn-relu',
    'conv3x3-bn-relu',
    'maxpool3x3',
    'output'
]


class ModelSpec_Modified:
    """Model specification given adjacency matrix and labeling."""

    def __init__(self, matrix, ops, data_format='channels_last'):
        """Initialize the module spec.

    Args:
      matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.

    Raises:
      ValueError: invalid matrix or ops
    """
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('matrix must be square')
        if shape[0] != len(ops):
            raise ValueError('length of ops must match matrix dimensions')
        if not is_upper_triangular(matrix):
            raise ValueError('matrix must be upper triangular')

        # Both the original and pruned matrices are deep copies of the matrix and
        # ops so any changes to those after initialization are not recognized by the
        # spec.
        self.original_matrix = copy.deepcopy(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(matrix)
        self.ops = copy.deepcopy(ops)
        self.valid_spec = True
        self.data_format = data_format

    def hash_spec(self, canonical_ops):
        """Computes the isomorphism-invariant graph hash of this spec.

    Args:
      canonical_ops: list of operations in the canonical ordering which they
        were assigned (i.e. the order provided in the config['available_ops']).

    Returns:
      MD5 hash of this spec which can be used to query the dataset.
    """
        # Invert the operations back to integer label indices used in graph gen.
        labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
        return graph_util.hash_module(self.matrix, labeling)


class NASBench101(SearchSpace):

    _name = 'nasbench101'

    def __init__(self, file_path, multi_fidelity=False, log_scale=True, negative=True, seed=None, ):
        super(NASBench101, self).__init__(file_path, dataset='cifar10', device='cpu')

        optim = 0.04944576819737756
        if log_scale:
            optim = np.log(optim)
        if negative:
            optim = -optim
        self.optim = optim

        self.seed = seed
        self.multi_fidelity = multi_fidelity
        self.log_scale = log_scale
        if self.multi_fidelity: self.api = api.NASBench(os.path.join(file_path, 'nasbench_full.tfrecord'), seed=0)
        else: self.api = api.NASBench(os.path.join(file_path, 'nasbench_only108.tfrecord'))
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.model_spec_list = []
        self.negative = negative
        # self.optimal_val = 0.04944576819737756  # lowest mean validation error
        # self.y_star_test = 0.056824247042338016  # lowest mean test error

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative
        self.seed = seed

    def _retrieve(self, G, budget, which='eval'):

        #  set random seed for evaluation
        if which == 'eval':
            seed_list = [0, 1, 2]
            if self.seed is None:
                seed = random.choice(seed_list)
            elif self.seed >= 3:
                seed = self.seed
            else:
                seed = seed_list[self.seed]
        else:
            # For testing, there should be no stochasticity
            seed = 3

        # input is a list of graphs [G1,G2, ....]
        if self.multi_fidelity is False:
            assert budget == 108
        # get node label and adjacency matrix
        node_labeling = list(nx.get_node_attributes(G, 'op_name').values())
        adjacency_matrix = np.array(nx.adjacency_matrix(G).todense())

        model_spec = ModelSpec_Modified(adjacency_matrix, node_labeling)
        try:
            # data = self.dataset.query(model_spec, epochs=budget)
            fixed_stat, computed_stat = self.api.get_metrics_from_spec(model_spec)
            data = {}
            data['module_adjacency'] = fixed_stat['module_adjacency']
            data['module_operations'] = fixed_stat['module_operations']
            data['trainable_parameters'] = fixed_stat['trainable_parameters']
            if seed is not None and seed >= 3:
                compute_data_all = computed_stat[budget]
                data['validation_accuracy'] = np.mean([cd['final_validation_accuracy'] for cd in compute_data_all])
                data['test_accuracy'] = np.mean([cd['final_test_accuracy'] for cd in compute_data_all])
                data['training_time'] = np.mean([cd['final_training_time'] for cd in compute_data_all])
            else:
                compute_data = computed_stat[budget][seed]
                data['validation_accuracy'] = compute_data['final_validation_accuracy']
                data['test_accuracy'] = compute_data['final_test_accuracy']
                data['training_time'] = compute_data['final_training_time']

        except api.OutOfDomainError:
            self.record_invalid(1, 1, 0)

            if self.log_scale:
                y_invalid = np.log(1)
            else:
                y_invalid = 1
            return y_invalid

        self.record_valid(data, model_spec)
        if which == 'eval':
            err = 1 - data["validation_accuracy"]
        elif which == 'test':
            err = 1 - data['test_accuracy']
        else:
            raise ValueError("Unknown query parameter: which = " + str(which))
        if self.log_scale:
            y = np.log(err)
        else:
            y = err
        if self.negative:
            y = -y

        cost = {'train_time': data['training_time']}
        return y, cost

    def eval(self, G, budget=108, n_repeat=1, use_banana=False):
        """
        todo: use_bananas has not been updated to return the training details in addition to the validation/test acc.
        """
        # input is a list of graphs [G1,G2, ....]
        if not isinstance(G, nx.Graph):
            G = self.to_networkx(G)
        if use_banana:
            return self.banana_retrieve(G, 'eval')
        if n_repeat == 1:
            return self._retrieve(G, budget, 'eval')
        return np.mean(np.array([self._retrieve(G, budget, 'eval')[0] for _ in range(n_repeat)])), [self._retrieve(G, budget, 'eval')[0] for _ in range(n_repeat)]

    def query(self, arch, budget=108, n_repeat=1, **kwargs):
        return self.eval(arch, budget=budget, n_repeat=n_repeat, **kwargs)

    def test(self, G, budget=108, n_repeat=1, use_banana=False):
        if not isinstance(G, nx.Graph):
            G = self.to_networkx(G)
        if use_banana:
            return self.banana_retrieve(G, 'test')
        return np.mean(np.array([self._retrieve(G, budget, 'test')[0] for _ in range(n_repeat)]))

    def banana_retrieve(self, G, which='eval'):
        patience = 50
        accs = []
        node_labeling = list(nx.get_node_attributes(G, 'op_name').values())
        adjacency_matrix = np.array(nx.adjacency_matrix(G).todense())

        model_spec = api.ModelSpec(adjacency_matrix, node_labeling)
        while len(accs) < 3 and patience > 0:
            patience -= 1
            if which == 'eval':
                acc = self.api.query(model_spec)['validation_accuracy']
            else:
                acc = self.api.query(model_spec)['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        err = round((1 - np.mean(accs)), 4)

        if self.log_scale:
            err = np.log(err)
        if self.negative:
            err = -err
        return err

    def record_invalid(self, valid, test, costs):

        self.y_valid.append(valid)
        self.y_test.append(test)
        self.costs.append(costs)
        self.model_spec_list.append({'adjacency': None, 'node_labels': None})

    def record_valid(self, data, model_spec):

        # record valid adjacency matrix and node labels
        self.model_spec_list.append({'adjacency': model_spec.original_matrix, 'node_labels': model_spec.original_ops})

        # compute mean test error for the final budget
        _, metrics = self.api.get_metrics_from_spec(model_spec)
        mean_test_error = 1 - np.mean([metrics[108][i]["final_test_accuracy"] for i in range(3)])
        self.y_test.append(mean_test_error)

        # compute validation error for the chosen budget
        valid_error = 1 - data["validation_accuracy"]
        self.y_valid.append(valid_error)

        runtime = data["training_time"]
        self.costs.append(runtime)

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.model_spec_list = []

    @staticmethod
    def get_configuration_space():
        # for unpruned graph
        cs = ConfigSpace.ConfigurationSpace()
        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
        return cs

    def to_networkx(self, arch, **kwargs):
        """Convert an architecture to a networkx graph"""
        adj, ops = arch
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        for i, n in enumerate(ops):
            G.nodes[i]['op_name'] = n
        return G

    def to_numpy(self, G, node_label='op_name'):
        """Convert a networkx graph to (adj, feat)"""
        """Convert networkx representation back to adj/feature array representation"""
        # todo: support graphs that are not necessarily node attributed and/or multiple node/edge attributes.
        adj = nx.to_numpy_array(G)
        # feature vector
        x = np.array([data[node_label] for n, data in G.nodes(data=True)])
        return x, adj

    def get_random_arch(self, prune=True, return_networkx=True, return_numpy=False):
        """Generate a random architecture from the search space"""
        patience = 100
        while patience > 0:
            nas101_cs = self.get_configuration_space()
            config = nas101_cs.sample_configuration()
            adjacency_matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
            idx = np.triu_indices(adjacency_matrix.shape[0], k=1)
            for i in range(VERTICES * (VERTICES - 1) // 2):
                row = idx[0][i]
                col = idx[1][i]
                adjacency_matrix[row, col] = config["edge_%d" % i]

            labeling = [config["op_node_%d" % _] for _ in range(5)]
            labeling = ['input'] + list(labeling) + ['output']
            if prune:
                try:
                    pruned_adjacency_matrix, pruned_labeling = self.prune((adjacency_matrix, labeling), return_networkx=False, return_numpy=True)
                except:
                    patience -= 1
                    continue
            else:
                pruned_adjacency_matrix, pruned_labeling = adjacency_matrix, labeling

            # skip only duplicating 2-node architecture
            if len(pruned_labeling) == 2:
                patience -= 1
                continue

            # skip invalid architectures whose number of edges exceed the max limit of 9
            if np.sum(pruned_adjacency_matrix) > MAX_EDGES or np.sum(pruned_adjacency_matrix) == 0:
                patience -= 1
                continue
            G = self.to_networkx((pruned_adjacency_matrix, pruned_labeling))
            if return_networkx and not return_numpy:
                return G
            elif return_numpy and not return_networkx:
                return pruned_adjacency_matrix, pruned_labeling
            else:
                return G, (pruned_adjacency_matrix, pruned_labeling)
        logging.warning('Unable to sample from the search space!')
        return None

    def prune(self, arch, return_networkx=True, return_numpy=False):
        """Prune the extraneous parts of the graph.

        General procedure:
          1) Remove parts of graph not connected to input.
          2) Remove parts of graph not connected to output.
          3) Reorder the vertices so that they are consecutive after steps 1 and 2.

        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        if isinstance(arch, nx.Graph):
            adj, ops = self.to_numpy(arch)
        else:
            adj, ops = arch[0], arch[1]
        num_vertices = np.shape(adj)[0]
        new_matrix = copy.deepcopy(adj)
        new_ops = copy.deepcopy(ops)
        # DFS forward from input
        visited_from_input = {0}
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if adj[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = {num_vertices - 1}
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if adj[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            new_matrix = None
            new_ops = None
            valid_spec = False
            return
        new_matrix = np.delete(new_matrix, list(extraneous), axis=0)
        new_matrix = np.delete(new_matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del new_ops[index]

        G = self.to_networkx((new_matrix, new_ops))
        if return_networkx and not return_numpy: return G
        elif return_numpy and not return_networkx: return new_matrix, new_ops
        else: return G, (new_matrix, new_ops)

    def find_node_approximate_archs(self, op_idx, ops_only=False):
        pass


if __name__ == '__main__':
    import pickle
    output_path = '../data/'
    # with open(os.path.join(output_path, 'valid_arch_samples_pruned'), 'rb') as outfile:
    #     res = pickle.load(outfile)
    #
    # idx = 1
    # A = res['model_graph_specs'][idx]['adjacency']
    # nl = res['model_graph_specs'][idx]['node_labels']
    A = np.array([[0, 1, 1, 0, 0, 1, 1],    # input layer
                  [0, 0, 0, 0, 0, 1, 0],    # 1x1 conv
                  [0, 0, 0, 1, 0, 0, 0],    # 3x3 conv
                  [0, 0, 0, 0, 1, 0, 0],    # 3x3 max-pool
                  [0, 0, 0, 0, 0, 1, 0],    # 3x3 conv
                  [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
                  [0, 0, 0, 0, 0, 0, 0]])   # output layer
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    nl = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
    for i, n in enumerate(nl):
        G.node[i]['op_name'] = n
    nascifar10 = NASBench101(file_path=output_path, seed=4)
    f = nascifar10.eval
    result = f(G)
