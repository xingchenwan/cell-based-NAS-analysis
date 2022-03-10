import numpy as np
import scipy.sparse as sp
import torch
import ConfigSpace
import networkx as nx
from utils import encode_onehot


# def encode_onehot(labels, benchmark='nasbench301'):
#     # classes = set(labels)
#     if benchmark == 'nasbench101':
#         classes = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
#     elif benchmark == 'nasbench201':
#         classes = ['input', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none', 'output']
#     elif benchmark in ['nasbench301', 'darts']:
#         classes = ['input0', 'input2', 'max_pool_3x3',
#                    'avg_pool_3x3',
#                    'skip_connect',
#                    'sep_conv_3x3',
#                    'sep_conv_5x5',
#                    'dil_conv_3x3',
#                    'dil_conv_5x5', 'output', 'add']
#     else:
#         classes = set(labels)
#     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
#                     enumerate(classes)}
#     labels_onehot = np.array(list(map(classes_dict.get, labels)),
#                              dtype=np.int32)
#     return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_matrix_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(matrix):
    """Row-normalize sparse matrix"""
    rowsum = np.array(matrix.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    matrix = r_mat_inv.dot(matrix)
    return matrix


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_matrix_to_torch_sparse_tensor(sparse_matrix):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64))
    values = torch.from_numpy(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_global_node(matrix, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if (ifAdj):
        matrix = np.column_stack((matrix, np.ones(matrix.shape[0], dtype=np.float32)))
        matrix = np.row_stack((matrix, np.zeros(matrix.shape[1], dtype=np.float32)))
        np.fill_diagonal(matrix, 1)
        matrix = matrix.T
    else:
        matrix = np.column_stack((matrix, np.zeros(matrix.shape[0], dtype=np.float32)))
        matrix = np.row_stack((matrix, np.zeros(matrix.shape[1], dtype=np.float32)))
        matrix[matrix.shape[0] - 1][matrix.shape[1] - 1] = 1
    return matrix


def zero_one_normalization(X, lower=None, upper=None):
    if lower is None:
        lower = np.min(X, axis=0)
    if upper is None:
        upper = np.max(X, axis=0)

    X_normalized = np.true_divide((X - lower), (upper - lower))

    return X_normalized, lower, upper


def zero_one_denormalization(X_normalized, lower, upper):
    return lower + (upper - lower) * X_normalized


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean


def rmse(pred, target) -> float:
    """Compute the root mean squared error"""
    assert pred.shape[0] == target.shape[0], 'predictant shape ' + \
                                             str(pred.shape[0]) + ' but target shape' + str(target.shape[0])
    n = pred.shape[0]
    return np.sqrt(np.sum((pred - target) ** 2) / n)


def nll(pred, pred_std, target) -> float:
    """Compute the negative log-likelihood (over the validation dataset)"""
    from scipy.stats import norm
    total_nll_origin = - np.mean(norm.logpdf(target, loc=pred, scale=pred_std))
    return total_nll_origin


def spearman(pred, target) -> float:
    """Compute the spearman correlation coefficient between prediction and target"""
    from scipy import stats
    coef_val, p_val = stats.spearmanr(pred, target)
    return coef_val


VERTICES = 7
MAX_EDGES = 9


def config_to_feature_adj(config_list, ignore_invalid=True, benchmark='nasbench301'):
    X_list = []
    adj_list = []
    valid_config_list = []
    for config in config_list:

        if benchmark == 'nasbench301':
            matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
            idx = np.triu_indices(matrix.shape[0], k=1)
            for i in range(VERTICES * (VERTICES - 1) // 2):
                row = idx[0][i]
                col = idx[1][i]
                matrix[row, col] = config["edge_%d" % i]

            labeling = [config["op_node_%d" % i] for i in range(5)]
            node_labels = ['input'] + list(labeling) + ['output']

            # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
            if np.sum(matrix) > MAX_EDGES and ignore_invalid:
                # print('invalid arch')
                continue

        elif benchmark == 'nasbench201':
            op_labeling = [config["edge_%d" % i] for i in range(len(config.keys()))]
            _, matrix, node_labels = create_nasbench201_graph_unpruned(op_labeling)

        onehot_archs_x = encode_onehot(node_labels, search_space=benchmark)
        arch_adj = add_global_node(matrix, ifAdj=True)
        onehot_features = add_global_node(onehot_archs_x, ifAdj=False)
        X_list.append(onehot_features)
        adj_list.append(arch_adj)
        valid_config_list.append(config)

    return X_list, adj_list, valid_config_list


def get_nas101_configuration_space():
    # NAS-CIFAR10 A
    nas101_cs = ConfigSpace.ConfigurationSpace()
    OPS = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']

    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", OPS))
    nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", OPS))
    for i in range(VERTICES * (VERTICES - 1) // 2):
        nas101_cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
    return nas101_cs


def get_nas201_configuration_space():
    # for unpruned graph
    cs = ConfigSpace.ConfigurationSpace()
    ops_choices = ['nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none']
    for i in range(6):
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, ops_choices))
    return cs


# Random to generate new graphs
def random_sampling(pool_size=100, benchmark='nasbench301', seed=0, ignore_invalid=True, ignore_repeat=False,
                    encode=True):
    candidate_configs = []
    op_label_list = []

    if benchmark == 'nasbench101':
        nas_cs = get_nas101_configuration_space()
    else:
        nas_cs = get_nas201_configuration_space()
        nas_cs.seed(seed)

    while len(candidate_configs) < pool_size:
        if benchmark == 'nasbench101':
            # generate random architecture for nasbench301
            config = nas_cs.sample_configuration()

            adjacency_matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
            idx = np.triu_indices(adjacency_matrix.shape[0], k=1)
            for i in range(VERTICES * (VERTICES - 1) // 2):
                row = idx[0][i]
                col = idx[1][i]
                adjacency_matrix[row, col] = config["edge_%d" % i]

            op_labeling = [config["op_node_%d" % i] for i in range(5)]

            # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
            if np.sum(adjacency_matrix) > MAX_EDGES and ignore_invalid:
                # print(f'invalid arch: {np.sum(adjacency_matrix)}')
                continue

            if ignore_repeat and op_labeling in op_label_list:
                continue

        elif benchmark == 'nasbench201':
            config = nas_cs.sample_configuration()
            op_labeling = [config["edge_%d" % i] for i in range(len(config.keys()))]
            # skip only duplicating architecture
            if ignore_repeat and op_labeling in op_label_list:
                continue

        # print(f'valid found, len{len(candidate_configs)}')
        candidate_configs.append(config)
        op_label_list.append(op_labeling)

    if encode:
        X_list, adj_list, valid_config_list = config_to_feature_adj(candidate_configs, ignore_invalid=False,
                                                                    benchmark=benchmark)
    else:
        X_list = candidate_configs
        adj_list = [0] * len(X_list)
        valid_config_list = candidate_configs

    return X_list, adj_list, valid_config_list


def create_nasbench201_graph_unpruned(op_node_labelling):
    assert len(op_node_labelling) == 6
    # the graph has 8 nodes (6 operation nodes + input + output)
    G = nx.DiGraph()
    edge_list = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7)]
    G.add_edges_from(edge_list)

    # assign node attributes and collate the information for nodes to be removed
    # (i.e. nodes with 'skip_connect' or 'none' label)
    node_labelling = ['input'] + op_node_labelling + ['output']
    for i, n in enumerate(node_labelling):
        G.nodes[i]['op_name'] = n

    # create the arch string for querying nasbench dataset
    arch_query_string = f'|{op_node_labelling[0]}~0|+' \
                        f'|{op_node_labelling[1]}~0|{op_node_labelling[2]}~1|+' \
                        f'|{op_node_labelling[3]}~0|{op_node_labelling[4]}~1|{op_node_labelling[5]}~2|'

    G.name = arch_query_string
    adj_matrix = np.array(nx.adjacency_matrix(G).todense())
    return G, adj_matrix, node_labelling


if __name__ == '__main__':
    nas201_cs = get_nas201_configuration_space()
