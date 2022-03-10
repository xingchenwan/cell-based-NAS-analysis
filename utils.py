from collections import namedtuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
OPS = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'none'
       ]
AUGMENTED_OPS = ['input1', 'input2'] + OPS + ['add', 'output']  # all possible cell op types including the non-op ones

NASBENCH_101_OPS = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
NASBENCH_201_OPS = ['input', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none', 'output']


def drop_nb101_op(graph_input, node_index, replace_with_skip=False):
    """Drop a node (i.e. operation) in the node-attributed NAS-Bench-101 cell
    replace_with_skip: remove the node altogether or replace with a skip connection
    """
    if isinstance(graph_input, nx.Graph):
        adj = nx.to_numpy_array(graph_input)
        # feature vector
        x = np.array([data['op_name'] for n, data in graph_input.nodes(data=True)])
    else:
        adj, x = graph_input

    adj_dropped = np.delete(adj, node_index, 0)
    adj_dropped = np.delete(adj_dropped, node_index, 1)
    x_dropped = np.delete(x, node_index, 0).tolist()

    if isinstance(graph_input, nx.Graph):
        G_dropped = nx.from_numpy_array(adj_dropped, create_using=nx.DiGraph)
        for i, n in enumerate(x_dropped):
            G_dropped.nodes[i]['op_name'] = n
            return G_dropped
    return adj_dropped, x_dropped


def compute_op_grad_norm(genotype: Genotype, ):
    pass


def plot_graph(G, name):
    labels = {}
    plt.figure(figsize=[4, 4, ])
    for node, data in G.nodes(data=True):
        labels[node] = data['op_name']
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, arrows=True, with_labels=True, labels=labels)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)
    y_max, y_min = max(y_values), min(y_values)
    y_margin = (y_max - y_min) * 0.25
    plt.ylim(y_min - y_margin, y_max + y_margin)
    #     plt.title(res['report_df']['support'].values[idx])
    plt.title(name)
    plt.show()


def archinfo2genotype(arch_info):
    """Takes an arch info object (i.e. the result object loaded from NB301 dataset) and return the genotype"""
    config = arch_info['optimized_hyperparamater_config']
    genotype = []
    for i, cell_type in enumerate(['normal', 'reduce']):
        genotype.append([])

        start = 0
        n = 2
        for node_idx in range(4):
            end = start + n
            # print(start, end)
            for j in range(start, end):
                key = 'NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type, j)
                if key in config:
                    genotype[i].append((config[key], j - start))

            if len(genotype[i]) != 2 * (node_idx + 1):
                print('this is not a valid darts arch')
                return config

            start = end
            n += 1

    return Genotype(
        normal=genotype[0],
        normal_concat=[2, 3, 4, 5],
        reduce=genotype[1],
        reduce_concat=[2, 3, 4, 5]
    )


def genotype2networkx(genotype, edge_attributed=False, return_joint_graph=False,
                      node_feature_name='node_feature',
                      edge_feature_name='edge_feature',
                      numeric_label=True):
    """
    Convert a genotype to (a pair) of networkx object.
    edge_attributed: whether the cell should be edge attributed (NB201/DARTS style: there are 4 operations nodes with
        2 operation (represented as edge attribute) on each + 2 input nodes + 1 output nodes).
        Otherwise, convert to a node-attributed representation (NB101 style): the attributed nodes represent the operations
        and the directed edges represent the information flow only.
    return_joint_graph: if True the return the normal and reduce cells as a single (disjointed) networkx graphs. otherwise
        return 2 separate networkx objects.
    numeric_label: whether use the index of the operation (a numeric value) instead of the string as the node/edge label
    """

    def _cell2graph_node(cell, concat):
        """Convert a genotype into a node-based networkx representation"""
        G = nx.DiGraph()
        n_nodes = (len(cell) // 2) * 3 + 3
        G.add_nodes_from(range(n_nodes), )
        n_ops = len(cell) // 2
        if numeric_label:
            G.nodes[0][node_feature_name] = AUGMENTED_OPS.index('input1')
            G.nodes[1][node_feature_name] = AUGMENTED_OPS.index('input2')
            G.nodes[n_nodes - 1][node_feature_name] = AUGMENTED_OPS.index('output')
        else:
            G.nodes[0][node_feature_name] = 'input1'
            G.nodes[1][node_feature_name] = 'input2'
            G.nodes[n_nodes - 1][node_feature_name] = 'output'
        for i in range(n_ops):
            if numeric_label:
                G.nodes[i * 3 + 2][node_feature_name] = AUGMENTED_OPS.index(cell[i * 2][0])
                G.nodes[i * 3 + 3][node_feature_name] = AUGMENTED_OPS.index(cell[i * 2 + 1][0])
                G.nodes[i * 3 + 4][node_feature_name] = AUGMENTED_OPS.index('add')
            else:
                G.nodes[i * 3 + 2][node_feature_name] = cell[i * 2][0]
                G.nodes[i * 3 + 3][node_feature_name] = cell[i * 2 + 1][0]
                G.nodes[i * 3 + 4][node_feature_name] = 'add'
            G.add_edge(i * 3 + 2, i * 3 + 4)
            G.add_edge(i * 3 + 3, i * 3 + 4)

        for i in range(n_ops):
            # Add the connections to the input
            for offset in range(2):
                if cell[i * 2 + offset][1] == 0:
                    G.add_edge(0, i * 3 + 2 + offset)
                elif cell[i * 2 + offset][1] == 1:
                    G.add_edge(1, i * 3 + 2 + offset)
                else:
                    k = cell[i * 2 + offset][1] - 2
                    # Add a connection from the output of another block
                    G.add_edge(int(k) * 3 + 4, i * 3 + 2 + offset)
        # Add connections to the output
        for i in concat:
            if i <= 1:
                G.add_edge(i, n_nodes - 1)  # Directly from either input to the output
            else:
                op_number = i - 2
                G.add_edge(op_number * 3 + 4, n_nodes - 1)
        # If remove the skip link nodes, do another sweep of the graph
        for j in range(n_nodes):
            try:
                G.nodes[j]
            except KeyError:
                continue
            if (numeric_label and G.nodes[j][node_feature_name] == AUGMENTED_OPS.index('skip_connect')) or \
                    (not numeric_label and G.nodes[j][node_feature_name] == 'skip_connect'):
                in_edges = list(G.in_edges(j))
                out_edge = list(G.out_edges(j))[0][1]  # There should be only one out edge really...
                for in_edge in in_edges:
                    G.add_edge(in_edge[0], out_edge)
                G.remove_node(j)
            elif (numeric_label and G.nodes[j][node_feature_name] == AUGMENTED_OPS.index('none')) or \
                    (not numeric_label and G.nodes[j][node_feature_name] == 'none'):
                G.remove_node(j)
        for j in range(n_nodes):
            try:
                G.nodes[j]
            except KeyError:
                continue

            if G.nodes[j][node_feature_name] not in ['input1', 'input2', AUGMENTED_OPS.index('input1'), AUGMENTED_OPS.index('input2')]:
                # excepting the input nodes, if the node has no incoming edge, remove it
                if len(list(G.in_edges(j))) == 0:
                    G.remove_node(j)
            elif G.nodes[j][node_feature_name] != 'output' or G.nodes[j][node_feature_name] != AUGMENTED_OPS.index('output'):
                # excepting the output nodes, if the node has no outgoing edge, remove it
                if len(list(G.out_edges(j))) == 0:
                    G.remove_node(j)
            elif G.nodes[j][node_feature_name] == 'add' or G.nodes[j][node_feature_name] != AUGMENTED_OPS.index('add'):
                # If add has one incoming edge only, remove the node
                in_edges = list(G.in_edges(j))
                out_edges = list(G.out_edges(j))
                if len(in_edges) == 1 and len(out_edges) == 1:
                    G.add_edge(in_edges[0][0], out_edges[0][1])
                    G.remove_node(j)
        return G

    def _cell2graph_edge(cell, concat):
        """Convert a genotype into a edge-based networkx representation"""
        G = nx.DiGraph()
        n_nodes = 2 + 4 + 1     # two inputs, 4 ops and 1 output nodes
        n_ops = len(cell)       # number of operation nodes
        G.add_nodes_from(range(n_nodes))
        if numeric_label:
            G.nodes[0][edge_feature_name] = AUGMENTED_OPS.index('input1')
            G.nodes[1][edge_feature_name] = AUGMENTED_OPS.index('input2')
            G.nodes[n_nodes - 1][edge_feature_name] = AUGMENTED_OPS.index('output')
        else:
            G.nodes[0][edge_feature_name] = 'input1'
            G.nodes[1][edge_feature_name] = 'input2'
            G.nodes[n_nodes - 1][edge_feature_name] = 'output'
        for i in range(4):
            G.nodes[2 + i][edge_feature_name] = str(i)

        for i in range(n_ops):
            if cell[i][0] in ['none', AUGMENTED_OPS.index('none')]:
                continue
            node_to_connect = i // 2 + 2
            G.add_edge(cell[i][1], node_to_connect, op_name=cell[i][0])

        # add connections to the output
        for i in concat:
            G.add_edge(i, n_nodes - 1)
        return G

    if edge_attributed:
        G_normal = _cell2graph_edge(genotype.normal, genotype.normal_concat)
        G_reduce = _cell2graph_edge(genotype.reduce, genotype.reduce_concat)
    else:
        G_normal = _cell2graph_node(genotype.normal, genotype.normal_concat)
        G_reduce = _cell2graph_node(genotype.reduce, genotype.reduce_concat)

    if return_joint_graph:
        return nx.disjoint_union(G_normal, G_reduce)
    else:
        return G_normal, G_reduce


def encode_onehot(labels, search_space='nasbench101'):
    # classes = set(labels)
    if search_space == 'nasbench101':
        classes = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
    elif search_space == 'nasbench201':
        classes = ['input', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3', 'skip_connect', 'none', 'output']
    elif search_space in ['nasbench301', 'darts']:
        classes = AUGMENTED_OPS
    else:
        classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def pad_graphs(adj_list, feature_list, search_space='nasbench101', add_global_node=True):
    """Pad the adjacency and feature matrices"""
    if search_space == 'nasbench101':
        max_node = 7
    else:
        max_node = max([a.shape[0] for a in adj_list])
    n = len(adj_list)
    assert len(feature_list) == n
    all_archs_adj_list = []
    all_archs_features_list = []
    for j in range(n):
        padded_adj = np.zeros((max_node, max_node))
        padded_adj[:adj_list[j].shape[0], :adj_list[j].shape[1]] = adj_list[j]
        padded_feat = np.zeros((max_node, feature_list[j].shape[1]))
        # padded_feat[:, 0] = 1.
        padded_feat[:feature_list[j].shape[0], :feature_list[j].shape[1]] = feature_list[j]
        if add_global_node:
            padded_adj = do_add_global_node(padded_adj, ifAdj=True)
            padded_feat = do_add_global_node(padded_feat, ifAdj=False)
        all_archs_features_list.append(np.copy(padded_feat))
        all_archs_adj_list.append(np.copy(padded_adj))
    return all_archs_adj_list, all_archs_features_list


def do_add_global_node(matrix, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if ifAdj:
        matrix = np.column_stack((matrix, np.ones(matrix.shape[0], dtype=np.float32)))
        matrix = np.row_stack((matrix, np.zeros(matrix.shape[1], dtype=np.float32)))
        np.fill_diagonal(matrix, 1)
        matrix = matrix.T
    else:
        matrix = np.column_stack((matrix, np.zeros(matrix.shape[0], dtype=np.float32)))
        matrix = np.row_stack((matrix, np.zeros(matrix.shape[1], dtype=np.float32)))
        matrix[matrix.shape[0] - 1][matrix.shape[1] - 1] = 1
    return matrix


def prepare_data_for_gcn(search_space, data, add_global_node=True):
    """Prepare data for GCN model fitting and/or prediction. i.e. one-hot transform the categorical features, add
    global nodes and pad the adjacency/feature matrices to the smallest common dimensions."""
    if isinstance(data[0], nx.Graph):
        adj_list, feature_list = [], []
        for i in range(len(data)):
            adj_list.append(nx.to_numpy_array(data[i]))
            x = np.array([data['op_name'] for n, data in data[i].nodes(data=True)])
            feature_list.append(x)
    else:
        adj_list, feature_list = data
        if not isinstance(adj_list, list): adj_list = [adj_list]
        if not isinstance(feature_list, list): feature_list = [feature_list]
    n = len(adj_list)

    # transform the feature_list into onehot
    all_archs_features_list = []
    all_archs_adj_list = []
    for j in range(n):
        one_hot_x = encode_onehot(feature_list[j], search_space=search_space)
        # if add_global_node:
        #     arch_adj = do_add_global_node(adj_list[j], ifAdj=True)
        #     one_hot_x = do_add_global_node(one_hot_x, ifAdj=False)
        # else:
        arch_adj = adj_list[j]
        all_archs_features_list.append(one_hot_x)
        all_archs_adj_list.append(arch_adj)
    # pad the feature matrix/adjacency with zer
    adjs, feats = pad_graphs(all_archs_adj_list, all_archs_features_list, search_space=search_space, add_global_node=add_global_node)
    return feats, adjs