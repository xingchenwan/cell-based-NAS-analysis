import networkx as nx
import numpy as np
from typing import List

from search_spaces.nas301 import PRIMITIVES


def summarize_nb301_subgraphs_edge_level(G_list: List[nx.DiGraph]):
    """Summarize edge level properties of important node-level subgraphs of the NB301 graphs.
    Key statistics:
        - number of nodes/edges
        - distribution of the edge labels (i.e. the operations) in the subgraph
        - number of disconnected components
        - distribution of the depths of the edges
    """
    def summarize_single_graph(G: nx.DiGraph):
        nnodes, nedges = G.number_of_nodes(), G.number_of_edges()
        if nnodes > 0:
            n_components = nx.number_weakly_connected_components(G)
            ops, depths, node_identity = [], [], []
            for ei, eo, data in G.edges(data=True):
                # is_normal = True if data['isNormal'] else False
                if isinstance(data['op_name'], int): ops.append(PRIMITIVES[data['op_name']])
                else: ops.append(data['op_name'])
                edge_depth = min(G.nodes()[ei]['depth'], G.nodes()[eo]['depth'])
                node_identity.append(G.nodes()[ei]['node_name'])
                depths.append(edge_depth)
        else:
            n_components = 0
            ops, depths, node_identity = [], [], []

        res = {
            'nnodes': nnodes,
            'nedges': nedges,
            'n_components': n_components,
            'node_identity': node_identity,
            'ops': ops,     # the latter two quantities are distributions.
            'depths': depths,
            # 'isNormal': is_normal
        }
        return res

    nnodes, nedges, n_components, ops, depths, node_identities = [], [], [], [], [], []

    for i, g in enumerate(G_list):
        res = summarize_single_graph(g)
        nnodes.append(res['nnodes'])
        nedges.append(res['nedges'])
        n_components.append(res['n_components'])
        ops += res['ops']
        depths += res['depths']
        node_identities += res['node_identity']
        # is_normal.append(res['is_normal'])
    return {
        'nnodes': nnodes,
        'nedges': nedges,
        'n_components': n_components,
        'ops': ops,
        'depths': depths,
        'node_identities': node_identities
    }

