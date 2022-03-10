import torch
from abc import abstractmethod
import networkx as nx
from typing import List
import torch.nn as nn
import numpy as np


class BasePredictor(nn.Module):

    def __init__(self):
        super(BasePredictor, self).__init__()

    @abstractmethod
    def fit(self, x_train: List[nx.Graph], y_train: torch.Tensor):
        """
        Train the predictor on x_train and y_train. note that fit overwrites any data already fit to the predictor
        :param x_train: a list of networkx graphs
        :param y_train: torch.Tensor representing the training targets
        :return: None
        """
        raise NotImplementedError

    # @abstractmethod
    # def update(self, x_update: List[nx.Graph], y_update: torch.Tensor):
    #     """
    #     Similar to fit, but append the x_update and y_update to the existing train data (if any, if there
    #     is no training data, this simply performs fit)
    #     :param x_update: a list of dgl graphs
    #     :param y_update: torch.Tensor representing the training targets
    #     :return: None
    #     """
    #     raise NotImplementedError

    @abstractmethod
    def predict(self, x_eval, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        Predict the graphs in x_eval using the predictor
        :param x_eval: list of dgl graphs. The test graphs on which we predict
        :param kwargs:
        :return: (mean, variance) torch.Tensor of the same shape of x_eval
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """To accommodate the format of feature_tensor and adj_tensor during inference time"""
        x = torch.clone(x).detach().numpy()
        adj = torch.clone(adj).detach().numpy()
        feature = {'op_name': x}
        networkx_graph = [self.array2networkx(feature, adj)]
        return self.predict(networkx_graph)

    @staticmethod
    def array2networkx(x: np.array, adj: np.array,):
        """Convert array representations of node/edge features into networkx graph"""
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        if x is not None:
            for n in G.nodes():
                for k, v in x.items():
                    G.nodes[n][k] = v[n]
        G = G.to_directed()
        return G

    @staticmethod
    def networkx2array(graph: nx.Graph, node_label='op_name',):
        """Convert networkx representation back to adj/feature array representation"""
        #todo: support graphs that are not necessarily node attributed and/or multiple node/edge attributes.
        adj = nx.to_numpy_array(graph)
        # feature vector
        x = np.array([data[node_label] for n, data in graph.nodes(data=True)])
        return x, adj
