import torch
import torch.nn as nn
import torch.nn.functional as F
from surrogates.base_predictor import BasePredictor
from torch.utils.data import DataLoader
from .utils import AverageMeterGroup
import numpy as np
import torch.optim as optim
import networkx as nx
import math


def normalize_adj(adj):
    # Row-normalize matrix
    # return adj
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))


def accuracy_mse(prediction, target, scale=100.):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


class DirectedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)

    def forward(self, inputs, adj):
        norm_adj = normalize_adj(adj)
        # norm_adj = adj
        output1 = F.relu(torch.matmul(norm_adj.double(), torch.matmul(inputs.double(), self.weight1.double())))
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        output2 = F.relu(torch.matmul(inv_norm_adj, torch.matmul(inputs.double(), self.weight2.double())))
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NeuralPredictorModel(nn.Module):
    def __init__(self, initial_hidden=-1, gcn_hidden=64, gcn_layers=4, linear_hidden=64):
        super().__init__()
        self.gcn = [DirectedGraphConvolution(initial_hidden if i == 0 else gcn_hidden, gcn_hidden)
                    for i in range(gcn_layers)]
        self.gcn = nn.ModuleList(self.gcn)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(gcn_hidden, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

    def forward(self, inputs):
        # out = x
        # numv = x.shape[0]
        numv, adj, out = inputs["num_vertices"], inputs["adjacency"], inputs["operations"]
        gs = adj.size(1)  # graph node number

        adj_with_diag = normalize_adj(adj + torch.eye(gs, device=adj.device))  # assuming diagonal is not 1
        # adj_with_diag = normalize_adj(adj)
        for layer in self.gcn:
            out = layer(out, adj_with_diag)
        out = graph_pooling(out, numv)
        # out = self.fc1(out.float())
        # out = self.dropout(out)
        out = self.fc2(out.float()).view(-1)
        return out


class GCNPredictor(BasePredictor):
    def __init__(self, encoding_type='gcn', ss_type=None, ):
        super(GCNPredictor, self).__init__()
        self.encoding_type = encoding_type
        # self.need_separate_hpo = need_separate_hpo
        self.hyperparams = None

        if ss_type is not None:
            self.ss_type = ss_type
        self.default_hyperparams = {'gcn_hidden': 64,
                                    'batch_size': 10,
                                    'epochs': 300,
                                    'lr': 1e-3,
                                    'wd': 0.}

    def get_model(self, **kwargs):
        if self.ss_type == 'nasbench101': initial_hidden = 5 + 1
        else:  initial_hidden = 7
        predictor = NeuralPredictorModel(initial_hidden=initial_hidden)
        return predictor

    def forward(self, x: torch.Tensor, adj: torch.Tensor, detach=False):
        """Called on as "predict" but on a single graph"""
        self.model.eval()
        encoded = {
            'num_vertices': adj.shape[0],
            'adjacency': adj.double(),
            'operations': x.double(),
            'val_acc': float(0.0)
        }
        test_data = [encoded]
        test_data = np.array(test_data)
        test_data_loader = DataLoader(test_data, batch_size=1)
        pred = []
        if detach:
            with torch.no_grad():
                for _, batch in enumerate(test_data_loader):
                    prediction = self.model(batch)
                    pred.append(prediction.cpu())

            pred = torch.cat(pred)
            return pred * self.std + self.mean, None

        pred = []
        for _, batch in enumerate(test_data_loader):
            prediction = self.model(batch)
            pred.append(prediction)
        pred = torch.cat(pred)
        return pred * self.std + self.mean, None

    def fit(self, xtrain, ytrain, train_info=None, seed=0):
        if isinstance(xtrain[0], nx.Graph):
            xtrain = [self.networkx2array(x) for x in xtrain]

        # get hyperparameters
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        gcn_hidden = self.hyperparams['gcn_hidden']
        batch_size = self.hyperparams['batch_size']
        epochs = self.hyperparams['epochs']
        lr = self.hyperparams['lr']
        wd = self.hyperparams['wd']

        # get mean and std, normlize accuracies
        self.mean = torch.mean(ytrain)
        self.std = torch.std(ytrain)
        ytrain_normed = (ytrain - self.mean)/self.std
        # encode data in gcn format
        train_data = []
        # for i, (x, adj) in enumerate(xtrain):
        for i in range(len(xtrain[0])):
            encoded = {
                'num_vertices': xtrain[1][i].shape[0],
                'adjacency': xtrain[1][i],
                'operations': xtrain[0][i],
                'val_acc': float(ytrain_normed[i])
            }
            # encoded = encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
            # encoded['val_acc'] = float(ytrain_normed[i])
            train_data.append(encoded)
        train_data = np.array(train_data)

        self.model = self.get_model(gcn_hidden=gcn_hidden)
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        # self.model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        self.model.train()

        for _ in range(epochs):
            meters = AverageMeterGroup()
            lr = optimizer.param_groups[0]["lr"]
            for _, batch in enumerate(data_loader):
                target = batch["val_acc"].float()
                prediction = self.model(batch)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                mse = accuracy_mse(prediction, target)
                meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))

            # lr_scheduler.step()
        train_pred = torch.squeeze(self.predict(xtrain, detach=True)[0])
        train_error = torch.mean(torch.abs(train_pred-ytrain))
        return train_error

    def predict(self, x_eval, info=None, eval_batch_size=64, detach=False):
        if isinstance(x_eval[0], nx.Graph):
            x_eval = [self.networkx2array(x) for x in x_eval]

        test_data = []
        for i in range(len(x_eval[0])):
        # for i, (x, adj) in enumerate(x_eval):
            encoded = {
                'num_vertices': x_eval[1][i].shape[0],
                'adjacency': x_eval[1][i],
                'operations': x_eval[0][i],
                'val_acc': float(0.0)
            }
            # encoded = encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
            # encoded['val_acc'] = float(ytrain_normed[i])
            test_data.append(encoded)
        test_data = np.array(test_data)
        test_data_loader = DataLoader(test_data, batch_size=eval_batch_size)

        self.model.eval()
        pred = []
        if detach:
            with torch.no_grad():
                for _, batch in enumerate(test_data_loader):
                    prediction = self.model(batch)
                    pred.append(prediction.cpu())

            pred = torch.cat(pred)
            return pred * self.std + self.mean, None

        pred = []
        for _, batch in enumerate(test_data_loader):
            prediction = self.model(batch)
            pred.append(prediction)
        pred = torch.cat(pred)
        return pred * self.std + self.mean, None
