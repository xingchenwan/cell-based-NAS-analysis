import time
import numpy as np

import torch
import torch.optim as optim

from scipy import optimize
from .models import GCN
from .train_val import train
from .utils import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization, rmse, nll, spearman, add_global_node
from .bayesian_linear_regression import BayesianLinearRegression, Prior
from utils import encode_onehot
from surrogates.base_predictor import BasePredictor
from typing import List
import networkx as nx


class GCN_BLR(BasePredictor):

    def __init__(self,
                 batch_size=10,
                 num_epochs=500,
                 learning_rate=0.001,
                 n_units=64,
                 alpha=1.0, beta=1000,
                 prior=None,
                 normalize_output=True,
                 seed=None,
                 search_space: str = 'nasbench101',
                 # transform_one_hot: bool = True,
                 # global_node: bool = True,
                 verbose=False,
                 gpu=False):
        """
        GCN + BLR surrogate for NAS

        Parameters
        ----------
        batch_size: int
            Batch size for training the neural network
        num_epochs: int
            Number of epochs for training
        learning_rate: float
            Initial learning rate for Adam
        n_units: int
            Number of units for the gcn layers in the model
        alpha: float
            Hyperparameter of the Bayesian linear regression
        beta: float
            Hyperparameter of the Bayesian linear regression
        prior: Prior object
            Prior for alpa and beta. If set to None the default prior is used
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        rng: np.random.RandomState
            Random number generator
        """
        super(GCN_BLR, self).__init__()

        if seed is None: self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:  self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.verbose = verbose
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.X = None
        self.y = None
        self.network = None

        self.normalize_output = normalize_output

        if prior is None: self.prior = Prior(rng=self.rng)
        else:  self.prior = prior

        # Network hyper parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.n_units = n_units
        self.network = None
        self.model = None

        self.alpha = alpha
        self.beta = beta
        self.hypers = None

        # check device
        self.gpu = gpu
        if self.gpu: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = 'cpu'
        self.search_space = search_space
        # self.transform_one_hot = transform_one_hot
        # self.global_node = global_node
        # self.max_dim = None

    def fit(self, x_train, y_train: torch.Tensor, do_optimize=True):
        if isinstance(x_train[0], nx.Graph):
            transformed_data = [self.networkx2array(x, 'op_name') for x in x_train]
            transformed_data = np.array(transformed_data, dtype=np.object)
            X_list, adj_list = transformed_data[:, 0].tolist(), transformed_data[:, 1].tolist()
        else:
            X_list, adj_list = x_train
        y_list = y_train.detach().numpy().tolist()
        self.fit_(X_list, adj_list, y_list, do_optimize=do_optimize)

    def fit_(self, X_list, adj_list, y_list, do_optimize=True):
        """
        Trains GCN on the provided data.
        Same as the original fit module but it takes feature_list and adj_list
        Parameters
        ----------
        X_list: list of N entries (n_nodes, d)
            architecture node feature
        adj_list: list of N entries (n_nodes, n_nodes)
            architecture adjacency matrix
        y_list: list of N entries (1,)
            architecture validation accuracy
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters are used.
        """
        n = len(X_list)
        assert len(adj_list) == n, 'shape mismatch!'
        all_archs_features_list = X_list
        all_archs_adj_list = adj_list
        # for j in range(n):
        #     if self.transform_one_hot:
        #         one_hot_x = encode_onehot(X_list[j], search_space=self.search_space)
        #     else:
        #         one_hot_x = X_list[j]
        #     if self.global_node:
        #         arch_adj = add_global_node(adj_list[j], ifAdj=True)
        #         one_hot_x = add_global_node(one_hot_x, ifAdj=False)
        #     else:
        #         arch_adj = adj_list[j]
        #     all_archs_features_list.append(one_hot_x)
        #     all_archs_adj_list.append(arch_adj)
        # # pad the feature matrix/adjacency with zeros, where applicable
        # self.max_dim = max([a.shape[0] for a in all_archs_features_list])
        # for j in range(n):
        #     padded_adj = np.zeros((self.max_dim, self.max_dim))
        #     padded_adj[:all_archs_adj_list[j].shape[0], :all_archs_adj_list[j].shape[1]] = all_archs_adj_list[j]
        #     padded_feat = np.zeros((self.max_dim, all_archs_features_list[j].shape[1]))
        #     # padded_feat[:, 0] = 1.
        #     padded_feat[:all_archs_features_list[j].shape[0], :all_archs_features_list[j].shape[1]] = all_archs_features_list[j]
        #     if self.global_node:
        #         padded_adj = add_global_node(padded_adj, ifAdj=True)
        #         padded_feat = add_global_node(padded_feat, ifAdj=False)
        #     all_archs_features_list[j] = np.copy(padded_feat)
        #     all_archs_adj_list[j] = np.copy(padded_adj)

        start_time = time.time()

        # Normalize ouputs
        y_array = np.array(y_list)
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y_array)
        else:
            self.y = y_array

        self.y_min = np.min(self.y)
        self.X_list = all_archs_features_list
        self.adj_list = all_archs_adj_list
        train_X_tensor = torch.tensor(all_archs_features_list, dtype=torch.float32)
        train_adj_tensor = torch.tensor(all_archs_adj_list,  dtype=torch.float32)
        train_y_tensor = torch.tensor(self.y, dtype=torch.float32)
        train_data = (train_adj_tensor, train_X_tensor, train_y_tensor)
        batch_size = min(self.batch_size, train_X_tensor.shape[0])

        if do_optimize:
            # Create the neural network
            n_features = train_X_tensor.shape[-1]
            network = GCN(nfeat=n_features, nhid=self.n_units)

            self.network = network.to(self.device)
            optimizer = optim.Adam(self.network.parameters(),
                                   lr=self.init_learning_rate)

            # Start training
            lc = []
            for epoch in range(self.num_epochs):

                epoch_start_time = time.time()

                # Train model for 1 epoch
                epoch_loss = train(self.network, optimizer, train_data, batch_size)
                lc.append(epoch_loss)

                curtime = time.time()
                epoch_time = curtime - epoch_start_time
                total_time = curtime - start_time
                if self.verbose:
                    print(f'Epoch {epoch}: mse={epoch_loss}, time {epoch_time:.3f}s, total time {total_time:.3f}s')

        # Train BLR
        # Design matrix
        network = self.network
        if self.gpu:
            network.cpu()
        self.Theta = network.basis_funcs(train_X_tensor, train_adj_tensor).data.numpy()

        if do_optimize:
            # Optimize hyperparameters of the Bayesian linear regression
            p0 = self.prior.sample_from_prior(n_samples=1)
            res = optimize.fmin(self.negative_mll, p0, disp=self.verbose)
            self.alpha = np.exp(res[0])
            self.beta = 1 / np.exp(res[1])
            self.hypers = [self.alpha, self.beta]
            if self.verbose:
                print(f'Optimise BLR hyperparameters: hypers_learnt={self.hypers}')
        else:
            self.hypers = [self.alpha, self.beta]
            if self.verbose:
                print(f'Dont optimise BLR hyperparameters, reuse hypers={self.hypers}')

        # Instantiate a model for each hyperparameter configuration
        model = BayesianLinearRegression(alpha=self.hypers[0],
                                         beta=self.hypers[1],)
        model.train(self.Theta, self.y, do_optimize=False)
        self.model = model

    def negative_mll(self, theta):
        """
        Negative marginla Log likelihood of the data marginalised over the weights w. See chapter 3.5 of
        the book by Bishop of an derivation.

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        """
        if np.any(theta == np.inf):
            return -np.inf

        if np.any((-10 > theta) + (theta > 10)):
            return -np.inf

        alpha = np.exp(theta[0])
        beta = 1 / np.exp(theta[1])

        D = self.Theta.shape[1]
        N = self.Theta.shape[0]

        K = beta * np.dot(self.Theta.T, self.Theta)
        K += np.eye(self.Theta.shape[1]) * alpha
        K_inv = np.linalg.inv(K)
        m = beta * np.dot(K_inv, self.Theta.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.Theta, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K) + 1e-10)

        if np.any(np.isnan(mll)):
            return 1e25
        return - mll

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, adj):
        """Default mode of forwarding -- does not break the computing graphs"""
        if self.gpu: network = self.network.cuda()
        else: network = self.network
        network.eval()
        if x.ndimension() == 2: x = x.unsqueeze(0)
        if adj.ndimension() == 2: adj = adj.unsqueeze(0)
        test_Theta = network.basis_funcs(x, adj)
        mu, _ = self.model.predict(test_Theta.detach().numpy())
        return mu

    def predict(self, x_eval, **kwargs) -> (torch.Tensor, torch.Tensor):
        if isinstance(x_eval[0], nx.Graph):
            transformed_data = [self.networkx2array(x, 'op_name') for x in x_eval]
            transformed_data = np.array(transformed_data, dtype=np.object)
            X_list, adj_list = transformed_data[:, 0].tolist(), transformed_data[:, 1].tolist()
        else:
            X_list, adj_list = x_eval
        return self.predict_(X_list, adj_list, )

    def predict_(self, X_test_list, adj_test_list):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test_list: list of N entries (n_nodes, d)
            test architecture node feature
        adj_test_list: list of N entries (n_nodes, n_nodes)
            test architecture adjacency matrix
        """
        n = len(X_test_list)
        assert len(X_test_list) == n, 'shape mismatch!'

        all_archs_features_list = X_test_list
        all_archs_adj_list = adj_test_list
        # for j in range(n):
        #     if self.transform_one_hot:
        #         one_hot_x = encode_onehot(X_test_list[j], search_space=self.search_space)
        #     else:
        #         one_hot_x = X_test_list[j]
        #     if self.global_node:
        #         arch_adj = add_global_node(adj_test_list[j], ifAdj=True)
        #         one_hot_x = add_global_node(one_hot_x, ifAdj=False)
        #     else:
        #         arch_adj = adj_test_list[j]
        #     all_archs_features_list.append(one_hot_x)
        #     all_archs_adj_list.append(arch_adj)
        # # pad the feature matrix/adjacency with zeros, where applicable
        # for j in range(n):
        #     padded_adj = np.zeros((self.max_dim, self.max_dim))
        #     padded_adj[:all_archs_adj_list[j].shape[0], :all_archs_adj_list[j].shape[1]] = all_archs_adj_list[j]
        #     padded_feat = np.zeros((self.max_dim, all_archs_features_list[j].shape[1]))
        #     padded_feat[:all_archs_features_list[j].shape[0], :all_archs_features_list[j].shape[1]] = all_archs_features_list[j]
        #     if self.global_node:
        #         padded_adj = add_global_node(padded_adj, ifAdj=True)
        #         padded_feat = add_global_node(padded_feat, ifAdj=False)
        #     all_archs_features_list[j] = np.copy(padded_feat)
        #     all_archs_adj_list[j] = np.copy(padded_adj)

        # for j in range(n):
        #     if self.transform_one_hot:
        #         X_test_list[j] = encode_onehot(X_test_list[j], benchmark=self.search_space)
        #     if self.global_node:
        #         adj_test_list[j] = add_global_node(adj_test_list[j], ifAdj=True)
        #         X_test_list[j] = add_global_node(X_test_list[j], ifAdj=False)

        if self.gpu: network = self.network.cuda()
        else: network = self.network

        test_X_tensor = torch.tensor(all_archs_features_list, dtype=torch.float32)
        test_adj_tensor = torch.tensor(all_archs_adj_list, dtype=torch.float32)

        # Get Design matrix for test data
        network.eval()
        test_Theta = network.basis_funcs(test_X_tensor, test_adj_tensor).data.numpy()

        # Marginalise predictions over hyperparameters of the BLR
        mu, var = self.model.predict(test_Theta)

        # Clip negative variances and set them to the smallest
        # positive float value
        if var.shape[0] == 1:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
        else:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
            var[np.where((var < np.finfo(var.dtype).eps) & (var > -np.finfo(var.dtype).eps))] = 0

        if self.normalize_output:
            mu = zero_mean_unit_var_denormalization(mu, self.y_mean, self.y_std)
            var *= self.y_std ** 2

        return mu, var

    def validate(self, X_test_list, adj_test_list, y_test_list):
        """
        Validate GCN on the provided data.
        Parameters
        ----------
        X_test_list: list of N entries (n_nodes, d)
            test architecture node feature
        adj_test_list: list of N entries (n_nodes, n_nodes)
            test architecture adjacency matrix
        y_test_list: list of N entries (1,)
            test architecture validation accuracy
        """

        m_pred, v_pred = self.predict(X_test_list, adj_test_list)
        y_test_array = np.array(y_test_list)

        rmse_results = rmse(m_pred, y_test_array)
        rank_corr = spearman(m_pred, y_test_array)
        negative_log_likelihood = nll(m_pred, np.sqrt(v_pred), y_test_array)

        return rmse_results, negative_log_likelihood, rank_corr


# if __name__ == '__main__':
#     import pickle
#
#     # Load data
#     pickle_file = './data/valid_arch_samples'
#     with open(pickle_file, "rb") as file:
#         archs_repo = pickle.load(file)
#
#     n_total = 700
#     n_train = 50
#     global_node = True
#     batch_size = 128
#     seed = 42
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     archs = archs_repo['model_graph_specs'][:n_total]
#     archs_adj = [arch['adjacency'] for arch in archs]
#     archs_x = [arch['node_labels'] for arch in archs]
#     all_archs_val = archs_repo['validation_err'][:n_total]
#
#     all_archs_features_list = []
#     all_archs_adj_list = []
#     for i in range(n_total):
#         arch = archs[i]
#         onehot_archs_x = encode_onehot(arch['node_labels'])
#         if global_node:
#             arch_adj = add_global_node(arch['adjacency'], ifAdj=True)
#             onehot_features = add_global_node(onehot_archs_x, ifAdj=False)
#         else:
#             arch_adj = arch['adjacency']
#             onehot_features = onehot_archs_x
#         all_archs_features_list.append(onehot_features)
#         all_archs_adj_list.append(arch_adj)
#
#     X_train_list = all_archs_features_list[:n_train]
#     adj_train_list = all_archs_adj_list[:n_train]
#     y_train_list = all_archs_val[:n_train]
#
#     X_test_list = all_archs_features_list[n_train:]
#     adj_test_list = all_archs_adj_list[n_train:]
#     y_test_list = all_archs_val[n_train:]
#     # define and train surrogate
#     surrogate = GCN_BLR(batch_size=batch_size, num_epochs=100, learning_rate=0.001, n_units=64, normalize_output=False,
#                         seed=seed, verbose=True)
#
#     surrogate.train(X_train_list, adj_train_list, y_train_list)
#     rmse, negative_log_likelihood, rank_corr = surrogate.validate(X_test_list, adj_test_list, y_test_list)
#     print(f'rmse={rmse}, nll={negative_log_likelihood}, rank_cor{rank_corr}')
