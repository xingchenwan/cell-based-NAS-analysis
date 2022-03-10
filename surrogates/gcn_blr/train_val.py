from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .utils import add_global_node
from .models import GCN

import pickle
# from .utils import encode_onehot
from scipy import stats


def iterate_minibatches(input_adj, input_features, targets, batchsize, shuffle=False):
    assert input_features.shape[0] == targets.shape[0], \
       f"The number of training points is not the same. {input_features.shape[0]} , {targets.shape[0]}"
    rng = np.random.RandomState(0)
    if shuffle:
        indices = np.arange(input_features.shape[0])
        rng.shuffle(indices)

    for start_idx in range(0, input_features.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield input_adj[excerpt], input_features[excerpt], targets[excerpt]


def train(model, optimizer, train_data, batch_size):
    train_adj_tensor, train_features_tensor, train_y_tensor = train_data
    train_err = 0
    train_batches = 0
    for batch in iterate_minibatches(train_adj_tensor, train_features_tensor, train_y_tensor, batch_size, shuffle=True):
        adj, features, target = batch
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.mse_loss(output.flatten(), target)
        loss.backward()
        optimizer.step()

        train_err += loss
        train_batches += 1

    epoch_loss = train_err / train_batches

    return epoch_loss.cpu().detach().numpy()


def valid(model, valid_data, batch_size):
    valid_adj_tensor, valid_features_tensor, valid_y_tensor = valid_data
    all_outputs = []
    all_targets = []
    for batch in iterate_minibatches(valid_adj_tensor, valid_features_tensor, valid_y_tensor, batch_size,
                                     shuffle=False):
        adj, features, target = batch
        model.eval()
        output = model(features, adj)
        all_outputs.append(output.cpu().detach().numpy())
        all_targets.append(target.cpu().detach().numpy())

    all_outputs_array = np.vstack(all_outputs).flatten()
    all_targets_array = np.vstack(all_targets).flatten()
    rank_coeff, p_val = stats.spearmanr(all_outputs_array, all_targets_array)

    return rank_coeff
#
#
# if __name__ == '__main__':
#     # Training settings
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='Disables CUDA training.')
#     parser.add_argument('--fastmode', action='store_true', default=False,
#                         help='Validate during training pass.')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed.')
#     parser.add_argument('--epochs', type=int, default=100,
#                         help='Number of epochs to train.')
#     parser.add_argument('--lr', type=float, default=0.001,
#                         help='Initial learning rate.')
#     parser.add_argument('--weight_decay', type=float, default=0,
#                         help='Weight decay (L2 loss on parameters). e.g. 5e-4')
#     parser.add_argument('--hidden', type=int, default=64,
#                         help='Number of hidden units.')
#
#     args = parser.parse_args()
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
#
#     # Load data
#     pickle_file = '../data/valid_arch_samples'
#     with open(pickle_file, "rb") as file:
#         archs_repo = pickle.load(file)
#
#     n_total = 500
#     n_train = 200
#     global_node = True
#     batch_size = 128
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
#     train_features_tensor = torch.Tensor(all_archs_features_list[:n_train])
#     train_adj_tensor = torch.Tensor(all_archs_adj_list[:n_train])
#     train_y_tensor = torch.Tensor(all_archs_val[:n_train])
#
#     val_features_tensor = torch.Tensor(all_archs_features_list[n_train:])
#     val_adj_tensor = torch.Tensor(all_archs_adj_list[n_train:])
#     val_y_tensor = torch.Tensor(all_archs_val[n_train:])
#
#     n_features = train_features_tensor.shape[-1]
#
#     # Model and optimizer
#     model = GCN(nfeat=n_features)
#     optimizer = optim.Adam(model.parameters(),
#                            lr=args.lr, weight_decay=args.weight_decay)
#
#     train_data = (train_adj_tensor, train_features_tensor, train_y_tensor)
#     valid_data = (val_adj_tensor, val_features_tensor, val_y_tensor)
#     for epoch in range(args.epochs):
#         # Train model
#         epoch_loss = train(model, optimizer, train_data, batch_size)
#         val_score = valid(model, valid_data, batch_size)
#
#         print(f'Epoch {epoch}: mse={epoch_loss}, rank_score={val_score}')
#
#     print("Optimization Finished!")
