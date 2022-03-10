import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid=64):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.fc = nn.Linear(nhid, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = x[:,x.size()[1]-1,:]
        x = self.fc(x)
        return x
        # return self.sigmoid(x2)

    def basis_funcs(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = x[:,x.size()[1]-1,:]
        return x



# if __name__ == '__main__':
#     import pickle
#     from .utils import encode_onehot, add_global_node
#     import torch
#     import numpy as np
#     pickle_file = '../data/valid_arch_samples'
#     with open(pickle_file, "rb") as file:
#         archs_repo = pickle.load(file)
#
#
#     n_total = 100
#     n_train = 50
#     global_node = True
#     batch_size = 20
#     seed = 42
#     np.random.seed(seed)
#     torch.manual_seed(seed)
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
#     # Model
#     model = GCN(nfeat=n_features)
#
#     features = train_features_tensor[:batch_size]
#     adj = train_adj_tensor[:batch_size]
#     model.train()
#     output = model(features, adj)
#     print(f'{output.detach().numpy().T}')
