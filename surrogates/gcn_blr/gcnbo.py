from .gcn_blr import GCN_BLR
from .acquisitions import EI
from .utils import config_to_feature_adj, random_sampling
from tqdm import tqdm
import torch


class GCNBO:

    def __init__(self, objective_func, objective_func_name, init_configs, acquisition ='EI', candicate_pool_size=100,
                 bo_batch=1, train_batch_size=128, seed=0, update_freq=10,):

        surrogate = GCN_BLR(batch_size=train_batch_size, num_epochs=100, learning_rate=0.001,
                            n_units=64, normalize_output=False, seed=seed, verbose=False)

        self.surrogate = surrogate
        self.objective_func = objective_func
        self.objective_func_name = objective_func_name
        self.acquisition = acquisition
        self.bo_batch = bo_batch
        self.pool_size = candicate_pool_size
        self.update_freq = update_freq
        self.ignore_invalid = True
        self.f_update_interval = 20

        # prepare initial data for BO
        self.y_list = []
        if init_configs is not None:
            self.X_list, self.adj_list, valid_init_configs = config_to_feature_adj(init_configs, ignore_invalid=self.ignore_invalid)
            for config in valid_init_configs:
                out = self.objective_func(config)
                self.y_list.append(out)

    def optimise(self, n_iterations):
        # train surrogate based on the initial data
        self.surrogate.train(self.X_list, self.adj_list, self.y_list, do_optimize=True)

        x_f = []
        y_f = torch.tensor([])

        for i in tqdm(range(n_iterations)):

            if self.acquisition == 'EI':
                acq = EI(self.surrogate)

            candidates = random_sampling(pool_size=self.pool_size, benchmark=self.objective_func_name, ignore_invalid=self.ignore_invalid)

            next_X_list, next_adj_list, next_arch_config_list, acq_scores = acq.optimise(candidates=candidates, top_n = self.bo_batch)
            next_y_list = [self.objective_func(next_arch_config) for next_arch_config in next_arch_config_list]

            self.X_list += next_X_list
            self.adj_list += next_adj_list
            self.y_list += next_y_list

            print(f'gcn_bo: itr{i}, next_y_list={next_y_list}, best_y={min(self.y_list)}, blr_hyper={self.surrogate.hypers}')
            if i % self.update_freq == 0:
                self.surrogate.train(self.X_list, self.adj_list, self.y_list, do_optimize=True)
                print('update gcn')
            else:
                self.surrogate.train(self.X_list, self.adj_list, self.y_list, do_optimize=False)
