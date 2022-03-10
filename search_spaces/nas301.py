import torch
import numpy as np
from .search_space import SearchSpace
import random
import networkx as nx
import matplotlib.pyplot as plt
import os
try:
    import nasbench301 as nb
except:
    print('Importing NAS-Bench-301 API failed')
from collections import namedtuple
from copy import deepcopy
from darts_cnn.train_class import Train
from darts_cnn.eval_class import Eval
from darts_cnn.eval_class_imagenet import EvalImageNet

# from darts_cnn.model import NetworkCIFAR as Network

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
]
PRIMITIVES_NO_NONE = PRIMITIVES[:-1] # none is a valid primitive not for nb301, but for original darts search space
# EXTENDED_PRIMITIVES  = deepcopy(PRIMITIVES) + ['none']
N_TOWERS = 4
OPS_dict = {i: op for i, op in enumerate(PRIMITIVES)}
INIT_CHANNELS = 32
LAYERS = 8
INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'

PLOTTING_SHORTHAND = {  # shorthand for plotting -- to avoid cluttered networkx figures
    'sep_conv_3x3': 's3',
    'sep_conv_5x5': 's5',
    'dil_conv_3x3': 'd3',
    'dil_conv_5x5': 'd5',
    'skip_connect': 'skip',
    'max_pool_3x3': 'mp3',
    'avg_pool_3x3': 'ap3',
    'none': 'none',
    '': '',
}


class NASBench301(SearchSpace):
    _name = 'nasbench301'

    def __init__(self,
                 file_path='../nasbench301/nasbench301/nb_models_0.9/',
                 image_path='./',
                 arch_save_path='./output/',
                 dataset='cifar10',
                 device='cpu',
                 log_scale=True, negative=True,
                 external_surrogate=None,
                 ):
        """
        Load the surrogate models in NAS-Bench-301 for DARTS predicting architecture performance
        file_path: the folder containing trained surrogate models for NAS-Bench-301
        arch_weigh_path: the folder containing trained surrogate models for NAS-Bench-301
        dataset: the vision dataset to choose. select from 'cifar10', 'cifar100', and 'ImageNet16'.
        verbose: verbose model to be passed for NAS-Bench-201 API creation
        device: 'cpu' or 'cuda'. The location to load the models to. Note even on a cuda-enabled device, you
            need to set device to 'cuda' to use the GPU.
        """
        assert dataset in ['cifar10']
        if device == 'cuda' and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        super(NASBench301, self).__init__(file_path, dataset, device)

        if file_path is not None:
            try:
                performance_model = nb.load_ensemble(os.path.join(file_path, 'xgb_v0.9'))
                runtime_model = nb.load_ensemble(os.path.join(file_path, 'lgb_runtime_v0.9'))
                self.performance_model = performance_model
                self.runtime_model = runtime_model
                self.no_api = False
            except Exception as e:
                print(f'Loading NB301 failed. API-less mode. Error message is {e}')
                self.no_api = True
        else:
            print('NB301 file not specified. API-less mode.')
            self.no_api = True

        self.seed = 0  # TODO to be determined --> check what's used in NASBench301
        # self.arch_weigh_path = arch_weigh_path
        # self.image_path = image_path

        self.num_ops = len(PRIMITIVES)
        self.op_spots = N_TOWERS * 2
        self.image_path = image_path
        self.arch_save_path = arch_save_path
        if dataset == 'cifar10':
            self.CIFAR_CLASSES = 10
        elif dataset == 'cifar100':
            self.CIFAR_CLASSES = 100
        elif dataset == 'ImageNet16-120':
            self.CIFAR_CLASSES = 120  # not really cifar...

        self.log_scale = log_scale
        self.negative = negative
        self.external_surrogate = external_surrogate

    def query(self, arch, seed=None, full_result=False, with_noise=True):
        """Given a arch_str, return its a dictionary of results including accuracy, loss, flops, nparams and its trained
            weights loaded into a pytorch model.
        arch: a list of 2 lists [[normal_cell], [reduction_cell]] of tuples
        e.g.[[(0, 0), (1, 1), (2, 0), (0, 1), (0, 1), (3, 0), (3, 0), (2, 2)],
        [(4, 0), (4, 1), (3, 2), (4, 1), (4, 0), (3, 2), (3, 2), (4, 1)]],
        for each tuple (op_id, input_feature_node_id)
        seed: seed for the network training/surrogate prediction.
        full_result: whether we need to retrain the arch from scratch to obtain its output logits
        external_surrogate: whether use another surrogate model to query performances. for now expected to be a BANANAS
            surrogate
            # todo: ensure the external_surrogate argument is compatible with other surrogate chocies.
        """
        if self.no_api and self.external_surrogate is None:
            return np.nan

        if isinstance(arch, Genotype):
            genotype_config = arch
        else:
            genotype_config = self.to_genotype(arch)
        if seed is None:
            seed = 42

        # is_random = seed if seed is not None else True
        # get a dictionary of accuracy, loss and etc.
        arch_results = {}
        if self.external_surrogate is not None:
            encoded = np.array(self.encode_paths(arch))
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)
            acc_genotype = np.exp(-self.external_surrogate.predict(encoded)[0]) * 100
        else:
            acc_genotype = self.performance_model.predict(config=genotype_config, representation="genotype",
                                                          with_noise=with_noise)
        # runtime_genotype = self.runtime_model.predict(config=genotype_config, representation="genotype")
        # arch_results['cost'] = runtime_genotype
        res = 1. - acc_genotype / 100

        if self.log_scale:
            res = np.log(res)
        if self.negative:
            res = -res
        return res

    def train(self, arch, epochs: int = 100, seed: int = None, gpu_id: int = 0, **kwargs):
        """Train from scratch of an arch -- this actually trains a genotype cell from scratch, and is thus what
        NB301 aims to approximate cheaply
        epochs: number of training epochs

        Modify the training hyperparameters at ./darts_cnn/train_class.py
        """
        if not isinstance(arch, Genotype): arch = self.to_genotype(arch)
        arch_id = genotype2str(arch, flatten=True)
        if not os.path.exists(os.path.join(self.arch_save_path, 'arch_weights')):
            os.makedirs(os.path.join(self.arch_save_path, 'arch_weights'))

        arch_save_path = os.path.join(self.arch_save_path, 'arch_weights', arch_id+'_weights.pt')

        trainer = Train(image_data_path=self.image_path)
        if seed is None: seed = np.random.randint(0, 1e6)
        stats, model = trainer.main(arch, epochs=epochs, seed=seed, gpu=gpu_id, save=arch_save_path, **kwargs)
        return stats, model

    def evaluate(self, arch, epochs: int = 600, seed: int = None, gpu_id: int = 0, dataset='cifar10',
                 test_mode=False, **kwargs):
        """
        Train from scratch for the final architecture evaluation.
        """
        if not isinstance(arch, Genotype): arch = self.to_genotype(arch)
        arch_id = genotype2str(arch, flatten=True)
        if not os.path.exists(os.path.join(self.arch_save_path, 'arch_eval_weights')):
            os.makedirs(os.path.join(self.arch_save_path, 'arch_eval_weights'))

        arch_save_path = os.path.join(self.arch_save_path, 'arch_eval_weights', arch_id+'_eval_weights.pt')

        if dataset in ['cifar10', 'cifar100']: trainer = Eval(image_data_path=self.image_path, dataset=dataset)
        elif dataset.lower() == 'imagenet': trainer = EvalImageNet(image_data_path=self.image_path,)
        else: raise ValueError(f"Unknown dataset {dataset}")
        if seed is None: seed = np.random.randint(0, 1e6)
        stats, model = trainer.main(arch, epochs=epochs, seed=seed, gpu=gpu_id, save=arch_save_path, test_mode=test_mode, **kwargs)
        return stats, model

    def evaluate_list(self, archs, epochs: int = 600, seed: int = None, gpu_id: int = 0, dataset='cifar10', **kwargs):
        archs = [self.to_genotype(a) for a in archs]
        if not os.path.exists(os.path.join(self.arch_save_path, 'arch_eval_weights')):
            os.makedirs(os.path.join(self.arch_save_path, 'arch_eval_weights'))
        arch_save_path = os.path.join(self.arch_save_path, 'arch_eval_weights', 'list_eval_weights.pt')
        trainer = Eval(image_data_path=self.image_path, dataset=dataset)
        if seed is None: seed = np.random.randint(0, 1e6)
        stats, model = trainer.main(archs, epochs=epochs, seed=seed, gpu=gpu_id, save=arch_save_path, **kwargs)
        return stats, model

    def get_random_arch(self, return_string=True, same_normal_reduce=True, ):
        """
        n: number of random samples to yield
        same_arch (bool): whether to use the same architecture for the normal cell and the reduction cell
        """

        """Generate a list of 2 tuples, consisting of the random DARTS Genotype and DiGraph"""

        def _sample():
            normal = []
            reduction = []
            for i in range(N_TOWERS):
                ops = np.random.choice(range(len(PRIMITIVES_NO_NONE)), N_TOWERS)

                # input nodes for conv
                nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
                # input nodes for reduce
                if same_normal_reduce:
                    nodes_in_reduce = nodes_in_normal
                    normal.extend([(PRIMITIVES_NO_NONE[ops[0]], nodes_in_normal[0]), (PRIMITIVES_NO_NONE[ops[1]], nodes_in_normal[1])])
                    reduction.extend(
                        [(PRIMITIVES_NO_NONE[ops[0]], nodes_in_reduce[0],), (PRIMITIVES_NO_NONE[ops[1]], nodes_in_reduce[1])])
                else:
                    op_in_reduce = np.random.choice(range(len(PRIMITIVES_NO_NONE)), N_TOWERS)
                    nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)
                    normal.extend([(PRIMITIVES_NO_NONE[ops[0]], nodes_in_normal[0]), (PRIMITIVES_NO_NONE[ops[1]], nodes_in_normal[1])])
                    reduction.extend(
                        [(PRIMITIVES_NO_NONE[op_in_reduce[0]], nodes_in_reduce[0],),
                         (PRIMITIVES_NO_NONE[op_in_reduce[1]], nodes_in_reduce[1])])

            darts_genotype = Genotype(normal=normal, normal_concat=range(2, 2 + N_TOWERS),
                                      reduce=reduction, reduce_concat=range(2, 2 + N_TOWERS))
            return darts_genotype

        # obtain a randomly sampled genotype
        genotype = _sample()
        if return_string:
            arch_str = genotype2str(genotype)
            return genotype, arch_str
        return genotype

    def sample_from_anasod(self, anasod_normal: np.ndarray,
                           anasod_reduce: np.ndarray = None,
                           as_probability=True,
                           return_string=True,
                           return_genotype=False,
                           seed=None,
                           *args, **kwargs):
        """Sample a randomly wired architecture from ANASOD encoding. If anasod_reduce is not supplied, then the cell
        will use the same encoding for both normal and reduce cells"""

        if return_genotype is False and return_string is False:
            raise ValueError("Either or both of return_string and return_genotype need to be True!")

        def _sample_fixedop(ops_norm, ops_reduce, ):
            # Generate a randomly wired Genotype but with the operations chosen
            normal = []
            reduction = []
            for i in range(N_TOWERS):
                # input nodes for conv
                nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
                nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)
                normal.extend([(ops_norm[0], nodes_in_normal[0]), (ops_norm[1], nodes_in_normal[1])])
                reduction.extend([(ops_reduce[0], nodes_in_reduce[0]), (ops_reduce[1], nodes_in_reduce[1])])
                # Pop the first two elements
                ops_norm = ops_norm[2:]
                ops_reduce = ops_reduce[2:]

            darts_genotype = Genotype(normal=normal, normal_concat=list(range(2, 2 + N_TOWERS)),
                                      reduce=reduction, reduce_concat=list(range(2, 2 + N_TOWERS)))
            return darts_genotype

        N_OPS = N_TOWERS * 2
        assert len(anasod_normal) == len(PRIMITIVES_NO_NONE)
        if anasod_reduce is None:
            anasod_reduce = anasod_normal
        if as_probability:
            assert np.isclose(np.sum(anasod_normal), 1.)
            assert np.isclose(np.sum(anasod_reduce), 1.)
            anasod_normal = np.round(anasod_normal * N_OPS).astype(int)
            anasod_reduce = np.round(anasod_reduce * N_OPS).astype(int)
        else:
            anasod_normal = np.anasod_normal(anasod_normal).astype(int)
            anasod_reduce = np.round(anasod_reduce).astype(int)

        # Generate the ops from x
        ops_norm = []
        ops_reduce = []
        for idx in range(len(anasod_normal)):
            ops_norm += [PRIMITIVES_NO_NONE[idx]] * anasod_normal[idx]
            ops_reduce += [PRIMITIVES_NO_NONE[idx]] * anasod_reduce[idx]
        # Randomly shuffle the order
        if seed is None:
            np.random.shuffle(ops_norm)
            np.random.shuffle(ops_reduce)
        else:
            np.random.RandomState(seed).shuffle(ops_norm)
            np.random.RandomState(seed).shuffle(ops_reduce)

        genotype = _sample_fixedop(ops_norm, ops_reduce, )
        arch_string = self.to_str(genotype)
        if return_string and not return_genotype:
            return arch_string
        if return_genotype and not return_string:
            return genotype
        else:
            return genotype, arch_string

    def get_encoding(self, arch, encoding='anasod', normal_only=False):
        """Get encoding of a DARTS cell. Possible choices: adjacency encoding or the ANASOD encoding.
        normal_only: only return the encoding for the normal cell and that of the reduce cell is discarded."""

        def _from_ops(ops):
            op_type, count = np.unique(ops, return_counts=True)
            op_occurrences = dict(
                zip(op_type.tolist(), count))
            res = np.zeros(len(PRIMITIVES))
            for k, v in op_occurrences.items():
                try:
                    idx = PRIMITIVES.index(k)
                    res[idx] = v
                except ValueError:
                    continue
            return np.array(res)

        if not isinstance(arch, Genotype):
            genotype = self.to_genotype(arch)
        else:
            genotype = arch
        assert encoding in ['anasod']  # todo: implement the adjacency encoding, if necessary
        ops_norm = [g[0] for g in genotype.normal]
        ops_reduce = [g[0] for g in genotype.reduce]
        res_norm = _from_ops(ops_norm)
        res_reduce = _from_ops(ops_reduce)
        if normal_only:
            return res_norm
        return res_norm, res_reduce

    # functions for path encoding proposed in BANANAS
    def get_paths(self, arch):
        """ return all paths from input to output """
        if not isinstance(arch, Genotype):
            arch = self.to_genotype(arch)
        arch = convert_genotype_to_compact(arch)

        path_builder = [[[], [], [], []], [[], [], [], []]]
        paths = [[], []]

        for i, cell in enumerate(arch):
            for j in range(len(PRIMITIVES)):
              if cell[j][0] == 0:
                  path = [INPUT_1, PRIMITIVES[cell[j][1]]]
                  path_builder[i][j//2].append(path)
                  paths[i].append(path)
              elif cell[j][0] == 1:
                  path = [INPUT_2, PRIMITIVES[cell[j][1]]]
                  path_builder[i][j//2].append(path)
                  paths[i].append(path)
              else:
                  for path in path_builder[i][cell[j][0] - 2]:
                      path = [*path, PRIMITIVES[cell[j][1]]]
                      path_builder[i][j//2].append(path)
                      paths[i].append(path)

        # check if there are paths of length >=5
        contains_long_path = [False, False]
        if max([len(path) for path in paths[0]]) >= 5:
            contains_long_path[0] = True
        if max([len(path) for path in paths[1]]) >= 5:
            contains_long_path[1] = True

        return paths, contains_long_path

    def get_path_indices(self, arch, long_paths=True):
        """
        compute the index of each path
        There are 4 * (8^0 + ... + 8^4) paths total
        If long_paths = False, we give a single boolean to all paths of
        size 4, so there are only 4 * (1 + 8^0 + ... + 8^3) paths
        """
        paths, contains_long_path = self.get_paths(arch)
        normal_paths, reduce_paths = paths
        num_ops = len(PRIMITIVES)
        """
        Compute the max number of paths per input per cell.
        Since there are two cells and two inputs per cell, 
        total paths = 4 * max_paths
        """
        if not long_paths:
            max_paths = 1 + sum([num_ops ** i for i in range(N_TOWERS)])
        else:
            max_paths = sum([num_ops ** i for i in range(N_TOWERS + 1)])
        path_indices = []

        # set the base index based on the cell and the input
        for i, paths in enumerate((normal_paths, reduce_paths)):
            for path in paths:
                index = i * 2 * max_paths
                if path[0] == INPUT_2:
                    index += max_paths

                # recursively compute the index of the path
                for j in range(N_TOWERS + 1):
                    if j == len(path) - 1:
                        path_indices.append(index)
                        break
                    elif j == (N_TOWERS - 1) and not long_paths:
                        path_indices.append(2 * (i + 1) * max_paths - 1)
                        break
                    else:
                        index += num_ops ** j * (PRIMITIVES.index(path[j + 1]) + 1)

        return tuple(path_indices), contains_long_path

    def encode_paths(self, arch, long_paths=True):
        # output one-hot encoding of paths
        path_indices, _ = self.get_path_indices(arch, long_paths=long_paths)
        num_ops = len(PRIMITIVES)

        if not long_paths:
            max_paths = 1 + sum([num_ops ** i for i in range(N_TOWERS)])
        else:
            max_paths = sum([num_ops ** i for i in range(N_TOWERS + 1)])

        path_encoding = np.zeros(4 * max_paths)
        for index in path_indices:
            path_encoding[index] = 1
        return path_encoding

    def encode_freq_paths(self, arch, cutoff=100, filepath='~/naszilla/darts/counts.npy'):
        # natural cutoffs 32, 288, 2336
        path_indices, _ = self.get_path_indices(arch, long_paths=True)
        counts = np.load(os.path.expanduser(filepath))
        sorted_indices = np.flip(np.argsort(counts))

        num_ops = len(PRIMITIVES)
        max_paths = sum([num_ops ** i for i in range(N_TOWERS + 1)])
        cutoff = min(4 * max_paths, cutoff)
        encoding = np.zeros(cutoff)

        for i, index in enumerate(sorted_indices[:cutoff]):
            if index in path_indices:
                encoding[i] = 1
        return encoding

    def path_distance(self, arch1, arch2):
        # compute the distance between two architectures
        # by comparing their path encodings
        return np.sum(np.array(self.encode_paths(arch1) != np.array(self.encode_paths(arch2))))

    def encode_adj(self, arch, flatten=True):
        if not isinstance(arch, Genotype):
            arch = self.to_genotype(arch)
        # extract the normal and reduce cells
        normal, reduce = arch.normal, arch.reduce
        # convert the normal_name and reduce_name to normal and reduce
        normal_code = [
            (int(i[1]), PRIMITIVES.index(i[0]))
            for i in normal
        ]
        reduce_code = [
            (int(i[1]), PRIMITIVES.index(i[0]))
            for i in reduce
        ]
        normal_ops, normal_adj = build_mat_encoding(normal_code, )
        reduce_ops, reduce_adj = build_mat_encoding(reduce_code, )
        if flatten:
            encode = np.concatenate([
                np.reshape(normal_ops, -1), np.reshape(normal_adj, -1),
                np.reshape(reduce_ops, -1), np.reshape(reduce_adj, -1)
            ])
            return encode
        return normal_adj, normal_ops, reduce_adj, reduce_ops

    def find_node_proximate_archs(self, arch, op_idx, ops_only=False, ):
        """
        A NASBench-301/DARTS cell has 8 operations (or 16 if we consider both normal and reduction cells. Given an op_idx
        (0-7 for normal or 8-15 for reduction), this def enumerates all the architectures induced by changing that
        edge either by changing the operation or the wiring.
        ops_only: only change the operation but not perturb the wiring
        """
        assert op_idx < 16, f'{op_idx} is out of range!'
        primitives_no_none = [p for p in PRIMITIVES if p != 'none']
        proximate_archs = []
        if not isinstance(arch, Genotype):
            arch = self.to_genotype(arch)
        # possible node ops
        if op_idx < 8:  # normal cell
            op = arch.normal[op_idx][0]
        else:  # reduce cell
            op = arch.reduce[op_idx - 8][0]
        # architectures that only have the cell op being different
        for option in primitives_no_none:
            if option == op:  # same as existing operation
                continue
            candidate = deepcopy(arch)
            if op_idx < 8:
                candidate.normal[op_idx] = (option, candidate.normal[op_idx][1])
            else:
                candidate.reduce[op_idx - 8] = (option, candidate.reduce[op_idx - 8][1])
            proximate_archs.append(candidate)

        # architectures that have the end connection being different
        if not ops_only:
            if op_idx < 8:  # normal cell
                location = op_idx // 2  # the op cell index of the current op in question.
                curent_loc = arch.normal[op_idx][1]
                if op_idx % 2:
                    other_location = arch.normal[op_idx - 1][1]
                else:
                    other_location = arch.normal[op_idx + 1][1]
            else:  # reduce cell
                location = (op_idx - 8) // 2  # the op cell index of the current op in question.
                curent_loc = arch.reduce[op_idx - 8][1]
                if op_idx % 2:
                    other_location = arch.reduce[op_idx - 8 - 1][1]
                else:
                    other_location = arch.reduce[op_idx - 8 + 1][1]
            for candidate_location in range(location + 2):
                if candidate_location == curent_loc or candidate_location == other_location:
                    continue
                candidate = deepcopy(arch)
                if op_idx < 8:
                    candidate.normal[op_idx] = (candidate.normal[op_idx][0], candidate_location)
                else:
                    candidate.reduce[op_idx - 8] = (candidate.reduce[op_idx - 8][0], candidate_location)
                proximate_archs.append(candidate)
        return proximate_archs

    def get_neighbours(self, arch, return_string=True, return_genotype=False, mutate_normal_only=True,
                       shuffle=True):
        """Get neighbours of an architecture.

        mutate_normal_only: only allow neighbours from the normal cell (reduce cell is frozen)
        """
        op_nbhd = []
        edge_nbhd = []
        if not isinstance(arch, Genotype):
            arch = self.to_genotype(arch)
        arch_compact = convert_genotype_to_compact(arch)
        if mutate_normal_only:  # only mutate the first cell (i.e. the normal cell)
            max_cell = 1
        else:
            max_cell = 2

        for i in range(max_cell):
            cell = arch_compact[i]
            for j, pair in enumerate(cell):
                # mutate the op
                available = [op for op in range(len(PRIMITIVES_NO_NONE)) if op != pair[1]]
                for op in available:
                    new_arch = deepcopy(make_compact_mutable(arch_compact))
                    new_arch[i][j][1] = op
                    op_nbhd.append({'spec': new_arch})

                # mutate the edge
                other = j + 1 - 2 * (j % 2)
                available = [edge for edge in range(j // 2 + 2)
                             if edge not in [cell[other][0], pair[0]]]

                for edge in available:
                    new_arch = deepcopy(make_compact_mutable(arch_compact))
                    new_arch[i][j][0] = edge
                    edge_nbhd.append({'spec': new_arch})
        if shuffle:
            random.shuffle(edge_nbhd)
            random.shuffle(op_nbhd)
        # 112 in edge nbhd, 24 in op nbhd
        # alternate one edge nbr per 4 op nbrs
        nbrs = []
        op_idx = 0
        for i in range(len(edge_nbhd)):
            nbrs.append(edge_nbhd[i])
            for j in range(4):
                nbrs.append(op_nbhd[op_idx])
                op_idx += 1
        nbrs = [*nbrs, *op_nbhd[op_idx:]]
        genotypes = [convert_compact_to_genotype(a['spec']) for a in nbrs]
        strings = [self.to_str(genotype) for genotype in genotypes]
        if return_string and not return_genotype:
            return strings
        if return_genotype and not return_string:
            return genotypes
        return strings, genotypes

    def to_networkx_edge(self, arch, return_reduction=True, disjoint_union=False,
                         numeric_features=False, no_concat=False,
                         with_more_info=True, identify_ops=False,
                         compute_importance_weight=False):
        """Convert an arch or genotype of DARTS into a pair of **edge-attributed** networkx graphs
        return_reduction: whether to return both normal and reduction cells
        disjoint_union: whether to return the two cells as a disjoint-unioned single graph instead of a tuple of 2 graphs
        numeric_features: whether to save the node/edge features in a numeric manner
        identify_ops: whether to identify the operation nodes with different op names (i.e. 2,3,4,5). If False, they
            will be given a generic name
        compute_importance_weight: whether to calculate the importance weight of each edge in the architecture based
            on the counterfactual explanation. Enabling this might lead to significant slowdown in computing time.
        """
        if not isinstance(arch, Genotype):
            genotype = self.to_genotype(arch)
        else:
            genotype = arch

        def _cell2graph_edge(cell, concat):
            G = nx.DiGraph()
            n_nodes = 6 if no_concat else 7
            G.add_nodes_from(range(n_nodes))
            G.nodes[0]['node_name'] = 0 if numeric_features else 'input1'
            G.nodes[1]['node_name'] = 1 if numeric_features else 'input2'

            for i in range(2, 2 + 4):
                if identify_ops:
                    if numeric_features:
                        G.nodes[i]['node_name'] = i
                    else:
                        G.nodes[i]['node_name'] = str(i - 2)
                else:
                    if numeric_features:
                        G.nodes[i]['node_name'] = 2
                    else:
                        G.nodes[i]['node_name'] = 'op'

            if not no_concat:  # no need to add the output node if no_concat is enabled
                G.nodes[6]['node_name'] = 6 if numeric_features else 'output'
            for i, (op, wiring) in enumerate(cell):
                block = i // 2 + 2
                if numeric_features:
                    G.add_edge(wiring, block, op_name=PRIMITIVES.index(op), edge_order=i)
                else:
                    G.add_edge(wiring, block, op_name=op, edge_order=i)
            if not no_concat:
                for node in concat:
                    if numeric_features:
                        G.add_edge(node, 6, op_name=len(PRIMITIVES) + 1, )
                    else:
                        G.add_edge(node, 6, op_name='')
            return G

        def add_more_info(g: nx.DiGraph, isNormal=True):
            """Annotate the graph node and edges with additional information that may be relevant for subsequent analysis"""
            for i, (node, data) in enumerate(g.nodes(data=True)):
                # print(node, data)
                if data['node_name'] in ['input1', 0]:  # input1
                    g.nodes[i]['dist_input1'] = 0
                    g.nodes[i]['dist_input2'] = np.nan
                elif data['node_name'] in ['input2', 1]:  # input2
                    g.nodes[i]['dist_input1'] = np.nan
                    g.nodes[i]['dist_input2'] = 0
                else:  # compute the shortest distance to both input nodes
                    try:
                        dist_input1 = len(nx.shortest_path(g, source=0, target=i)) - 1
                    except nx.NetworkXNoPath:
                        dist_input1 = np.nan
                    try:
                        dist_input2 = len(nx.shortest_path(g, source=1, target=i)) - 1
                    except nx.NetworkXNoPath:
                        dist_input2 = np.nan
                    g.nodes[i]['dist_input1'] = dist_input1
                    g.nodes[i]['dist_input2'] = dist_input2
                g.nodes[i]['in_degrees'] = g.in_degree(i)
                g.nodes[i]['out_degrees'] = g.out_degree(i)
                g.nodes[i]['depth'] = np.nanmin([g.nodes[i]['dist_input1'], g.nodes[i]['dist_input2']])
                g.nodes[i]['isNormal'] = isNormal
            return g

        def get_edge_importance(normal: nx.DiGraph, reduce: nx.DiGraph = None):
            sensitivities = np.array(
                [self.get_edge_perturbation(genotype, op, ops_only=False) for op in range(16)]).flatten()
            for ei, eo, data in normal.edges(data=True):
                if 'edge_order' in data.keys():
                    normal.edges()[(ei, eo)]['weight'] = sensitivities[data['edge_order']]
            if reduce is not None:
                for ei, eo, data in reduce.edges(data=True):
                    if 'edge_order' in data.keys():
                        reduce.edges()[(ei, eo)]['weight'] = sensitivities[data['edge_order'] + 8]
                return normal, reduce
            return normal

        G_normal = _cell2graph_edge(genotype.normal, genotype.normal_concat)
        if with_more_info:
            G_normal = add_more_info(G_normal)
        if return_reduction:
            G_reduce = _cell2graph_edge(genotype.reduce, genotype.reduce_concat)
            if with_more_info:
                G_reduce = add_more_info(G_reduce, isNormal=False)
            if disjoint_union:
                G = nx.disjoint_union(G_normal, G_reduce)
                return G
            if compute_importance_weight:
                G_normal, G_reduce = get_edge_importance(G_normal, G_reduce)
            return G_normal, G_reduce
        if compute_importance_weight:
            G_normal = get_edge_importance(G_normal)
        return G_normal

    def to_networkx(self, arch, return_reduction=True, remove_skip=True, **kwargs) -> [nx.DiGraph, nx.DiGraph]:
        """Convert an arch or genotype of a DARTS cell into a pair of networkx graphs -- to node attributed graph.
        Note: to convert into edge-attributed networkx graph, use to_networkx_edge() call above.
        """
        if not isinstance(arch, Genotype):
            genotype = self.to_genotype(arch)
        else:
            genotype = arch

        def _cell2graph(cell, concat):
            G = nx.DiGraph()
            n_nodes = (len(cell) // 2) * 3 + 3
            G.add_nodes_from(range(n_nodes), op_name=None)
            n_ops = len(cell) // 2
            G.nodes[0]['op_name'] = 'input1'
            G.nodes[1]['op_name'] = 'input2'
            G.nodes[n_nodes - 1]['op_name'] = 'output'
            for i in range(n_ops):
                G.nodes[i * 3 + 2]['op_name'] = cell[i * 2][0]
                G.nodes[i * 3 + 3]['op_name'] = cell[i * 2 + 1][0]
                G.nodes[i * 3 + 4]['op_name'] = 'add'
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
            if remove_skip:
                for j in range(n_nodes):
                    try:
                        G.nodes[j]
                    except KeyError:
                        continue
                    if G.nodes[j]['op_name'] == 'skip_connect':
                        in_edges = list(G.in_edges(j))
                        out_edge = list(G.out_edges(j))[0][1]  # There should be only one out edge really...
                        for in_edge in in_edges:
                            G.add_edge(in_edge[0], out_edge)
                        G.remove_node(j)
                    elif G.nodes[j]['op_name'] == 'none':
                        G.remove_node(j)
                for j in range(n_nodes):
                    try:
                        G.nodes[j]
                    except KeyError:
                        continue

                    if G.nodes[j]['op_name'] not in ['input1', 'input2']:
                        # excepting the input nodes, if the node has no incoming edge, remove it
                        if len(list(G.in_edges(j))) == 0:
                            G.remove_node(j)
                    elif G.nodes[j]['op_name'] != 'output':
                        # excepting the output nodes, if the node has no outgoing edge, remove it
                        if len(list(G.out_edges(j))) == 0:
                            G.remove_node(j)
                    elif G.nodes[j]['op_name'] == 'add':  # If add has one incoming edge only, remove the node
                        in_edges = list(G.in_edges(j))
                        out_edges = list(G.out_edges(j))
                        if len(in_edges) == 1 and len(out_edges) == 1:
                            G.add_edge(in_edges[0][0], out_edges[0][1])
                            G.remove_node(j)

            return G

        G_normal = _cell2graph(genotype.normal, genotype.normal_concat)
        try:
            G_reduce = _cell2graph(genotype.reduce, genotype.reduce_concat)
        except AttributeError:
            G_reduce = None
        if return_reduction and G_reduce is not None:
            return G_normal, G_reduce
        else:
            return G_normal, None

    @staticmethod
    def to_str(genotype: Genotype):
        """alias for genotype2str"""
        return genotype2str(genotype)

    @staticmethod
    def to_genotype(arch):
        """convert an arch representation to genotype"""
        if hasattr(arch, 'normal') and hasattr(arch, 'reduce') and hasattr(arch, 'normal_concat') and hasattr(arch,
                                                                                                              'reduce_concat'):
            return arch
        genotype_config = Genotype(
            normal=[(OPS_dict[item[0]], item[1]) for item in arch[0]],
            normal_concat=[2, 3, 4, 5],
            reduce=[(OPS_dict[item[0]], item[1]) for item in arch[1]],
            reduce_concat=[2, 3, 4, 5]
        )
        return genotype_config

    @staticmethod
    def get_arch_list(arch):
        # convert tuple to list so that it is mutable
        arch_list = []
        for cell in arch:
            arch_list.append([])
            for pair in cell:
                arch_list[-1].append([])
                for num in pair:
                    arch_list[-1][-1].append(num)
        return arch_list

    def mutate(self, arch, edits=1, return_string=True, return_genotype=False, mutate_normal_only=True,
               **kwargs):
        """Mutate a DARTS architecture"""
        # todo
        raise NotImplementedError

    def get_edge_perturbation(self, arch, op_idx, ops_only=False, return_info=False):
        """This computes the counterfactual change in NB301 prediction if op_idx edge is changed with another operation,
        or if the wiring is changed from the present.
        return_info: whether return additional information
        """
        if self.no_api:
            print('Unable to execute get_important_subgraphs as I am under no-API mode!')
            return
        if not isinstance(arch, Genotype):
            arch = self.to_genotype(arch)
        candidates = self.find_node_proximate_archs(arch, op_idx, ops_only=ops_only)
        ori_perf = self.query(arch, with_noise=False)
        if self.negative: ori_perf = -ori_perf
        if self.log_scale: ori_perf = np.exp(ori_perf)
        perf = np.array([self.query(c, with_noise=False) for c in candidates])
        if self.negative: perf = -perf
        if self.log_scale: perf = np.exp(perf)
        perf = perf - ori_perf
        return np.mean(perf)

    def get_important_subgraphs(self, arch, threshold=0.0015, ops_only=False,
                                return_weights=False,
                                by='more'):
        """Find important subgraphs/edges via counterfactuals.
        return a pair of networkx objects, which denote the important subgraphs in the normal and reduce cells.
        """
        assert by in ['more', 'less']
        assert threshold > 0, 'threshold should be a float magnitude larger than 0!'
        if self.no_api:
            print('Unable to execute get_important_subgraphs as I am under no-API mode!')
            return
        if not isinstance(arch, nx.DiGraph):
            if not isinstance(arch, Genotype):
                arch = self.to_genotype(arch)
            # convert to edge graphs in networkx format
            edge_graph_n, edge_graph_r = self.to_networkx_edge(arch, return_reduction=True, disjoint_union=False,
                                                               numeric_features=True, no_concat=True)
        else:  # in case the architecture passed is already the edge-attributed networkx graph, no further preprocessing required
            edge_graph_n, edge_graph_r = arch
        # negative to flip to the sign, such that the good important features get positive signs for both normal and reduce cells
        sensitivities = np.array([self.get_edge_perturbation(arch, op, ops_only=ops_only) for op in
                                  range(edge_graph_n.number_of_edges() + edge_graph_r.number_of_edges())]).flatten()

        def _get_important_subgraph(edge_graph, sensitivities):
            if by == 'more':
                important_edge_idx = np.argwhere(sensitivities >= threshold).flatten()
            else:
                important_edge_idx = np.argwhere(sensitivities <= -threshold).flatten()
            # bad_edge_idx = np.argwhere(sensitivities <= -min_perturb_mag).flatten()
            edges_to_remove = []
            for ei, eo, data in edge_graph.edges(data=True):
                if 'edge_order' in data.keys() and data['edge_order'] not in important_edge_idx:
                    # print(ei, eo)
                    edges_to_remove.append((ei, eo))
            edge_graph.remove_edges_from(edges_to_remove)
            edge_graph.remove_nodes_from(list(nx.isolates(edge_graph)))

            if return_weights:
                weights = sensitivities[important_edge_idx]
                return edge_graph, weights
            return edge_graph

        impt_subgraph_n = _get_important_subgraph(edge_graph_n, sensitivities[:8])
        impt_subgraph_r = _get_important_subgraph(edge_graph_r, sensitivities[8:])
        return impt_subgraph_n, impt_subgraph_r

    def sample_random_subgraph(self, arch, n_edges_n: int, n_edges_r: int = None, numeric_feature=False):
        """Sample a subgraph by randomly sample 'nedges' edges from the graph. Used as contrast for the important
        subgraphs obtained above

        This def partiions nedges into nedges_n and nedges_r uniformly, and then sample respective number of edges
        in both the normal and reduce cell
        """
        if n_edges_r is None:
            n_edges = n_edges_n
            if n_edges > 0:
                n_edges_n = np.random.randint(n_edges)
            else:
                n_edges_n = 0
            n_edges_r = n_edges - n_edges_n

        if not isinstance(arch, nx.DiGraph):
            if not isinstance(arch, Genotype):
                arch = self.to_genotype(arch)
            # convert to edge graphs in networkx format
            edge_graph_n, edge_graph_r = self.to_networkx_edge(arch, return_reduction=True, disjoint_union=False,
                                                               numeric_features=numeric_feature, no_concat=True)
        else:  # in case the architecture passed is already the edge-attributed networkx graph, no further preprocessing required
            edge_graph_n, edge_graph_r = arch

        def _form_subgraph(edge_graph, n_edges):
            all_edges = list(edge_graph.edges())
            edges_idx_to_retain = np.random.choice(len(all_edges), size=n_edges, replace=False)
            edges_to_remove = [edge for i, edge in enumerate(all_edges) if i not in edges_idx_to_retain]
            edge_graph.remove_edges_from(edges_to_remove)
            edge_graph.remove_nodes_from(list(nx.isolates(edge_graph)))
            return edge_graph

        sampled_subgraph_n = _form_subgraph(edge_graph_n, n_edges=n_edges_n) if n_edges_n > 0 else nx.DiGraph()
        sampled_subgraph_r = _form_subgraph(edge_graph_r, n_edges=n_edges_r) if n_edges_r > 0 else nx.DiGraph()
        # return empty graph for the else part
        return sampled_subgraph_n, sampled_subgraph_r

    def plot(self, arch, edge_weight=None, thres=None, plot_normal_only=False):
        """Visualise the cells via networkx. Note that this currently only supports the visualisation for the
         edge_attributed graphs."""
        if not isinstance(arch, Genotype):
            arch = self.to_genotype(arch)
        cell_normal, cell_reduce = self.to_networkx_edge(arch, return_reduction=True, disjoint_union=False)

        def plot_cell(cell, ew=None):

            pos = nx.spring_layout(cell)
            node_color = ['gray'] * len(cell)
            M = cell.number_of_edges()
            #     edge_colors = [] * M
            edge_alphas = np.array([0.1] * M)
            edge_colors = ['gray'] * M
            #     edge_alphas = [(5) / (M + 4) for i in range(M)]
            if ew is None:
                ew = np.array([1.] * M)
                scaled_edge_weights = np.array([1.] * M)
            else:
                edge_weight_abs = np.abs(ew)
                scaled_edge_weights = 0.1 + (edge_weight_abs - min(edge_weight_abs)) / (
                        max(edge_weight_abs) - min(edge_weight_abs)) * 0.9

            node_labels = {}
            for i, (node, data) in enumerate(cell.nodes(data=True)):
                if 'input' in str(data['node_name']):
                    node_color[i] = 'orange'
                elif 'output' in str(data['node_name']):
                    node_color[i] = 'lightblue'
                node_labels.update({node: data['node_name']})

            nx.draw_networkx_nodes(cell, pos, node_color=node_color)
            for i, (in_node, out_node, props) in enumerate(cell.edges(data=True)):
                if 'edge_order' in props:
                    idx = props['edge_order']
                    edge_alphas[i] = abs(float(scaled_edge_weights[idx]))
                    if thres is not None and abs(ew[idx]) < thres:
                        edge_colors[i] = 'gray'  # unimportant edges
                    elif ew[idx] > 0:
                        edge_colors[i] = 'green'
                    elif ew[idx] < 0:
                        edge_colors[i] = 'red'  # important edges, but in opposing directions
                    else:
                        edge_colors[i] = 'gray'
                    # print(in_node, out_node, props['op_name'], edge_weight[idx], scaled_graph_norms[idx])

            edges = nx.draw_networkx_edges(
                cell,
                pos,
                arrowstyle="->",
                arrowsize=10,
                edge_color=edge_colors,
                edge_cmap=plt.cm.Blues,
                width=2,
            )
            for in_node, out_node, props in cell.edges(data=True):
                if 'edge_order' in props:
                    i = props['edge_order']
                    edges[i].set_alpha(edge_alphas[i])

            nx.draw_networkx_labels(cell, pos, labels=node_labels)
            edge_labels = nx.get_edge_attributes(cell, 'op_name')
            try:
                edge_labels = {k: PLOTTING_SHORTHAND[v] for k, v in edge_labels.items()}
            except IndexError:
                edge_labels = {k: PLOTTING_SHORTHAND[PRIMITIVES[v]] for k, v in edge_labels.items()}

            nx.draw_networkx_edge_labels(cell, pos=pos, edge_labels=edge_labels)

        plt.figure(figsize=[4, 4])
        plot_cell(cell_normal, ew=edge_weight[:8] if edge_weight is not None else None)
        plt.show()
        if not plot_normal_only:
            plt.figure(figsize=[4, 4])
            plot_cell(cell_reduce, ew=edge_weight[8:] if edge_weight is not None else None)
            plt.show()


def filestr2genotypes(file_str: str):
    """Convert a 32 digit file string into 2 lists
    This format is commonly used as a path name to store a NB301 string."""
    normal, reduce = file_str[:len(file_str)//2], file_str[len(file_str)//2:]
    normal_list = [(int(normal[n]), int(normal[n+1])) for n in range(0, len(normal), 2)]
    reduce_list = [(int(reduce[n]), int(reduce[n+1])) for n in range(0, len(reduce), 2)]
    return [normal_list, reduce_list]


def genotype2str(genotype: Genotype, flatten=False):
    """Convert a genotype to string representation"""
    normal_cell = genotype.normal
    reduce_cell = genotype.reduce
    normal_list = [(PRIMITIVES.index(item[0]), item[1]) for item in normal_cell]
    reduce_list = [(PRIMITIVES.index(item[0]), item[1]) for item in reduce_cell]
    if flatten:
        normal_list = np.array(normal_list).flatten()
        reduce_list = np.array(reduce_list).flatten()
        id_list = np.array([normal_list, reduce_list]).flatten().astype(np.str).tolist()
        id_str = "".join(id_list)
        return id_str
    return [normal_list, reduce_list]


def convert_genotype_to_compact(genotype):
    """ Converts Genotype to the compact representation """
    compact = []

    for i, cell_type in enumerate(['normal', 'reduce']):
        cell = eval('genotype.' + cell_type)
        compact.append([])

        for j in range(N_TOWERS * 2):
            compact[i].append((cell[j][1], PRIMITIVES.index(cell[j][0])))

    compact_tuple = (tuple(compact[0]), tuple(compact[1]))
    return compact_tuple


def make_compact_mutable(compact):
    # convert tuple to list so that it is mutable
    arch_list = []
    for cell in compact:
        arch_list.append([])
        for pair in cell:
            arch_list[-1].append([])
            for num in pair:
                arch_list[-1][-1].append(num)
    return arch_list


def convert_compact_to_genotype(compact):
    """ Converts the compact representation to a Genotype """
    genotype = []

    for i in range(2):
        cell = compact[i]
        genotype.append([])

        for j in range(N_TOWERS * 2):
            genotype[i].append((PRIMITIVES[cell[j][1] - 1], cell[j][0]))

    return Genotype(
        normal=genotype[0],
        normal_concat=list(range(2, 2 + N_TOWERS)),
        reduce=genotype[1],
        reduce_concat=list(range(2, 2 + N_TOWERS)),
    )


def process(geno):
    for i, item in enumerate(geno):
        geno[i] = tuple(geno[i])
    return geno


def transform_operations(ops):
    transform_dict = {'c_k-2': 0, 'c_k-1': 1, 'max_pool_3x3': 2, 'avg_pool_3x3': 3, 'skip_connect': 4,
                      'sep_conv_3x3': 5, 'sep_conv_5x5': 6, 'dil_conv_3x3': 7, 'dil_conv_5x5': 8, 'output': 9}

    ops_array = np.zeros([11, 11], dtype='int8')
    for row, op in enumerate(ops):
        ops_array[row, op] = 1
    return ops_array


def build_mat_encoding(normal):
    adj = torch.zeros(11, 11)
    ops = torch.zeros(11, 11)
    block_0 = (normal[0], normal[1])
    prev_b0_n1, prev_b0_n2 = block_0[0][0], block_0[1][0]
    prev_b0_o1, prev_b0_o2 = block_0[0][1], block_0[1][1]

    block_1 = (normal[2], normal[3])
    prev_b1_n1, prev_b1_n2 = block_1[0][0], block_1[1][0]
    prev_b1_o1, prev_b1_o2 = block_1[0][1], block_1[1][1]

    block_2 = (normal[4], normal[5])
    prev_b2_n1, prev_b2_n2 = block_2[0][0], block_2[1][0]
    prev_b2_o1, prev_b2_o2 = block_2[0][1], block_2[1][1]

    block_3 = (normal[6], normal[7])
    prev_b3_n1, prev_b3_n2 = block_3[0][0], block_3[1][0]
    prev_b3_o1, prev_b3_o2 = block_3[0][1], block_3[1][1]

    adj[2][-1] = 1
    adj[3][-1] = 1
    adj[4][-1] = 1
    adj[5][-1] = 1
    adj[6][-1] = 1
    adj[7][-1] = 1
    adj[8][-1] = 1
    adj[9][-1] = 1

    # B0
    adj[prev_b0_n1][2] = 1
    adj[prev_b0_n2][3] = 1

    # B1
    if prev_b1_n1 == 2:
        adj[2][4] = 1
        adj[3][4] = 1
    else:
        adj[prev_b1_n1][4] = 1

    if prev_b1_n2 == 2:
        adj[2][5] = 1
        adj[3][5] = 1
    else:
        adj[prev_b1_n2][5] = 1

    # B2
    if prev_b2_n1 == 2:
        adj[2][6] = 1
        adj[3][6] = 1
    elif prev_b2_n1 == 3:
        adj[4][6] = 1
        adj[5][6] = 1
    else:
        adj[prev_b2_n1][6] = 1

    if prev_b2_n2 == 2:
        adj[2][7] = 1
        adj[3][7] = 1
    elif prev_b2_n2 == 3:
        adj[4][7] = 1
        adj[5][7] = 1
    else:
        adj[prev_b2_n2][7] = 1

    # B3
    if prev_b3_n1 == 2:
        adj[2][8] = 1
        adj[3][8] = 1
    elif prev_b3_n1 == 3:
        adj[4][8] = 1
        adj[5][8] = 1
    elif prev_b3_n1 == 4:
        adj[6][8] = 1
        adj[7][8] = 1
    else:
        adj[prev_b3_n1][8] = 1

    if prev_b3_n2 == 2:
        adj[2][9] = 1
        adj[3][9] = 1
    elif prev_b3_n2 == 3:
        adj[4][9] = 1
        adj[5][9] = 1
    elif prev_b3_n2 == 4:
        adj[6][9] = 1
        adj[7][9] = 1
    else:
        adj[prev_b3_n2][9] = 1

    ops[0][0] = 1
    ops[1][1] = 1
    ops[-1][-1] = 1
    ops[2][prev_b0_o1+2] = 1
    ops[3][prev_b0_o2+2] = 1
    ops[4][prev_b1_o1+2] = 1
    ops[5][prev_b1_o2+2] = 1
    ops[6][prev_b2_o1+2] = 1
    ops[7][prev_b2_o2+2] = 1
    ops[8][prev_b3_o1+2] = 1
    ops[9][prev_b3_o2+2] = 1
    return ops, adj